"""
评测用的 litellm 适配层（τ-bench 的用户模拟器和单 Agent 基线都通过 litellm 调模型）。

做三件事：
1. 让 litellm 走百炼的 OpenAI 兼容接口：provider 用 "openai"，模型名 qwen-plus，地址和 Key 通过环境变量给 litellm；
2. 清洗发给模型的消息：τ-bench 把 litellm 返回的 message.model_dump() 原样放回对话历史，里面有
   function_call / provider_specific_fields 等多余字段，百炼的兼容接口不一定接受，这里只保留标准字段；
3. 统计 token：按"谁在调用"（agent = 单 Agent 基线，user = 用户模拟器）分别累计，用来估算花费。
   τ-bench 自带的花费统计依赖 litellm 的价格表，里面没有百炼模型，算出来是 0，所以自己数。
"""

import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")   # 不去网上拉价格表
os.environ.setdefault("LITELLM_LOG", "ERROR")

from dotenv import load_dotenv  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / "agent" / ".env")

DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
PROVIDER = "openai"                 # litellm 的 provider：OpenAI 兼容协议

USAGE: dict[str, dict[str, int]] = defaultdict(lambda: {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0})

_KEEP = ("role", "content", "tool_calls", "tool_call_id", "name")


def configure_litellm() -> None:
    """把百炼的地址和 Key 交给 litellm（OpenAI 兼容方式）。"""
    os.environ["OPENAI_API_KEY"] = os.getenv("DASHSCOPE_API_KEY", "")
    os.environ["OPENAI_API_BASE"] = os.getenv("BASE_URL", DEFAULT_BASE_URL)
    os.environ["OPENAI_BASE_URL"] = os.environ["OPENAI_API_BASE"]
    import litellm
    litellm.suppress_debug_info = True


def _clean_tool_calls(tool_calls):
    out = []
    for tc in tool_calls or []:
        fn = tc.get("function") or {}
        out.append({"id": tc.get("id"), "type": "function",
                    "function": {"name": fn.get("name"), "arguments": fn.get("arguments") or "{}"}})
    return out


def clean_messages(messages: list[dict]) -> list[dict]:
    cleaned = []
    for m in messages:
        m = dict(m) if isinstance(m, dict) else m.model_dump()
        c = {k: m[k] for k in _KEEP if m.get(k) is not None}
        if c.get("tool_calls"):
            c["tool_calls"] = _clean_tool_calls(c["tool_calls"])
        else:
            c.pop("tool_calls", None)
        if c.get("role") != "tool":
            c.pop("name", None)            # name 只在 tool 消息里有意义
        if c.get("content") is None and not c.get("tool_calls"):
            c["content"] = ""
        cleaned.append(c)
    return cleaned


def make_completion(who: str):
    """返回一个替代 litellm.completion 的函数：清洗消息 + 记账。"""
    import litellm

    def completion(*args, messages, **kwargs):
        # litellm 1.106 遇到带 tools 的请求会去导入它的 MCP 网关模块，该模块要求更新版本的 mcp 包，
        # 和项目里 langchain-mcp-adapters 用的 mcp 1.x 冲突（ImportError）。项目用不到这个网关，跳过它。
        kwargs.setdefault("_skip_mcp_handler", True)
        res = litellm.completion(*args, messages=clean_messages(messages), **kwargs)
        u = USAGE[who]
        u["calls"] += 1
        usage = getattr(res, "usage", None)
        if usage is not None:
            u["prompt_tokens"] += int(getattr(usage, "prompt_tokens", 0) or 0)
            u["completion_tokens"] += int(getattr(usage, "completion_tokens", 0) or 0)
        return res
    return completion


def install() -> None:
    """替换 τ-bench 模块里引用的 completion（它们是 from litellm import completion，按模块属性查找）。"""
    configure_litellm()
    import tau_bench.agents.tool_calling_agent as tca
    import tau_bench.envs.user as user_mod
    tca.completion = make_completion("agent")
    user_mod.completion = make_completion("user")


def snapshot() -> dict:
    return {k: dict(v) for k, v in USAGE.items()}


def diff(before: dict, after: dict) -> dict:
    out = {}
    for who, v in after.items():
        b = before.get(who, {})
        out[who] = {k: v[k] - b.get(k, 0) for k in v}
    return out
