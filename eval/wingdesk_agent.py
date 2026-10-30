"""
把翼枢的多 Agent 图接到 τ-bench 上：实现 τ-bench 的 Agent 接口（solve）。

一道题的流程：
    env.reset(task) → 模拟用户说第一句话
    循环：
      用户的话 → run_turn(多 Agent 图) → 图里的工具调用都经 TauBenchBackend 变成 env.step(工具动作)
               → 本轮最终回复 → env.step(respond 动作) → 模拟用户的下一句话
      直到：模拟用户说 ###STOP###（任务完成），或 Agent 调了 transfer_to_human_agents（转人工，本题结束），
            或动作数达到上限
    结束时 τ-bench 比对数据库最终状态的哈希、检查回复里有没有该说的信息，给出 reward（1 或 0）。

评测模式和产品模式的差异（都在 mode="eval" 里处理）：
- 工具：τ-bench 原版 14 个工具（含 user_id 参数），不经过 MCP / MySQL，也没有 UserIdInjector；
- 确认：没有 interrupt 确认卡片，Agent 必须在对话里列出详情、得到 yes 再执行写操作（τ-bench 政策原文）；
- 不用 Neo4j 航线工具、不用 RAG（政策直接注入提示词），不读写缓存和记忆——只评 Agent 本身。
"""

import asyncio
import sys
import time
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "agent") not in sys.path:
    sys.path.insert(0, str(ROOT / "agent"))

from langchain_core.callbacks import UsageMetadataCallbackHandler      # noqa: E402
from langchain_core.messages import convert_to_openai_messages         # noqa: E402

from backends.taubench_backend import TauBenchBackend                  # noqa: E402
from core.workflow.graph_manager import AgentGraphManager             # noqa: E402
from core.workflow.runner import make_config, run_turn                 # noqa: E402
from tools.taubench_tools import build_taubench_tools                  # noqa: E402

EVAL_USER = "eval"            # checkpointer 的 thread_id 前缀；评测模式下工具不读它


class WingDeskAgent:
    """接口与 tau_bench.agents.base.Agent 一致（solve），另提供异步版本 asolve。"""

    def __init__(self, llm: Any, router_llm: Any = None):
        self.llm = llm
        self.router_llm = router_llm

    def solve(self, env, task_index: Optional[int] = None, max_num_steps: int = 30) -> dict:
        return asyncio.run(self.asolve(env, task_index, max_num_steps))

    async def asolve(self, env, task_index: Optional[int] = None, max_num_steps: int = 30) -> dict:
        from tau_bench.types import RESPOND_ACTION_NAME, Action

        backend = TauBenchBackend(env)
        tools = build_taubench_tools(backend, env.tools_info)
        graph = AgentGraphManager(tools=tools, llm=self.llm, router_llm=self.router_llm,
                                  mode="eval", local_tools={}).build_graph()
        usage_cb = UsageMetadataCallbackHandler()
        session = f"task{task_index}-{time.time_ns()}"

        reset = await asyncio.to_thread(env.reset, task_index)
        obs = reset.observation
        reward, info, stop_reason = 0.0, reset.info.model_dump(), "max_steps"
        turns: list[dict] = []

        while len(env.actions) < max_num_steps:
            turn = await run_turn(graph, EVAL_USER, session, obs, callbacks=[usage_cb])
            turns.append({"user": obs, "route": turn.route,
                          "tools": [c["name"] for c in turn.tool_calls], "agent": turn.text})
            if backend.done:                                   # 本轮里调用了 transfer_to_human_agents
                reward = backend.last_response.reward
                info = {**info, **backend.last_response.info.model_dump()}
                stop_reason = "transfer"
                break
            resp = await asyncio.to_thread(
                env.step, Action(name=RESPOND_ACTION_NAME, kwargs={"content": turn.text or "(no response)"}))
            obs = resp.observation
            if resp.done:                                      # 模拟用户说了 ###STOP###
                reward = resp.reward
                info = {**info, **resp.info.model_dump()}
                stop_reason = "user_stop"
                break

        state = await graph.aget_state(make_config(EVAL_USER, session))
        messages = convert_to_openai_messages(state.values.get("messages", []))
        return {"reward": reward, "info": info, "messages": messages, "turns": turns,
                "stop_reason": stop_reason, "usage": dict(usage_cb.usage_metadata)}
