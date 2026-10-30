"""
MCPToolRegistry：全局唯一的 MCP 工具注册表。

旧实现的问题：每个子 Agent 每处理一次请求都新建一个 MultiServerMCPClient，并调用 client.get_tools()。
而 get_tools() 是"无状态"模式——langchain-mcp-adapters 的源码注释写着
"A new session will be created for each tool call"，也就是**每一次工具调用都会拉起一个新的 stdio 子进程**，
做完一次握手，用完就退出。延迟高，还会不断地创建、销毁进程和数据库连接。

新实现：
- FastAPI 启动时（lifespan）调用一次 start()：为每个 MCP Server 打开一个长连接 session，加载一次工具列表；
- 之后所有 Agent 调用 get_tools(白名单) 拿工具，复用同一个 session（同一个子进程、同一个数据库连接池）；
- 应用关闭时 stop() 关闭 session，子进程退出。

另外做了两件事：
- 用户相关工具的 user_id 参数从"模型可见的参数列表"里去掉，由 UserIdInjector 注入（见 interceptors.py）；
- 配置里相对路径的 cwd 按项目根目录解析，"python" 换成当前解释器。以前必须在项目根目录启动后端，
  否则 cwd="agent" 找不到目录；现在从哪里启动都可以。
"""

import copy
import json
import logging
import sys
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Iterable

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

from core.mcp.interceptors import USER_SCOPED_TOOLS, UserIdInjector

log = logging.getLogger(__name__)

AGENT_DIR = Path(__file__).resolve().parents[2]
PROJECT_ROOT = AGENT_DIR.parent
DEFAULT_CONFIG = AGENT_DIR / "config" / "mcp_servers.json"


def _hide_user_id(tool: BaseTool) -> BaseTool:
    """返回一个去掉 user_id 参数的工具副本：模型看到的工具定义里没有 user_id，自然也没法填别人的。"""
    schema = tool.args_schema
    if not isinstance(schema, dict) or "user_id" not in schema.get("properties", {}):
        return tool
    schema = copy.deepcopy(schema)
    schema["properties"].pop("user_id")
    if "user_id" in schema.get("required", []):
        schema["required"].remove("user_id")
    return tool.model_copy(update={"args_schema": schema})


class MCPToolRegistry:
    def __init__(self, config_path: str | Path = DEFAULT_CONFIG) -> None:
        self.config_path = Path(config_path)
        self._stack: AsyncExitStack | None = None
        self._tools: dict[str, BaseTool] = {}

    @property
    def started(self) -> bool:
        return self._stack is not None

    def _load_servers(self, env: dict | None = None) -> dict:
        servers = json.loads(self.config_path.read_text(encoding="utf-8")).get("mcpServers", {})
        for cfg in servers.values():
            if cfg.get("transport") == "stdio":
                if cfg.get("command") in ("python", "python3"):
                    cfg["command"] = sys.executable                 # 子进程用和后端同一个 conda 环境
                if cfg.get("cwd") and not Path(cfg["cwd"]).is_absolute():
                    cfg["cwd"] = str(PROJECT_ROOT / cfg["cwd"])
                if env:                                             # 测试时用来切换到 airline_test 库
                    cfg["env"] = {**cfg.get("env", {}), **env}
        return servers

    async def start(self, env: dict | None = None) -> None:
        if self.started:
            return
        servers = self._load_servers(env)
        client = MultiServerMCPClient(servers)
        stack = AsyncExitStack()
        tools: dict[str, BaseTool] = {}
        try:
            for name in servers:
                session = await stack.enter_async_context(client.session(name))   # 长连接，直到 stop()
                for t in await load_mcp_tools(session, server_name=name, tool_interceptors=[UserIdInjector()]):
                    tools[t.name] = _hide_user_id(t) if t.name in USER_SCOPED_TOOLS else t
        except BaseException:
            await stack.aclose()
            raise
        self._stack, self._tools = stack, tools
        log.info("MCP 已连接 %d 个 server，共 %d 个工具：%s", len(servers), len(tools), ", ".join(tools))

    async def stop(self) -> None:
        if self._stack is not None:
            await self._stack.aclose()
        self._stack, self._tools = None, {}

    def get_tools(self, names: Iterable[str] | None = None) -> list[BaseTool]:
        """按白名单取工具（每个 Agent 只拿自己需要的）；names=None 返回全部。未启动时返回空列表。"""
        if names is None:
            return list(self._tools.values())
        return [self._tools[n] for n in names if n in self._tools]

    def tool_names(self) -> list[str]:
        return list(self._tools)


# 全局单例：FastAPI lifespan 里 start()/stop()，各 Agent 直接 import 使用
mcp_registry = MCPToolRegistry()

# 兼容旧名字（core/__init__.py 等处 import 了 MCPManager）
MCPManager = MCPToolRegistry
