"""
MCPToolRegistry：全局唯一的 MCP 工具注册表，负责 MCP Server 子进程的连接、健康检查和自动重连。

为什么用长连接：langchain-mcp-adapters 的 client.get_tools() 是无状态模式，源码注释写着
"A new session will be created for each tool call"——每一次工具调用都会拉起一个新的 stdio 子进程、重新握手。
这里在应用启动时为每个 MCP Server 打开一个长连接 session，所有 Agent 复用同一个子进程和数据库连接池。

连接的生命周期：
- 每个 server 由一个常驻的 supervisor 任务持有连接，`async with client.session(name)` 的进入和退出都在这个任务里。
  stdio 客户端内部用 anyio 的 task group / cancel scope，进入和退出必须在同一个任务里，否则会报
  "Attempted to exit cancel scope in a different task"。所以别的任务只能"请求重连"，由 supervisor 自己关旧连接、开新连接。
- 子进程崩溃导致连接异常退出时，supervisor 按指数退避（1 秒起，最长 30 秒）自动重建。
- 健康检查：每 30 秒 ping 一次（5 秒超时）。子进程卡死时连接不会抛异常，只能靠 ping 发现；ping 失败就请求重连。
- 工具不直接持有 ClientSession，而是持有 _SessionProxy，调用时才取当前的 session。
  重连后，已经建好的 Agent 图和工具对象都不用重建。

工具调用失败时：
- 每次调用有总超时（默认 30 秒）；
- 连接类错误（连接关闭、管道断开、超时）会触发重连。只读工具在新连接上自动重试一次；写工具不重试——
  请求可能已经在旧连接上执行成功，重试可能把同一个操作做两次，所以把"结果未知"告诉模型，由它先查询确认。

另外：
- 用户相关工具的 user_id 参数从"模型可见的参数列表"里去掉，由 UserIdInjector 注入（见 interceptors.py）；
- 配置里相对路径的 cwd 按项目根目录解析，"python" 换成当前解释器，从哪个目录启动后端都可以。
"""

import asyncio
import copy
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import anyio
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp.shared.exceptions import McpError

from backends.base import WRITE_TOOLS
from core import clock
from core.mcp.interceptors import USER_SCOPED_TOOLS, UserIdInjector

log = logging.getLogger(__name__)

AGENT_DIR = Path(__file__).resolve().parents[2]
PROJECT_ROOT = AGENT_DIR.parent
DEFAULT_CONFIG = AGENT_DIR / "config" / "mcp_servers.json"

CONNECTION_CLOSED = -32000          # MCP SDK 在连接断开时给未完成请求返回的错误码（mcp.types.CONNECTION_CLOSED）
CALL_TIMEOUT = 30.0                 # 单次工具调用的总超时（秒）
CONNECT_TIMEOUT = 20.0              # 等待连接就绪的最长时间（秒）
PING_INTERVAL, PING_TIMEOUT = 30.0, 5.0
MAX_BACKOFF = 30.0


class MCPConnectionError(RuntimeError):
    """工具服务连接不可用。错误文本会作为工具结果交给模型，所以写成模型能据此行动的说明。"""


def is_connection_error(exc: BaseException) -> bool:
    """是否是"连接出了问题"（需要重连），而不是工具本身返回的业务错误。"""
    if isinstance(exc, (anyio.ClosedResourceError, anyio.BrokenResourceError, anyio.EndOfStream,
                        ConnectionError, TimeoutError, asyncio.TimeoutError)):
        return True
    return isinstance(exc, McpError) and getattr(exc.error, "code", None) == CONNECTION_CLOSED


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


class _SessionProxy:
    """工具里保存的"session"其实是它：call_tool 时取当前存活的连接，连接类错误时请求重连并按规则重试。
    list_tools 等其他方法原样转发给当前连接。"""

    def __init__(self, registry: "MCPToolRegistry", server: str) -> None:
        self._registry, self._server = registry, server

    def __getattr__(self, item: str) -> Any:
        session = self._registry.current_session(self._server)
        if session is None:
            raise MCPConnectionError(f"MCP server {self._server} 未连接")
        return getattr(session, item)

    async def call_tool(self, name: str, arguments: dict | None = None, *args, **kwargs):
        retryable = name not in WRITE_TOOLS
        for attempt in (1, 2):
            session = await self._registry.wait_session(self._server)
            try:
                return await asyncio.wait_for(session.call_tool(name, arguments, *args, **kwargs),
                                              self._registry.call_timeout)
            except Exception as exc:
                if not is_connection_error(exc):
                    raise
                self._registry.request_restart(self._server, session, f"{name} 调用失败：{type(exc).__name__}")
                if not retryable:
                    raise MCPConnectionError(
                        f"工具服务连接中断，{name} 的执行结果未知。请先查询订单确认是否已经生效，不要直接重试。") from exc
                if attempt == 2:
                    raise MCPConnectionError(f"工具服务暂时不可用（{type(exc).__name__}），请稍后再试。") from exc
                log.warning("只读工具 %s 因连接问题失败，重连后重试一次", name)


class MCPToolRegistry:
    def __init__(self, config_path: str | Path = DEFAULT_CONFIG, *, call_timeout: float = CALL_TIMEOUT,
                 ping_interval: float = PING_INTERVAL) -> None:
        self.config_path = Path(config_path)
        self.call_timeout, self.ping_interval = call_timeout, ping_interval
        self._client: MultiServerMCPClient | None = None
        self._servers: dict = {}
        self._tools: dict[str, BaseTool] = {}
        self._sessions: dict[str, Any] = {}
        self._ready: dict[str, asyncio.Event] = {}
        self._restart: dict[str, asyncio.Event] = {}
        self._tasks: list[asyncio.Task] = []
        self._stopping = False
        self.stats: dict[str, dict] = {}

    @property
    def started(self) -> bool:
        return bool(self._tasks)

    def _load_servers(self, env: dict | None = None) -> dict:
        servers = json.loads(self.config_path.read_text(encoding="utf-8")).get("mcpServers", {})
        for cfg in servers.values():
            if cfg.get("transport") == "stdio":
                if cfg.get("command") in ("python", "python3"):
                    cfg["command"] = sys.executable                 # 子进程用和后端同一个 conda 环境
                if cfg.get("cwd") and not Path(cfg["cwd"]).is_absolute():
                    cfg["cwd"] = str(PROJECT_ROOT / cfg["cwd"])
                # 子进程的环境：时钟模式与父进程保持一致（测试强制 fixed 时，MCP Server 也必须是 fixed）；
                # env 参数在测试时用来切换到 airline_test 库
                cfg["env"] = {clock.ENV_KEY: clock.mode(), **cfg.get("env", {}), **(env or {})}
        return servers

    # ------------------------------------------------------------ 启停
    async def start(self, env: dict | None = None) -> None:
        if self.started:
            return
        self._stopping = False
        self._servers = self._load_servers(env)
        self._client = MultiServerMCPClient(self._servers)
        for name in self._servers:
            self._ready[name], self._restart[name] = asyncio.Event(), asyncio.Event()
            self._sessions[name] = None
            self.stats[name] = {"connected": False, "restarts": 0, "last_error": None, "last_ping_ms": None}
            self._tasks.append(asyncio.create_task(self._supervise(name), name=f"mcp-supervisor-{name}"))
        try:
            tools: dict[str, BaseTool] = {}
            for name in self._servers:
                try:
                    await asyncio.wait_for(self._ready[name].wait(), CONNECT_TIMEOUT)
                except asyncio.TimeoutError:
                    raise RuntimeError(f"MCP server {name} 启动失败：{self.stats[name]['last_error']}") from None
                for t in await load_mcp_tools(_SessionProxy(self, name), server_name=name,
                                              tool_interceptors=[UserIdInjector()]):
                    tools[t.name] = _hide_user_id(t) if t.name in USER_SCOPED_TOOLS else t
        except BaseException:
            await self.stop()
            raise
        self._tools = tools
        self._tasks.append(asyncio.create_task(self._health_loop(), name="mcp-health"))
        log.info("MCP 已连接 %d 个 server，共 %d 个工具：%s", len(self._servers), len(tools), ", ".join(tools))

    async def stop(self) -> None:
        self._stopping = True
        for ev in self._restart.values():
            ev.set()                                   # 让 supervisor 在自己的任务里关闭连接
        tasks, self._tasks = self._tasks, []
        for t in tasks:
            if t.get_name() == "mcp-health":
                t.cancel()
        if tasks:
            _, pending = await asyncio.wait(tasks, timeout=10)
            for t in pending:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tools, self._sessions = {}, {}

    # ------------------------------------------------------------ 连接管理
    async def _supervise(self, name: str) -> None:
        backoff, connected_before = 1.0, False
        while not self._stopping:
            self._restart[name].clear()
            try:
                async with self._client.session(name) as session:
                    self._sessions[name] = session
                    st = self.stats[name]
                    st["connected"] = True
                    if connected_before:
                        st["restarts"] += 1
                        log.warning("MCP server %s 已重新连接（累计 %d 次）", name, st["restarts"])
                    connected_before, backoff = True, 1.0
                    self._ready[name].set()
                    await self._restart[name].wait()
            except asyncio.CancelledError:
                raise
            except Exception as exc:                   # 子进程崩溃、握手失败、管道断开……
                self.stats[name]["last_error"] = f"{type(exc).__name__}: {exc}"[:300]
                log.warning("MCP server %s 连接异常：%s", name, self.stats[name]["last_error"])
            finally:
                self._ready[name].clear()
                self._sessions[name] = None
                self.stats[name]["connected"] = False
            if self._stopping:
                break
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, MAX_BACKOFF)

    def request_restart(self, name: str, session: Any, reason: str) -> None:
        """请求重连。只有出问题的正是当前连接时才生效，避免几个并发失败的调用把新连接也重启掉。"""
        if self._sessions.get(name) is not session or self._stopping:
            return
        log.warning("请求重连 MCP server %s：%s", name, reason)
        self.stats[name]["last_error"] = reason
        self.stats[name]["connected"] = False
        self._ready[name].clear()
        self._restart[name].set()

    def current_session(self, name: str):
        return self._sessions.get(name)

    async def wait_session(self, name: str, timeout: float = CONNECT_TIMEOUT):
        if name not in self._ready or self._stopping:
            raise MCPConnectionError("工具服务未启动")
        try:
            await asyncio.wait_for(self._ready[name].wait(), timeout)
        except asyncio.TimeoutError:
            raise MCPConnectionError("工具服务正在重新连接，请稍后再试。") from None
        return self._sessions[name]

    async def _ping(self, name: str, timeout: float) -> None:
        session = self._sessions.get(name)
        if session is None:
            return
        t0 = time.perf_counter()
        try:
            await asyncio.wait_for(session.send_ping(), timeout)
            self.stats[name]["last_ping_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        except Exception as exc:
            self.request_restart(name, session, f"健康检查 ping 失败：{type(exc).__name__}")

    async def _health_loop(self) -> None:
        while not self._stopping:
            await asyncio.sleep(self.ping_interval)
            for name in self._servers:
                await self._ping(name, PING_TIMEOUT)

    async def check(self, timeout: float = 2.0) -> dict:
        """主动检查：立刻 ping 每个连接再返回状态（/api/health 用）。只读缓存的状态会滞后到下一次定时 ping。"""
        await asyncio.gather(*(self._ping(name, timeout) for name in self._servers))
        return self.health()

    def health(self) -> dict:
        servers = {n: dict(st) for n, st in self.stats.items()}
        return {"healthy": self.started and all(s["connected"] for s in servers.values()),
                "tools": len(self._tools), "servers": servers}

    # ------------------------------------------------------------ 工具
    def get_tools(self, names: Iterable[str] | None = None) -> list[BaseTool]:
        """按白名单取工具（每个 Agent 只拿自己需要的）；names=None 返回全部。未启动时返回空列表。"""
        if names is None:
            return list(self._tools.values())
        return [self._tools[n] for n in names if n in self._tools]

    def tool_names(self) -> list[str]:
        return list(self._tools)


# 全局单例：FastAPI lifespan 里 start()/stop()，各 Agent 直接 import 使用
mcp_registry = MCPToolRegistry()

# 别名：core/__init__.py 等处以 MCPManager 的名字导入
MCPManager = MCPToolRegistry
