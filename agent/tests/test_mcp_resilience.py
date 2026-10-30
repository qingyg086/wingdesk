"""
MCP 连接的容错：连接类错误的识别、只读工具重连后重试、写工具不重试、调用超时；
以及真实场景——把 MCP Server 子进程杀掉后，工具调用和健康检查都能触发自动重连。
"""

import asyncio
import os
import signal
import subprocess
import time

import anyio
import pytest
from mcp.shared.exceptions import McpError
from mcp.types import ErrorData

from core.mcp.mcp_manager import (CONNECTION_CLOSED, MCPConnectionError, MCPToolRegistry, _SessionProxy,
                                  is_connection_error)

CLOSED = McpError(ErrorData(code=CONNECTION_CLOSED, message="Connection closed"))


# ------------------------------------------------------------------ 替身
class FakeSession:
    def __init__(self, behavior: str = "ok"):
        self.behavior, self.calls = behavior, []

    async def call_tool(self, name, arguments=None, **kwargs):
        self.calls.append(name)
        if self.behavior == "closed":
            raise CLOSED
        if self.behavior == "hang":
            await asyncio.sleep(10)
        if self.behavior == "business-error":
            raise ValueError("Error: reservation not found")
        return f"ok:{name}"


class FakeRegistry:
    """按顺序提供连接：请求重连后换下一个。"""
    call_timeout = 0.2

    def __init__(self, *sessions):
        self.sessions, self.restarted = list(sessions), []

    async def wait_session(self, name):
        return self.sessions[0]

    def current_session(self, name):
        return self.sessions[0]

    def request_restart(self, name, session, reason):
        self.restarted.append(session)
        self.sessions.pop(0)


# ------------------------------------------------------------------ 单元测试
def test_connection_errors_are_told_apart_from_business_errors():
    for exc in (CLOSED, anyio.ClosedResourceError(), anyio.BrokenResourceError(), BrokenPipeError(),
                ConnectionResetError(), asyncio.TimeoutError()):
        assert is_connection_error(exc), exc
    for exc in (ValueError("bad date"), McpError(ErrorData(code=-32602, message="Invalid params"))):
        assert not is_connection_error(exc), exc


@pytest.mark.asyncio
async def test_read_tool_is_retried_once_on_a_new_connection():
    dead, fresh = FakeSession("closed"), FakeSession()
    reg = FakeRegistry(dead, fresh)
    assert await _SessionProxy(reg, "airline").call_tool("search_direct_flight", {}) == "ok:search_direct_flight"
    assert reg.restarted == [dead] and fresh.calls == ["search_direct_flight"]


@pytest.mark.asyncio
async def test_write_tool_is_not_retried():
    """写操作可能已经在旧连接上执行成功，重试会做两次：只请求重连，把"结果未知"交给模型。"""
    dead, fresh = FakeSession("closed"), FakeSession()
    reg = FakeRegistry(dead, fresh)
    with pytest.raises(MCPConnectionError, match="执行结果未知"):
        await _SessionProxy(reg, "airline").call_tool("book_reservation", {})
    assert reg.restarted == [dead] and fresh.calls == []


@pytest.mark.asyncio
async def test_hung_call_times_out_and_reconnects():
    hung, fresh = FakeSession("hang"), FakeSession()
    reg = FakeRegistry(hung, fresh)
    t0 = time.perf_counter()
    assert await _SessionProxy(reg, "airline").call_tool("get_reservation_details", {}) == "ok:get_reservation_details"
    assert time.perf_counter() - t0 < 2 and reg.restarted == [hung]


@pytest.mark.asyncio
async def test_business_errors_do_not_trigger_reconnect():
    s = FakeSession("business-error")
    reg = FakeRegistry(s)
    with pytest.raises(ValueError):
        await _SessionProxy(reg, "airline").call_tool("get_reservation_details", {})
    assert reg.restarted == []


@pytest.mark.asyncio
async def test_two_read_failures_give_up_with_a_clear_message():
    reg = FakeRegistry(FakeSession("closed"), FakeSession("closed"), FakeSession())
    with pytest.raises(MCPConnectionError, match="暂时不可用"):
        await _SessionProxy(reg, "airline").call_tool("search_direct_flight", {})


# ------------------------------------------------------------------ 真实子进程
def server_pid() -> int | None:
    out = subprocess.run(["pgrep", "-P", str(os.getpid()), "-f", "mcp_servers.airline_server"],
                         capture_output=True, text=True).stdout.split()
    return int(out[0]) if out else None


async def wait_until(cond, timeout=20.0):
    t0 = time.monotonic()
    while not cond():
        if time.monotonic() - t0 > timeout:
            return False
        await asyncio.sleep(0.1)
    return True


ARGS = {"origin": "JFK", "destination": "SEA", "date": "2024-05-20"}


def texts(result) -> list[str]:
    """工具返回的内容块带随机 id，只比较文本。"""
    return [b["text"] for b in result]


@pytest.mark.asyncio
async def test_tool_call_recovers_after_server_process_is_killed(_imported_once):
    reg = MCPToolRegistry(call_timeout=5, ping_interval=3600)        # 关掉定时 ping，只靠调用失败触发重连
    await reg.start(env={"MYSQL_DATABASE": "airline_test"})
    try:
        tool = reg.get_tools(["search_direct_flight"])[0]
        first = await tool.ainvoke(ARGS)
        pid = server_pid()
        assert pid, "没找到 MCP Server 子进程"
        os.kill(pid, signal.SIGKILL)
        second = await tool.ainvoke(ARGS)                            # 同一个工具对象，Agent 图不用重建
        assert texts(second) == texts(first) and "HAT069" in texts(first)[0]
        assert reg.stats["airline"]["restarts"] == 1 and server_pid() not in (None, pid)
        assert reg.health()["healthy"]
    finally:
        await reg.stop()
    assert server_pid() is None                                      # stop() 之后子进程退出


@pytest.mark.asyncio
async def test_health_check_reconnects_without_any_tool_call(_imported_once):
    reg = MCPToolRegistry(call_timeout=5, ping_interval=0.3)
    await reg.start(env={"MYSQL_DATABASE": "airline_test"})
    try:
        pid = server_pid()
        os.kill(pid, signal.SIGKILL)
        assert await wait_until(lambda: reg.stats["airline"]["restarts"] >= 1 and reg.health()["healthy"])
        assert server_pid() not in (None, pid)
        assert await wait_until(lambda: reg.stats["airline"]["last_ping_ms"] is not None)   # 新连接 ping 正常
        assert await reg.get_tools(["search_direct_flight"])[0].ainvoke(ARGS)
    finally:
        await reg.stop()


@pytest.mark.asyncio
async def test_health_check_endpoint_detects_failure_immediately(_imported_once):
    """/api/health 调用 check()：主动 ping，杀掉子进程后立刻报告不健康，而不是等下一次定时 ping。"""
    reg = MCPToolRegistry(call_timeout=5, ping_interval=3600)
    await reg.start(env={"MYSQL_DATABASE": "airline_test"})
    try:
        assert (await reg.check())["healthy"]
        os.kill(server_pid(), signal.SIGKILL)
        await asyncio.sleep(0.2)
        assert not (await reg.check())["healthy"]
        assert await wait_until(lambda: reg.stats["airline"]["restarts"] >= 1 and reg.health()["healthy"])
    finally:
        await reg.stop()

