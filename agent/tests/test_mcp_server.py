"""
MCP 层集成测试：真实拉起 airline_server 子进程（连 airline_test 库），通过 LangGraph 的 ToolNode 调用工具，
验证：工具清单、user_id 对模型不可见、强制注入当前用户、未登录拒绝执行、订单越权拦截、全程只有一个子进程。
"""

import asyncio
import os
import subprocess

import pytest
from langchain_core.messages import AIMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from backends.base import ALL_TOOLS
from core.mcp.interceptors import USER_SCOPED_TOOLS
from core.mcp.mcp_manager import MCPToolRegistry

MIA, LUCAS = "mia_li_3668", "lucas_wilson_8118"
MIA_RES, LUCAS_RES = "NO6JO3", "I6XC2H"


class Harness:
    """启动注册表 + 一个只有 ToolNode 的最小图（和 create_react_agent 里执行工具的方式相同）。"""

    async def __aenter__(self):
        self.reg = MCPToolRegistry()
        await self.reg.start(env={"MYSQL_DATABASE": "airline_test"})
        g = StateGraph(MessagesState)
        g.add_node("tools", ToolNode(self.reg.get_tools()))
        g.add_edge(START, "tools")
        g.add_edge("tools", END)
        self.graph = g.compile()
        return self

    async def __aexit__(self, *exc):
        await self.reg.stop()

    async def call(self, name, args=None, user=None):
        msg = AIMessage(content="", tool_calls=[{"name": name, "args": args or {}, "id": "call_1"}])
        cfg = {"configurable": {"user_id": user}} if user else {}
        out = (await self.graph.ainvoke({"messages": [msg]}, config=cfg))["messages"][-1]
        content = out.content if isinstance(out.content, str) else "".join(b.get("text", "") for b in out.content)
        return content, out.status


def mcp_children() -> int:
    """当前进程拉起的 airline_server 子进程个数（macOS / Linux 的 ps 都支持这个写法）。"""
    out = subprocess.run(["ps", "-A", "-o", "ppid=,command="], capture_output=True, text=True).stdout
    me = str(os.getpid())
    return sum(1 for line in out.splitlines()
               if line.split(None, 1)[0] == me and "mcp_servers.airline_server" in line)


@pytest.fixture()
def fresh_db(rw):
    """借用 rw 夹具：每个测试前重新导入 airline_test。"""
    return rw


@pytest.mark.asyncio
async def test_tools_listed_and_user_id_hidden(fresh_db):
    async with Harness() as h:
        assert sorted(h.reg.tool_names()) == sorted(ALL_TOOLS)
        for tool in h.reg.get_tools():
            params = convert_to_openai_tool(tool)["function"]["parameters"].get("properties", {})
            if tool.name in USER_SCOPED_TOOLS:
                assert "user_id" not in params, tool.name           # 模型看不到 user_id
        assert "reservation_id" in convert_to_openai_tool(h.reg.get_tools(["cancel_reservation"])[0])[
            "function"]["parameters"]["properties"]


@pytest.mark.asyncio
async def test_user_id_injected_and_overridden(fresh_db):
    async with Harness() as h:
        text, _ = await h.call("get_user_details", {}, user=MIA)
        assert '"first_name": "Mia"' in text
        # 模型硬塞别人的 user_id：被拦截器改回当前用户
        text, _ = await h.call("get_user_details", {"user_id": LUCAS}, user=MIA)
        assert '"first_name": "Mia"' in text and "Lucas" not in text


@pytest.mark.asyncio
async def test_fail_closed_without_login(fresh_db):
    async with Harness() as h:
        text, status = await h.call("get_user_details", {}, user=None)
        assert status == "error" and "未登录" in text
        text, _ = await h.call("list_all_airports", {}, user=None)        # 与用户无关的工具不受影响
        assert '"JFK": "New York"' in text


@pytest.mark.asyncio
async def test_reservation_ownership(fresh_db):
    async with Harness() as h:
        text, _ = await h.call("get_reservation_details", {"reservation_id": MIA_RES}, user=MIA)
        assert f'"reservation_id": "{MIA_RES}"' in text
        # 别人的订单：返回和"订单不存在"完全相同的文本
        text, _ = await h.call("get_reservation_details", {"reservation_id": LUCAS_RES}, user=MIA)
        assert text == "Error: user not found"
        text, _ = await h.call("cancel_reservation", {"reservation_id": LUCAS_RES}, user=MIA)
        assert text == "Error: reservation not found"
        text, _ = await h.call("get_reservation_details", {"reservation_id": LUCAS_RES}, user=LUCAS)
        assert '"status": "cancelled"' not in text                        # 别人的订单没有被取消


@pytest.mark.asyncio
async def test_single_subprocess_for_many_calls(fresh_db):
    """30 次并发调用之后，仍然只有 1 个 MCP 子进程（无状态模式下每次调用都会拉起一个新的子进程）。"""
    async with Harness() as h:
        await asyncio.gather(*[h.call("search_direct_flight",
                                      {"origin": "JFK", "destination": "SEA", "date": "2024-05-20"}, user=MIA)
                               for _ in range(30)])
        assert mcp_children() == 1
    await asyncio.sleep(0.5)
    assert mcp_children() == 0                                            # stop() 之后子进程退出


@pytest.mark.asyncio
async def test_write_tool_through_mcp(fresh_db):
    """写工具走完整链路：发代金券给当前用户（模型传的 user_id 被忽略）。"""
    async with Harness() as h:
        text, _ = await h.call("send_certificate", {"user_id": LUCAS, "amount": 100}, user=MIA)
        assert text.startswith("Certificate certificate_") and f"user {MIA}" in text
