"""
Agent 层离线测试：用脚本化假模型驱动真实的图 + 真实的 MCP Server + airline_test 库。
测的是"机制"是否正确：路由、interrupt 暂停与恢复、确认/拒绝、checkpointer 多轮、粘性路由、越权、会话隔离。
"""

import json

import pytest
from langchain_core.messages import HumanMessage, ToolMessage

from agents.orchestrator import RouteDecision
from backends.mysql_backend import MySQLBackend
from core.mcp.mcp_manager import MCPToolRegistry
from core.workflow.graph_manager import AgentGraphManager
from core.workflow.runner import make_config, pending_confirmation, run_turn
from fake_llm import ScriptedLLM, say, tool_call

MIA, LUCAS = "mia_li_3668", "lucas_wilson_8118"
DATE = "2024-05-20"


def R(target, reason="test"):
    return RouteDecision(target=target, reason=reason)


class World:
    """一次测试的环境：MCP 注册表（连 airline_test）+ 假模型 + 编好的图。"""

    def __init__(self, mode="product"):
        self.mode = mode
        self.llm = ScriptedLLM()
        self.db = MySQLBackend("airline_test")

    async def __aenter__(self):
        self.reg = MCPToolRegistry()
        await self.reg.start(env={"MYSQL_DATABASE": "airline_test"})
        tools = {t.name: t for t in self.reg.get_tools()}
        self.graph = AgentGraphManager(tools=tools, llm=self.llm, mode=self.mode, local_tools={}).build_graph()
        return self

    async def __aexit__(self, *exc):
        await self.reg.stop()

    async def state(self, user, session):
        return (await self.graph.aget_state(make_config(user, session))).values

    def booking_call(self, user_id_claimed=LUCAS):
        fl = json.loads(self.db.search_direct_flight("JFK", "SEA", DATE))[0]
        price = fl["prices"]["economy"]
        return tool_call("book_reservation", user_id=user_id_claimed, origin="JFK", destination="SEA",
                         flight_type="one_way", cabin="economy",
                         flights=[{"flight_number": fl["flight_number"], "date": DATE}],
                         passengers=[{"first_name": "Mia", "last_name": "Li", "dob": "1990-04-05"}],
                         payment_methods=[{"payment_id": "credit_card_4421486", "amount": price}],
                         total_baggages=0, nonfree_baggages=0, insurance="no")

    def reservations(self, user):
        return json.loads(self.db.get_user_details(user))["reservations"]


@pytest.mark.asyncio
async def test_route_to_flight_search_and_reply(rw):
    async with World() as w:
        w.llm.routes = [R("flight_search_agent")]
        w.llm.script = [tool_call("search_direct_flight", origin="JFK", destination="SEA", date=DATE),
                        say("为您找到以下航班……")]
        turn = await run_turn(w.graph, MIA, "s1", "帮我查 5 月 20 日纽约到西雅图的航班")
        assert turn.route == ["orchestrator", "flight_search_agent"]
        assert [c["name"] for c in turn.tool_calls] == ["search_direct_flight"]
        assert turn.text == "为您找到以下航班……" and turn.confirm is None
        tool_msgs = [m for m in (await w.state(MIA, "s1"))["messages"] if isinstance(m, ToolMessage)]
        assert "HAT" in str(tool_msgs[0].content)          # 真实查到了航班


@pytest.mark.asyncio
async def test_write_waits_for_confirmation_then_executes(rw):
    async with World() as w:
        before = w.reservations(MIA)
        w.llm.routes = [R("booking_agent")]
        w.llm.script = [w.booking_call(), say("预订成功！")]
        turn = await run_turn(w.graph, MIA, "s1", "就订这个")
        # 1) 停在确认这一步：有确认摘要，数据库里还没有新订单
        assert turn.confirm and turn.confirm["action"] == "book_reservation"
        assert "预订机票：JFK → SEA" in turn.confirm["summary_zh"]
        assert w.reservations(MIA) == before
        assert await pending_confirmation(w.graph, make_config(MIA, "s1"))
        # 2) 用户回复"确认"→ 恢复执行，订单写入当前登录用户（模型传的 lucas 被拦截器改掉）
        turn = await run_turn(w.graph, MIA, "s1", "确认")
        assert turn.resumed and turn.route == ["confirm", "booking_agent"]
        assert turn.text == "预订成功！"
        assert len(w.reservations(MIA)) == len(before) + 1
        assert await pending_confirmation(w.graph, make_config(MIA, "s1")) is None


@pytest.mark.asyncio
async def test_reject_with_feedback_does_not_write(rw):
    async with World() as w:
        before = w.reservations(MIA)
        w.llm.routes = [R("booking_agent")]
        w.llm.script = [w.booking_call(), say("好的，为您换成商务舱再报价。")]
        await run_turn(w.graph, MIA, "s1", "就订这个")
        turn = await run_turn(w.graph, MIA, "s1", "等一下，我想换成商务舱")
        assert w.reservations(MIA) == before                     # 没有写库
        last_seen = w.llm.agent_inputs[-1]                       # 模型恢复后看到的工具结果
        tm = [m for m in last_seen if isinstance(m, ToolMessage)][-1]
        assert "用户未确认" in str(tm.content) and "换成商务舱" in str(tm.content)
        assert turn.text.startswith("好的")


@pytest.mark.asyncio
async def test_confirm_button_overrides_text(rw):
    """前端点按钮：confirm=False 时，即使文字是"好的"也算拒绝。"""
    async with World() as w:
        before = w.reservations(MIA)
        w.llm.routes = [R("booking_agent")]
        w.llm.script = [w.booking_call(), say("已取消。")]
        await run_turn(w.graph, MIA, "s1", "订")
        await run_turn(w.graph, MIA, "s1", "好的", confirm=False)
        assert w.reservations(MIA) == before


@pytest.mark.asyncio
async def test_multi_turn_history_and_sticky_routing(rw):
    async with World() as w:
        w.llm.routes = [R("flight_search_agent"), R("flight_search_agent")]
        w.llm.script = [tool_call("search_direct_flight", origin="JFK", destination="SEA", date=DATE),
                        say("有 HAT069 等航班。"), say("HAT069 是早上出发的。")]
        await run_turn(w.graph, MIA, "s1", "查 5/20 纽约到西雅图")
        await run_turn(w.graph, MIA, "s1", "第一个几点出发？")
        # 路由器第二轮看到了"当前处理中的 Agent"和第一轮的对话
        router_prompt = w.llm.router_inputs[1][-1].content
        assert "当前处理中的 Agent：flight_search_agent" in router_prompt
        assert "查 5/20 纽约到西雅图" in router_prompt
        # Agent 第二轮看到了第一轮的工具结果（checkpointer 保存了完整历史，没有靠拼提示词）
        second_turn_input = w.llm.agent_inputs[-1]
        assert any(isinstance(m, ToolMessage) for m in second_turn_input)
        assert sum(isinstance(m, HumanMessage) for m in second_turn_input) == 2
        assert (await w.state(MIA, "s1"))["active_agent"] == "flight_search_agent"


@pytest.mark.asyncio
async def test_router_failure_falls_back(rw):
    async with World() as w:
        w.llm.routes = [RuntimeError("bad json"), R("booking_agent"), RuntimeError("timeout")]
        w.llm.script = [say("请问您想咨询什么？"), say("请问要订哪天的？"), say("好的，5 月 20 日。")]
        t1 = await run_turn(w.graph, MIA, "s1", "你好")
        assert t1.route[-1] == "policy_agent"                    # 没有进行中的任务 → 政策咨询兜底
        await run_turn(w.graph, MIA, "s1", "我要订票")
        t3 = await run_turn(w.graph, MIA, "s1", "20 号")
        assert t3.route[-1] == "booking_agent"                   # 有进行中的任务 → 继续交给它


@pytest.mark.asyncio
async def test_cannot_read_other_users_reservation(rw):
    async with World() as w:
        w.llm.routes = [R("reservation_agent")]
        w.llm.script = [tool_call("get_reservation_details", reservation_id="I6XC2H"), say("没有找到该订单。")]
        await run_turn(w.graph, MIA, "s1", "查一下订单 I6XC2H")        # I6XC2H 属于 lucas
        tm = [m for m in (await w.state(MIA, "s1"))["messages"] if isinstance(m, ToolMessage)][-1]
        assert "Error: user not found" in str(tm.content)


@pytest.mark.asyncio
async def test_sessions_isolated_between_users(rw):
    """两个用户用同一个 session_id，对话历史互不可见（thread_id = user_id:session_id）。"""
    async with World() as w:
        w.llm.routes = [R("policy_agent"), R("policy_agent")]
        w.llm.script = [say("A"), say("B")]
        await run_turn(w.graph, MIA, "default_session", "我是 mia 的秘密问题")
        await run_turn(w.graph, LUCAS, "default_session", "你好")
        lucas_view = w.llm.agent_inputs[-1]
        assert not any("mia 的秘密问题" in str(m.content) for m in lucas_view)


@pytest.mark.asyncio
async def test_eval_mode_writes_without_interrupt(rw):
    """评测模式（τ-bench）：不弹确认卡片，确认由对话完成（模拟用户在对话里说 yes）。"""
    async with World(mode="eval") as w:
        before = w.reservations(MIA)
        w.llm.routes = [R("booking_agent")]
        w.llm.script = [w.booking_call(), say("done")]
        turn = await run_turn(w.graph, MIA, "s1", "yes, book it")
        assert turn.confirm is None and len(w.reservations(MIA)) == len(before) + 1
