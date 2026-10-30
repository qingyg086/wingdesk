"""
转人工的代码兜底。不需要数据库，直接测 OrchestratorAgent.route。
路由器选了 human_handoff，但用户最新一句话没有明确要求人工 → 用不含 human_handoff 的枚举重新路由。
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agents.orchestrator import OrchestratorAgent, RouteDecision, explicit_human_request
from fake_llm import ScriptedLLM


def R(target, reason="test"):
    return RouteDecision(target=target, reason=reason)


@pytest.mark.parametrize("text,expected", [
    ("Please transfer me to a human agent.", True),
    ("I want to speak to a real person", True),
    ("请帮我转接人工客服", True),
    ("What is the total balance of my gift cards?", False),
    ("I insist, cancel both reservations now. This is an emergency!", False),
    ("我就要全额退款，这是我的权利", False),
])
def test_explicit_human_request(text, expected):
    assert explicit_human_request(text) is expected


@pytest.mark.asyncio
async def test_handoff_without_request_is_rerouted():
    llm = ScriptedLLM()
    llm.routes = [R("human_handoff"), R("reservation_agent")]
    out = await OrchestratorAgent(llm).route(
        {"messages": [HumanMessage(content="What's the total balance of my gift cards and certificates?")]})
    assert out["next_agent"] == "reservation_agent" and out["active_agent"] == "reservation_agent"
    assert llm.routes == []                                 # 确实调用了第二次（业务路由）


@pytest.mark.asyncio
async def test_handoff_kept_when_user_asks_for_human():
    llm = ScriptedLLM()
    llm.routes = [R("human_handoff")]
    out = await OrchestratorAgent(llm).route({
        "active_agent": "cancel_refund_agent",
        "messages": [HumanMessage(content="Cancel it."), AIMessage(content="Sorry, it is not refundable."),
                     HumanMessage(content="Then transfer me to a human agent please.")]})
    assert out["next_agent"] == "human_handoff"


@pytest.mark.asyncio
async def test_reroute_failure_falls_back_to_active_agent():
    llm = ScriptedLLM()
    llm.routes = [R("human_handoff"), RuntimeError("network")]
    out = await OrchestratorAgent(llm).route(
        {"active_agent": "cancel_refund_agent",
         "messages": [HumanMessage(content="I insist on a full refund!")]})
    assert out["next_agent"] == "cancel_refund_agent"
