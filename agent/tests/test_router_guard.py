"""
转人工的代码兜底。不需要数据库，直接测 OrchestratorAgent.route。
路由器选了 human_handoff，但用户最新一句话没有明确要求人工 → 用不含 human_handoff 的枚举重新路由。
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agents.orchestrator import OrchestratorAgent, RouteDecision, direct_human_request, explicit_human_request, flight_policy_question
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
    ("请不要转人工，只告诉我宠物乘机的规定", False),
    ("Do not transfer me to a human agent; show my reservation.", False),
    ("My manager asked me to cancel both bookings.", False),
    ("请停止自动客服，帮我接通真人客服", True),
    ("我需要人工客服", True),
    ("我不想要人工客服", False),
    ("帮我查人工值机的开放时间", False),
    ("我需要了解人工值机流程", False),
    ("I need a representative to discuss my booking.", True),
    ("Please connect me to customer service.", True),
    ("I need to know the customer service hours.", False),
    ("如果我需要人工客服，该怎么联系？", False),
    ("怎样转人工客服？", False),
    ("为什么页面上有‘找人工客服’按钮？", False),
    ("I want a human-readable receipt.", False),
    ("I want customer service hours.", False),
    ("How do I speak to a human agent?", False),
    ("不如转人工吧", True),
    ("No need to wait—transfer me to a human agent.", True),
    ("我要转人工办理退票", True),
    ("让我跟真人客服说话", True),
    ("I do not want to speak to a human agent.", False),
])
def test_explicit_human_request(text, expected):
    assert explicit_human_request(text) is expected


def test_direct_human_request_requires_an_actual_request():
    assert direct_human_request("我需要人工客服")
    assert not direct_human_request("人工智能会处理这个问题吗？")
    assert not direct_human_request("My manager asked me to cancel both bookings.")
    assert not direct_human_request("帮我查人工值机的开放时间")
    assert direct_human_request("Please connect me to customer service.")
    assert not direct_human_request("怎样转人工客服？")
    assert not direct_human_request("I want customer service hours.")
    assert not direct_human_request("I want a human-readable receipt.")
    assert direct_human_request("不如转人工吧")


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
    assert llm.routes == [R("human_handoff")]  # 明确请求无需调用路由模型


@pytest.mark.asyncio
async def test_clear_baggage_policy_question_skips_flight_router():
    llm = ScriptedLLM()
    llm.routes = [R("flight_search_agent")]
    agent = OrchestratorAgent(llm)
    out = await agent.route({"active_agent": "flight_search_agent",
                             "messages": [HumanMessage(content="查明天去西雅图的航班"),
                                          AIMessage(content="找到了三个直飞选项。"),
                                          HumanMessage(content="随身行李额度是多少？")]})
    assert out["next_agent"] == "policy_agent" and llm.routes == [R("flight_search_agent")]
    assert not flight_policy_question("我想订票，能免费托运几件行李？")
    assert not flight_policy_question("查明天去西雅图的航班，同时告诉我随身行李规定")
    assert not flight_policy_question("查一下有哪些航班可以免费托运行李？")
    assert not flight_policy_question("帮我订机票，随身行李可以带几件？")
    assert not flight_policy_question("我这张票的行李额度是多少？")
    assert not flight_policy_question("托运行李能加到我的票上吗？")
    assert not flight_policy_question("我不想问行李额度是多少")
    assert not flight_policy_question("My checked baggage on order 123?")


@pytest.mark.asyncio
async def test_reroute_failure_falls_back_to_active_agent():
    llm = ScriptedLLM()
    llm.routes = [R("human_handoff"), RuntimeError("network")]
    out = await OrchestratorAgent(llm).route(
        {"active_agent": "cancel_refund_agent",
         "messages": [HumanMessage(content="I insist on a full refund!")]})
    assert out["next_agent"] == "cancel_refund_agent"
