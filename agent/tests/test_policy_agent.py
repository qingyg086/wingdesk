"""Boundary enforcement, failures, and the real parent-graph SSE path (offline)."""
import json

import pytest
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from agents.policy import Decision, render, FAILED, UNKNOWN, UNAVAILABLE
from agents.specialists import build_specialists
from core.workflow.state import AgentState
from core.workflow.runner import stream_turn
from fake_llm import ScriptedLLM, say

SOURCE = {'text': '仅限直飞，不允许转机。', 'title': '儿童政策', 'section': '条件',
          'citation': 'children#条件', 'category': '仿真扩充政策'}


@pytest.mark.parametrize('decision', [
    {'behavior': 'answer', 'sources': [99]}, {'behavior': 'answer', 'sources': [-1]},
    {'behavior': 'answer'}, {'behavior': 'answer', 'sources': [0, 0]},
    {'behavior': 'clarify', 'question': '编造规定'},
])
def test_invalid_selection_fails_closed(decision):
    assert render(Decision(**decision), [SOURCE]) == FAILED


def test_full_section_and_negation_preserved_and_partial_is_explicit():
    text = render(Decision(behavior='partial', sources=[0]), [SOURCE])
    assert SOURCE['text'] in text and '其余内容没有明确依据' in text
    assert '《儿童政策》' in text and '演示政策' in text


def make_agent(script, payload=None, raises=False):
    @tool
    def query_vector_db(query: str) -> str:
        """Retrieve policy."""
        if raises:
            raise RuntimeError('secret-host')
        return json.dumps(payload or {'status': 'candidates', 'sources': [SOURCE]})
    return build_specialists({}, ScriptedLLM(script=[say(s) for s in script]),
                             local_tools={'query_vector_db': query_vector_db})['policy_agent']


@pytest.mark.asyncio
@pytest.mark.parametrize('payload,raises,expected', [
    ({'status': 'no_evidence'}, False, UNKNOWN),
    ({'status': 'unavailable'}, False, UNAVAILABLE),
    (None, True, UNAVAILABLE),
    ({'status': 'garbage'}, False, FAILED),
])
async def test_failures_do_not_call_model_or_expose_secrets(payload, raises, expected):
    result = await make_agent([], payload, raises).ainvoke({'messages': [HumanMessage('能转机吗')]})
    assert result['messages'][-1].content == expected


@pytest.mark.asyncio
@pytest.mark.parametrize('response', ['可以，费用100美元。', '{"behavior":"answer","sources":[0],"answer":"编造"}',
                                       '{"behavior":"answer","sources":[true]}'])
async def test_free_text_and_extra_fields_never_reach_answer(response):
    result = await make_agent([response]).ainvoke({'messages': [HumanMessage('能转机吗')]})
    assert result['messages'][-1].content == FAILED


@pytest.mark.asyncio
async def test_stream_only_emits_verified_reply_once_and_records_tool_and_history():
    builder = StateGraph(AgentState)
    builder.add_node('policy_agent', make_agent(['{"behavior":"answer","sources":[0]}', '{"approved":true}']))
    builder.add_edge(START, 'policy_agent')
    builder.add_edge('policy_agent', END)
    graph = builder.compile(checkpointer=MemorySaver())
    events = [e async for e in stream_turn(graph, 'u', 's', '能转机吗？') if e]
    tokens = ''.join(e['text'] for e in events if e['type'] == 'token')
    assert tokens == render(Decision(behavior='answer', sources=[0]), [SOURCE])
    assert events[-1]['text'] == tokens
    assert len([e for e in events if e['type'] == 'tool']) == 1
    state = await graph.aget_state({'configurable': {'thread_id': 'u:s'}})
    assert state.values['messages'][-1].content == tokens
    assert 'behavior' not in tokens


@pytest.mark.asyncio
async def test_reviewer_rejects_related_but_not_answering_evidence():
    result = await make_agent(['{"behavior":"answer","sources":[0]}', '{"approved":false}']).ainvoke(
        {'messages': [HumanMessage('无人陪伴服务费支持什么汇率？')]})
    assert result['messages'][-1].content == UNKNOWN


@pytest.mark.parametrize('query,expected', [
    ('托运行李买多了，能把件数减掉吗？', True),
    ('按规定可以减少行李吗', True),
    ('能帮我减掉托运行李吗？', False),
    ('订单 ABC123 能减少行李吗？', False),
    ('减少行李', False),
    ('能减少行李吗，顺便帮我改签', False),
])
def test_permission_question_does_not_intercept_execution(query, expected):
    from agents.orchestrator import baggage_reduction_policy_question
    assert baggage_reduction_policy_question(query) is expected


def test_unused_model_fields_cannot_change_the_rendered_response():
    assert render(Decision(behavior='unknown', sources=[99], question='item'), [SOURCE]) == UNKNOWN
    assert render(Decision(behavior='partial', sources=[0], question='pet'), [SOURCE]) == render(
        Decision(behavior='partial', sources=[0]), [SOURCE])


@pytest.mark.asyncio
async def test_model_wall_clock_timeout_cancels_a_stalled_call(monkeypatch):
    import asyncio
    from agents import policy
    cancelled = []
    class StalledModel:
        async def ainvoke(self, messages, config):
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.append(True)
    monkeypatch.setattr(policy, 'DECISION_TIMEOUT_SECONDS', 0.01)
    with pytest.raises(TimeoutError):
        await policy.call_decision(StalledModel(), [])
    assert cancelled == [True]
