"""Policy retrieval regressions: labels, source provenance, provider failure and safe fallback."""
import json
from types import SimpleNamespace

import pytest

from core import policy_retrieval as retrieval
from tools import vector_tool
from scripts.bench_policy_retrieval import load_cases, summary


def hit():
    return {'id': 'a', 'source': 'extended/01_carry_on.md', 'section': '随身行李件数',
            'title': '随身行李政策', 'category': 'extended', 'text': '候选资料', 'score': 0.9}


def test_evaluation_labels_resolve_and_questions_do_not_overlap():
    rows, fingerprint = retrieval.corpus()
    cases = load_cases(rows, 'dev') + load_cases(rows, 'holdout')
    assert len(cases) == len({c['query'].strip() for c in cases}) == 48
    assert len({r['id'] for r in rows}) == len(rows)
    assert len(fingerprint) == 64
    assert sum(c['kind'] == 'answerable' for c in cases) == 36


def test_metrics_do_not_count_unknown_questions_as_retrieval_success():
    rows = [{'kind':'answerable','first_relevant_rank':2,'latency_ms':10},
            {'kind':'answerable','first_relevant_rank':None,'latency_ms':20},
            {'kind':'unanswerable','first_relevant_rank':1,'latency_ms':30}]
    result = summary(rows)
    assert result['hit_at_1'] == 0 and result['hit_at_3'] == .5 and result['mrr_at_10'] == .25


def test_evidence_preserves_section_and_simulation_label_without_confidence_claim():
    result = vector_tool.evidence_payload([hit()])
    assert result['status'] == 'candidates'
    assert result['sources'][0]['citation'] == 'extended/01_carry_on.md#随身行李件数'
    assert result['sources'][0]['category'] == '仿真扩充政策'
    assert 'score' not in result['sources'][0]
    assert vector_tool.evidence_payload([])['status'] == 'no_evidence'


def test_reranker_failure_falls_back_to_measured_dense_path(monkeypatch):
    calls = []
    class Retriever:
        def search(self, query, mode, limit):
            calls.append(mode)
            if mode == 'hybrid_rerank':
                raise TimeoutError('provider down')
            return [hit()]
    monkeypatch.setenv('POLICY_RETRIEVAL_MODE', 'hybrid_rerank')
    monkeypatch.setattr(vector_tool, 'get_retriever', Retriever)
    result = json.loads(vector_tool.query_vector_db.invoke({'query':'行李规定'}))
    assert calls == ['hybrid_rerank','vector']
    assert result['degraded'] is True and result['sources']


def test_database_outage_is_distinct_from_missing_evidence(monkeypatch):
    def unavailable():
        raise RuntimeError('private-host secret-token')
    monkeypatch.setattr(vector_tool, 'get_retriever', unavailable)
    result = vector_tool.query_vector_db.invoke({'query':'行李规定'})
    assert json.loads(result)['status'] == 'unavailable'
    assert 'private-host' not in result and 'secret-token' not in result


@pytest.mark.parametrize('indices', [[0,0],[99],[0]])
def test_invalid_reranker_response_is_rejected(monkeypatch, indices):
    import dashscope
    monkeypatch.setattr(dashscope.TextReRank,'call', lambda **kw: SimpleNamespace(
        status_code=200,output={'results':[{'index':i,'relevance_score':.8} for i in indices]}))
    with pytest.raises(ValueError):
        retrieval.rerank('测试',[hit(),dict(hit(),id='b')],2)


@pytest.mark.skipif(__import__('os').getenv('POLICY_INTEGRATION_TEST') != '1',
                    reason='set POLICY_INTEGRATION_TEST=1 with a local Milvus server')
def test_native_index_version_survives_creation_and_reopen(monkeypatch):
    """Exercise the server schema: create_collection(description=...) alone was ignored."""
    import uuid
    rows = [dict(hit(), id='checked', text='托运行李每件超额收费50美元', source='core/checked.md'),
            dict(hit(), id='carry', text='随身行李可以放在座位下方', source='extended/carry.md')]
    for row in rows:
        row.pop('score')
    fingerprint = 'a' * 64
    name = 'policy_test_' + uuid.uuid4().hex
    class Embedder:
        def embed_documents(self, texts):
            return [[1.] + [0.] * (retrieval.DIM-1) for _ in texts]
        def embed_query(self, text):
            return [1.] + [0.] * (retrieval.DIM-1)
    monkeypatch.setattr(retrieval, 'embeddings', Embedder)
    db = retrieval.client()
    reader = None
    try:
        assert retrieval.build_index(db, rows, fingerprint, name) == name
        # A second invocation must validate and reuse the existing index.
        assert retrieval.build_index(db, rows, fingerprint, name) == name
        monkeypatch.setattr(retrieval,'corpus',lambda:(rows,fingerprint))
        monkeypatch.setattr(retrieval,'collection_name',lambda _:name)
        retrieval.get_retriever.cache_clear()
        reader = retrieval.get_retriever()
        hits = reader.search('托运行李超额收费', 'bm25')
        assert hits and hits[0]['id'] == 'checked'
    finally:
        retrieval.get_retriever.cache_clear()
        if reader:
            reader.db.close()
        if db.has_collection(name):
            db.drop_collection(name)
        db.close()


# ------------------------------------------------------------------ 政策 Agent：代码保证先检索再回答
def _recording_tool(queries):
    from langchain_core.tools import tool

    @tool
    def query_vector_db(query: str) -> str:
        """检索航空政策证据。"""
        queries.append(query)
        return json.dumps(vector_tool.evidence_payload([hit()]), ensure_ascii=False)
    return query_vector_db


def _policy_agent(llm, queries, mode='product'):
    from agents.specialists import build_specialists
    return build_specialists({}, llm, mode=mode, local_tools={'query_vector_db': _recording_tool(queries)})['policy_agent']


@pytest.mark.asyncio
async def test_policy_agent_retrieves_with_users_words_and_only_renders_evidence():
    from langchain_core.messages import HumanMessage
    from fake_llm import ScriptedLLM, say
    llm, queries = ScriptedLLM(script=[say('{"behavior":"answer","sources":[0]}'), say('{"approved":true}')]), []
    question = '无人陪伴能转机吗？'
    result = await _policy_agent(llm, queries).ainvoke({'messages': [HumanMessage(question)]})
    assert queries == [question]
    assert '候选资料' in llm.agent_inputs[0][-1].content
    assert [type(m).__name__ for m in result['messages']] == ['HumanMessage', 'AIMessage', 'ToolMessage', 'AIMessage']
    assert '候选资料' in result['messages'][-1].content
    assert '演示政策' in result['messages'][-1].content
    assert 'behavior' not in result['messages'][-1].content


@pytest.mark.asyncio
async def test_clarification_keeps_multiple_turns_and_new_topic_resets():
    from langchain_core.messages import HumanMessage
    from fake_llm import ScriptedLLM, say
    llm, queries = ScriptedLLM(script=[
        say('{"behavior":"clarify","question":"baggage_type"}'), say('{"approved":true}'),
        say('{"behavior":"clarify","question":"baggage_allowance"}'), say('{"approved":true}'),
        say('{"behavior":"answer","sources":[0]}'), say('{"approved":true}'),
        say('{"behavior":"unknown"}')]), []
    agent = _policy_agent(llm, queries)
    messages = []
    for question in ['行李能带几件？', '托运的', '普通会员经济舱', '签证多少钱？']:
        result = await agent.ainvoke({'messages': messages + [HumanMessage(question)]})
        messages = result['messages']
    assert queries == ['行李能带几件？', '行李能带几件？\n补充：托运的',
                       '行李能带几件？\n补充：托运的\n补充：普通会员经济舱', '签证多少钱？']


@pytest.mark.asyncio
async def test_eval_mode_is_unchanged():
    from langchain_core.messages import HumanMessage
    from fake_llm import ScriptedLLM, say
    llm, queries = ScriptedLLM(script=[say('政策中没有相关规定。')]), []
    await _policy_agent(llm, queries, mode='eval').ainvoke({'messages': [HumanMessage('行李能带几件？')]})
    assert queries == []


def test_answer_checks_flag_made_up_citations_and_unsupported_claims():
    from scripts.bench_policy_answers import load_titles, score_row
    titles = load_titles()
    assert {'托运行李政策', '危险品与限制携带物品'} <= titles and len(titles) == 25
    rule = {'behavior': 'unknown', 'must_include_any': [['没有', '未']], 'must_not_include': ['禁止使用']}
    bad = score_row({'answer': '飞行中禁止使用充电宝。《旅客运输总条件》第 5.2 节', 'tools': []}, rule, titles)
    assert not bad['passed'] and bad['fabricated_citations'] == ['旅客运输总条件']
    assert '未检索' in bad['problems']
    good = score_row({'answer': '现有政策资料没有规定飞行中能否使用。《危险品与限制携带物品》#锂电池与充电宝',
                      'tools': [{'name': 'query_vector_db', 'content': '{}'}]}, rule, titles)
    assert good['passed'], good['problems']


def test_real_section_names_in_brackets_are_not_counted_as_made_up():
    from scripts.bench_policy_answers import load_sections, load_titles, score_row
    rule = {'behavior': 'answer', 'must_include_any': [['申报']]}
    row = {'answer': '需在值机时申报。《锂电池与充电宝》', 'tools': [{'name': 'query_vector_db', 'content': '{}'}]}
    assert score_row(row, rule, load_titles(), load_sections())['passed']
    assert not score_row(dict(row, answer='需在值机时申报。《锂电池运输条例》'), rule, load_titles(), load_sections())['passed']
