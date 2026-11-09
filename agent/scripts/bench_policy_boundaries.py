"""Fresh boundary acceptance through the real router, policy subgraph, streaming and checkpoint.
No cache or business tools. Run once with --output (existing results are never overwritten).
"""
import argparse
import asyncio
import hashlib
import json
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'agent'))
from scripts.bench_policy_answers import score_row, load_titles, load_sections, write_new

SUITE = ROOT / 'eval/policy_boundary_acceptance.json'


async def run(output):
    if output.exists():
        raise ValueError('output exists')
    from core.llm import get_llm, get_router_llm
    from core.policy_retrieval import corpus
    from core.workflow.graph_manager import AgentGraphManager
    from core.workflow.runner import run_turn, make_config
    from langchain_core.messages import ToolMessage
    from tools.vector_tool import query_vector_db
    graph = AgentGraphManager(tools={}, llm=get_llm(), router_llm=get_router_llm(),
                              local_tools={'query_vector_db': query_vector_db}).build_graph()
    rows = []
    for case in json.loads(SUITE.read_text())['cases']:
        session = uuid.uuid4().hex
        turns = []
        for query in case['turns']:
            start = time.perf_counter()
            result = await asyncio.wait_for(run_turn(graph, 'policy_acceptance', session, query), timeout=120)
            turns.append({'query': query, 'answer': result.text, 'route': result.route,
                          'latency_ms': round((time.perf_counter()-start)*1000)})
        state = await graph.aget_state(make_config('policy_acceptance', session))
        row = {**case, 'answer': result.text, 'results': turns,
               'audit': [m.additional_kwargs['policy_audit'] for m in state.values['messages']
                         if 'policy_audit' in m.additional_kwargs],
               'tools': [{'name': m.name, 'content': m.content} for m in state.values['messages']
                         if isinstance(m, ToolMessage)]}
        # Partial cases must also have real retrieval, like answer cases.
        rule = {**case, 'behavior': 'answer' if case['behavior'] == 'partial' else case['behavior']}
        row['check'] = score_row(row, rule, load_titles(), load_sections())
        problems = row['check']['problems']
        if any('policy_agent' not in t['route'] for t in turns):
            problems.append('未进入政策 Agent')
        if case.get('first_turn_contains') and case['first_turn_contains'] not in turns[0]['answer']:
            problems.append('第一轮未正确追问')
        if any(json.loads(t['content']).get('status') == 'unavailable' for t in row['tools']
               if t['name'] == 'query_vector_db'):
            problems.append('检索服务不可用')
        row['check']['passed'] = not problems
        rows.append(row)
        print(case['id'], row['check'], flush=True)
        with output.with_suffix('.progress.jsonl').open('a', encoding='utf-8') as progress:
            progress.write(json.dumps(row, ensure_ascii=False) + '\n')
    files = ['agent/agents/policy.py', 'agent/agents/specialists.py', 'agent/core/workflow/runner.py',
             'agent/agents/orchestrator.py', 'agent/tools/vector_tool.py', 'agent/core/policy_retrieval.py', 'agent/scripts/bench_policy_boundaries.py']
    summary = {'passed': sum(r['check']['passed'] for r in rows), 'total': len(rows)}
    write_new(output, {'metadata': {'run_at_utc': datetime.now(timezone.utc).isoformat(),
                                   'suite_sha256': hashlib.sha256(SUITE.read_bytes()).hexdigest(),
                                   'corpus_sha256': corpus()[1],
                                   'source_sha256': {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files},
                                   'note': 'Real router + policy retrieval + model + stream_turn + MemorySaver; no cache/business tools. Keyword checks require manual review.'},
                       'summary': summary, 'rows': rows})
    print(summary)
    if summary['passed'] != summary['total']:
        raise RuntimeError('Acceptance failed; inspect recorded results')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    asyncio.run(run(parser.parse_args().output))
