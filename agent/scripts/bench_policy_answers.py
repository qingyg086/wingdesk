"""Run the real policy specialist on the answer-behavior cases and score the answers automatically.

The checks (eval/policy_answer_checks.json) flag obvious failures: no retrieval, citations of documents
that are not in the knowledge base, missing key facts, forbidden claims (made-up prices or rules), and
clarification cases that do not ask anything. Passing the checks does not prove an answer is fully right,
so the full answers are kept for manual review. No router, cache or business tools are involved.

    python agent/scripts/bench_policy_answers.py --output eval/results/my_policy_answers.json
    python agent/scripts/bench_policy_answers.py --rescore eval/results/20261108_policy_answers_final.json
"""
import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'agent'))
CASES = ROOT / 'eval/policy_retrieval_cases.json'
CHECKS = ROOT / 'eval/policy_answer_checks.json'
ANSWERABLE_SAMPLE = {'R01', 'R02', 'R08', 'R20', 'R25', 'R30'}
CITATION = re.compile(r'《([^》]+)》')
RETRIEVAL_TOOL = 'query_vector_db'


def load_titles() -> set[str]:
    """知识库里每份政策文档的标题（文件的一级标题）。只读文件，不依赖检索服务，重新打分时也能用。"""
    titles = set()
    for path in sorted((ROOT / 'data/airline/policy').rglob('*.md')):
        heading = next((ln for ln in path.read_text(encoding='utf-8').splitlines() if ln.startswith('# ')), None)
        if heading:
            titles.add(heading[2:].strip())
    return titles


def load_sections() -> set[str]:
    """政策文档里的小节标题（## 级）。模型有时把小节名写在书名号里，这引用的是真实内容，不算编造。"""
    return {ln[3:].strip() for path in (ROOT / 'data/airline/policy').rglob('*.md')
            for ln in path.read_text(encoding='utf-8').splitlines() if ln.startswith('## ')}


def score_row(row: dict, rule: dict, titles: set[str], sections: frozenset | set = frozenset()) -> dict:
    answer = row.get('answer') or ''
    behavior = rule.get('behavior', 'answer')
    problems = []
    if '这次未能可靠核实' in answer or '政策查询暂时不可用' in answer:
        problems.append('处理故障，不能计作正常拒答')
    retrieved = any(t['name'] == RETRIEVAL_TOOL for t in row.get('tools', []))
    if behavior in ('answer', 'unknown') and not retrieved:
        problems.append('未检索')
    fake = sorted({c for c in CITATION.findall(answer)
                   if not any(c.strip().startswith(t) for t in titles) and c.strip() not in sections})
    if fake:
        problems.append('引用不存在的文档：' + '、'.join(fake))
    for group in rule.get('must_include_any', []):
        if not any(k in answer for k in group):
            problems.append('缺少：' + ' / '.join(group))
    for bad in rule.get('must_not_include', []):
        if bad in answer:
            problems.append('出现不应有的内容：' + bad)
    for pattern in rule.get('must_not_regex', []):
        if re.search(pattern, answer):
            problems.append(f'出现不应有的内容：/{pattern}/')
    if behavior == 'clarify' and '？' not in answer and '?' not in answer:
        problems.append('没有追问')
    return {'behavior': behavior, 'passed': not problems, 'problems': problems,
            'retrieved': retrieved, 'fabricated_citations': fake}


def score_rows(rows: list[dict], titles: set[str]) -> dict:
    rules = json.loads(CHECKS.read_text(encoding='utf-8'))['cases']
    sections = load_sections()
    for row in rows:
        row['check'] = score_row(row, rules[row['id']], titles, sections)
    total, passed = Counter(), Counter()
    for row in rows:
        total[row['check']['behavior']] += 1
        passed[row['check']['behavior']] += row['check']['passed']
    return {'passed': sum(passed.values()), 'total': len(rows),
            'by_behavior': {b: f'{passed[b]}/{total[b]}' for b in ('answer', 'unknown', 'clarify')},
            'fabricated_citation_answers': sum(bool(r['check']['fabricated_citations']) for r in rows),
            'answer_or_unknown_without_retrieval': sum(r['check']['behavior'] != 'clarify' and not r['check']['retrieved']
                                                       for r in rows),
            'checks_sha256': hashlib.sha256(CHECKS.read_bytes()).hexdigest()}


def report(rows: list[dict], summary: dict) -> None:
    for row in rows:
        mark = 'PASS' if row['check']['passed'] else 'FAIL'
        print(f"{row['id']} {row['check']['behavior']:<8} {mark}  {'；'.join(row['check']['problems'])}")
    print(f"\n通过 {summary['passed']}/{summary['total']}：有答案 {summary['by_behavior']['answer']}，"
          f"资料不足 {summary['by_behavior']['unknown']}，需追问 {summary['by_behavior']['clarify']}；"
          f"引用不存在文档的回答 {summary['fabricated_citation_answers']} 条；"
          f"应检索未检索 {summary['answer_or_unknown_without_retrieval']} 条")


def write_new(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as file:          # never overwrite an earlier run
        file.write(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')


async def run(output: Path) -> None:
    if output.exists():
        raise ValueError('output exists; use a new filename')
    from agents.specialists import build_specialists
    from core.llm import get_llm
    from core.policy_retrieval import corpus
    from langchain_core.messages import HumanMessage, ToolMessage
    from tools.vector_tool import query_vector_db

    cases = [c for c in json.loads(CASES.read_text(encoding='utf-8'))['queries']
             if c['kind'] != 'answerable' or c['id'] in ANSWERABLE_SAMPLE]
    agent = build_specialists({}, get_llm(0.0), local_tools={RETRIEVAL_TOOL: query_vector_db})['policy_agent']
    rows = []
    for case in cases:
        start = time.perf_counter()
        result = await agent.ainvoke({'messages': [HumanMessage(case['query'])]}, {'recursion_limit': 20})
        messages = result['messages']
        tools = [m for m in messages if isinstance(m, ToolMessage)]
        rows.append(dict(case, answer=messages[-1].content, audit=messages[-1].additional_kwargs.get('policy_audit'), latency_ms=round((time.perf_counter() - start) * 1000),
                         tools=[{'name': m.name, 'content': m.content} for m in tools]))
        print(f"{case['id']} recorded; tools={len(tools)}", flush=True)
    unavailable = [r['id'] for r in rows for t in r['tools']
                   if t['name'] == RETRIEVAL_TOOL and json.loads(t['content'])['status'] == 'unavailable']
    summary = score_rows(rows, load_titles())
    metadata = {'run_at_utc': datetime.now(timezone.utc).isoformat(), 'model': os.getenv('MODEL', 'qwen-plus'),
                'mode': os.getenv('POLICY_RETRIEVAL_MODE', 'hybrid_rerank'),
                'dataset_sha256': hashlib.sha256(CASES.read_bytes()).hexdigest(), 'corpus_sha256': corpus()[1],
                'source_sha256': {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in
                                  ['agent/agents/specialists.py', 'agent/tools/vector_tool.py',
                                   'agent/core/policy_retrieval.py', 'agent/agents/policy.py', 'agent/core/workflow/runner.py']},
                'note': 'Direct product policy specialist with real retrieval and model; excludes router, cache '
                        'and business tools. Automatic checks flag obvious failures only; review answers manually.'}
    write_new(output, {'metadata': metadata, 'summary': summary, 'rows': rows})
    report(rows, summary)
    if unavailable or any(not r['answer'] for r in rows) or summary['passed'] != summary['total']:
        raise RuntimeError(f'acceptance failed ({summary["passed"]}/{summary["total"]}); unavailable={unavailable}; inspect results')


def rescore(path: Path, output: Path | None) -> None:
    data = json.loads(path.read_text(encoding='utf-8'))
    summary = score_rows(data['rows'], load_titles())
    report(data['rows'], summary)
    if output:
        write_new(output, {'metadata': {**data['metadata'], 'rescored_from': str(path.relative_to(ROOT)),
                                        'rescored_at_utc': datetime.now(timezone.utc).isoformat()},
                           'summary': summary, 'rows': data['rows']})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    parser.add_argument('--rescore', type=Path, help='score an existing result file without calling any model')
    args = parser.parse_args()
    if args.rescore:
        rescore(args.rescore.resolve(), args.output)
    elif args.output:
        asyncio.run(run(args.output))
    else:
        parser.error('--output or --rescore is required')
