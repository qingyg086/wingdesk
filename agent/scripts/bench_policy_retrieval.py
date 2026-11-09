"""Frozen question-level comparison of policy retrieval. No production collection is changed."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'agent'))
from core.policy_retrieval import (ANALYZER, CANDIDATES, DENSE_INDEX, MODEL, MODES, RRF_K,
                                   PolicyRetriever, build_index, client, corpus)
DATA = ROOT / 'eval/policy_retrieval_cases.json'


def load_cases(rows, split):
    cases = json.loads(DATA.read_text())['queries']
    sections = {(row['source'], row['section']) for row in rows}
    seen = set()
    for case in cases:
        if case['id'] in seen or not case['query'].strip():
            raise ValueError('duplicate ID or empty query')
        seen.add(case['id'])
        if case['kind'] == 'answerable' and not case['relevant']:
            raise ValueError('answerable question without evidence')
        for expected in case['relevant']:
            if (expected['source'], expected['section']) not in sections:
                raise ValueError(f"unknown section: {case['id']} {expected}")
    return [case for case in cases if case['split'] == split]


def summary(rows):
    answerable = [r for r in rows if r['kind'] == 'answerable']
    latency = sorted(r['latency_ms'] for r in rows)
    return {'queries': len(rows), 'answerable': len(answerable),
            'hit_at_1': sum(r['first_relevant_rank'] == 1 for r in answerable) / len(answerable),
            'hit_at_3': sum(r['first_relevant_rank'] is not None and r['first_relevant_rank'] <= 3 for r in answerable) / len(answerable),
            'mrr_at_10': sum(1 / r['first_relevant_rank'] if r['first_relevant_rank'] else 0 for r in answerable) / len(answerable),
            'p50_ms': statistics.median(latency), 'p95_ms': latency[max(0, math.ceil(len(latency)*.95)-1)]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=['dev', 'holdout'], default='dev')
    parser.add_argument('--modes', nargs='+', choices=MODES, default=['vector'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output already exists; use a new filename to preserve evidence')
    rows, fingerprint = corpus()
    cases = load_cases(rows, args.split)
    commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    db = client()
    server_version = db.get_server_version()
    name = f'policy_eval_{fingerprint[:8]}_{int(time.time())}'
    results = []
    try:
        build_index(db, rows, fingerprint, name)
        retriever = PolicyRetriever(db, name)
        # Warm connections on a fixed corpus title, never on an evaluation question.
        for mode in args.modes:
            retriever.search('订票政策', mode, limit=10)
        for i, case in enumerate(cases):
            # Rotate mode order to avoid always measuring one model first.
            modes = args.modes[i % len(args.modes):] + args.modes[:i % len(args.modes)]
            for mode in modes:
                start = time.perf_counter()
                hits = retriever.search(case['query'], mode, limit=10)
                elapsed = (time.perf_counter() - start) * 1000
                relevant = {(x['source'], x['section']) for x in case['relevant']}
                rank = next((j for j, hit in enumerate(hits, 1) if (hit['source'], hit['section']) in relevant), None)
                results.append(dict(case, mode=mode, first_relevant_rank=rank, latency_ms=round(elapsed, 2),
                                    hits=[{k:v for k,v in hit.items() if k != 'text'} for hit in hits]))
            print(f'{i+1}/{len(cases)} complete', flush=True)
    finally:
        if db.has_collection(name):
            db.drop_collection(name)
        db.close()
    metadata = {'run_at_utc': datetime.now(timezone.utc).isoformat(), 'split': args.split,
                'git_commit': commit,
                'dataset_sha256': hashlib.sha256(DATA.read_bytes()).hexdigest(), 'corpus_sha256': fingerprint,
                'retriever_sha256': hashlib.sha256((ROOT/'agent/core/policy_retrieval.py').read_bytes()).hexdigest(),
                'embedding_model': MODEL, 'document_embedding_role': 'document', 'query_embedding_role': 'query',
                'analyzer': ANALYZER, 'dense_index': DENSE_INDEX, 'candidates_per_branch': CANDIDATES,
                'rrf_k': RRF_K, 'reranker': os.getenv('POLICY_RERANK_MODEL', 'qwen3-rerank'), 'latency': 'warm per query incl live embedding/rerank APIs; excludes indexing; retrieve top 10',
                'milvus_version': server_version, 'note': 'fresh isolated index; original splitter and dense settings; not online traffic'}
    summaries = {mode: summary([r for r in results if r['mode']==mode]) for mode in args.modes}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as file:
        file.write(json.dumps(dict(metadata=metadata,summary=summaries,rows=results),ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summaries, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
