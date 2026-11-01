"""Validate the opt-in semantic-cache configuration on unseen questions.

The public preset questions are inserted into a temporary Milvus Lite collection.
Real DashScope query embeddings are generated for each question, then replayed through
SemanticCache.get_cache. No production collection or business database is changed.

Run from the repository root with DASHSCOPE_API_KEY in agent/.env:
    python agent/scripts/cache_holdout_validation.py
    python agent/scripts/cache_holdout_validation.py --output eval/results/my_cache_holdout.json
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / "agent" / ".env")
sys.path.insert(0, str(ROOT / "app"))

from infra.cache import (  # noqa: E402
    EMBEDDING_DIMS,
    EMBEDDING_MODEL,
    SIMILARITY_THRESHOLD,
    SemanticCache,
    cache_mode,
    is_cacheable,
    normalize_query,
)

PRESET_PATH = ROOT / "data" / "airline" / "cache" / "preset_qa.json"
DEV_PATH = ROOT / "data" / "airline" / "cache" / "threshold_eval.json"
HOLDOUT_PATH = ROOT / "data" / "airline" / "cache" / "holdout_eval.json"
OUT_DIR = ROOT / "eval" / "results"
VALIDATED_MODE = "semantic"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_snapshot() -> tuple[str | None, bool | None]:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                            text=True, check=False)
    status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True,
                            text=True, check=False)
    return (commit.stdout.strip() if commit.returncode == 0 else None,
            bool(status.stdout) if status.returncode == 0 else None)


def load_data() -> tuple[list[dict], list[dict]]:
    """Reject overlap before any embedding call, so this remains a real holdout."""
    preset = json.loads(PRESET_PATH.read_text(encoding="utf-8"))["items"]
    dev = json.loads(DEV_PATH.read_text(encoding="utf-8"))["queries"]
    holdout = json.loads(HOLDOUT_PATH.read_text(encoding="utf-8"))["queries"]
    answers = {item["id"] for item in preset}
    if len(answers) != len(preset):
        raise ValueError("duplicate preset answer IDs")
    seen_before = {normalize_query(question)
                   for item in preset for question in [item["question"], *item.get("aliases", [])]}
    seen_before.update(normalize_query(case["text"]) for case in dev)
    seen_holdout: set[str] = set()
    case_ids: set[str] = set()
    for case in holdout:
        if case["id"] in case_ids:
            raise ValueError(f"duplicate holdout ID: {case['id']}")
        case_ids.add(case["id"])
        norm = normalize_query(case["text"])
        if not norm or norm in seen_holdout or norm in seen_before:
            raise ValueError(f"holdout question is empty, repeated, or seen in development: {case['id']}")
        seen_holdout.add(norm)
        if case["label"] == "positive":
            if case.get("expected") not in answers:
                raise ValueError(f"unknown expected answer: {case['id']}")
        elif case["label"] not in {"hard_negative", "easy_negative"}:
            raise ValueError(f"invalid label: {case['id']}")
    if not holdout or not any(q["label"] == "positive" for q in holdout) or not any(
            q["label"] != "positive" for q in holdout):
        raise ValueError("holdout must contain positive and negative cases")
    return preset, holdout


class PrecomputedEmbeddings:
    """Replay real model vectors while exercising the production cache path."""

    def __init__(self, vectors: dict[str, list[float]]):
        self.vectors = vectors

    async def aembed_query(self, text: str) -> list[float]:
        return self.vectors[normalize_query(text)]


def _nearest(query: np.ndarray, entries: np.ndarray) -> tuple[int, float]:
    scores = entries @ query
    index = int(np.argmax(scores))
    return index, float(scores[index])


def _summary(rows: list[dict]) -> dict:
    positives = [row for row in rows if row["label"] == "positive"]
    negatives = [row for row in rows if row["label"] != "positive"]
    correct = sum(row["outcome"] == "correct_hit" for row in positives)
    wrong = sum(row["outcome"] == "wrong_answer" for row in positives)
    false_hits = sum(row["outcome"] == "false_hit" for row in negatives)
    hits = sum(row["actual_answer_id"] is not None for row in rows)
    return {
        "total": len(rows),
        "positives": len(positives),
        "correct_hits": correct,
        "positive_misses": len(positives) - correct - wrong,
        "wrong_answers": wrong,
        "negatives": len(negatives),
        "hard_negatives": sum(row["label"] == "hard_negative" for row in negatives),
        "false_hits": false_hits,
        "hard_false_hits": sum(row["label"] == "hard_negative" and row["outcome"] == "false_hit"
                               for row in negatives),
        "recall": round(correct / len(positives), 4),
        "precision": round(correct / hits, 4) if hits else 1.0,
        "false_hit_rate": round(false_hits / len(negatives), 4),
        "guard_rejections": sum(not row["cacheable"] for row in rows),
    }


async def validate(preset: list[dict], cases: list[dict]) -> dict:
    from langchain_community.embeddings import DashScopeEmbeddings
    from pymilvus import MilvusClient

    key = os.getenv("DASHSCOPE_API_KEY")
    if not key:
        raise RuntimeError("DASHSCOPE_API_KEY is required in agent/.env")
    entries = [(item["id"], question, item["answer"])
               for item in preset for question in [item["question"], *item.get("aliases", [])]]
    answer_by_id = {item["id"]: item["answer"] for item in preset}
    texts = [normalize_query(question) for _, question, _ in entries] + [
        normalize_query(case["text"]) for case in cases]
    print(f"Embedding {len(texts)} preset and holdout questions with {EMBEDDING_MODEL} "
          "using the production query-embedding method...", flush=True)
    started = time.perf_counter()
    model = DashScopeEmbeddings(model=EMBEDDING_MODEL, dashscope_api_key=key)
    vectors = []
    for index, text in enumerate(texts, 1):
        vectors.append(await model.aembed_query(text))
        if index % 20 == 0 or index == len(texts):
            print(f"  embedded {index}/{len(texts)}", flush=True)
    embed_ms = round((time.perf_counter() - started) * 1000)
    if len(vectors) != len(texts) or any(len(vec) != EMBEDDING_DIMS[EMBEDDING_MODEL] for vec in vectors):
        raise RuntimeError("embedding count or dimension differs from the configured cache model")
    mapping = dict(zip(texts, vectors))
    if len(mapping) != len(texts):
        raise ValueError("preset and holdout questions overlap after normalization")
    matrix = np.asarray(vectors, dtype=float)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    entry_matrix, case_matrix = matrix[:len(entries)], matrix[len(entries):]
    question_to_id = {normalize_query(question): answer_id for answer_id, question, _ in entries}
    rows = []
    with tempfile.TemporaryDirectory(prefix="nimbus-cache-holdout-") as directory:
        client = MilvusClient(str(Path(directory) / "cache.db"))
        cache = SemanticCache(threshold=SIMILARITY_THRESHOLD, model=EMBEDDING_MODEL,
                              mode=VALIDATED_MODE)
        await cache.initialize(client=client, embeddings=PrecomputedEmbeddings(mapping))
        if not cache.available:
            raise RuntimeError("temporary Milvus cache could not initialize")
        for _, question, answer in entries:
            await cache.set_cache(question, answer)
        seeded = client.query(collection_name="qa_semantic_cache", filter='scope == "public"',
                              output_fields=["question"], limit=len(entries) + 1)
        if len(seeded) != len(entries):
            raise RuntimeError(f"temporary cache contains {len(seeded)}/{len(entries)} preset entries")
        for case, query_vector in zip(cases, case_matrix):
            nearest_index, nearest_score = _nearest(query_vector, entry_matrix)
            hit = await cache.get_cache(case["text"], user_id="holdout-test-user")
            actual = question_to_id.get(normalize_query(hit["matched_question"])) if hit else None
            if hit and actual is None:
                raise RuntimeError(f"cache returned an unknown preset question for {case['id']}")
            answer_matches_preset = bool(hit and hit["answer"] == answer_by_id[actual])
            if case["label"] == "positive":
                outcome = "correct_hit" if actual == case["expected"] and answer_matches_preset else (
                    "positive_miss" if actual is None else "wrong_answer")
            else:
                outcome = "false_hit" if actual is not None else "correct_rejection"
            rows.append({
                "id": case["id"], "text": case["text"], "label": case["label"],
                "expected_answer_id": case.get("expected"), "actual_answer_id": actual,
                "outcome": outcome, "cacheable": is_cacheable(case["text"]),
                "answer_matches_preset": answer_matches_preset,
                "hit_level": hit["level"] if hit else None,
                "hit_similarity": round(hit["similarity"], 4) if hit else None,
                "nearest_answer_id": entries[nearest_index][0],
                "nearest_question": entries[nearest_index][1],
                "nearest_cosine": round(nearest_score, 4),
            })
        client.close()
    summary = _summary(rows)
    return {"summary": summary, "embedding_total_ms": embed_ms, "rows": rows}


def main(output: Path | None, min_recall: float) -> int:
    preset, cases = load_data()
    commit, dirty = _git_snapshot()
    metadata = {
        "started_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "git_commit": commit,
        "git_dirty": dirty,
        "preset_sha256": _sha256(PRESET_PATH),
        "development_sha256": _sha256(DEV_PATH),
        "holdout_sha256": _sha256(HOLDOUT_PATH),
        "cache_source_sha256": _sha256(ROOT / "app" / "infra" / "cache.py"),
        "validator_source_sha256": _sha256(Path(__file__).resolve()),
        "embedding_model": EMBEDDING_MODEL,
        "threshold": SIMILARITY_THRESHOLD,
        "embedding_dimensions": EMBEDDING_DIMS[EMBEDDING_MODEL],
        "embedding_provider": "DashScope via DashScopeEmbeddings.aembed_query",
        "cache_backend": "temporary Milvus Lite, COSINE FLAT index",
        "evaluated_cache_mode": VALIDATED_MODE,
        "configured_runtime_cache_mode": cache_mode(),
    }
    result = asyncio.run(validate(preset, cases))
    result["metadata"] = metadata
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    auto_output = output is None
    if auto_output:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        output = OUT_DIR / f"{stamp}_cache_holdout.json"
    else:
        output = output if output.is_absolute() else ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    s = result["summary"]
    gate = {"min_recall": min_recall, "no_wrong_answers": s["wrong_answers"] == 0,
            "no_false_hits": s["false_hits"] == 0, "passed": (s["recall"] >= min_recall
            and s["wrong_answers"] == 0 and s["false_hits"] == 0)}
    result["gate"] = gate
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if auto_output:
        with output.open("x", encoding="utf-8") as file:
            file.write(payload)
    else:
        output.write_text(payload, encoding="utf-8")
    print(f"Holdout {s['total']} cases: correct {s['correct_hits']}/{s['positives']} positives "
          f"(recall {s['recall']:.1%}); false hits {s['false_hits']}/{s['negatives']} negatives "
          f"(precision {s['precision']:.1%}); wrong answers {s['wrong_answers']}", flush=True)
    for row in result["rows"]:
        if row["outcome"] not in {"correct_hit", "correct_rejection"}:
            print(f"  {row['id']}: {row['outcome']}, nearest={row['nearest_answer_id']} "
                  f"({row['nearest_cosine']:.3f}), actual={row['actual_answer_id']}", flush=True)
    print(f"Gate: {'PASS' if gate['passed'] else 'FAIL'} "
          f"(recall >= {min_recall:.0%}, zero wrong answers, zero false hits)", flush=True)
    print(f"Saved {output.relative_to(ROOT) if output.is_relative_to(ROOT) else output}", flush=True)
    return 0 if gate["passed"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, help="result JSON path (default: unique UTC filename)")
    parser.add_argument("--min-recall", type=float, default=0.80,
                        help="minimum correct positive recall for a passing validation (default: 0.80)")
    args = parser.parse_args()
    if not 0 <= args.min_recall <= 1:
        parser.error("--min-recall must be between 0 and 1")
    raise SystemExit(main(args.output, args.min_recall))
