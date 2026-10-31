"""Compare route accuracy and latency on the same labeled conversation cases.

Run from the repository root with an API key in agent/.env:
    python agent/scripts/bench_router.py --models qwen-plus qwen-flash

Only the orchestrator runs. This benchmark does not call any airline tools or
write business data. Each case is a fresh, independent conversation snapshot.
"""

import argparse
import asyncio
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "agent"))

from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402

from agents.orchestrator import OrchestratorAgent, TARGETS  # noqa: E402
from core.clock import mode as clock_mode  # noqa: E402
from core.llm import DEFAULT_BASE_URL, get_llm  # noqa: E402

CASES_PATH = ROOT / "eval" / "router_cases.json"
OUT_DIR = ROOT / "eval" / "results"


def _git_snapshot() -> tuple[str | None, bool | None]:
    """Record the source revision without including file contents or secrets."""
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                            check=False)
    status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True,
                            check=False)
    return (commit.stdout.strip() if commit.returncode == 0 else None,
            bool(status.stdout) if status.returncode == 0 else None)


def _source_hash() -> str:
    digest = hashlib.sha256()
    for path in (ROOT / "agent" / "agents" / "orchestrator.py", ROOT / "agent" / "core" / "llm.py",
                 Path(__file__).resolve()):
        digest.update(str(path.relative_to(ROOT)).encode("utf-8") + b"\0" + path.read_bytes())
    return digest.hexdigest()


def _reserve_default_output() -> Path:
    """Create a unique result file so a rerun cannot replace prior evidence."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    for attempt in range(100):
        suffix = f"_{attempt}" if attempt else ""
        output = OUT_DIR / f"{stamp}_router_compare{suffix}.json"
        try:
            output.open("x", encoding="utf-8").close()
            return output
        except FileExistsError:
            continue
    raise FileExistsError("Could not reserve a unique router benchmark filename")


def _run_metadata(cases_path: Path, models: list[str]) -> dict:
    commit, dirty = _git_snapshot()
    return {
        "started_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "cases_sha256": hashlib.sha256(cases_path.read_bytes()).hexdigest(),
        "router_source_sha256": _source_hash(),
        "git_commit": commit,
        "git_dirty": dirty,
        "clock_mode": clock_mode(),
        "router_models": models,
        "api_endpoint_host": urlsplit(os.getenv("BASE_URL", DEFAULT_BASE_URL)).hostname,
    }


def load_cases(path: Path = CASES_PATH) -> list[dict]:
    cases = json.loads(path.read_text(encoding="utf-8"))
    ids = [case["id"] for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("router case IDs must be unique")
    for case in cases:
        if case["expected"] not in TARGETS or case.get("active_agent", "") not in ("", *TARGETS):
            raise ValueError(f"invalid agent in case {case['id']}")
        if len(case.get("history", [])) % 2:
            raise ValueError(f"history must contain user/assistant pairs: {case['id']}")
    return cases


def case_state(case: dict) -> dict:
    messages = []
    for index, content in enumerate(case.get("history", [])):
        messages.append(HumanMessage(content=content) if index % 2 == 0 else AIMessage(content=content))
    messages.append(HumanMessage(content=case["query"]))
    return {"active_agent": case.get("active_agent", ""), "messages": messages}


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def summarize(rows: list[dict]) -> dict:
    times = [row["elapsed_ms"] for row in rows]
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["group"]].append(row)
    return {
        "count": len(rows),
        "correct": sum(row["correct"] for row in rows),
        "accuracy": round(sum(row["correct"] for row in rows) / len(rows), 4),
        "p50_ms": round(statistics.median(times)),
        "p95_ms": round(percentile(times, 0.95)),
        "rule_fast_paths": sum(row["reason"].startswith("规则直达") for row in rows),
        "fallbacks": sum(row["fallback"] for row in rows),
        "by_group": {name: {"correct": sum(row["correct"] for row in group), "count": len(group)}
                     for name, group in groups.items()},
    }


async def benchmark(models: list[str], rounds: int, output: Path | None = None,
                    cases_path: Path = CASES_PATH) -> None:
    cases = load_cases(cases_path)
    results = {"cases": str(cases_path.relative_to(ROOT)), "rounds": rounds,
               "metadata": _run_metadata(cases_path, models), "models": {}}
    output = output or _reserve_default_output()
    output.parent.mkdir(parents=True, exist_ok=True)

    for model in models:
        agent = OrchestratorAgent(get_llm(temperature=0.0, model=model))
        rows = []
        print(f"\n{model}: {len(cases)} cases × {rounds} round(s)", flush=True)
        for round_index in range(rounds):
            for case in cases:
                start = time.perf_counter()
                decision = await agent.route(case_state(case))
                elapsed = (time.perf_counter() - start) * 1000
                row = {
                    "round": round_index + 1,
                    "id": case["id"],
                    "group": case["group"],
                    "expected": case["expected"],
                    "actual": decision["next_agent"],
                    "correct": decision["next_agent"] == case["expected"],
                    "elapsed_ms": round(elapsed),
                    "reason": decision["route_reason"],
                    "fallback": decision["route_reason"].startswith(("路由失败，", "重新路由失败，")),
                }
                rows.append(row)
                if not row["correct"]:
                    print(f"  {case['id']}: expected {row['expected']}, got {row['actual']}", flush=True)
            print(f"  round {round_index + 1}/{rounds} finished", flush=True)
        summary = summarize(rows)
        results["models"][model] = {"summary": summary, "rows": rows}
        output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  accuracy {summary['correct']}/{summary['count']} | "
              f"p50 {summary['p50_ms']} ms | p95 {summary['p95_ms']} ms | "
              f"fallbacks {summary['fallbacks']}", flush=True)
    print(f"Saved {output.relative_to(ROOT) if output.is_relative_to(ROOT) else output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["qwen-plus", "qwen-flash"])
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--cases", type=Path, default=CASES_PATH)
    parser.add_argument("--output", type=Path, help="explicit output path (may replace an existing file)")
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("--rounds must be >= 1")
    cases_path = args.cases if args.cases.is_absolute() else ROOT / args.cases
    output = (args.output if args.output.is_absolute() else ROOT / args.output) if args.output else None
    asyncio.run(benchmark(args.models, args.rounds, output, cases_path))
