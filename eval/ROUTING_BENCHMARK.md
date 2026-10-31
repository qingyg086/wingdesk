# Router latency and accuracy — 2026-11-01

## Change and scope

The product orchestrator uses `qwen-flash`; specialist agents still use `qwen-plus`.
Explicit requests for a human agent and standalone baggage-policy questions while
the flight-search agent is active take narrow rule paths. Ordinary follow-ups still
go through the model: a short reply may start a different task.

The model choice is configurable with `ROUTER_MODEL` in `agent/.env`. The benchmark
used the default Beijing DashScope endpoint. [Qwen Flash's official model page](https://help.aliyun.com/zh/model-studio/qwen-flash)
lists Function Calling support for that region; support differs by region.

## Labeled route cases

`agent/scripts/bench_router.py` ran the same prompts sequentially against both
models at temperature 0. The 36 development cases include first requests, ongoing
tasks, task switches, human handoff, and misleading mentions of a human agent.
The 20 separate holdout cases were prepared before the final model comparison.
Both sets include Chinese and English. Each case is one independent conversation
snapshot; this benchmark makes no airline-tool or database calls. Both models use
the same rule paths (5 development cases and 2 holdout cases).

| Set | Model | Correct | Route p50 | Route p95 |
|---|---|---:|---:|---:|
| Development (36) | `qwen-plus` | 36/36 | 1,506 ms | 2,164 ms |
| Development (36) | `qwen-flash` | 36/36 | 614 ms | 780 ms |
| Holdout (20) | `qwen-plus` | 20/20 | 1,528 ms | 2,066 ms |
| Holdout (20) | `qwen-flash` | 20/20 | 585 ms | 696 ms |

The measured route p50 fell 59% on the development set and 62% on the holdout
set. No model-call fallback errors occurred in these final runs. Raw cases and results:
[`router_cases.json`](router_cases.json), [`router_holdout.json`](router_holdout.json),
[`20261101_router_final_compare.json`](results/20261101_router_final_compare.json), and
[`20261101_router_final_holdout.json`](results/20261101_router_final_holdout.json).
The result files record the UTC run time, case and source hashes, model names,
API endpoint host, and each case's decision and elapsed time.

To repeat the route comparison with an API key in `agent/.env`:

```bash
python agent/scripts/bench_router.py --cases eval/router_cases.json
python agent/scripts/bench_router.py --cases eval/router_holdout.json
```

## End-to-end streaming

`bench_streaming.py` ran five read-only questions twice per configuration against
the real agent graph and local services. It measured the first specialist-node
event, first answer token, and total turn duration. The semantic cache and
long-term memory are bypassed by this script. Routes were identical in all ten
baseline/optimized pairs.

| Median over 10 turns | Before | After | Change |
|---|---:|---:|---:|
| First specialist event | 2,402 ms | 1,826 ms | −24.0% |
| First answer token | 3,745 ms | 3,081 ms | −17.7% |
| Total turn | 6,735 ms | 6,658 ms | −1.1% |

Raw results: [`baseline`](results/20261101_streaming_baseline.json) and
[`final optimized`](results/20261101_streaming_final_optimized.json). The final
optimized run recorded zero router fallbacks. The baseline predates the benchmark
metadata addition but uses the same five questions, two rounds, and graph path.
With the local services running, compare the product graph using:

```bash
ROUTER_MODEL=qwen-plus python agent/scripts/bench_streaming.py --rounds 2 --tag baseline
ROUTER_MODEL=qwen-flash python agent/scripts/bench_streaming.py --rounds 2 --tag optimized
```

These are small, single-pass measurements on handcrafted route cases and ten
streaming turns. Network and model variation can change the results, especially
full-turn duration, which includes specialist and tool work. They show no routing
regression in these cases, not a guaranteed production accuracy.
The τ-bench task-success evaluation was not rerun for this change.
