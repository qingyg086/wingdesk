"""
流式 vs 非流式的首字延迟对比（真实环境：qwen-plus + MySQL + Neo4j + Milvus）。

用法（在项目根目录，Docker 里的数据库需已启动）：
    python agent/scripts/bench_streaming.py              # 每个问题跑 2 轮
    python agent/scripts/bench_streaming.py --rounds 3

怎么比：每个问题在一个全新的会话里跑一遍 stream_turn（和网页走同一条代码路径），记录
    first_step  第一个"进入专业 Agent"事件的时间（用户看到"已转给 xx Agent"）
    first_token 第一个回复文字的时间 —— 真流式下用户看到第一个字的时间
    total       整轮结束的时间       —— 非流式（先跑完整轮再推文字）的首字时间 ≈ total
只跑只读问题（查航班、查订单、问政策），不会改动数据库。不经过语义缓存和长期记忆，只测 Agent 图本身。
"""

import argparse
import asyncio
import json
import logging
import os
import statistics
import sys
import time
import uuid
import warnings
from datetime import date
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "agent"))
logging.basicConfig(level=logging.WARNING)
logging.getLogger("core.mcp.interceptors").setLevel(logging.ERROR)

from core.clock import day_label                           # noqa: E402
from core.mcp.mcp_manager import mcp_registry              # noqa: E402
from core.workflow.graph_manager import AgentGraphManager  # noqa: E402
from core.workflow.runner import stream_turn               # noqa: E402

USER = "mia_li_3668"
QUESTIONS = [
    f"{day_label(5)}从纽约 JFK 飞西雅图 SEA 有哪些直飞航班？",          # 固定时钟下是 5 月 20 日
    f"{day_label(3)}从纽约 JFK 到洛杉矶 LAX，有没有转机一次的航班？",    # 固定时钟下是 5 月 18 日
    "帮我查一下我名下都有哪些订单。",
    "基础经济舱的票可以退吗？",
    "金卡会员坐经济舱能免费托运几件行李？",
]
OUT_DIR = ROOT / "eval" / "results"


async def measure(graph, question: str) -> dict:
    session = f"bench-{uuid.uuid4().hex[:8]}"
    t0 = time.perf_counter()
    first_step = first_token = None
    n_tokens, route, tools = 0, [], []
    async for ev in stream_turn(graph, USER, session, question):
        if not ev:
            continue
        now = (time.perf_counter() - t0) * 1000
        if ev["type"] == "node" and ev["key"] not in ("orchestrator", "confirm") and first_step is None:
            first_step = now
        elif ev["type"] == "token":
            n_tokens += 1
            if first_token is None:
                first_token = now
        elif ev["type"] == "done":
            route, tools = ev["route"], [c["name"] for c in ev["tool_calls"]]
    total = (time.perf_counter() - t0) * 1000
    return {"question": question, "route": route, "tools": tools, "token_events": n_tokens,
            "first_step_ms": round(first_step or 0), "first_token_ms": round(first_token or total),
            "total_ms": round(total)}


async def main(rounds: int):
    await mcp_registry.start()
    try:
        graph = AgentGraphManager().build_graph()
        rows = []
        for r in range(rounds):
            print(f"\n━━━━ 第 {r + 1}/{rounds} 轮 ━━━━")
            for q in QUESTIONS:
                row = await measure(graph, q)
                rows.append(row)
                print(f"  {q[:26]:<28} {' → '.join(row['route']):<38} "
                      f"转入Agent {row['first_step_ms']:>5} ms | 首字 {row['first_token_ms']:>5} ms | "
                      f"总耗时 {row['total_ms']:>5} ms | {row['token_events']} 段")
    finally:
        await mcp_registry.stop()

    ft = [r["first_token_ms"] for r in rows]
    tt = [r["total_ms"] for r in rows]
    fs = [r["first_step_ms"] for r in rows if r["first_step_ms"]]
    print("\n================ 汇总（%d 次）================" % len(rows))
    print(f"  看到'转入 xx Agent' 中位数   {statistics.median(fs):>6.0f} ms")
    print(f"  真流式首字      中位数       {statistics.median(ft):>6.0f} ms   平均 {statistics.mean(ft):.0f} ms")
    print(f"  非流式首字(=总耗时) 中位数   {statistics.median(tt):>6.0f} ms   平均 {statistics.mean(tt):.0f} ms")
    print(f"  首字延迟降低（中位数）       {1 - statistics.median(ft) / statistics.median(tt):.0%}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{date.today():%Y%m%d}_streaming.json"
    out.write_text(json.dumps({"rounds": rounds, "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"结果已保存：{out.relative_to(ROOT)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=2)
    asyncio.run(main(ap.parse_args().rounds))
