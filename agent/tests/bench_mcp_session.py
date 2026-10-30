"""
对比 MCP 工具调用的两种方式的延迟：
  A：MultiServerMCPClient.get_tools() 的无状态模式——每次工具调用都新建 session、拉起一个子进程
  B：MCPToolRegistry 的长连接 session——整个应用只拉起一个子进程

用法（在项目根目录，需要 MySQL 已启动）：
    python agent/tests/bench_mcp_session.py            # 默认每种方式调用 20 次
    python agent/tests/bench_mcp_session.py -n 50
"""

import argparse
import asyncio
import statistics
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
AGENT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT_DIR))

import logging                                                     # noqa: E402
logging.disable(logging.WARNING)

from langchain_mcp_adapters.client import MultiServerMCPClient     # noqa: E402

from core.mcp.mcp_manager import MCPToolRegistry                   # noqa: E402

ARGS = {"origin": "JFK", "destination": "SEA", "date": "2024-05-20"}


async def timed(coro_factory, n):
    samples = []
    for _ in range(n):
        t0 = time.perf_counter()
        await coro_factory()
        samples.append((time.perf_counter() - t0) * 1000)
    return samples


def report(name, samples):
    s = sorted(samples)
    p95 = s[min(len(s) - 1, int(len(s) * 0.95))]
    print(f"{name:<28} 平均 {statistics.mean(s):7.1f} ms   中位数 {statistics.median(s):7.1f} ms   P95 {p95:7.1f} ms")


async def main(n):
    reg = MCPToolRegistry()
    servers = reg._load_servers()

    # 方式 A：无状态 client，每次调用都拉起新的子进程
    old_client = MultiServerMCPClient(servers)
    old_tool = next(t for t in await old_client.get_tools() if t.name == "search_direct_flight")
    old = await timed(lambda: old_tool.ainvoke(ARGS), n)

    # 方式 B：长连接 session
    await reg.start()
    try:
        new_tool = reg.get_tools(["search_direct_flight"])[0]
        await new_tool.ainvoke(ARGS)                                 # 预热：第一次调用要建数据库连接池
        new = await timed(lambda: new_tool.ainvoke(ARGS), n)
    finally:
        await reg.stop()

    print(f"\nsearch_direct_flight（JFK→SEA 2024-05-20），每种方式调用 {n} 次：")
    report("A：每次调用新建 session", old)
    report("B：全局长连接 session", new)
    print(f"平均延迟降低 {statistics.mean(old) / statistics.mean(new):.1f} 倍")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=20)
    asyncio.run(main(ap.parse_args().n))
