"""
命令行入口：不启动 Web 服务，直接在终端里和多 Agent 系统对话，方便调试。

用法（在项目根目录）：
    python agent/main.py                                  # 交互模式，默认以 mia_li_3668 登录
    python agent/main.py --user lucas_wilson_8118         # 换一个登录用户
    python agent/main.py --query "帮我查 5 月 20 日纽约到西雅图的航班"   # 单次提问

交互模式里：遇到写操作会显示确认摘要，输入 y 确认、n 取消，也可以直接输入修改意见；输入 exit 退出。
"""

import argparse
import asyncio
import logging
import os
import sys
import uuid
from pathlib import Path

os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
sys.path.insert(0, str(Path(__file__).parent))
if hasattr(sys.stdin, "reconfigure"):
    sys.stdin.reconfigure(encoding="utf-8", errors="replace")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from core.mcp.mcp_manager import mcp_registry          # noqa: E402
from core.workflow.graph_manager import AgentGraphManager   # noqa: E402
from core.workflow.runner import run_turn               # noqa: E402

AGENT_ZH = {"flight_search_agent": "航班查询", "booking_agent": "订票", "reservation_agent": "订单管理",
            "cancel_refund_agent": "退票补偿", "policy_agent": "政策咨询", "human_handoff": "转人工"}


def show(turn):
    agent = turn.route[-1] if turn.route else ""
    tools = " → ".join(c["name"] for c in turn.tool_calls)
    print(f"  路由: {' → '.join(turn.route)}（{AGENT_ZH.get(agent, agent)}）" + (f"  工具: {tools}" if tools else ""))
    print(f"客服> {turn.text}\n")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", default="mia_li_3668", help="登录用户 ID")
    ap.add_argument("--session", default=None, help="会话 ID，默认随机")
    ap.add_argument("--query", default=None, help="单次提问后退出")
    ap.add_argument("--log", default="WARNING")
    args = ap.parse_args()
    logging.basicConfig(level=args.log.upper(), format="%(levelname)s %(name)s: %(message)s")

    session = args.session or f"cli-{uuid.uuid4().hex[:8]}"
    await mcp_registry.start()
    try:
        graph = AgentGraphManager().build_graph()
        print(f" 航空客服 CLI | 登录用户：{args.user} | 会话：{session} | 输入 exit 退出\n")
        queries = [args.query] if args.query else None
        while True:
            if queries is not None:
                if not queries:
                    break
                text = queries.pop(0)
                print(f"用户> {text}")
            else:
                text = input("用户> ").strip()
                if text.lower() in ("exit", "quit", "q"):
                    break
                if not text:
                    continue
            turn = await run_turn(graph, args.user, session, text)
            show(turn)
            while turn.confirm and queries is None:       # 有待确认的写操作
                ans = input("确认执行吗？(y = 确认 / n = 取消 / 或直接输入修改意见) ").strip()
                confirm = True if ans.lower() in ("y", "yes") else False if ans.lower() in ("n", "no") else None
                turn = await run_turn(graph, args.user, session, ans or "n", confirm=confirm)
                show(turn)
    finally:
        await mcp_registry.stop()


if __name__ == "__main__":
    asyncio.run(main())
