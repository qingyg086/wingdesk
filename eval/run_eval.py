"""
τ-bench airline 跑分脚本。

用法（在项目根目录；需要 agent/.env 里的 DASHSCOPE_API_KEY；不需要启动任何数据库）：
    python eval/run_eval.py --agent multi  --task-ids 0 1 2 3 4        # 翼枢多 Agent，先跑 5 题
    python eval/run_eval.py --agent single --task-ids 0 1 2 3 4        # τ-bench 自带的单 Agent 基线
    python eval/run_eval.py --agent multi  --start 0 --end 50          # 全量 50 题
    python eval/run_eval.py --agent multi  --lang zh ...               # 中文任务（需要 eval/tasks_zh.json）

两种 Agent 用同一套环境、同一批任务、同一个模型（默认 qwen-plus），模拟用户也是 qwen-plus：
    single：tau_bench.agents.tool_calling_agent.ToolCallingAgent（官方代码，英文原版政策做系统提示词）
    multi ：eval/wingdesk_agent.py，翼枢的 orchestrator + 5 个专业 Agent（评测模式）

输出：每题打印 [OK]/[FAIL]；最后打印成功率、平均轮数、平均工具调用数、token 与花费估算；
结果逐题写进 eval/results/<日期>_<agent>_<lang>_<model>.json（跑一半中断也不丢）。

花费估算：按 --price-in / --price-out（元 / 百万 token）计算，默认值是填写时查到的 qwen-plus 价格，
**以百炼控制台的账单为准**。token 数是实测的。
"""

import argparse
import asyncio
import json
import os
import sys
import time
import traceback
from datetime import datetime
from math import comb
from pathlib import Path

# 评测一律用 τ-bench 的固定时间（2024-05-15 15:00），不受 agent/.env 里 WINGDESK_CLOCK=live 的影响
os.environ["WINGDESK_CLOCK"] = "fixed"

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "agent")]

import llm_shim  # noqa: E402  先于 tau_bench 导入：设置 litellm 环境变量

OUT_DIR = ROOT / "eval" / "results"
TASKS_ZH = ROOT / "eval" / "tasks_zh.json"
ZH_USER_RULE = "\n- 请全程使用简体中文与客服交流（客服说英文时也用中文回复）。"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", choices=["multi", "single"], default="multi")
    ap.add_argument("--lang", choices=["en", "zh"], default="en")
    ap.add_argument("--task-ids", type=int, nargs="+")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=5, help="不含 end；全量用 --end 50")
    ap.add_argument("--trials", type=int, default=1, help=">1 时计算 pass^k")
    ap.add_argument("--model", default="qwen-plus")
    ap.add_argument("--user-model", default="qwen-plus")
    ap.add_argument("--max-steps", type=int, default=30, help="每题最多动作数（工具调用 + 回复），与 τ-bench 默认一致")
    ap.add_argument("--price-in", type=float, default=0.8, help="输入价格，元/百万 token")
    ap.add_argument("--price-out", type=float, default=2.0, help="输出价格，元/百万 token")
    ap.add_argument("--tag", default="", help="附加在结果文件名里，便于区分不同实验")
    return ap.parse_args()


def make_env(args, task_index: int):
    from tau_bench.envs.airline import MockAirlineDomainEnv
    env = MockAirlineDomainEnv(user_strategy="llm", user_model=args.user_model,
                               user_provider=llm_shim.PROVIDER, task_index=task_index)
    if args.lang == "zh":
        zh = json.loads(TASKS_ZH.read_text(encoding="utf-8"))["instructions"]
        env.tasks = [t.model_copy(update={"instruction": zh[str(i)]}) if str(i) in zh else t
                     for i, t in enumerate(env.tasks)]
        orig = env.user.build_system_prompt
        env.user.build_system_prompt = lambda instruction: orig(instruction) + ZH_USER_RULE
    return env


def instrument(env) -> list:
    """记录 Agent 实际执行的动作。env.step 在算分时会递归调用自身重放标准答案，用深度计数把那部分排除掉。"""
    recorded, depth = [], [0]
    orig = env.step

    def step(action):
        if depth[0] == 0:
            recorded.append({"name": action.name, "kwargs": action.kwargs})
        depth[0] += 1
        try:
            return orig(action)
        finally:
            depth[0] -= 1
    env.step = step
    return recorded


def make_multi_llm(model: str):
    import os
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(api_key=os.getenv("DASHSCOPE_API_KEY"), model=model,
                      base_url=os.getenv("BASE_URL", llm_shim.DEFAULT_BASE_URL),
                      temperature=0.0, timeout=120, max_retries=3,
                      stream_usage=True)          # 流式调用时也返回 token 用量


async def run_one(args, agent, task_index: int, trial: int) -> dict:
    before = llm_shim.snapshot()        # 放在建环境之前：τ-bench 建环境时用户模拟器就会先调一次模型
    t0 = time.perf_counter()
    env = make_env(args, task_index)
    actions = instrument(env)
    task = env.tasks[task_index]
    row = {"task_id": task_index, "trial": trial, "user_id": task.user_id,
           "expected_actions": [{"name": a.name, "kwargs": a.kwargs} for a in task.actions],
           "expected_outputs": task.outputs}
    try:
        if args.agent == "multi":
            res = await agent.asolve(env, task_index, max_num_steps=args.max_steps)
            agent_usage = {"prompt_tokens": sum(v.get("input_tokens", 0) for v in res["usage"].values()),
                           "completion_tokens": sum(v.get("output_tokens", 0) for v in res["usage"].values())}
            row.update(reward=res["reward"], stop_reason=res["stop_reason"], turns=res["turns"],
                       traj=res["messages"], reward_info=res["info"].get("reward_info"))
        else:
            res = await asyncio.to_thread(agent.solve, env, task_index, args.max_steps)
            row.update(reward=res.reward, traj=res.messages, reward_info=res.info.get("reward_info"),
                       stop_reason="done" if res.info.get("reward_info") else "max_steps")
            agent_usage = None
    except Exception as exc:
        row.update(reward=0.0, error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        agent_usage = None
    used = llm_shim.diff(before, llm_shim.snapshot())
    if agent_usage is None:
        agent_usage = {k: used.get("agent", {}).get(k, 0) for k in ("prompt_tokens", "completion_tokens")}
    row["usage"] = {"agent": agent_usage,
                    "user": {k: used.get("user", {}).get(k, 0) for k in ("prompt_tokens", "completion_tokens")}}
    row["seconds"] = round(time.perf_counter() - t0, 1)
    row["actions"] = actions
    row["n_user_turns"] = sum(a["name"] == "respond" for a in actions)
    row["n_tool_calls"] = sum(a["name"] != "respond" for a in actions)
    return row


def cost(usage: dict, args) -> float:
    return (usage["prompt_tokens"] * args.price_in + usage["completion_tokens"] * args.price_out) / 1e6


def summarize(rows: list[dict], args) -> dict:
    n = len(rows)
    by_task: dict[int, list[float]] = {}
    for r in rows:
        by_task.setdefault(r["task_id"], []).append(r["reward"])
    pass_k = {}
    for k in range(1, args.trials + 1):
        vals = [comb(sum(1 for x in v if x == 1), k) / comb(len(v), k) for v in by_task.values() if len(v) >= k]
        pass_k[f"pass^{k}"] = round(sum(vals) / len(vals), 4) if vals else None
    tot = {w: {"prompt_tokens": sum(r["usage"][w]["prompt_tokens"] for r in rows),
               "completion_tokens": sum(r["usage"][w]["completion_tokens"] for r in rows)} for w in ("agent", "user")}
    return {
        "n_runs": n, "n_tasks": len(by_task),
        "success": sum(r["reward"] == 1 for r in rows), "avg_reward": round(sum(r["reward"] for r in rows) / n, 4),
        **pass_k,
        "avg_user_turns": round(sum(r["n_user_turns"] for r in rows) / n, 1),
        "avg_tool_calls": round(sum(r["n_tool_calls"] for r in rows) / n, 1),
        "avg_seconds": round(sum(r["seconds"] for r in rows) / n, 1),
        "errors": sum(1 for r in rows if r.get("error")),
        "stop_reasons": {s: sum(1 for r in rows if r.get("stop_reason") == s)
                         for s in sorted({r.get("stop_reason") for r in rows if r.get("stop_reason")})},
        "tokens": tot,
        "cost_yuan": {w: round(cost(tot[w], args), 4) for w in tot},
    }


async def main():
    args = parse_args()
    llm_shim.install()
    if args.lang == "zh" and not TASKS_ZH.exists():
        sys.exit(f"缺少 {TASKS_ZH.relative_to(ROOT)}，中文评测需要先生成中文任务。")

    ids = args.task_ids or list(range(args.start, args.end))
    if args.agent == "multi":
        from wingdesk_agent import WingDeskAgent
        agent = WingDeskAgent(llm=make_multi_llm(args.model))
    else:
        from tau_bench.agents.tool_calling_agent import ToolCallingAgent
        probe = make_env(args, ids[0])
        agent = ToolCallingAgent(tools_info=probe.tools_info, wiki=probe.wiki, model=args.model,
                                 provider=llm_shim.PROVIDER, temperature=0.0)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    name = f"{stamp}_{args.agent}_{args.lang}_{args.model}" + (f"_{args.tag}" if args.tag else "")
    out = OUT_DIR / f"tau_{name}.json"
    meta = {"agent": args.agent, "lang": args.lang, "model": args.model, "user_model": args.user_model,
            "task_ids": ids, "trials": args.trials, "max_steps": args.max_steps,
            "price_yuan_per_million": {"input": args.price_in, "output": args.price_out}}
    print(f"▶ {args.agent} | {args.lang} | 模型 {args.model} | 模拟用户 {args.user_model} | 任务 {ids} × {args.trials} 轮")

    rows = []
    for trial in range(args.trials):
        for i in ids:
            row = await run_one(args, agent, i, trial)
            rows.append(row)
            tok = row["usage"]["agent"]["prompt_tokens"] + row["usage"]["agent"]["completion_tokens"]
            mark = "[OK]" if row["reward"] == 1 else "[FAIL]"
            extra = f" | 出错 {row['error'][:80]}" if row.get("error") else ""
            print(f"{mark} task {i:>2} (trial {trial}) | {row['n_user_turns']} 轮对话 {row['n_tool_calls']} 次工具 | "
                  f"Agent {tok} tokens | {row['seconds']}s | 结束原因 {row.get('stop_reason')}{extra}")
            out.write_text(json.dumps({"meta": meta, "summary": summarize(rows, args), "rows": rows},
                                      ensure_ascii=False, indent=1, default=str), encoding="utf-8")

    s = summarize(rows, args)
    print("\n================ 汇总 ================")
    print(f"成功 {s['success']}/{s['n_runs']}（pass^1 = {s['pass^1']:.0%}）" +
          "".join(f"  {k} = {v:.0%}" for k, v in s.items() if k.startswith("pass^") and k != "pass^1" and v is not None))
    print(f"平均每题：{s['avg_user_turns']} 轮对话，{s['avg_tool_calls']} 次工具调用，{s['avg_seconds']} 秒；出错 {s['errors']} 题")
    print(f"结束原因：{s['stop_reasons']}")
    for w, label in (("agent", "被测 Agent"), ("user", "模拟用户")):
        t = s["tokens"][w]
        print(f"{label}：输入 {t['prompt_tokens']:,} / 输出 {t['completion_tokens']:,} tokens，约 ¥{s['cost_yuan'][w]:.2f}")
    total = s["cost_yuan"]["agent"] + s["cost_yuan"]["user"]
    print(f"合计约 ¥{total:.2f}；按此估算 50 题约 ¥{total / max(s['n_runs'], 1) * 50:.2f}（价格以百炼控制台为准）")
    print(f"结果已保存：{out.relative_to(ROOT)}")


if __name__ == "__main__":
    asyncio.run(main())
