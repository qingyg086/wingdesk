"""
τ-bench 评测链路的离线测试：假模型 + 脚本化的模拟用户，跑真实的 τ-bench 环境和打分逻辑。

验证的是"接线"是否正确，不是模型好不好：
- 多 Agent 图的工具调用确实变成了 env.step，改动了 τ-bench 自己的数据，哈希比对能拿到 1 分；
- 不执行写操作 → 数据库状态和标准答案不一致 → 0 分；
- 调用 transfer_to_human_agents 后本题立即结束，不再把回复发给模拟用户；
- 记录的动作里不混入 τ-bench 算分时重放的标准答案。
没有安装 tau-bench 时自动跳过（安装方法见 eval/README.md）。
"""

import sys
from pathlib import Path

import pytest

pytest.importorskip("tau_bench")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "eval"))

from fake_llm import ScriptedLLM, say, tool_call   # noqa: E402
from test_graph import R                           # noqa: E402


class ScriptedUser:
    """替代 τ-bench 的 LLM 用户模拟器：按顺序说出预先写好的话。"""

    def __init__(self, lines):
        self.lines = list(lines)
        self.heard: list[str] = []          # Agent 发给用户的每一条回复

    def reset(self, instruction=None):
        return self.lines.pop(0)

    def step(self, content):
        self.heard.append(content)
        return self.lines.pop(0)

    def get_total_cost(self):
        return 0.0


def make_env(task_index, user_lines):
    from tau_bench.envs.airline import MockAirlineDomainEnv
    env = MockAirlineDomainEnv(user_strategy="human", user_model="none", task_index=task_index)
    env.user = ScriptedUser(user_lines)
    return env


@pytest.mark.asyncio
async def test_cancel_task_scores_one():
    """task 1 的标准答案是 cancel_reservation(Z7GOZK)：Agent 照做 → reward = 1。"""
    from wingdesk_agent import WingDeskAgent
    from run_eval import instrument

    env = make_env(1, ["Hi, I'm olivia_gonzalez_2305. Please cancel reservation Z7GOZK.", "yes", "###STOP###"])
    recorded = instrument(env)
    llm = ScriptedLLM()
    llm.routes = [R("cancel_refund_agent"), R("cancel_refund_agent")]
    llm.script = [tool_call("get_reservation_details", reservation_id="Z7GOZK"),
                  say("Reservation Z7GOZK can be cancelled. Do you confirm? (yes/no)"),
                  tool_call("cancel_reservation", reservation_id="Z7GOZK"),
                  say("Your reservation Z7GOZK has been cancelled.")]
    res = await WingDeskAgent(llm=llm).asolve(env, 1)

    assert res["reward"] == 1.0 and res["stop_reason"] == "user_stop"
    assert [t["route"] for t in res["turns"]] == [["orchestrator", "cancel_refund_agent"]] * 2
    assert env.user.heard[-1] == "Your reservation Z7GOZK has been cancelled."
    assert [a["name"] for a in recorded] == ["get_reservation_details", "respond", "cancel_reservation", "respond"]
    assert res["info"]["reward_info"]["info"]["r_actions"] == 1.0


@pytest.mark.asyncio
async def test_missing_write_scores_zero():
    """同一题，Agent 没有执行取消 → 数据库状态与标准答案不一致 → reward = 0。"""
    from wingdesk_agent import WingDeskAgent

    env = make_env(1, ["Cancel reservation Z7GOZK please, user olivia_gonzalez_2305.", "###STOP###"])
    llm = ScriptedLLM()
    llm.routes = [R("cancel_refund_agent")]
    llm.script = [say("Sorry, this reservation cannot be cancelled.")]
    res = await WingDeskAgent(llm=llm).asolve(env, 1)
    assert res["reward"] == 0.0
    assert res["info"]["reward_info"]["info"]["r_actions"] == 0.0


@pytest.mark.asyncio
async def test_transfer_ends_task_immediately():
    """task 13 的标准答案是转人工（不改数据）：调用 transfer 后立即结束打分，不再给用户发消息。
    用户没要求人工时，路由器选 human_handoff 会被代码改成业务 Agent，由业务 Agent 按政策自己转人工。"""
    from wingdesk_agent import WingDeskAgent

    env = make_env(13, ["I want to change my flight in reservation XEWRD9 to a nonstop one. I'm james_lee_6136."])
    llm = ScriptedLLM()
    llm.routes = [R("human_handoff"), R("reservation_agent")]
    llm.script = [tool_call("transfer_to_human_agents", summary="Partially used reservation XEWRD9."),
                  say("I have transferred you to a human agent.")]
    res = await WingDeskAgent(llm=llm).asolve(env, 13)
    assert res["stop_reason"] == "transfer" and res["reward"] == 1.0
    assert env.user.heard == []                       # 转人工后没有再调用模拟用户


def test_summary_pass_k():
    """pass^k：同一题跑 k 次全部成功的概率（τ-bench 论文的定义），用组合数无偏估计。"""
    from run_eval import summarize

    class A:
        trials, price_in, price_out = 2, 0.8, 2.0
    rows = [{"task_id": 0, "reward": 1.0}, {"task_id": 0, "reward": 1.0},
            {"task_id": 1, "reward": 1.0}, {"task_id": 1, "reward": 0.0}]
    for r in rows:
        r.update(usage={"agent": {"prompt_tokens": 0, "completion_tokens": 0},
                        "user": {"prompt_tokens": 0, "completion_tokens": 0}},
                 n_user_turns=1, n_tool_calls=1, seconds=1.0)
    s = summarize(rows, A())
    assert s["pass^1"] == 0.75 and s["pass^2"] == 0.5
