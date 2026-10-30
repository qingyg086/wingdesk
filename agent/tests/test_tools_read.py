"""
只读工具一致性测试：同样的输入，MySQLBackend 与 τ-bench 原版工具的返回必须**逐字相同**（字符串完全一致）。
"""

import itertools
import json

from helpers import call_both

AIRPORTS = ["ATL", "BOS", "CLT", "DEN", "DFW", "DTW", "EWR", "IAH", "JFK", "LAS",
            "LAX", "LGA", "MCO", "MIA", "MSP", "ORD", "PHL", "PHX", "SEA", "SFO"]
PAIRS = [(a, b) for a, b in itertools.product(AIRPORTS, AIRPORTS) if a != b]       # 380 个起降组合
# 05-01 / 05-08：触发"次日日期不补 0"特性；05-15：系统"今天"；05-20：有大量第二段次日出发的方案；
# 05-30：次日 05-31 不在数据里；外加 3 个不规范的日期
SEARCH_DATES = ["2024-05-01", "2024-05-08", "2024-05-15", "2024-05-20", "2024-05-30"]
BAD_DATES = ["2024-05-2", "2024-5-15", "2024-06-01"]


def test_get_user_details_all_users(ro, ref_tools):
    backend, data = ro
    for uid in list(data["users"]) + ["no_such_user_123"]:
        mine, ref = call_both(backend, data, ref_tools, "get_user_details", user_id=uid)
        assert mine == ref, uid


def test_get_reservation_details_all_reservations(ro, ref_tools):
    backend, data = ro
    for rid in list(data["reservations"]) + ["ZZZZZZ", "hathat"]:
        mine, ref = call_both(backend, data, ref_tools, "get_reservation_details", reservation_id=rid)
        assert mine == ref, rid


def test_list_all_airports(ro, ref_tools):
    backend, data = ro
    mine, ref = call_both(backend, data, ref_tools, "list_all_airports")
    assert mine == ref


def test_search_direct_flight(ro, ref_tools):
    backend, data = ro
    hits = 0
    for (o, d), date in itertools.product(PAIRS, SEARCH_DATES + BAD_DATES):
        mine, ref = call_both(backend, data, ref_tools, "search_direct_flight",
                              origin=o, destination=d, date=date)
        assert mine == ref, (o, d, date)
        hits += mine != "[]"
    assert hits > 100            # 确认测到了有结果的情况，而不是全是空列表


def test_search_onestop_flight(ro, ref_tools):
    """380 个组合 × 5 个日期，并统计第二段次日出发（原版特性 NOTE 2 生效）的方案数。"""
    backend, data = ro
    total = quirk_mismatch_date = 0
    for (o, d), date in itertools.product(PAIRS, SEARCH_DATES):
        mine, ref = call_both(backend, data, ref_tools, "search_onestop_flight",
                              origin=o, destination=d, date=date)
        assert mine == ref, (o, d, date)
        for leg1, leg2 in json.loads(mine):
            total += 1
            quirk_mismatch_date += leg1["date"] != leg2["date"]
    assert total > 1000
    assert quirk_mismatch_date > 0        # 确认"第二段次日出发"的分支也被比对到了
    print(f"\n中转方案共 {total} 个，其中第二段次日出发 {quirk_mismatch_date} 个")


def test_onestop_quirk_unpadded_date(ro, ref_tools):
    """原版特性：5 月 1 日出发、第一段次日到达时，第二段日期拼成 '2024-05-2'，这些方案被漏掉。
    这里证明 MySQL 后端也"漏掉"了它们（照搬原版），而不是意外修好。"""
    backend, data = ro
    overnight = [f for f in data["flights"].values() if "+1" in f["scheduled_arrival_time_est"]]
    f1 = overnight[0]
    mine = json.loads(backend.search_onestop_flight(f1["origin"], "SEA" if f1["destination"] != "SEA" else "BOS",
                                                    "2024-05-01"))
    assert all(leg1["flight_number"] != f1["flight_number"] for leg1, _ in mine)


def test_calculate(ro, ref_tools):
    backend, data = ro
    for expr in ["2 + 2", "100 * 3 + 30 * 3", "(1200 - 300) / 4", "10 / 3", "-5 + 2", "1/0", "1.0/0",
                 "2 +", "abc", "3 * (2 + 1", "  7  "]:
        mine, ref = call_both(backend, data, ref_tools, "calculate", expression=expr)
        if ref.startswith("Error") and "invalid syntax" in ref or "was never closed" in ref:
            assert mine.startswith("Error"), expr           # 语法错误：两边都报错即可，报错文本可以不同
        else:
            assert mine == ref, expr
    # 与原版的有意差异：不支持 **（防止 9**9**9**9 这类表达式把 CPU 跑满）
    assert ref_tools["calculate"](data, expression="2**10") == "1024.0"
    assert backend.calculate("2**10").startswith("Error")


def test_think_and_transfer(ro, ref_tools):
    backend, data = ro
    assert call_both(backend, data, ref_tools, "think", thought="x") == ("", "")
    mine, ref = call_both(backend, data, ref_tools, "transfer_to_human_agents", summary="need help")
    assert mine == ref == "Transfer successful"
