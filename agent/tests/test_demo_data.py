"""
演示数据：时间平移 + 滚动生成航班（agent/database/demo_data.py），以及实时时钟下 MySQL 后端的日期处理。

数据库测试使用独立的库 airline_demo_test，测完删除，不影响 airline 和 airline_test。
"""

import json
from datetime import date, datetime, timedelta

import pytest

import demo_data as dd
from core import clock
from import_airline import connect

DEMO_DB = "airline_demo_test"
NOW = datetime(2026, 10, 30, 9, 30)            # 假定的"现在"（美东时间）
OFFSET = (NOW.date() - date(2024, 5, 15)).days
HORIZON = 20                                    # 原始数据平移后覆盖到今天往后 15 天，窗口 20 天才需要新生成


# ------------------------------------------------------------------ 时钟
def test_clock_modes(monkeypatch):
    monkeypatch.setenv(clock.ENV_KEY, "fixed")
    assert clock.now() == datetime(2024, 5, 15, 15, 0) and clock.is_fixed()
    assert clock.now_text().startswith("2024-05-15 15:00:00（星期三")
    assert clock.day_label(5) == "5 月 20 日"
    monkeypatch.setenv(clock.ENV_KEY, "live")
    assert not clock.is_fixed() and clock.now().year >= 2026
    assert clock.now(fixed=True) == clock.FIXED_NOW                # 评测模式强制固定时间
    monkeypatch.setenv(clock.ENV_KEY, "tomorrow")
    with pytest.raises(ValueError):
        clock.mode()


# ------------------------------------------------------------------ 纯函数
def test_scheduled_times_handles_next_day_arrival():
    dep, arr = dd.scheduled_times(date(2026, 11, 1), "22:30:00", "01:15:00+1")
    assert dep == datetime(2026, 11, 1, 22, 30) and arr == datetime(2026, 11, 2, 1, 15)


def test_generated_instance_is_deterministic_and_plausible():
    pool = dd.load_pools()["HAT001"]
    a = dd.generate_instance("HAT001", date(2026, 11, 3), pool)
    assert a == dd.generate_instance("HAT001", date(2026, 11, 3), pool)        # 同一航班同一天，结果相同
    assert a != dd.generate_instance("HAT001", date(2026, 11, 4), pool)
    lo = {c: min(p["prices"][c] for p in pool) for c in dd.CABINS}
    hi = {c: max(p["prices"][c] for p in pool) for c in dd.CABINS}
    for c in dd.CABINS:
        assert a[f"seats_{c}"] >= 0
        assert lo[c] * 0.9 - 1 <= a[f"price_{c}"] <= hi[c] * 1.1 + 1


def test_day_of_status_rates():
    day, dep, arr = date(2026, 11, 3), datetime(2026, 11, 3, 8), datetime(2026, 11, 3, 11)
    statuses = [dd.day_of_status(f"HAT{i:03d}", day, dep, arr)["status"] for i in range(1, 1001)]
    assert set(statuses) == {"on time", "delayed", "cancelled"}
    assert 0.05 <= statuses.count("delayed") / 1000 <= 0.15
    assert 0.01 <= statuses.count("cancelled") / 1000 <= 0.06


def test_progress_through_the_day():
    dep, arr = datetime(2026, 11, 3, 8), datetime(2026, 11, 3, 11)
    assert dd.progress("on time", None, dep, arr, datetime(2026, 11, 3, 7)) is None
    flying = dd.progress("on time", None, dep, arr, datetime(2026, 11, 3, 9))
    assert flying["status"] == "flying" and flying["actual_departure_time_est"] == dep
    assert dd.progress("flying", dep, None, arr, datetime(2026, 11, 3, 10)) is None
    landed = dd.progress("flying", dep, None, arr, datetime(2026, 11, 3, 12))
    assert landed["status"] == "landed" and landed["actual_arrival_time_est"] == arr


def test_rewind_to_current_time():
    sched = datetime(2026, 11, 3, 14, 0)
    dep, arr = datetime(2026, 11, 3, 14, 50), datetime(2026, 11, 3, 17, 0)
    back = dd.rewind("landed", dep, arr, None, sched, datetime(2026, 11, 3, 9))        # 还没起飞
    assert back["status"] == "delayed" and back["estimated_departure_time_est"] == dep
    assert back["actual_departure_time_est"] is None
    mid = dd.rewind("landed", dep, arr, None, sched, datetime(2026, 11, 3, 16))        # 在空中
    assert mid["status"] == "flying" and mid["estimated_arrival_time_est"] == arr
    assert dd.rewind("landed", dep, arr, None, sched, datetime(2026, 11, 3, 18)) is None


# ------------------------------------------------------------------ 数据库
def q(sql, *args):
    conn = connect()
    try:
        with conn.cursor() as cur:
            cur.execute(f"USE {DEMO_DB}")
            cur.execute(sql, args)
            return cur.fetchall()
    finally:
        conn.close()


@pytest.fixture(scope="module")
def demo_db():
    res = dd.rebase(DEMO_DB, now=NOW, horizon=HORIZON, verbose=False)
    yield res
    conn = connect()
    with conn.cursor() as cur:
        cur.execute(f"DROP DATABASE IF EXISTS {DEMO_DB}")
    conn.close()


def test_rebase_shifts_everything_by_the_same_offset(demo_db):
    assert demo_db["offset_days"] == OFFSET
    assert demo_db["generated"] == 5 * 300           # 今天往后第 16–20 天是新生成的
    assert q("SELECT MIN(date) FROM flight_instances")[0][0] == date(2024, 5, 1) + timedelta(days=OFFSET)
    # 订单的航段日期和下单时间平移了同样的天数（与原始 JSON 对比）
    raw = json.loads((dd.DATA / "reservations.json").read_text(encoding="utf-8"))["AIXC49"]
    (created,), = q("SELECT created_at FROM reservations WHERE reservation_id='AIXC49'")
    assert created == datetime.fromisoformat(raw["created_at"]) + timedelta(days=OFFSET)
    legs = q("SELECT date FROM reservation_flights WHERE reservation_id='AIXC49' ORDER BY seq")
    assert [d for (d,) in legs] == [date.fromisoformat(f["date"]) + timedelta(days=OFFSET) for f in raw["flights"]]
    # 每个订单航段都能找到对应的航班实例
    assert q("SELECT COUNT(*) FROM reservation_flights rf LEFT JOIN flight_instances fi "
             "ON fi.flight_number=rf.flight_number AND fi.date=rf.date WHERE fi.date IS NULL")[0][0] == 0


def test_window_is_filled_and_today_is_not_bookable(demo_db):
    today = NOW.date()
    assert q("SELECT COUNT(*) FROM flight_instances WHERE date=%s AND status='available'", today)[0][0] == 0
    for d in range(1, HORIZON + 1):
        n = q("SELECT COUNT(*) FROM flight_instances WHERE date=%s AND status='available'",
              today + timedelta(days=d))[0][0]
        assert n == 300, f"今天往后第 {d} 天只有 {n} 个可订航班"
    # 今天的航班状态和当前时刻一致：没有"未起飞却已降落 / 飞行中"，也没有"该起飞了还在准点"
    assert q("SELECT COUNT(*) FROM flight_instances WHERE status IN ('landed', 'flying') "
             "AND actual_departure_time_est > %s", NOW)[0][0] == 0
    assert q("SELECT COUNT(*) FROM flight_instances WHERE status='landed' AND actual_arrival_time_est > %s",
             NOW)[0][0] == 0
    assert q("SELECT COUNT(*) FROM flight_instances WHERE status IN ('on time', 'delayed') "
             "AND estimated_departure_time_est <= %s", NOW)[0][0] == 0
    assert demo_db["rewound"] > 0
    meta = dict(q("SELECT k, v FROM demo_meta"))
    assert meta["offset_days"] == str(OFFSET) and meta["anchor_date"] == today.isoformat()


def test_roll_is_idempotent_and_moves_forward(demo_db):
    assert dd.roll(DEMO_DB, now=NOW, horizon=HORIZON) == {"generated": 0, "day_of": 0, "progressed": 0}
    later = NOW + timedelta(days=2)
    stats = dd.roll(DEMO_DB, now=later, horizon=HORIZON)
    assert stats["generated"] == 600                  # 窗口往后移了 2 天
    assert stats["day_of"] == 600                      # 明天、后天的航班到了起飞当天（或已过去）
    assert q("SELECT COUNT(*) FROM flight_instances WHERE date<%s AND status IN "
             "('available', 'on time', 'delayed', 'flying')", later.date() - timedelta(days=1))[0][0] == 0


def test_mysql_backend_with_live_clock(demo_db, monkeypatch):
    """实时时钟下：直飞查询能查到平移后的日期；中转方案的次日日期按真实日历计算；新订单的下单时间是"现在"。"""
    from backends.mysql_backend import MySQLBackend
    monkeypatch.setenv(clock.ENV_KEY, "live")
    monkeypatch.setattr(clock, "now", lambda fixed=None: clock.FIXED_NOW if fixed else NOW)
    db = MySQLBackend(DEMO_DB, pool_size=1)
    day = (NOW.date() + timedelta(days=5)).isoformat()
    assert json.loads(db.search_direct_flight("JFK", "SEA", day))
    codes = [c for (c,) in q("SELECT code FROM airports")]
    plans = [p for o in codes for dst in codes if o != dst
             for p in json.loads(db.search_onestop_flight(o, dst, day))]
    assert plans
    for leg1, leg2 in plans:
        assert leg1["status"] == leg2["status"] == "available"
        expected = (date.fromisoformat(day) + timedelta(days=1)).isoformat() \
            if "+1" in leg1["scheduled_arrival_time_est"] else day
        assert leg2["date"] == expected
    assert any(leg2["date"] != leg1["date"] for leg1, leg2 in plans), "没有覆盖到次日出发的第二段"
