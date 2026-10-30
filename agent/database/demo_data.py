"""
演示数据：时间平移 + 滚动生成航班。只用于演示库（默认 airline）；测试库 airline_test 和 τ-bench 评测不受影响。

用法（在项目根目录）：
    python agent/database/demo_data.py rebase     # 重新导入原始数据，并整体平移到"今天"（会清掉演示库里新产生的订单）
    python agent/database/demo_data.py roll       # 补齐未来航班、推进航班状态（可重复执行；后端在 live 时钟下每小时自动执行）
    python agent/database/demo_data.py status     # 查看平移基准、航班日期范围和今天的航班状态

时间平移（rebase）：原始数据的设定时间是 2024-05-15 15:00（美东时间）。计算 offset = 今天 - 2024-05-15（天），
把航班实例的日期和时间、订单航段日期、订单下单时间都加上 offset。数据之间的相对关系不变：
原来"5 天后起飞"的订单平移后还是 5 天后起飞，"下单 24 小时内可取消"之类的规则照样成立。
原始数据记录的是下午 3 点时的航班状态，平移后再按当前时刻对齐：还没到起飞时间的"已降落 / 飞行中"
退回"准点 / 延误"（保留原来的延误时长），已起飞未到达的"已降落"改为"飞行中"。

滚动生成（roll）：
1. 补齐 [今天, 今天 + horizon] 内缺失的航班实例（状态 available）。余座和价格以原始数据里同一航班
   某个可订日期为基准，加小幅扰动（余座 ±3，价格 ±10%）。随机数种子是"航班号 + 日期"，重复执行结果相同。
2. 推进状态，规则与原始数据一致（当天的航班不再开放预订）：
   - 日期 <= 今天、仍是 available 的实例：约 10% 延误（30–240 分钟）、3% 取消，其余为准点；
     清空余座和价格，写入预计起降时间；
   - 准点 / 延误 / 飞行中的实例：起飞时间已过 → 飞行中；到达时间已过 → 已降落。
生成的航班数据都是仿真数据。
"""

import argparse
import asyncio
import json
import logging
import os
import random
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pymysql

sys.path.insert(0, str(Path(__file__).resolve().parent))         # import_airline
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # agent/（core.clock）

from import_airline import DATA, DB_NAME, connect, import_all    # noqa: E402
from core import clock                                            # noqa: E402

log = logging.getLogger(__name__)

BASE_DATE = clock.FIXED_NOW.date()          # 原始数据的"今天"：2024-05-15
HORIZON_DAYS = 60
DELAY_RATE, CANCEL_RATE = 0.10, 0.03
CABINS = ("basic_economy", "economy", "business")
CLEARED = {**{f"seats_{c}": None for c in CABINS}, **{f"price_{c}": None for c in CABINS}}
# 平移时要整体加 offset 的列（NULL 加天数仍是 NULL）
SHIFT_COLUMNS = {
    "flight_instances": ("date", "actual_departure_time_est", "actual_arrival_time_est",
                         "estimated_departure_time_est", "estimated_arrival_time_est"),
    "reservation_flights": ("date",),
    "reservations": ("created_at",),
}
META_DDL = "CREATE TABLE IF NOT EXISTS demo_meta (k VARCHAR(32) PRIMARY KEY, v VARCHAR(64) NOT NULL)"


def default_database() -> str:
    return os.getenv("MYSQL_DATABASE", DB_NAME)


# ------------------------------------------------------------------ 纯函数（便于单元测试）
def scheduled_times(day: date, dep: str, arr: str) -> tuple[datetime, datetime]:
    """计划起降时间。到达时间带 "+1" 表示次日到达。"""
    t = lambda s: datetime.strptime(s, "%H:%M:%S").time()
    plus_one = arr.endswith("+1")
    return (datetime.combine(day, t(dep)),
            datetime.combine(day + timedelta(days=1 if plus_one else 0), t(arr.removesuffix("+1"))))


def _rng(flight_number: str, day: date, salt: str) -> random.Random:
    return random.Random(f"{flight_number}|{day.isoformat()}|{salt}")


def load_pools() -> dict[str, list[dict]]:
    """每个航班在原始数据里的可订日期（available），作为生成余座和价格的基准。"""
    flights = json.loads((DATA / "flights.json").read_text(encoding="utf-8"))
    return {fn: [x for _, x in sorted(v["dates"].items()) if x["status"] == "available"]
            for fn, v in flights.items()}


def generate_instance(flight_number: str, day: date, pool: list[dict]) -> dict:
    """生成一个可订的航班实例。"""
    rng = _rng(flight_number, day, "generate")
    base = rng.choice(pool)
    row = {"flight_number": flight_number, "date": day, "status": "available"}
    for c in CABINS:
        row[f"seats_{c}"] = max(0, base["available_seats"][c] + rng.randint(-3, 3))
        row[f"price_{c}"] = max(1, round(base["prices"][c] * rng.uniform(0.9, 1.1)))
    return row


def day_of_status(flight_number: str, day: date, dep: datetime, arr: datetime) -> dict:
    """航班到了起飞当天：停止售票，按概率变成准点、延误或取消。返回要更新的字段。"""
    rng = _rng(flight_number, day, "day-of")
    r = rng.random()
    if r < CANCEL_RATE:
        return {"status": "cancelled", **CLEARED,
                "estimated_departure_time_est": None, "estimated_arrival_time_est": None}
    delay = timedelta(minutes=rng.randint(30, 240)) if r < CANCEL_RATE + DELAY_RATE else timedelta(0)
    return {"status": "delayed" if delay else "on time", **CLEARED,
            "estimated_departure_time_est": dep + delay, "estimated_arrival_time_est": arr + delay}


def progress(status: str, actual_dep, est_dep, est_arr, now: datetime) -> dict | None:
    """准点 / 延误 / 飞行中的航班随时间推进。不需要变化时返回 None。"""
    dep, arr = actual_dep or est_dep, est_arr
    if dep is None or arr is None:
        return None
    if arr <= now:
        return {"status": "landed", "actual_departure_time_est": dep, "actual_arrival_time_est": arr,
                "estimated_departure_time_est": None, "estimated_arrival_time_est": None}
    if dep <= now and status != "flying":
        return {"status": "flying", "actual_departure_time_est": dep,
                "estimated_departure_time_est": None, "estimated_arrival_time_est": arr}
    return None


def rewind(status: str, actual_dep, actual_arr, est_arr, sched_dep: datetime, now: datetime) -> dict | None:
    """平移后按当前时刻对齐：时间还没到的"已降落 / 飞行中"往回退。不需要变化时返回 None。"""
    if status not in ("landed", "flying") or actual_dep is None:
        return None
    arr = actual_arr or est_arr
    if actual_dep > now:
        late = actual_dep - sched_dep > timedelta(minutes=15)
        return {"status": "delayed" if late else "on time", "estimated_departure_time_est": actual_dep,
                "estimated_arrival_time_est": arr, "actual_departure_time_est": None, "actual_arrival_time_est": None}
    if status == "landed" and arr is not None and arr > now:
        return {"status": "flying", "estimated_arrival_time_est": arr, "actual_arrival_time_est": None}
    return None


# ------------------------------------------------------------------ 数据库操作
def _update(cur, flight_number: str, day: date, fields: dict) -> None:
    sets = ", ".join(f"{k}=%s" for k in fields)
    cur.execute(f"UPDATE flight_instances SET {sets} WHERE flight_number=%s AND date=%s",
                (*fields.values(), flight_number, day))


def _set_meta(cur, **kv) -> None:
    cur.execute(META_DDL)
    for k, v in kv.items():
        cur.execute("REPLACE INTO demo_meta (k, v) VALUES (%s, %s)", (k, str(v)))


def read_meta(database: str | None = None) -> dict:
    conn = connect()
    try:
        with conn.cursor() as cur:
            cur.execute(f"USE {database or default_database()}")
            cur.execute(META_DDL)
            cur.execute("SELECT k, v FROM demo_meta")
            return dict(cur.fetchall())
    finally:
        conn.close()


def roll(database: str | None = None, now: datetime | None = None, horizon: int = HORIZON_DAYS) -> dict:
    """补齐未来航班 + 推进状态，一个事务内完成。返回各类变更的数量。"""
    database = database or default_database()
    now = now or clock.now(fixed=False)
    today = now.date()
    stats = {"generated": 0, "day_of": 0, "progressed": 0}
    conn = connect()
    try:
        with conn.cursor(pymysql.cursors.DictCursor) as cur:
            cur.execute(f"USE {database}")
            cur.execute("SELECT flight_number, scheduled_departure_time_est AS dep, "
                        "scheduled_arrival_time_est AS arr FROM flights")
            sched = {r["flight_number"]: (r["dep"], r["arr"]) for r in cur.fetchall()}

            # 1. 补齐 [今天, 今天 + horizon] 缺失的航班实例
            cur.execute("SELECT flight_number, date FROM flight_instances WHERE date BETWEEN %s AND %s",
                        (today, today + timedelta(days=horizon)))
            have = {(r["flight_number"], r["date"]) for r in cur.fetchall()}
            pools = load_pools()
            rows = [generate_instance(fn, today + timedelta(days=d), pools[fn])
                    for d in range(horizon + 1) for fn in sorted(sched)
                    if (fn, today + timedelta(days=d)) not in have]
            if rows:
                cols = list(rows[0])
                cur.executemany(f"INSERT INTO flight_instances ({', '.join(cols)}) "
                                f"VALUES ({', '.join(['%s'] * len(cols))})", [tuple(r[c] for c in cols) for r in rows])
            stats["generated"] = len(rows)

            # 2. 到了起飞当天还开放预订的航班：停售，变成准点 / 延误 / 取消
            cur.execute("SELECT flight_number, date FROM flight_instances WHERE date <= %s AND status='available'",
                        (today,))
            for r in cur.fetchall():
                dep, arr = scheduled_times(r["date"], *sched[r["flight_number"]])
                _update(cur, r["flight_number"], r["date"], day_of_status(r["flight_number"], r["date"], dep, arr))
                stats["day_of"] += 1

            # 3. 准点 / 延误 / 飞行中 → 飞行中 / 已降落
            cur.execute("SELECT flight_number, date, status, actual_departure_time_est AS a_dep, "
                        "estimated_departure_time_est AS e_dep, estimated_arrival_time_est AS e_arr "
                        "FROM flight_instances WHERE date <= %s AND status IN ('on time', 'delayed', 'flying')",
                        (today,))
            for r in cur.fetchall():
                upd = progress(r["status"], r["a_dep"], r["e_dep"], r["e_arr"], now)
                if upd:
                    _update(cur, r["flight_number"], r["date"], upd)
                    stats["progressed"] += 1

            _set_meta(cur, last_roll_at=now.isoformat())
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return stats


def rebase(database: str | None = None, now: datetime | None = None, horizon: int = HORIZON_DAYS,
           verbose: bool = True) -> dict:
    """重新导入原始数据 → 整体平移到今天 → 滚动生成。返回 offset 和滚动统计。"""
    database = database or default_database()
    now = now or clock.now(fixed=False)
    offset = (now.date() - BASE_DATE).days
    if offset < 0:
        raise ValueError(f"今天 {now.date()} 早于原始数据的基准日期 {BASE_DATE}")
    if not import_all(database, verbose=verbose):
        raise RuntimeError("导入原始数据失败")
    conn = connect()
    try:
        with conn.cursor() as cur:
            cur.execute(f"USE {database}")
            # 航班实例的日期是主键的一部分，且被订单航段外键引用：平移期间暂停外键检查，
            # 并按日期从大到小更新，避免平移后的日期和尚未平移的行撞主键
            cur.execute("SET FOREIGN_KEY_CHECKS=0")
            for table, cols in SHIFT_COLUMNS.items():
                sets = ", ".join(f"{c} = DATE_ADD({c}, INTERVAL %s DAY)" for c in cols)
                order = " ORDER BY date DESC" if table == "flight_instances" else ""
                cur.execute(f"UPDATE {table} SET {sets}{order}", (offset,) * len(cols))
            cur.execute("SET FOREIGN_KEY_CHECKS=1")
            cur.execute("SELECT COUNT(*) FROM reservation_flights rf LEFT JOIN flight_instances fi "
                        "ON fi.flight_number = rf.flight_number AND fi.date = rf.date WHERE fi.date IS NULL")
            orphans = cur.fetchone()[0]
            if orphans:
                raise RuntimeError(f"平移后有 {orphans} 个订单航段找不到对应的航班实例")
            # 按当前时刻对齐：只可能影响昨天和今天的航班
            cur.execute("SELECT fi.flight_number, fi.date, fi.status, fi.actual_departure_time_est, "
                        "fi.actual_arrival_time_est, fi.estimated_arrival_time_est, f.scheduled_departure_time_est, "
                        "f.scheduled_arrival_time_est FROM flight_instances fi JOIN flights f USING (flight_number) "
                        "WHERE fi.date >= %s AND fi.status IN ('landed', 'flying')",
                        (now.date() - timedelta(days=1),))
            rewound = 0
            for fn, day, status, a_dep, a_arr, e_arr, s_dep, s_arr in cur.fetchall():
                upd = rewind(status, a_dep, a_arr, e_arr, scheduled_times(day, s_dep, s_arr)[0], now)
                if upd:
                    _update(cur, fn, day, upd)
                    rewound += 1
            _set_meta(cur, base_date=BASE_DATE.isoformat(), offset_days=offset,
                      anchor_date=(BASE_DATE + timedelta(days=offset)).isoformat(), rebased_at=now.isoformat())
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    stats = roll(database, now, horizon)
    return {"offset_days": offset, "rewound": rewound, **stats}


def summary(database: str | None = None, now: datetime | None = None) -> dict:
    database = database or default_database()
    today = (now or clock.now(fixed=False)).date()
    conn = connect()
    try:
        with conn.cursor() as cur:
            cur.execute(f"USE {database}")
            cur.execute(META_DDL)
            cur.execute("SELECT k, v FROM demo_meta")
            meta = dict(cur.fetchall())
            cur.execute("SELECT MIN(date), MAX(date) FROM flight_instances")
            first, last = cur.fetchone()
            cur.execute("SELECT status, COUNT(*) FROM flight_instances WHERE date=%s GROUP BY status", (today,))
            today_status = dict(cur.fetchall())
            cur.execute("SELECT COUNT(DISTINCT date) FROM flight_instances WHERE date > %s AND status='available'",
                        (today,))
            bookable_days = cur.fetchone()[0]
    finally:
        conn.close()
    return {"meta": meta, "first_date": str(first), "last_date": str(last),
            "today": str(today), "today_status": today_status, "bookable_days_ahead": bookable_days}


# ------------------------------------------------------------------ 后端定时任务
async def roll_periodically(interval_seconds: int = 3600) -> None:
    """后端在 live 时钟下运行：启动时滚动一次，之后每隔 interval_seconds 再滚动。
    演示库还没平移过（仍是 2024 年的原始数据）时只提示、不改数据。"""
    while True:
        try:
            meta = await asyncio.to_thread(read_meta)
            if "offset_days" not in meta:
                log.warning("演示库还没有做时间平移，航班仍停留在 2024 年。"
                            "请运行 python agent/database/demo_data.py rebase")
                return
            stats = await asyncio.to_thread(roll)
            log.info("演示数据滚动完成：%s", stats)
        except Exception:
            log.exception("演示数据滚动失败")
        await asyncio.sleep(interval_seconds)


def main() -> None:
    ap = argparse.ArgumentParser(description="演示数据：时间平移 + 滚动生成航班")
    ap.add_argument("command", choices=["rebase", "roll", "status"])
    ap.add_argument("--database", default=None, help="默认取 agent/.env 里的 MYSQL_DATABASE（airline）")
    ap.add_argument("--horizon", type=int, default=HORIZON_DAYS, help="生成到今天往后多少天，默认 60")
    args = ap.parse_args()

    if clock.is_fixed():
        print(f"[WARN] {clock.ENV_KEY} 不是 live：后端仍会按 2024-05-15 运行。"
              f"演示时请在 agent/.env 里设置 {clock.ENV_KEY}=live")
    if args.command == "rebase":
        res = rebase(args.database, horizon=args.horizon)
        print(f"[OK] 平移 {res['offset_days']} 天；{res['rewound']} 个航班按当前时刻对齐；新生成 {res['generated']} 个航班实例，"
              f"{res['day_of']} 个进入起飞当天，{res['progressed']} 个状态推进")
    elif args.command == "roll":
        print(f"[OK] {roll(args.database, horizon=args.horizon)}")
    s = summary(args.database)
    print(f"平移信息：{s['meta']}")
    print(f"航班日期范围：{s['first_date']} ~ {s['last_date']}；今天 {s['today']}（美东时间）")
    print(f"今天的航班状态：{s['today_status']}；往后可预订的天数：{s['bookable_days_ahead']}")


if __name__ == "__main__":
    main()
