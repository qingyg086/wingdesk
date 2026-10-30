"""
把 τ-bench airline 数据（data/airline/*.json）导入 MySQL 的 airline 库。

用法（在项目根目录）：
    python agent/database/import_airline.py                      # 导入正式库 airline
    python agent/database/import_airline.py --database airline_test   # 导入测试库（pytest 会自动调用）

- 可重复执行：先执行 airline_schema.sql（DROP + CREATE），再全量导入。
- 整个导入在一个事务里完成：中途出错会回滚，不会留下"导了一半"的数据。
- 导入后自动核对行数，与 τ-bench 原始 JSON 对不上就以非 0 退出码结束。
- 连接信息读 agent/.env 里的 MYSQL_HOST / MYSQL_PORT / MYSQL_USER / MYSQL_PASSWORD。
"""

import json
import os
import sys
import time
from pathlib import Path

import pymysql
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]          # 项目根目录
DATA = ROOT / "data" / "airline"
SCHEMA = Path(__file__).resolve().parent / "airline_schema.sql"
DB_NAME = "airline"
BATCH = 1000
ADDRESS_KEYS = ["address1", "address2", "city", "country", "state", "zip"]   # 原版 JSON 的字段顺序

load_dotenv(ROOT / "agent" / ".env")


def connect():
    return pymysql.connect(
        host=os.getenv("MYSQL_HOST", "localhost"),
        port=int(os.getenv("MYSQL_PORT", "3306")),
        user=os.getenv("MYSQL_USER", "root"),
        password=os.getenv("MYSQL_PASSWORD", ""),
        charset="utf8mb4",
        autocommit=False,
    )


def run_schema(cur, database=DB_NAME):
    """执行建表脚本。去掉 -- 注释行后按分号拆成单条语句（脚本里的字符串不含分号）。
    database 不是 airline 时，把脚本里的库名替换掉（测试库用）。"""
    lines = [ln for ln in SCHEMA.read_text(encoding="utf-8").splitlines()
             if not ln.lstrip().startswith("--")]
    sql = "\n".join(lines)
    if database != DB_NAME:
        sql = (sql.replace(f"CREATE DATABASE IF NOT EXISTS {DB_NAME} ", f"CREATE DATABASE IF NOT EXISTS {database} ")
                  .replace(f"USE {DB_NAME};", f"USE {database};"))
    for stmt in sql.split(";"):
        if stmt.strip():
            cur.execute(stmt)


def dt(s):
    """'2024-05-02T03:10:19' -> '2024-05-02 03:10:19'；None 原样返回。"""
    return s.replace("T", " ") if s else None


def insert(cur, table, cols, rows):
    sql = f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join(['%s'] * len(cols))})"
    for i in range(0, len(rows), BATCH):
        cur.executemany(sql, rows[i:i + BATCH])
    return len(rows)


def build_rows(users, flights, reservations, airports):
    r = {k: [] for k in ["users", "payment_methods", "saved_passengers", "airports", "flights",
                         "flight_instances", "reservations", "reservation_flights",
                         "reservation_passengers", "payment_history"]}

    codes = [c for c in airports if not c.startswith("_")]     # airports_zh.json 的顺序 = τ-bench 原版顺序
    for seq, code in enumerate(codes):
        a = airports[code]
        r["airports"].append((code, seq, a["city"], a["city_zh"], a["name_en"], a["name_zh"]))

    user_seq = {}  # reservation_id -> 在用户 reservations 列表里的位置
    for uid, u in users.items():
        addr = u["address"]
        r["users"].append((uid, u["name"]["first_name"], u["name"]["last_name"], u["email"], u["dob"],
                           *(addr[k] for k in ADDRESS_KEYS), u["membership"]))
        for seq, (pid, p) in enumerate(u["payment_methods"].items()):
            r["payment_methods"].append((pid, uid, seq, p["source"], p.get("brand"),
                                         p.get("last_four"), p.get("amount")))
        for p in u["saved_passengers"]:
            r["saved_passengers"].append((uid, p["first_name"], p["last_name"], p["dob"]))
        for seq, rid in enumerate(u["reservations"]):
            user_seq[rid] = seq

    for fn, f in flights.items():
        r["flights"].append((fn, f["origin"], f["destination"],
                             f["scheduled_departure_time_est"], f["scheduled_arrival_time_est"]))
        for date, d in f["dates"].items():
            seats, prices = d.get("available_seats", {}), d.get("prices", {})
            r["flight_instances"].append((
                fn, date, d["status"],
                seats.get("basic_economy"), seats.get("economy"), seats.get("business"),
                prices.get("basic_economy"), prices.get("economy"), prices.get("business"),
                dt(d.get("actual_departure_time_est")), dt(d.get("actual_arrival_time_est")),
                dt(d.get("estimated_departure_time_est")), dt(d.get("estimated_arrival_time_est")),
            ))

    for rid, x in reservations.items():
        r["reservations"].append((rid, x["user_id"], user_seq[rid], x["origin"], x["destination"],
                                  x["flight_type"], x["cabin"], dt(x["created_at"]),
                                  x["total_baggages"], x["nonfree_baggages"], x["insurance"],
                                  x.get("status", "active")))
        for seq, s in enumerate(x["flights"]):
            r["reservation_flights"].append((rid, seq, s["flight_number"], s["date"],
                                             s["origin"], s["destination"], s["price"]))
        for seq, p in enumerate(x["passengers"]):
            r["reservation_passengers"].append((rid, seq, p["first_name"], p["last_name"], p["dob"]))
        for p in x["payment_history"]:
            r["payment_history"].append((rid, p["payment_id"], p["amount"]))
    return r


COLUMNS = {
    "users": ["user_id", "first_name", "last_name", "email", "dob", *ADDRESS_KEYS, "membership"],
    "payment_methods": ["payment_id", "user_id", "seq", "source", "brand", "last_four", "amount"],
    "saved_passengers": ["user_id", "first_name", "last_name", "dob"],
    "airports": ["code", "seq", "city", "city_zh", "name_en", "name_zh"],
    "flights": ["flight_number", "origin", "destination",
                "scheduled_departure_time_est", "scheduled_arrival_time_est"],
    "flight_instances": ["flight_number", "date", "status",
                         "seats_basic_economy", "seats_economy", "seats_business",
                         "price_basic_economy", "price_economy", "price_business",
                         "actual_departure_time_est", "actual_arrival_time_est",
                         "estimated_departure_time_est", "estimated_arrival_time_est"],
    "reservations": ["reservation_id", "user_id", "user_seq", "origin", "destination", "flight_type",
                     "cabin", "created_at", "total_baggages", "nonfree_baggages", "insurance", "status"],
    "reservation_flights": ["reservation_id", "seq", "flight_number", "date",
                            "origin", "destination", "price"],
    "reservation_passengers": ["reservation_id", "seq", "first_name", "last_name", "dob"],
    "payment_history": ["reservation_id", "payment_id", "amount"],
}

# 插入顺序要满足外键依赖：先被引用的表，后引用别人的表
ORDER = ["airports", "users", "payment_methods", "saved_passengers", "flights", "flight_instances",
         "reservations", "reservation_flights", "reservation_passengers", "payment_history"]

# 检查 *_zh 生成列没有 NULL（出现 NULL 说明数据里有 enums_zh.json 没覆盖的枚举值）
ZH_CHECKS = {
    "users": ["membership_zh"],
    "payment_methods": ["source_zh"],
    "flight_instances": ["status_zh"],
    "reservations": ["flight_type_zh", "cabin_zh", "insurance_zh", "status_zh"],
}


def import_all(database=DB_NAME, verbose=True):
    """导入并核对，返回是否全部通过。pytest 里用 import_all("airline_test", verbose=False)。"""
    out = print if verbose else (lambda *a, **k: None)
    t0 = time.time()
    load = lambda name: json.loads((DATA / name).read_text(encoding="utf-8"))
    users, flights = load("users.json"), load("flights.json")
    reservations, airports = load("reservations.json"), load("airports_zh.json")
    rows = build_rows(users, flights, reservations, airports)

    env_db = os.getenv("MYSQL_DATABASE")
    if database == DB_NAME and env_db != DB_NAME:
        out(f"[WARN]  agent/.env 里 MYSQL_DATABASE={env_db}，后端要用航空数据需改成 {DB_NAME}")

    conn = connect()
    try:
        with conn.cursor() as cur:
            out(f"执行建表脚本 {SCHEMA.name}（库 {database}）…")
            run_schema(cur, database)     # DDL 在 MySQL 里会隐式提交，所以放在事务之前
            conn.commit()
            out("导入数据（单个事务）…")
            cur.execute(f"USE {database}")
            for t in ORDER:
                n = insert(cur, t, COLUMNS[t], rows[t])
                out(f"  {t:<24}{n:>6} 行")
            conn.commit()
    except Exception:
        conn.rollback()
        out("[FAIL] 导入失败，已回滚")
        raise

    # ---------------- 核对 ----------------
    ok = True
    with conn.cursor() as cur:
        out("核对行数（期望值由原始 JSON 计算）…")
        for t in ORDER:
            cur.execute(f"SELECT COUNT(*) FROM {t}")
            n, exp = cur.fetchone()[0], len(rows[t])
            ok &= n == exp
            out(f"  {'[OK]' if n == exp else '[FAIL]'} {t:<24}{n:>6} / 期望 {exp}")
        for t, cols in ZH_CHECKS.items():
            for c in cols:
                cur.execute(f"SELECT COUNT(*) FROM {t} WHERE {c} IS NULL")
                n = cur.fetchone()[0]
                ok &= n == 0
                if n:
                    out(f"  [FAIL] {t}.{c} 有 {n} 行为 NULL，enums_zh.json 缺少对照")
        cur.execute("SELECT status, status_zh, COUNT(*) FROM flight_instances "
                    "GROUP BY status, status_zh ORDER BY COUNT(*) DESC")
        out("  航班状态分布：" + "，".join(f"{s}/{z} {n}" for s, z, n in cur.fetchall()))
    conn.close()
    out(f"{'导入完成' if ok else '[FAIL] 核对未通过'}，用时 {time.time() - t0:.1f}s")
    return ok


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--database", default=DB_NAME, help="目标库名，默认 airline；测试用 airline_test")
    args = ap.parse_args()
    sys.exit(0 if import_all(args.database) else 1)


if __name__ == "__main__":
    main()
