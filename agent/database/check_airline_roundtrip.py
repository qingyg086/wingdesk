"""
往返校验：从 MySQL 把每个用户、订单、航班实例拼回 τ-bench 的 JSON 结构，和原始 JSON 逐个比对。

用法（在项目根目录，先跑过 import_airline.py）：
    python agent/database/check_airline_roundtrip.py

为什么要做：一个 JSON 文档被拆成了多张关系表，拆的时候可能丢字段、类型变了（DECIMAL / DATE）、
列表顺序乱了。全部比对一致，才能保证 MySQLBackend 返回的结果和 τ-bench 原版工具一致。
MySQLBackend 用的是同一套拼装逻辑。
"""

import decimal
import json
import os
import sys
from pathlib import Path

import pymysql
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "airline"
load_dotenv(ROOT / "agent" / ".env")
ADDRESS_KEYS = ["address1", "address2", "city", "country", "state", "zip"]   # 原版 JSON 的字段顺序


def first_diff(a, b, path="$"):
    """找出两个 JSON 对象第一处不一致的位置（包括 key 顺序不同），方便定位问题。"""
    if isinstance(a, dict) and isinstance(b, dict):
        if list(a) != list(b):
            return f"{path} 的字段或字段顺序不同：MySQL={list(a)}，原版={list(b)}"
        for k in a:
            d = first_diff(a[k], b[k], f"{path}.{k}")
            if d:
                return d
        return None
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return f"{path} 的长度不同：MySQL={len(a)}，原版={len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            d = first_diff(x, y, f"{path}[{i}]")
            if d:
                return d
        return None
    if a != b or type(a) is not type(b):
        return f"{path} 的值不同：MySQL={a!r}，原版={b!r}"
    return None


def num(v):
    """DECIMAL(10,2) -> 整数就还原成 int，否则 float（τ-bench 原版 JSON 里是数字）。"""
    if isinstance(v, decimal.Decimal):
        return int(v) if v == int(v) else float(v)
    return v


def iso(v):
    return v.isoformat() if v is not None else None


class Loader:
    def __init__(self, conn):
        self.cur = conn.cursor(pymysql.cursors.DictCursor)

    def q(self, sql, *args):
        self.cur.execute(sql, args)
        return self.cur.fetchall()

    def user(self, uid):
        u = self.q("SELECT * FROM users WHERE user_id=%s", uid)[0]
        pms = {}
        for p in self.q("SELECT * FROM payment_methods WHERE user_id=%s AND status='active' ORDER BY seq", uid):
            d = {"source": p["source"]}
            if p["source"] == "credit_card":
                d.update(brand=p["brand"], last_four=p["last_four"])
            else:
                d["amount"] = num(p["amount"])
            d["id"] = p["payment_id"]
            pms[p["payment_id"]] = d
        saved = [{"first_name": x["first_name"], "last_name": x["last_name"], "dob": iso(x["dob"])}
                 for x in self.q("SELECT * FROM saved_passengers WHERE user_id=%s ORDER BY id", uid)]
        rids = [x["reservation_id"] for x in
                self.q("SELECT reservation_id FROM reservations WHERE user_id=%s ORDER BY user_seq", uid)]
        return {
            "name": {"first_name": u["first_name"], "last_name": u["last_name"]},
            "address": {k: u[k] for k in ADDRESS_KEYS},
            "email": u["email"], "dob": iso(u["dob"]), "payment_methods": pms,
            "saved_passengers": saved, "membership": u["membership"], "reservations": rids,
        }

    def reservation(self, rid):
        r = self.q("SELECT * FROM reservations WHERE reservation_id=%s", rid)[0]
        flights = [{"origin": x["origin"], "destination": x["destination"],
                    "flight_number": x["flight_number"], "date": iso(x["date"]), "price": x["price"]}
                   for x in self.q("SELECT * FROM reservation_flights WHERE reservation_id=%s ORDER BY seq", rid)]
        pax = [{"first_name": x["first_name"], "last_name": x["last_name"], "dob": iso(x["dob"])}
               for x in self.q("SELECT * FROM reservation_passengers WHERE reservation_id=%s ORDER BY seq", rid)]
        pay = [{"payment_id": x["payment_id"], "amount": num(x["amount"])}
               for x in self.q("SELECT * FROM payment_history WHERE reservation_id=%s ORDER BY id", rid)]
        out = {
            "reservation_id": rid, "user_id": r["user_id"], "origin": r["origin"],
            "destination": r["destination"], "flight_type": r["flight_type"], "cabin": r["cabin"],
            "flights": flights, "passengers": pax, "payment_history": pay,
            "created_at": iso(r["created_at"]), "total_baggages": r["total_baggages"],
            "nonfree_baggages": r["nonfree_baggages"], "insurance": r["insurance"],
        }
        if r["status"] != "active":      # 原版只有取消后才出现 status 字段
            out["status"] = r["status"]
        return out

    def flight_instance(self, x):
        d = {"status": x["status"]}
        if x["status"] == "available":
            d["available_seats"] = {c: x[f"seats_{c}"] for c in ("basic_economy", "economy", "business")}
            d["prices"] = {c: x[f"price_{c}"] for c in ("basic_economy", "economy", "business")}
        for k in ("actual_departure_time_est", "actual_arrival_time_est",
                  "estimated_departure_time_est", "estimated_arrival_time_est"):
            if x[k] is not None:
                d[k] = iso(x[k])
        return d


def main():
    load = lambda n: json.loads((DATA / n).read_text(encoding="utf-8"))
    U, R, F = load("users.json"), load("reservations.json"), load("flights.json")
    conn = pymysql.connect(host=os.getenv("MYSQL_HOST", "localhost"),
                           port=int(os.getenv("MYSQL_PORT", "3306")),
                           user=os.getenv("MYSQL_USER", "root"),
                           password=os.getenv("MYSQL_PASSWORD", ""),
                           database="airline", charset="utf8mb4")
    L = Loader(conn)

    # 用 json.dumps 比较：连字段顺序也要一致
    same = lambda a, b: json.dumps(a) == json.dumps(b)
    bad_u = [k for k in U if not same(L.user(k), U[k])]
    bad_r = [k for k in R if not same(L.reservation(k), R[k])]
    if bad_u:
        print(f"   用户 {bad_u[0]}：{first_diff(L.user(bad_u[0]), U[bad_u[0]])}")
    if bad_r:
        print(f"   订单 {bad_r[0]}：{first_diff(L.reservation(bad_r[0]), R[bad_r[0]])}")
    bad_f = [(x["flight_number"], iso(x["date"])) for x in L.q("SELECT * FROM flight_instances")
             if L.flight_instance(x) != F[x["flight_number"]]["dates"][iso(x["date"])]]
    conn.close()

    for name, total, bad in [("用户", len(U), bad_u), ("订单", len(R), bad_r),
                             ("航班实例", sum(len(f["dates"]) for f in F.values()), bad_f)]:
        print(f"{'[OK]' if not bad else '[FAIL]'} {name}：{total - len(bad)}/{total} 与原版 JSON 一致"
              + (f"，不一致示例 {bad[:3]}" if bad else ""))
    sys.exit(0 if not (bad_u or bad_r or bad_f) else 1)


if __name__ == "__main__":
    main()
