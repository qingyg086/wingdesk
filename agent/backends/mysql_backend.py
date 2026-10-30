"""
MySQLBackend：产品模式的航空业务后端，读写 MySQL 的 airline 库。

设计要点：
1. 连接池：DBUtils.PooledDB。MCP Server 进程里多个工具调用并发执行（各自在线程池里跑），
   每次从池里借一个连接、用完归还，避免每次调用都重新建 TCP 连接和认证。
2. 事务：每个写操作在一个事务里完成（例如订票 = 写订单 + 航段 + 乘客 + 支付流水 + 扣礼品卡余额），
   任何一步失败都回滚，不会出现"订单写了一半"。
3. 行锁：写操作开头对用户行和支付方式行加 SELECT ... FOR UPDATE，
   同一用户的并发订票会排队执行，防止同一张礼品卡被重复扣款（超卖 / 双花）。
4. 每次借出连接都以 commit/rollback 结束：InnoDB 默认 REPEATABLE READ，
   连接复用时如果不结束事务，会一直读到旧快照。
5. 与 τ-bench 原版逐字一致：返回的 JSON 结构、字段顺序、错误文本都照搬原版，
   连原版的几个"特性"也照搬（见下方 NOTE），保证产品模式和评测模式下 Agent 看到的结果相同。
   只有"在多用户的真实数据库里会出错"的地方才偏离原版（订单号/代金券号生成、代金券软删除）。
"""

import copy
import json
import os
import random
import re
import string
from contextlib import contextmanager
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Callable, Optional

import pymysql
from dotenv import load_dotenv

from backends.base import AirlineBackend
from core import clock

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

CABINS = ("basic_economy", "economy", "business")
ADDRESS_KEYS = ("address1", "address2", "city", "country", "state", "zip")
TIME_KEYS = ("actual_departure_time_est", "actual_arrival_time_est",
             "estimated_departure_time_est", "estimated_arrival_time_est")
_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


# ------------------------------------------------------------------------------ 小工具
def _num(v):
    """DECIMAL -> int（整数）或 float，还原成 τ-bench JSON 里的数字类型。"""
    if isinstance(v, Decimal):
        return int(v) if v == int(v) else float(v)
    return v


def _iso(v):
    return v.isoformat() if v is not None else None


def _next_day(date: str, compat: bool) -> str:
    """次日日期。compat=True 时照搬 τ-bench 原版的写法（年月写死为 2024-05、日期不补 0，见 search_onestop_flight）。"""
    if compat:
        return f"2024-05-{int(date[-2:]) + 1}"
    return (datetime.strptime(date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")


def _dec(v) -> Decimal:
    """写库前把 int/float 转成 Decimal，避免 float 二进制误差（如 0.1 + 0.2）。"""
    return Decimal(str(v))


def _valid_date(s) -> bool:
    """严格校验 YYYY-MM-DD。
    τ-bench 用字符串精确匹配日期（dict 的 key），而 MySQL 会把 '2024-05-2' 自动转成 2024-05-02 去匹配。
    不先校验的话，MySQL 后端会"意外修好"原版中转搜索的补 0 bug，结果就和原版对不上了。"""
    if not isinstance(s, str) or not _DATE_RE.fullmatch(s):
        return False
    try:
        datetime.strptime(s, "%Y-%m-%d")
        return True
    except ValueError:
        return False


# ------------------------------------------------------------------------------ 编号生成
def random_reservation_id(cur) -> str:
    """产品模式：6 位大写字母+数字，与原数据格式一致，查重后返回。"""
    alphabet = string.ascii_uppercase + string.digits
    while True:
        rid = "".join(random.choices(alphabet, k=6))
        cur.execute("SELECT 1 FROM reservations WHERE reservation_id=%s", (rid,))
        if not cur.fetchone():
            return rid


def random_certificate_id(cur, user_id: str) -> str:
    """产品模式：certificate_ + 7 位数字，全局查重。"""
    while True:
        pid = f"certificate_{random.randint(1000000, 9999999)}"
        cur.execute("SELECT 1 FROM payment_methods WHERE payment_id=%s", (pid,))
        if not cur.fetchone():
            return pid


def taubench_reservation_id(cur) -> str:
    """τ-bench 原版规则：依次尝试 HATHAT / HATHAU / HATHAV（原版假设每个任务最多订 3 次）。
    只用于一致性测试；产品环境多用户共享一个库，用这个规则会撞号。"""
    rid = "HATHAT"
    for nxt in ("HATHAU", "HATHAV"):
        cur.execute("SELECT 1 FROM reservations WHERE reservation_id=%s", (rid,))
        if not cur.fetchone():
            return rid
        rid = nxt
    return rid


def taubench_certificate_id(cur, user_id: str) -> Optional[str]:
    """τ-bench 原版规则：certificate_3221322 / 3221323 / 3221324 中第一个该用户还没有的。只用于测试。"""
    for n in (3221322, 3221323, 3221324):
        pid = f"certificate_{n}"
        cur.execute("SELECT 1 FROM payment_methods WHERE payment_id=%s", (pid,))
        if not cur.fetchone():
            return pid
    return None


class MySQLBackend(AirlineBackend):
    def __init__(self, database: Optional[str] = None, *, pool_size: int = 10,
                 new_reservation_id: Callable = random_reservation_id,
                 new_certificate_id: Callable = random_certificate_id):
        from dbutils.pooled_db import PooledDB

        self.database = database or os.getenv("MYSQL_DATABASE", "airline")
        self._new_reservation_id = new_reservation_id
        self._new_certificate_id = new_certificate_id
        self._pool = PooledDB(
            creator=pymysql, maxconnections=pool_size, mincached=1, blocking=True,
            ping=1,                         # 借出前检查连接是否存活（MySQL 会断开长时间空闲的连接）
            host=os.getenv("MYSQL_HOST", "localhost"),
            port=int(os.getenv("MYSQL_PORT", "3306")),
            user=os.getenv("MYSQL_USER", "root"),
            password=os.getenv("MYSQL_PASSWORD", ""),
            database=self.database, charset="utf8mb4", autocommit=False,
            cursorclass=pymysql.cursors.DictCursor,
        )

    # ============================================================== 连接与事务
    @contextmanager
    def _tx(self):
        """借一个连接，开一个事务。正常结束 commit，异常 rollback，最后归还连接。"""
        conn = self._pool.connection()
        try:
            cur = conn.cursor()
            yield cur
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()          # PooledDB 的 close() 是"归还到池里"，不是真的断开

    def _q(self, cur, sql, *args):
        cur.execute(sql, args)
        return cur.fetchall()

    def _q1(self, cur, sql, *args):
        cur.execute(sql, args)
        return cur.fetchone()

    # ============================================================== 从多张表拼回 τ-bench 的 JSON 结构
    def _load_user(self, cur, user_id, lock=False):
        """lock=True 时对用户行和支付方式行加行锁（写操作用）。"""
        fu = " FOR UPDATE" if lock else ""
        u = self._q1(cur, f"SELECT * FROM users WHERE user_id=%s{fu}", user_id)
        if not u:
            return None
        pms = {}
        for p in self._q(cur, f"SELECT * FROM payment_methods WHERE user_id=%s AND status='active' "
                              f"ORDER BY seq{fu}", user_id):
            d = {"source": p["source"]}
            if p["source"] == "credit_card":
                d.update(brand=p["brand"], last_four=p["last_four"])
            else:
                d["amount"] = _num(p["amount"])
            d["id"] = p["payment_id"]
            pms[p["payment_id"]] = d
        saved = [{"first_name": x["first_name"], "last_name": x["last_name"], "dob": _iso(x["dob"])}
                 for x in self._q(cur, "SELECT * FROM saved_passengers WHERE user_id=%s ORDER BY id", user_id)]
        rids = [x["reservation_id"] for x in self._q(
            cur, "SELECT reservation_id FROM reservations WHERE user_id=%s ORDER BY user_seq", user_id)]
        return {
            "name": {"first_name": u["first_name"], "last_name": u["last_name"]},
            "address": {k: u[k] for k in ADDRESS_KEYS},
            "email": u["email"], "dob": _iso(u["dob"]), "payment_methods": pms,
            "saved_passengers": saved, "membership": u["membership"], "reservations": rids,
        }

    def _load_reservation(self, cur, reservation_id, lock=False):
        fu = " FOR UPDATE" if lock else ""
        r = self._q1(cur, f"SELECT * FROM reservations WHERE reservation_id=%s{fu}", reservation_id)
        if not r:
            return None
        flights = [{"origin": x["origin"], "destination": x["destination"],
                    "flight_number": x["flight_number"], "date": _iso(x["date"]), "price": x["price"]}
                   for x in self._q(cur, "SELECT * FROM reservation_flights WHERE reservation_id=%s "
                                         "ORDER BY seq", reservation_id)]
        pax = [{"first_name": x["first_name"], "last_name": x["last_name"], "dob": _iso(x["dob"])}
               for x in self._q(cur, "SELECT * FROM reservation_passengers WHERE reservation_id=%s "
                                     "ORDER BY seq", reservation_id)]
        pay = [{"payment_id": x["payment_id"], "amount": _num(x["amount"])}
               for x in self._q(cur, "SELECT * FROM payment_history WHERE reservation_id=%s ORDER BY id",
                                reservation_id)]
        out = {
            "reservation_id": r["reservation_id"], "user_id": r["user_id"], "origin": r["origin"],
            "destination": r["destination"], "flight_type": r["flight_type"], "cabin": r["cabin"],
            "flights": flights, "passengers": pax, "payment_history": pay,
            "created_at": _iso(r["created_at"]), "total_baggages": r["total_baggages"],
            "nonfree_baggages": r["nonfree_baggages"], "insurance": r["insurance"],
        }
        if r["status"] != "active":        # 原版只有取消后才出现 status 字段
            out["status"] = r["status"]
        return out

    @staticmethod
    def _flight_base(f):
        return {"flight_number": f["flight_number"], "origin": f["origin"], "destination": f["destination"],
                "scheduled_departure_time_est": f["scheduled_departure_time_est"],
                "scheduled_arrival_time_est": f["scheduled_arrival_time_est"]}

    @staticmethod
    def _instance(x):
        """flight_instances 的一行 -> 原版 flights.json 里 dates[date] 的结构。"""
        d = {"status": x["status"]}
        if x["status"] == "available":
            d["available_seats"] = {c: x[f"seats_{c}"] for c in CABINS}
            d["prices"] = {c: x[f"price_{c}"] for c in CABINS}
        for k in TIME_KEYS:
            if x[k] is not None:
                d[k] = _iso(x[k])
        return d

    def _load_flight(self, cur, flight_number):
        f = self._q1(cur, "SELECT * FROM flights WHERE flight_number=%s", flight_number)
        return self._flight_base(f) if f else None

    def _load_instance(self, cur, flight_number, date):
        if not _valid_date(date):
            return None
        x = self._q1(cur, "SELECT * FROM flight_instances WHERE flight_number=%s AND date=%s",
                     flight_number, date)
        return self._instance(x) if x else None

    def _active_payment(self, cur, user_id, payment_id):
        p = self._q1(cur, "SELECT * FROM payment_methods WHERE payment_id=%s AND user_id=%s "
                          "AND status='active' FOR UPDATE", payment_id, user_id)
        if p and p["amount"] is not None:
            p["amount"] = _num(p["amount"])
        return p

    def get_reservation_owner(self, reservation_id: str) -> Optional[str]:
        """产品模式的越权校验用（τ-bench 没有登录概念，所以不在 AirlineBackend 接口里）。"""
        with self._tx() as cur:
            r = self._q1(cur, "SELECT user_id FROM reservations WHERE reservation_id=%s", reservation_id)
        return r["user_id"] if r else None

    # ============================================================== 只读工具
    def get_user_details(self, user_id: str) -> str:
        with self._tx() as cur:
            u = self._load_user(cur, user_id)
        return json.dumps(u) if u else "Error: user not found"

    def get_reservation_details(self, reservation_id: str) -> str:
        with self._tx() as cur:
            r = self._load_reservation(cur, reservation_id)
        # NOTE: 原版找不到订单时返回的就是 "user not found"（原版的笔误），照搬
        return json.dumps(r) if r else "Error: user not found"

    def list_all_airports(self) -> str:
        with self._tx() as cur:
            rows = self._q(cur, "SELECT code, city FROM airports ORDER BY seq")
        return json.dumps({r["code"]: r["city"] for r in rows})

    def search_direct_flight(self, origin: str, destination: str, date: str) -> str:
        if not _valid_date(date):
            return json.dumps([])
        with self._tx() as cur:
            rows = self._q(cur, """
                SELECT f.flight_number, f.origin, f.destination,
                       f.scheduled_departure_time_est, f.scheduled_arrival_time_est, fi.*
                FROM flights f JOIN flight_instances fi ON fi.flight_number = f.flight_number
                WHERE f.origin=%s AND f.destination=%s AND fi.date=%s AND fi.status='available'
                ORDER BY f.flight_number""", origin, destination, date)
        return json.dumps([{**self._flight_base(r), **self._instance(r)} for r in rows])

    def search_onestop_flight(self, origin: str, destination: str, date: str) -> str:
        """逐行照搬原版算法（只是数据从 MySQL 取）。原版有两个特性，按用户决定照搬：
        NOTE 1：第二段的日期 date2 用 f"2024-05-{int(date[-2:])+1}" 拼接，没有补 0，
                5 月 1-8 日出发、第一段次日到达时，会拼出 '2024-05-2' 这种不存在的日期，方案被漏掉。
        NOTE 2：判断第二段是否可订用的是 date2，但返回结果里第二段的状态/余座/价格读的是 date（第一段的日期）。
        这两个特性只在固定时钟（τ-bench 兼容）模式下保留；实时时钟模式下数据日期是平移过的，
        按正确的日期计算次日，第二段的信息也读 date2。
        """
        compat = clock.is_fixed()
        with self._tx() as cur:
            f1s = [self._flight_base(f) for f in self._q(
                cur, "SELECT * FROM flights WHERE origin=%s ORDER BY flight_number", origin)]
            f2s = [self._flight_base(f) for f in self._q(
                cur, "SELECT * FROM flights WHERE destination=%s ORDER BY flight_number", destination)]
            # 预先把可能用到的航班实例一次查出来，避免在双重循环里逐条查库（N+1 查询）
            cand_dates = {date}
            try:
                cand_dates.add(_next_day(date, compat))
            except ValueError:
                pass
            cand_dates = [d for d in cand_dates if _valid_date(d)]
            fns = list({f["flight_number"] for f in f1s + f2s})
            inst = {}
            if fns and cand_dates:
                ph_f, ph_d = ", ".join(["%s"] * len(fns)), ", ".join(["%s"] * len(cand_dates))
                for x in self._q(cur, f"SELECT * FROM flight_instances WHERE flight_number IN ({ph_f}) "
                                      f"AND date IN ({ph_d})", *fns, *cand_dates):
                    inst[(x["flight_number"], _iso(x["date"]))] = self._instance(x)

        results = []
        for flight1 in f1s:
            for flight2 in f2s:
                if flight1["destination"] != flight2["origin"]:
                    continue
                date2 = (_next_day(date, compat)
                         if "+1" in flight1["scheduled_arrival_time_est"] else date)   # NOTE 1
                if flight1["scheduled_arrival_time_est"] > flight2["scheduled_departure_time_est"]:
                    continue
                d1, d2 = inst.get((flight1["flight_number"], date)), inst.get((flight2["flight_number"], date2))
                if d1 and d2 and d1["status"] == "available" and d2["status"] == "available":
                    result1 = {**flight1, **d1, "date": date}
                    leg2 = inst[(flight2["flight_number"], date)] if compat else d2           # NOTE 2
                    result2 = {**flight2, **leg2, "date": date2}
                    results.append([result1, result2])
        return json.dumps(results)

    # ============================================================== 写工具
    def book_reservation(self, user_id, origin, destination, flight_type, cabin, flights, passengers,
                         payment_methods, total_baggages, nonfree_baggages, insurance) -> str:
        with self._tx() as cur:
            user = self._load_user(cur, user_id, lock=True)
            if not user:
                return "Error: user not found"
            if cabin not in CABINS:
                return f"Error: invalid cabin {cabin}"
            bad = [p for p in passengers if not _valid_date(p.get("dob"))]
            if bad:
                return f"Error: invalid date of birth {bad[0].get('dob')}"

            reservation_id = self._new_reservation_id(cur)
            created_at = clock.now().isoformat()
            # 与原版的构造方式完全一致，返回的 JSON 字段顺序也就一致
            reservation = {
                "reservation_id": reservation_id, "user_id": user_id, "origin": origin,
                "destination": destination, "flight_type": flight_type, "cabin": cabin,
                "flights": copy.deepcopy(flights), "passengers": passengers,
                "payment_history": payment_methods, "created_at": created_at,
                "total_baggages": total_baggages, "nonfree_baggages": nonfree_baggages,
                "insurance": insurance,
            }

            total_price = 0
            for flight in reservation["flights"]:
                fn = flight["flight_number"]
                base = self._load_flight(cur, fn)
                if not base:
                    return f"Error: flight {fn} not found"
                d = self._load_instance(cur, fn, flight["date"])
                if not d:
                    return f"Error: flight {fn} not found on date {flight['date']}"
                if d["status"] != "available":
                    return f"Error: flight {fn} not available on date {flight['date']}"
                if d["available_seats"][cabin] < len(passengers):
                    return f"Error: not enough seats on flight {fn}"
                flight["price"] = d["prices"][cabin]
                flight["origin"] = base["origin"]
                flight["destination"] = base["destination"]
                total_price += flight["price"] * len(passengers)

            if insurance == "yes":
                total_price += 30 * len(passengers)
            total_price += 50 * nonfree_baggages

            for pm in payment_methods:
                pid, amount = pm["payment_id"], pm["amount"]
                if pid not in user["payment_methods"]:
                    return f"Error: payment method {pid} not found"
                src = user["payment_methods"][pid]
                if src["source"] in ("gift_card", "certificate") and src["amount"] < amount:
                    return f"Error: not enough balance in payment method {pid}"
            paid = sum(p["amount"] for p in payment_methods)
            if paid != total_price:
                return f"Error: payment amount does not add up, total price is {total_price}, but paid {paid}"

            # ---- 校验全部通过，开始写（同一个事务）----
            for pm in payment_methods:
                src = user["payment_methods"][pm["payment_id"]]["source"]
                if src == "gift_card":
                    cur.execute("UPDATE payment_methods SET amount = amount - %s WHERE payment_id=%s",
                                (_dec(pm["amount"]), pm["payment_id"]))
                elif src == "certificate":
                    # 原版是从用户档案里删掉这张代金券；这里软删除，支付流水的外键仍然有效
                    cur.execute("UPDATE payment_methods SET status='used' WHERE payment_id=%s",
                                (pm["payment_id"],))
            # NOTE 3：原版订票不扣减航班余座（源码注释：假设座位更新需要时间），照搬
            seq = self._q1(cur, "SELECT COALESCE(MAX(user_seq), -1) + 1 AS n FROM reservations "
                                "WHERE user_id=%s", user_id)["n"]
            cur.execute("INSERT INTO reservations (reservation_id, user_id, user_seq, origin, destination, "
                        "flight_type, cabin, created_at, total_baggages, nonfree_baggages, insurance) "
                        "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                        (reservation_id, user_id, seq, origin, destination, flight_type, cabin,
                         created_at.replace("T", " "), total_baggages, nonfree_baggages, insurance))
            self._write_flights(cur, reservation_id, reservation["flights"])
            self._write_passengers(cur, reservation_id, passengers)
            for pm in payment_methods:
                self._append_payment(cur, reservation_id, pm["payment_id"], pm["amount"])
        return json.dumps(reservation)

    def update_reservation_flights(self, reservation_id, cabin, flights, payment_id) -> str:
        with self._tx() as cur:
            reservation = self._load_reservation(cur, reservation_id, lock=True)
            if not reservation:
                return "Error: reservation not found"
            if cabin not in CABINS:
                return f"Error: invalid cabin {cabin}"
            n_pax = len(reservation["passengers"])

            total_price = 0
            flights = copy.deepcopy(flights)
            for flight in flights:
                # 已有航段（同航班号、同日期、舱位不变）：保留原价，不按当前价格重算
                same = [f for f in reservation["flights"]
                        if f["flight_number"] == flight["flight_number"] and f["date"] == flight["date"]
                        and cabin == reservation["cabin"]]
                if same:
                    total_price += same[0]["price"] * n_pax
                    flight["price"] = same[0]["price"]
                    flight["origin"] = same[0]["origin"]
                    flight["destination"] = same[0]["destination"]
                    continue
                fn = flight["flight_number"]
                base = self._load_flight(cur, fn)
                if not base:
                    return f"Error: flight {fn} not found"
                d = self._load_instance(cur, fn, flight["date"])
                if not d:
                    return f"Error: flight {fn} not found on date {flight['date']}"
                if d["status"] != "available":
                    return f"Error: flight {fn} not available on date {flight['date']}"
                if d["available_seats"][cabin] < n_pax:
                    return f"Error: not enough seats on flight {fn}"
                flight["price"] = d["prices"][cabin]
                flight["origin"] = base["origin"]
                flight["destination"] = base["destination"]
                total_price += flight["price"] * n_pax
            total_price -= sum(f["price"] for f in reservation["flights"]) * n_pax

            pm = self._active_payment(cur, reservation["user_id"], payment_id)
            if not pm:
                return "Error: payment method not found"
            if pm["source"] == "certificate":
                return "Error: certificate cannot be used to update reservation"
            if pm["source"] == "gift_card" and pm["amount"] < total_price:
                return "Error: gift card balance is not enough"

            if pm["source"] == "gift_card":       # total_price 为负数时相当于退回礼品卡
                cur.execute("UPDATE payment_methods SET amount = amount - %s WHERE payment_id=%s",
                            (_dec(total_price), payment_id))
            # NOTE 4：原版改航班时不更新订单的 cabin 字段（即使传入了新舱位），照搬
            cur.execute("DELETE FROM reservation_flights WHERE reservation_id=%s", (reservation_id,))
            self._write_flights(cur, reservation_id, flights)
            reservation["flights"] = flights
            if total_price != 0:
                self._append_payment(cur, reservation_id, payment_id, total_price)
                reservation["payment_history"].append({"payment_id": payment_id, "amount": total_price})
        return json.dumps(reservation)

    def update_reservation_passengers(self, reservation_id, passengers) -> str:
        with self._tx() as cur:
            reservation = self._load_reservation(cur, reservation_id, lock=True)
            if not reservation:
                return "Error: reservation not found"
            if len(passengers) != len(reservation["passengers"]):
                return "Error: number of passengers does not match"
            bad = [p for p in passengers if not _valid_date(p.get("dob"))]
            if bad:
                return f"Error: invalid date of birth {bad[0].get('dob')}"
            cur.execute("DELETE FROM reservation_passengers WHERE reservation_id=%s", (reservation_id,))
            self._write_passengers(cur, reservation_id, passengers)
            reservation["passengers"] = passengers
        return json.dumps(reservation)

    def update_reservation_baggages(self, reservation_id, total_baggages, nonfree_baggages, payment_id) -> str:
        with self._tx() as cur:
            reservation = self._load_reservation(cur, reservation_id, lock=True)
            if not reservation:
                return "Error: reservation not found"
            total_price = 50 * max(0, nonfree_baggages - reservation["nonfree_baggages"])
            pm = self._active_payment(cur, reservation["user_id"], payment_id)
            if not pm:
                return "Error: payment method not found"
            if pm["source"] == "certificate":
                return "Error: certificate cannot be used to update reservation"
            if pm["source"] == "gift_card" and pm["amount"] < total_price:
                return "Error: gift card balance is not enough"

            cur.execute("UPDATE reservations SET total_baggages=%s, nonfree_baggages=%s WHERE reservation_id=%s",
                        (total_baggages, nonfree_baggages, reservation_id))
            reservation["total_baggages"] = total_baggages
            reservation["nonfree_baggages"] = nonfree_baggages
            if pm["source"] == "gift_card":
                cur.execute("UPDATE payment_methods SET amount = amount - %s WHERE payment_id=%s",
                            (_dec(total_price), payment_id))
            if total_price != 0:
                self._append_payment(cur, reservation_id, payment_id, total_price)
                reservation["payment_history"].append({"payment_id": payment_id, "amount": total_price})
        return json.dumps(reservation)

    def cancel_reservation(self, reservation_id) -> str:
        with self._tx() as cur:
            reservation = self._load_reservation(cur, reservation_id, lock=True)
            if not reservation:
                return "Error: reservation not found"
            # 冲正：对每一笔历史支付追加一条金额相反的记录。
            # NOTE 5：与原版一致，只记流水，不把钱加回礼品卡余额（原版也不加）
            refunds = [{"payment_id": p["payment_id"], "amount": -p["amount"]}
                       for p in reservation["payment_history"]]
            for r in refunds:
                self._append_payment(cur, reservation_id, r["payment_id"], r["amount"])
            cur.execute("UPDATE reservations SET status='cancelled' WHERE reservation_id=%s", (reservation_id,))
            reservation["payment_history"].extend(refunds)
            reservation["status"] = "cancelled"
        return json.dumps(reservation)

    def send_certificate(self, user_id, amount) -> str:
        with self._tx() as cur:
            if not self._q1(cur, "SELECT 1 FROM users WHERE user_id=%s FOR UPDATE", user_id):
                return "Error: user not found"
            payment_id = self._new_certificate_id(cur, user_id)
            if payment_id is None:
                return "Error: no certificate id available"
            seq = self._q1(cur, "SELECT COALESCE(MAX(seq), -1) + 1 AS n FROM payment_methods WHERE user_id=%s",
                           user_id)["n"]
            cur.execute("INSERT INTO payment_methods (payment_id, user_id, seq, source, amount) "
                        "VALUES (%s,%s,%s,'certificate',%s)", (payment_id, user_id, seq, _dec(amount)))
        return f"Certificate {payment_id} added to user {user_id} with amount {amount}."

    # ============================================================== 写库小工具
    def _write_flights(self, cur, reservation_id, flights):
        cur.executemany(
            "INSERT INTO reservation_flights (reservation_id, seq, flight_number, date, origin, destination, price) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s)",
            [(reservation_id, i, f["flight_number"], f["date"], f["origin"], f["destination"], f["price"])
             for i, f in enumerate(flights)])

    def _write_passengers(self, cur, reservation_id, passengers):
        cur.executemany(
            "INSERT INTO reservation_passengers (reservation_id, seq, first_name, last_name, dob) "
            "VALUES (%s,%s,%s,%s,%s)",
            [(reservation_id, i, p["first_name"], p["last_name"], p["dob"]) for i, p in enumerate(passengers)])

    def _append_payment(self, cur, reservation_id, payment_id, amount):
        cur.execute("INSERT INTO payment_history (reservation_id, payment_id, amount) VALUES (%s,%s,%s)",
                    (reservation_id, payment_id, _dec(amount)))
