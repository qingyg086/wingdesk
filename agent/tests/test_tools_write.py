"""
写工具一致性测试：两边从同一个初始状态出发，执行同一串操作，比对
  1. 每一步的返回值（成功时比 JSON 内容，失败时比错误文本）；
  2. 操作之后受影响的用户档案和订单（相当于比对"数据库最终状态"，τ-bench 评分就是看这个）。
"""

import json

from helpers import assert_same, call_both

DATE = "2024-05-20"


# ------------------------------------------------------------------ 从数据里挑测试用例
def pick_user(data, need=("gift_card", "certificate", "credit_card")):
    for uid, u in data["users"].items():
        sources = {p["source"] for p in u["payment_methods"].values()}
        if set(need) <= sources:
            return uid
    raise AssertionError("没有同时有这几种支付方式的用户")


def pm(data, uid, source):
    return next(p for p in data["users"][uid]["payment_methods"].values() if p["source"] == source)


def pick_flight(data, date=DATE, cabin="economy", seats=3, exclude=()):
    for fn, f in data["flights"].items():
        d = f["dates"].get(date, {})
        if fn not in exclude and d.get("status") == "available" and d["available_seats"][cabin] >= seats:
            return fn, f
    raise AssertionError("找不到可订航班")


def pick_reservation(data, cabin="economy", need_gift=True):
    """挑一个舱位符合、订票用户有礼品卡、单程、未取消的订单。"""
    for rid, r in data["reservations"].items():
        sources = {p["source"] for p in data["users"][r["user_id"]]["payment_methods"].values()}
        if r["cabin"] == cabin and r["flight_type"] == "one_way" and (not need_gift or "gift_card" in sources):
            return rid, r
    raise AssertionError("找不到合适的订单")


def check_state(backend, data, ref_tools, user_ids=(), reservation_ids=()):
    for uid in user_ids:
        assert_same(*call_both(backend, data, ref_tools, "get_user_details", user_id=uid), f"用户 {uid} 状态不一致")
    for rid in reservation_ids:
        assert_same(*call_both(backend, data, ref_tools, "get_reservation_details", reservation_id=rid),
                    f"订单 {rid} 状态不一致")


def booking_args(data, uid, fn, cabin="economy", n_pax=1, pay=None, insurance="no", nonfree=0):
    f = data["flights"][fn]
    price = f["dates"][DATE]["prices"][cabin]
    total = price * n_pax + (30 * n_pax if insurance == "yes" else 0) + 50 * nonfree
    pax = [{"first_name": "Test", "last_name": f"Passenger{i}", "dob": "1990-01-0" + str(i + 1)} for i in range(n_pax)]
    return dict(user_id=uid, origin=f["origin"], destination=f["destination"], flight_type="one_way",
                cabin=cabin, flights=[{"flight_number": fn, "date": DATE}], passengers=pax,
                payment_methods=pay(total) if pay else [], total_baggages=nonfree, nonfree_baggages=nonfree,
                insurance=insurance), total


# ------------------------------------------------------------------ 订票
def test_book_with_credit_card_then_second_booking(rw, ref_tools):
    """单张信用卡订票；同一用户再订一次，订单号应为 HATHAT、HATHAU（原版规则）。"""
    backend, data = rw
    uid = pick_user(data)
    cc = pm(data, uid, "credit_card")["id"]
    fn, _ = pick_flight(data)
    args, _ = booking_args(data, uid, fn, n_pax=2, insurance="yes", nonfree=1,
                           pay=lambda t: [{"payment_id": cc, "amount": t}])
    m1, r1 = call_both(backend, data, ref_tools, "book_reservation", **args)
    assert json.loads(r1)["reservation_id"] == "HATHAT"
    assert m1 == r1                     # 新订单的返回值逐字相同
    m2, r2 = call_both(backend, data, ref_tools, "book_reservation", **args)
    assert json.loads(m2)["reservation_id"] == "HATHAU"
    assert m2 == r2
    check_state(backend, data, ref_tools, [uid], ["HATHAT", "HATHAU"])


def test_book_with_split_payment(rw, ref_tools):
    """代金券 + 礼品卡 + 信用卡组合支付：礼品卡扣余额，代金券用后从档案消失（MySQL 里是软删除）。"""
    backend, data = rw
    uid = pick_user(data)
    cert, gift, cc = pm(data, uid, "certificate"), pm(data, uid, "gift_card"), pm(data, uid, "credit_card")
    fn, _ = pick_flight(data, cabin="business", seats=2)

    def pay(total):
        c = min(cert["amount"], total // 3)
        g = min(gift["amount"], total // 3)
        return [{"payment_id": cert["id"], "amount": c}, {"payment_id": gift["id"], "amount": g},
                {"payment_id": cc["id"], "amount": total - c - g}]

    args, _ = booking_args(data, uid, fn, cabin="business", n_pax=2, pay=pay)
    m, r = call_both(backend, data, ref_tools, "book_reservation", **args)
    assert not r.startswith("Error"), r
    assert m == r
    check_state(backend, data, ref_tools, [uid], ["HATHAT"])
    assert cert["id"] not in json.loads(backend.get_user_details(uid))["payment_methods"]


def test_book_errors(rw, ref_tools):
    """各种失败分支：错误文本必须和原版一致，而且失败后数据库不能有任何改动。"""
    backend, data = rw
    uid = pick_user(data)
    cc, gift = pm(data, uid, "credit_card")["id"], pm(data, uid, "gift_card")
    fn, _ = pick_flight(data)
    full = lambda t: [{"payment_id": cc, "amount": t}]
    base, total = booking_args(data, uid, fn, pay=full)

    landed = next((n, d) for n, f in data["flights"].items() for d, x in f["dates"].items() if x["status"] == "landed")
    tight = next((n, x["available_seats"]["economy"]) for n, f in data["flights"].items()
                 for d, x in f["dates"].items() if d == DATE and x["status"] == "available"
                 and 0 < x["available_seats"]["economy"] < 5)
    cases = [
        dict(base, user_id="nobody_000"),
        dict(base, flights=[{"flight_number": "HAT999", "date": DATE}]),
        dict(base, flights=[{"flight_number": fn, "date": "2024-05-2"}]),          # 不规范日期
        dict(base, flights=[{"flight_number": landed[0], "date": landed[1]}]),
        dict(base, flights=[{"flight_number": tight[0], "date": DATE}],
             passengers=[{"first_name": "A", "last_name": str(i), "dob": "1990-01-01"} for i in range(tight[1] + 1)]),
        dict(base, payment_methods=[{"payment_id": "credit_card_0000000", "amount": total}]),
        dict(base, payment_methods=[{"payment_id": gift["id"], "amount": gift["amount"] + 1}]),
        dict(base, payment_methods=[{"payment_id": cc, "amount": total - 1}]),
    ]
    for args in cases:
        m, r = call_both(backend, data, ref_tools, "book_reservation", **args)
        assert r.startswith("Error"), r
        assert m == r
    check_state(backend, data, ref_tools, [uid])
    assert backend.get_reservation_details("HATHAT") == "Error: user not found"   # 没有留下半个订单


# ------------------------------------------------------------------ 改航班 / 升舱
def test_update_flights_change_segment_and_upgrade(rw, ref_tools):
    backend, data = rw
    rid, r = pick_reservation(data)
    uid = r["user_id"]
    gift = pm(data, uid, "gift_card")["id"]
    seg = r["flights"][0]
    alt = next((n for n, f in data["flights"].items()
                if f["origin"] == seg["origin"] and f["destination"] == seg["destination"]
                and f["dates"].get(DATE, {}).get("status") == "available"), None)
    assert alt, "该航线 5-20 没有可订航班"
    # 1) 换成另一天的航班，礼品卡补差价（或退差价到礼品卡）
    m, rf = call_both(backend, data, ref_tools, "update_reservation_flights", reservation_id=rid,
                      cabin=r["cabin"], flights=[{"flight_number": alt, "date": DATE}], payment_id=gift)
    assert_same(m, rf, "换航班")
    # 2) 同一航班升舱到商务舱，信用卡或礼品卡付差价（原版不会更新订单的 cabin 字段——照搬）
    pay2 = next((p["id"] for p in data["users"][uid]["payment_methods"].values() if p["source"] == "credit_card"), gift)
    m, rf = call_both(backend, data, ref_tools, "update_reservation_flights", reservation_id=rid,
                      cabin="business", flights=[{"flight_number": alt, "date": DATE}], payment_id=pay2)
    assert_same(m, rf, "升舱")
    if not rf.startswith("Error"):
        assert json.loads(m)["cabin"] == r["cabin"]           # NOTE 4：cabin 字段没变
    check_state(backend, data, ref_tools, [uid], [rid])


def test_update_flights_errors(rw, ref_tools):
    backend, data = rw
    rid, r = pick_reservation(data)
    uid = r["user_id"]
    flights = [{"flight_number": f["flight_number"], "date": f["date"]} for f in r["flights"]]
    cert = next((p["id"] for p in data["users"][uid]["payment_methods"].values() if p["source"] == "certificate"), None)
    gift = pm(data, uid, "gift_card")["id"]
    cases = [dict(reservation_id="ZZZZZZ", cabin="economy", flights=flights, payment_id=gift),
             dict(reservation_id=rid, cabin=r["cabin"], flights=[{"flight_number": "HAT999", "date": DATE}], payment_id=gift),
             dict(reservation_id=rid, cabin=r["cabin"], flights=flights, payment_id="gift_card_0000000")]
    if cert:
        cases.append(dict(reservation_id=rid, cabin=r["cabin"], flights=flights, payment_id=cert))
    for args in cases:
        m, rf = call_both(backend, data, ref_tools, "update_reservation_flights", **args)
        assert rf.startswith("Error") and m == rf, (m, rf)
    check_state(backend, data, ref_tools, [uid], [rid])


# ------------------------------------------------------------------ 乘客、行李、取消、代金券
def test_update_passengers(rw, ref_tools):
    backend, data = rw
    rid, r = pick_reservation(data, need_gift=False)
    new = [{"first_name": "Ming", "last_name": "Li", "dob": "1995-06-15"} for _ in r["passengers"]]
    m, rf = call_both(backend, data, ref_tools, "update_reservation_passengers", reservation_id=rid, passengers=new)
    assert m == rf
    m, rf = call_both(backend, data, ref_tools, "update_reservation_passengers", reservation_id=rid,
                      passengers=new + new)
    assert m == rf == "Error: number of passengers does not match"
    check_state(backend, data, ref_tools, [r["user_id"]], [rid])


def test_update_baggages(rw, ref_tools):
    backend, data = rw
    rid, r = pick_reservation(data)
    uid = r["user_id"]
    gift = pm(data, uid, "gift_card")["id"]
    m, rf = call_both(backend, data, ref_tools, "update_reservation_baggages", reservation_id=rid,
                      total_baggages=r["total_baggages"] + 2, nonfree_baggages=r["nonfree_baggages"] + 1,
                      payment_id=gift)
    assert m == rf
    cert = next((p["id"] for p in data["users"][uid]["payment_methods"].values() if p["source"] == "certificate"), None)
    if cert:
        m, rf = call_both(backend, data, ref_tools, "update_reservation_baggages", reservation_id=rid,
                          total_baggages=5, nonfree_baggages=5, payment_id=cert)
        assert m == rf == "Error: certificate cannot be used to update reservation"
    check_state(backend, data, ref_tools, [uid], [rid])


def test_cancel_twice(rw, ref_tools):
    """取消两次：原版不检查是否已取消，第二次会把所有流水（包括第一次的冲正）再冲正一遍——照搬。"""
    backend, data = rw
    rid, r = pick_reservation(data, need_gift=False)
    for _ in range(2):
        m, rf = call_both(backend, data, ref_tools, "cancel_reservation", reservation_id=rid)
        assert m == rf
    m, rf = call_both(backend, data, ref_tools, "cancel_reservation", reservation_id="ZZZZZZ")
    assert m == rf == "Error: reservation not found"
    check_state(backend, data, ref_tools, [r["user_id"]], [rid])


def test_send_certificate(rw, ref_tools):
    backend, data = rw
    uid = pick_user(data)
    for amount in (100, 150):
        m, rf = call_both(backend, data, ref_tools, "send_certificate", user_id=uid, amount=amount)
        assert m == rf
    m, rf = call_both(backend, data, ref_tools, "send_certificate", user_id="nobody_000", amount=50)
    assert m == rf == "Error: user not found"
    check_state(backend, data, ref_tools, [uid])


def test_book_after_certificate_and_cancel(rw, ref_tools):
    """组合流程：发代金券 → 用代金券订票 → 取消订单。"""
    backend, data = rw
    uid = pick_user(data)
    call_both(backend, data, ref_tools, "send_certificate", user_id=uid, amount=500)
    cc = pm(data, uid, "credit_card")["id"]
    fn, _ = pick_flight(data)
    args, total = booking_args(data, uid, fn, pay=lambda t: [{"payment_id": "certificate_3221322", "amount": min(500, t)},
                                                              {"payment_id": cc, "amount": t - min(500, t)}])
    m, rf = call_both(backend, data, ref_tools, "book_reservation", **args)
    assert m == rf and not rf.startswith("Error")
    m, rf = call_both(backend, data, ref_tools, "cancel_reservation", reservation_id="HATHAT")
    assert_same(m, rf, "取消新订单")
    check_state(backend, data, ref_tools, [uid], ["HATHAT"])
