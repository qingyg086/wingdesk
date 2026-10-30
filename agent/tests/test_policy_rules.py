"""Policy checks reject unsafe writes and give the agent actionable errors."""

import json

import pytest

from core.policy_rules import validate_booking, validate_baggage_update
from tools.taubench_tools import build_taubench_tools
from mcp_servers import airline_server


USER = {"membership": "silver", "payment_methods": {
    "cert1": {"source": "certificate"}, "cert2": {"source": "certificate"},
    "card": {"source": "credit_card"}, "gift": {"source": "gift_card"},
}}


def booking(**changes):
    args = {"cabin": "economy", "passengers": [{"first_name": "A"}],
            "total_baggages": 2, "nonfree_baggages": 0,
            "payment_methods": [{"payment_id": "cert1", "amount": 100},
                                {"payment_id": "card", "amount": 50}]}
    args.update(changes)
    return args


def test_payment_limit_blocks_two_certificates():
    args = booking(payment_methods=[{"payment_id": "cert1", "amount": 100},
                                    {"payment_id": "cert2", "amount": 155}])
    assert "at most 1 certificate" in validate_booking(USER, args)
    assert validate_booking(USER, booking()) is None


def test_booking_baggage_allowance_is_per_passenger():
    args = booking(passengers=[{}, {}], total_baggages=5, nonfree_baggages=1)
    assert validate_booking(USER, args) is None  # 2 silver economy passengers: 4 free bags
    assert "must be 1" in validate_booking(USER, booking(**{**args, "nonfree_baggages": 2}))


def test_baggage_update_rejects_wrong_charge_and_uses_new_cabin():
    reservation = {"cabin": "basic_economy", "passengers": [{}]}
    args = {"total_baggages": 2, "nonfree_baggages": 1}
    assert "must be 0" in validate_baggage_update(USER, reservation, args, cabin="economy")
    assert validate_baggage_update(USER, reservation, {**args, "nonfree_baggages": 0}, cabin="economy") is None


def test_eval_tool_returns_error_without_executing_unsafe_booking():
    class FakeBackend:
        def __init__(self):
            self.env = type("Env", (), {"data": {"users": {"u1": USER}, "reservations": {}}})()
            self.calls = []

        def call(self, name, **kwargs):
            self.calls.append((name, kwargs))
            return "booked"

    backend = FakeBackend()
    schema = {"type": "object", "properties": {
        "user_id": {"type": "string"}, "cabin": {"type": "string"},
        "passengers": {"type": "array", "items": {"type": "object"}},
        "total_baggages": {"type": "integer"}, "nonfree_baggages": {"type": "integer"},
        "payment_methods": {"type": "array", "items": {"type": "object"}},
    }}
    tool = build_taubench_tools(backend, [{"function": {"name": "book_reservation", "parameters": schema}}])[
        "book_reservation"]
    bad = booking(payment_methods=[{"payment_id": "cert1", "amount": 100},
                                   {"payment_id": "cert2", "amount": 155}])
    assert "at most 1 certificate" in tool.invoke({"user_id": "u1", **bad})
    assert backend.calls == []
    assert tool.invoke({"user_id": "u1", **booking()}) == "booked"
    assert len(backend.calls) == 1


@pytest.mark.asyncio
async def test_product_tool_blocks_bad_booking_before_database_write(monkeypatch):
    class FakeBackend:
        called = False

        def get_user_details(self, user_id):
            return json.dumps(USER)

        def book_reservation(self, *args):
            self.called = True
            return "booked"

    fake = FakeBackend()
    monkeypatch.setattr(airline_server, "backend", lambda: fake)
    result = await airline_server.book_reservation(
        user_id="u1", origin="SFO", destination="JFK", flight_type="one_way", cabin="economy",
        flights=[], passengers=[airline_server.PassengerInfo(first_name="A", last_name="B", dob="1990-01-01")],
        payment_methods=[airline_server.PaymentItem(payment_id="cert1", amount=100),
                         airline_server.PaymentItem(payment_id="cert2", amount=155)],
        total_baggages=2, nonfree_baggages=0, insurance="no")
    assert "at most 1 certificate" in result
    assert fake.called is False


@pytest.mark.asyncio
async def test_product_tool_blocks_bad_baggage_update_before_database_write(monkeypatch):
    class FakeBackend:
        called = False

        def get_reservation_owner(self, reservation_id):
            return "u1"

        def get_user_details(self, user_id):
            return json.dumps(USER)

        def get_reservation_details(self, reservation_id):
            return json.dumps({"cabin": "economy", "passengers": [{}, {}]})

        def update_reservation_baggages(self, *args):
            self.called = True
            return "updated"

    fake = FakeBackend()
    monkeypatch.setattr(airline_server, "backend", lambda: fake)
    result = await airline_server.update_reservation_baggages(
        reservation_id="r1", total_baggages=5, nonfree_baggages=2,
        payment_id="card", user_id="u1")
    assert "must be 1" in result  # Two silver economy passengers get four free bags.
    assert fake.called is False


def test_eval_baggage_check_uses_cabin_after_flight_change():
    """τ-bench 改航班不更新订单 cabin。升舱后再加行李，要按新舱位算免费额度（task 4/5 的正确操作不能被拦）。"""
    reservation = {"user_id": "u1", "cabin": "basic_economy", "passengers": [{}]}

    class FakeBackend:
        def __init__(self):
            self.env = type("Env", (), {"data": {"users": {"u1": USER}, "reservations": {"r1": reservation}}})()
            self.calls = []

        def call(self, name, **kwargs):
            self.calls.append(name)
            return '{"reservation_id": "r1"}'          # 与原版一样：订单上的 cabin 不变

    backend = FakeBackend()
    obj = {"type": "object", "properties": {}}
    tools = build_taubench_tools(backend, [{"function": {"name": n, "parameters": obj}}
                                           for n in ("update_reservation_flights", "update_reservation_baggages")])
    bags = {"reservation_id": "r1", "total_baggages": 2, "nonfree_baggages": 0, "payment_id": "card"}
    # 升舱前：silver 基础经济舱每人免费 1 件，2 件应收 1 件 → 拦下
    assert "must be 1" in tools["update_reservation_baggages"].invoke(bags)
    # 升舱到经济舱后：每人免费 2 件 → 放行
    tools["update_reservation_flights"].invoke({"reservation_id": "r1", "cabin": "economy",
                                                "flights": [], "payment_id": "card"})
    assert tools["update_reservation_baggages"].invoke(bags) == '{"reservation_id": "r1"}'
    assert backend.calls == ["update_reservation_flights", "update_reservation_baggages"]


@pytest.mark.asyncio
async def test_product_baggage_check_uses_cabin_after_flight_change(monkeypatch):
    class FakeBackend:
        def get_reservation_owner(self, reservation_id):
            return "u1"

        def get_user_details(self, user_id):
            return json.dumps(USER)

        def get_reservation_details(self, reservation_id):
            return json.dumps({"cabin": "basic_economy", "passengers": [{}]})

        def update_reservation_flights(self, *args):
            return json.dumps({"reservation_id": "r9"})

        def update_reservation_baggages(self, *args):
            return "updated"

    monkeypatch.setattr(airline_server, "backend", lambda: FakeBackend())
    monkeypatch.setattr(airline_server, "_cabin_after_change", {})
    kw = dict(reservation_id="r9", total_baggages=2, nonfree_baggages=0, payment_id="card", user_id="u1")
    assert "must be 1" in await airline_server.update_reservation_baggages(**kw)
    await airline_server.update_reservation_flights(reservation_id="r9", cabin="economy", flights=[],
                                                    payment_id="card", user_id="u1")
    assert await airline_server.update_reservation_baggages(**kw) == "updated"
