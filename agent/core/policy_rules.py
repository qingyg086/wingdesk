"""Deterministic checks for the airline rules that must not depend on model recall.

These functions only inspect trusted user/reservation records and proposed tool
arguments. They return a useful tool error so the agent can correct its call.
"""

from collections import Counter


FREE_BAGS = {
    "regular": {"basic_economy": 0, "economy": 1, "business": 2},
    "silver": {"basic_economy": 1, "economy": 2, "business": 3},
    "gold": {"basic_economy": 2, "economy": 3, "business": 3},
}
PAYMENT_LIMITS = {"certificate": 1, "credit_card": 1, "gift_card": 3}


def expected_nonfree(total: int, membership: str, cabin: str, passengers: int) -> int:
    """Baggage counts are for the entire order; allowance is per passenger."""
    return max(0, total - FREE_BAGS[membership][cabin] * passengers)


def validate_booking(user: dict, args: dict) -> str | None:
    """Check payment-method counts and baggage charges before booking."""
    if not user:
        return "Error: user not found; booking was not executed"
    profile = user.get("payment_methods", {})
    counts = Counter()
    for payment in args.get("payment_methods", []):
        pid = payment.get("payment_id")
        method = profile.get(pid)
        if not method:
            return f"Error: payment method {pid} is not in the user's profile; booking was not executed"
        source = method.get("source")
        if source not in PAYMENT_LIMITS:
            return f"Error: unsupported payment method {pid}; booking was not executed"
        counts[source] += 1
        if counts[source] > PAYMENT_LIMITS[source]:
            return (f"Error: at most {PAYMENT_LIMITS[source]} {source} payment method(s) "
                    "per reservation; booking was not executed. Choose allowed payment methods and retry.")
    try:
        total = args["total_baggages"]
        nonfree = args["nonfree_baggages"]
        n_people = len(args["passengers"])
        if not isinstance(total, int) or total < 0 or not isinstance(nonfree, int) or not 1 <= n_people <= 5:
            return "Error: invalid baggage or passenger count; booking was not executed"
        expected = expected_nonfree(total, user["membership"], args["cabin"], n_people)
    except (KeyError, TypeError):
        return "Error: missing membership, cabin, or baggage data; booking was not executed"
    if nonfree != expected:
        return (f"Error: nonfree_baggages must be {expected} for {n_people} passenger(s), "
                f"membership={user['membership']}, cabin={args['cabin']}, total_baggages={total}; "
                "booking was not executed. Recalculate payment and retry.")
    return None


def validate_baggage_update(user: dict, reservation: dict, args: dict, *, cabin: str | None = None) -> str | None:
    """Check the proposed total against the order's free allowance."""
    if not user or not reservation:
        return "Error: user or reservation not found; baggage update was not executed"
    try:
        total = args["total_baggages"]
        nonfree = args["nonfree_baggages"]
        n_people = len(reservation["passengers"])
        if not isinstance(total, int) or total < 0 or not isinstance(nonfree, int) or n_people < 1:
            return "Error: invalid baggage or passenger count; baggage update was not executed"
        effective_cabin = cabin or reservation["cabin"]
        expected = expected_nonfree(total, user["membership"], effective_cabin, n_people)
    except (KeyError, TypeError):
        return "Error: missing membership, cabin, or baggage data; baggage update was not executed"
    if nonfree != expected:
        return (f"Error: nonfree_baggages must be {expected} for {n_people} passenger(s), "
                f"membership={user['membership']}, cabin={effective_cabin}, total_baggages={total}; "
                "baggage update was not executed. Recalculate payment and retry.")
    return None
