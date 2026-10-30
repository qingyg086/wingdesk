"""
航空客服 MCP Server（FastMCP，stdio 传输）：把 MySQLBackend 的 14 个方法暴露成 14 个 MCP 工具。

启动方式：由后端的 MCPToolRegistry 按 config/mcp_servers.json 拉起，整个应用生命周期只拉起一次。
单独调试：  cd agent && python -m mcp_servers.airline_server

几个要点：
1. stdio 传输下，stdout 是 JSON-RPC 协议通道，任何 print 都会破坏协议。日志一律写 stderr。
2. 工具函数是 async 的，真正查库的同步代码放到线程池里跑（anyio.to_thread），
   否则一个慢查询会卡住整个 Server 的事件循环，其他并发的工具调用都得等着。
3. 防越权（第二道防线）：涉及用户数据的工具都带 user_id 参数，这个参数由后端拦截器
   UserIdInjector 根据当前登录用户强制注入，模型看不到也改不了；涉及订单的工具还会检查
   订单是否属于该用户。不属于时返回和"订单不存在"完全相同的错误，不泄露订单是否存在。
4. 工具名、参数名、英文描述与 τ-bench 原版一致（search_onestop_flight 原版描述写成了
   "Search direct flights"，是原版的笔误，这里改正）。
"""

import functools
import json
import logging
import sys
from pathlib import Path
from typing import Annotated, Literal, Union

import anyio
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))          # agent/
from backends.base import safe_calculate                                # noqa: E402
from backends.mysql_backend import MySQLBackend                         # noqa: E402
from core.policy_rules import validate_booking, validate_baggage_update  # noqa: E402

logging.basicConfig(stream=sys.stderr, level=logging.INFO,
                    format="%(asctime)s [airline-mcp] %(levelname)s %(message)s")
log = logging.getLogger("airline-mcp")

mcp = FastMCP("AirlineMCPServer")
_backend: MySQLBackend | None = None
# 订单号 → 最近一次成功改签后的舱位。MySQLBackend 照搬了原版"改航班不更新订单 cabin 字段"的特性（NOTE 4），
# 行李校验要用新舱位算免费额度。存在进程内存里：Server 重启后丢失，此时退回用订单上的舱位（偏保守）。
_cabin_after_change: dict[str, str] = {}


def backend() -> MySQLBackend:
    global _backend
    if _backend is None:
        _backend = MySQLBackend()
        log.info("MySQLBackend 已连接，库 %s", _backend.database)
    return _backend


async def run(fn, *args, **kwargs):
    """把同步的数据库调用放到线程池执行，不阻塞事件循环。"""
    return await anyio.to_thread.run_sync(functools.partial(fn, *args, **kwargs))


async def owns(reservation_id: str, user_id: str) -> bool:
    return await run(backend().get_reservation_owner, reservation_id) == user_id


# ------------------------------------------------------------------ 参数类型（与 τ-bench 的 JSON Schema 对应）
Cabin = Literal["basic_economy", "economy", "business"]
Number = Union[int, float]          # 用 int|float 而不是 float：传 100 时保持整数 100，否则会变成 100.0
UserId = Annotated[str, Field(description="当前登录用户的 ID。由系统根据登录态自动注入，调用时无需也无法指定。")]
ResId = Annotated[str, Field(description="The reservation ID, such as 'ZFA04Y'.")]
PayId = Annotated[str, Field(description="The payment id stored in user profile, such as 'credit_card_7815826', "
                                         "'gift_card_7815826', 'certificate_7815826'.")]
Airport = Annotated[str, Field(description="The airport in three letters, such as 'JFK'.")]
Date = Annotated[str, Field(description="The date of the flight in the format 'YYYY-MM-DD', such as '2024-05-01'.")]


class FlightRef(BaseModel):
    flight_number: str = Field(description="Flight number, such as 'HAT001'.")
    date: str = Field(description="The date for the flight in the format 'YYYY-MM-DD', such as '2024-05-01'.")


class PassengerInfo(BaseModel):
    first_name: str = Field(description="The first name of the passenger, such as 'Noah'.")
    last_name: str = Field(description="The last name of the passenger, such as 'Brown'.")
    dob: str = Field(description="The date of birth of the passenger in the format 'YYYY-MM-DD', such as '1990-01-01'.")


class PaymentItem(BaseModel):
    payment_id: str = Field(description="The payment id stored in user profile, such as 'credit_card_7815826', "
                                        "'gift_card_7815826', 'certificate_7815826'.")
    amount: Number = Field(description="The amount to be paid.")


def dump(items):
    return [i.model_dump() for i in items]


# ================================================================== 只读工具
@mcp.tool()
async def get_user_details(user_id: UserId) -> str:
    """Get the details of an user, including their reservations."""
    return await run(backend().get_user_details, user_id)


@mcp.tool()
async def get_reservation_details(reservation_id: Annotated[str, Field(description="The reservation id, such as '8JX2WO'.")],
                                  user_id: UserId) -> str:
    """Get the details of a reservation."""
    if not await owns(reservation_id, user_id):
        return "Error: user not found"           # 与原版"订单不存在"的返回完全相同
    return await run(backend().get_reservation_details, reservation_id)


@mcp.tool()
async def list_all_airports() -> str:
    """List all airports and their cities."""
    return await run(backend().list_all_airports)


@mcp.tool()
async def search_direct_flight(origin: Airport, destination: Airport, date: Date) -> str:
    """Search direct flights between two cities on a specific date."""
    return await run(backend().search_direct_flight, origin, destination, date)


@mcp.tool()
async def search_onestop_flight(origin: Airport, destination: Airport, date: Date) -> str:
    """Search one-stop flights (with exactly one connection) between two cities on a specific date."""
    return await run(backend().search_onestop_flight, origin, destination, date)


# ================================================================== 写工具（Agent 侧在执行前会经过人工确认）
@mcp.tool()
async def book_reservation(
    user_id: UserId,
    origin: Annotated[str, Field(description="The IATA code for the origin city, such as 'SFO'.")],
    destination: Annotated[str, Field(description="The IATA code for the destination city, such as 'JFK'.")],
    flight_type: Literal["one_way", "round_trip"],
    cabin: Cabin,
    flights: Annotated[list[FlightRef], Field(description="An array of objects containing details about each piece of flight.")],
    passengers: Annotated[list[PassengerInfo], Field(description="An array of objects containing details about each passenger.")],
    payment_methods: Annotated[list[PaymentItem], Field(description="An array of objects containing details about each payment method.")],
    total_baggages: Annotated[int, Field(description="The total number of baggage items included in the reservation.")],
    nonfree_baggages: Annotated[int, Field(description="The number of non-free baggage items included in the reservation.")],
    insurance: Literal["yes", "no"],
) -> str:
    """Book a reservation."""
    user_result = await run(backend().get_user_details, user_id)
    if user_result.startswith("Error"):
        return user_result
    error = validate_booking(json.loads(user_result), {
        "cabin": cabin, "passengers": passengers, "payment_methods": dump(payment_methods),
        "total_baggages": total_baggages, "nonfree_baggages": nonfree_baggages,
    })
    if error:
        return error
    return await run(backend().book_reservation, user_id, origin, destination, flight_type, cabin,
                     dump(flights), dump(passengers), dump(payment_methods), total_baggages,
                     nonfree_baggages, insurance)


@mcp.tool()
async def update_reservation_flights(
    reservation_id: ResId,
    cabin: Cabin,
    flights: Annotated[list[FlightRef], Field(description="An array of objects containing details about each piece of "
                                              "flight in the ENTIRE new reservation. Even if the a flight segment is "
                                              "not changed, it should still be included in the array.")],
    payment_id: PayId,
    user_id: UserId,
) -> str:
    """Update the flight information of a reservation."""
    if not await owns(reservation_id, user_id):
        return "Error: reservation not found"
    result = await run(backend().update_reservation_flights, reservation_id, cabin, dump(flights), payment_id)
    if not result.startswith("Error"):
        _cabin_after_change[reservation_id] = cabin
    return result


@mcp.tool()
async def update_reservation_passengers(
    reservation_id: ResId,
    passengers: Annotated[list[PassengerInfo], Field(description="An array of objects containing details about each passenger.")],
    user_id: UserId,
) -> str:
    """Update the passenger information of a reservation."""
    if not await owns(reservation_id, user_id):
        return "Error: reservation not found"
    return await run(backend().update_reservation_passengers, reservation_id, dump(passengers))


@mcp.tool()
async def update_reservation_baggages(
    reservation_id: ResId,
    total_baggages: Annotated[int, Field(description="The updated total number of baggage items included in the reservation.")],
    nonfree_baggages: Annotated[int, Field(description="The updated number of non-free baggage items included in the reservation.")],
    payment_id: PayId,
    user_id: UserId,
) -> str:
    """Update the baggage information of a reservation."""
    if not await owns(reservation_id, user_id):
        return "Error: reservation not found"
    user_result, reservation_result = await run(backend().get_user_details, user_id), await run(
        backend().get_reservation_details, reservation_id)
    if user_result.startswith("Error") or reservation_result.startswith("Error"):
        return "Error: user or reservation not found"
    error = validate_baggage_update(json.loads(user_result), json.loads(reservation_result), {
        "total_baggages": total_baggages, "nonfree_baggages": nonfree_baggages,
    }, cabin=_cabin_after_change.get(reservation_id))
    if error:
        return error
    return await run(backend().update_reservation_baggages, reservation_id, total_baggages,
                     nonfree_baggages, payment_id)


@mcp.tool()
async def cancel_reservation(reservation_id: ResId, user_id: UserId) -> str:
    """Cancel the whole reservation."""
    if not await owns(reservation_id, user_id):
        return "Error: reservation not found"
    return await run(backend().cancel_reservation, reservation_id)


@mcp.tool()
async def send_certificate(user_id: UserId,
                           amount: Annotated[Number, Field(description="Certificate amount to send.")]) -> str:
    """Send a certificate to a user. Be careful!"""
    return await run(backend().send_certificate, user_id, amount)


# ================================================================== 不访问数据库的工具
@mcp.tool()
async def calculate(expression: Annotated[str, Field(description="The mathematical expression to calculate, such as "
                                                    "'2 + 2'. The expression can contain numbers, operators "
                                                    "(+, -, *, /), parentheses, and spaces.")]) -> str:
    """Calculate the result of a mathematical expression."""
    return safe_calculate(expression)          # 纯计算，不需要连数据库


@mcp.tool()
async def think(thought: Annotated[str, Field(description="A thought to think about.")]) -> str:
    """Use the tool to think about something. It will not obtain new information or change the database, but just
    append the thought to the log. Use it when complex reasoning is needed."""
    return ""


@mcp.tool()
async def transfer_to_human_agents(summary: Annotated[str, Field(description="A summary of the user's issue.")]) -> str:
    """Transfer the user to a human agent, with a summary of the user's issue. Only transfer if the user explicitly
    asks for a human agent, or if the user's issue cannot be resolved by the agent with the available tools."""
    return "Transfer successful"


if __name__ == "__main__":
    log.info("Airline MCP Server 启动（stdio）")
    mcp.run(transport="stdio")
