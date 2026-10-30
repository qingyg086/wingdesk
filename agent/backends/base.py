"""
航空业务后端的抽象接口：14 个方法，与 τ-bench airline 的 14 个工具一一对应。

为什么要有这一层：
- 产品模式：MCP Server → MySQLBackend → MySQL（真实数据库、事务、连接池）
- 评测模式：Agent 工具 → TauBenchBackend → τ-bench env.step（用原版工具逻辑和数据，保证评分有效）
同一套 Agent 代码，换一个后端就能在两种模式下运行；测试时也可以用同样的输入调用两个后端，比对结果。

约定：
- 所有方法的参数名、返回值与 τ-bench 原版工具完全一致：成功时返回 JSON 字符串，
  失败时返回以 "Error: " 开头的字符串（原版就是这样，Agent 会把错误信息读给模型）。
- 业务规则（能不能退票、改签）不在后端检查——原版工具也不检查，靠 Agent 遵守政策。
"""

import ast
import operator
from abc import ABC, abstractmethod
from typing import Any

# 写操作工具：执行前需要用户确认（interrupt 用这个列表）
WRITE_TOOLS = {
    "book_reservation", "update_reservation_flights", "update_reservation_passengers",
    "update_reservation_baggages", "cancel_reservation", "send_certificate",
}

ALL_TOOLS = [
    "get_user_details", "get_reservation_details", "list_all_airports",
    "search_direct_flight", "search_onestop_flight", "calculate", "think",
    "book_reservation", "update_reservation_flights", "update_reservation_passengers",
    "update_reservation_baggages", "cancel_reservation", "send_certificate",
    "transfer_to_human_agents",
]


class AirlineBackend(ABC):
    # ------------------------------------------------------------------ 只读
    @abstractmethod
    def get_user_details(self, user_id: str) -> str: ...

    @abstractmethod
    def get_reservation_details(self, reservation_id: str) -> str: ...

    @abstractmethod
    def list_all_airports(self) -> str: ...

    @abstractmethod
    def search_direct_flight(self, origin: str, destination: str, date: str) -> str: ...

    @abstractmethod
    def search_onestop_flight(self, origin: str, destination: str, date: str) -> str: ...

    # ------------------------------------------------------------------ 写
    @abstractmethod
    def book_reservation(self, user_id: str, origin: str, destination: str, flight_type: str,
                         cabin: str, flights: list[dict], passengers: list[dict],
                         payment_methods: list[dict], total_baggages: int, nonfree_baggages: int,
                         insurance: str) -> str: ...

    @abstractmethod
    def update_reservation_flights(self, reservation_id: str, cabin: str, flights: list[dict],
                                   payment_id: str) -> str: ...

    @abstractmethod
    def update_reservation_passengers(self, reservation_id: str, passengers: list[dict]) -> str: ...

    @abstractmethod
    def update_reservation_baggages(self, reservation_id: str, total_baggages: int,
                                    nonfree_baggages: int, payment_id: str) -> str: ...

    @abstractmethod
    def cancel_reservation(self, reservation_id: str) -> str: ...

    @abstractmethod
    def send_certificate(self, user_id: str, amount: float) -> str: ...

    # ------------------------------------------------------------------ 不访问数据的工具
    # 产品模式下这几个和数据库无关，默认实现放在基类；TauBenchBackend 会覆盖成 env.step，
    # 因为评测时每一次工具调用（包括 think）都要记进 τ-bench 的动作日志。
    def calculate(self, expression: str) -> str:
        return safe_calculate(expression)

    def think(self, thought: str) -> str:
        return ""

    def transfer_to_human_agents(self, summary: str) -> str:
        return "Transfer successful"


# ======================================================================================
# calculate：原版用 eval()。虽然限制了字符集，但 "9**9**9**9" 只用合法字符就能把 CPU 跑满（DoS）。
# 这里改为 AST 白名单求值：只允许数字、+ - * /、正负号和括号。
# 与原版的差异：不支持 **（原版支持）；语法错误时的报错文本可能不同。数值结果一致。
# ======================================================================================
_ALLOWED_CHARS = set("0123456789+-*/(). ")
_BINOPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _eval_node(node: ast.AST) -> Any:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        return _BINOPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval_node(node.operand))
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
        raise ValueError("exponent (**) is not supported")
    raise ValueError(f"unsupported expression: {type(node).__name__}")


def safe_calculate(expression: str) -> str:
    if not all(ch in _ALLOWED_CHARS for ch in expression):
        return "Error: invalid characters in expression"
    try:
        # eval() 会忽略开头的空格，ast.parse 不会（会报 unexpected indent），这里对齐原版行为
        tree = ast.parse(expression.lstrip(" "), filename="<string>", mode="eval")
        return str(round(float(_eval_node(tree)), 2))
    except Exception as e:  # 与原版一致：任何异常都转成 "Error: ..." 返回给模型
        return f"Error: {e}"
