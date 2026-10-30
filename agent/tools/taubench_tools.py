"""
评测模式的工具：直接用 τ-bench 原版的工具定义（名称、描述、参数 JSON Schema，含 user_id），
调用时转给 TauBenchBackend → env.step。

和产品模式的区别：
- 产品模式的工具来自 MCP Server（MySQL），user_id 由系统注入、模型看不到；
- 评测模式按 τ-bench 规则，Agent 要先在对话里问到用户 ID，再自己填进工具参数，所以 schema 保留 user_id。
"""

from typing import Any

from langchain_core.tools import StructuredTool
from core.policy_rules import validate_booking, validate_baggage_update


def build_taubench_tools(backend: Any, tools_info: list[dict]) -> dict:
    """tools_info：env.tools_info（OpenAI function calling 格式）。返回 {工具名: LangChain 工具}。

    规则校验：订票、改行李在真正调用 env.step 之前先用 core.policy_rules 校验，
    不合规就返回错误给模型、不执行，也不会出现在 τ-bench 的动作记录里。
    舱位跟踪：τ-bench 原版 update_reservation_flights 不会更新订单的 cabin 字段，
    升舱后订单上仍是旧舱位。如果按订单上的舱位算免费行李额度，会把"升舱 + 加行李"的正确操作也拦下来
    （task 4、5）。所以这里记下每个订单最近一次成功改签后的新舱位，校验行李时优先用它。
    """
    tools = {}
    cabin_after_change: dict[str, str] = {}      # 订单号 → 改签后的舱位（每个评测环境一份）
    for info in tools_info:
        fn = info["function"]
        name = fn["name"]

        def _run(_name=name, **kwargs) -> str:
            data = backend.env.data
            if _name == "book_reservation":
                error = validate_booking(data["users"].get(kwargs.get("user_id")), kwargs)
                if error:
                    return error
            elif _name == "update_reservation_baggages":
                reservation_id = kwargs.get("reservation_id")
                reservation = data["reservations"].get(reservation_id)
                user = data["users"].get(reservation["user_id"]) if reservation else None
                error = validate_baggage_update(user, reservation, kwargs,
                                                cabin=cabin_after_change.get(reservation_id))
                if error:
                    return error
            result = backend.call(_name, **kwargs)
            if _name == "update_reservation_flights" and not str(result).startswith("Error"):
                cabin_after_change[kwargs.get("reservation_id")] = kwargs.get("cabin")
            return result

        tools[name] = StructuredTool.from_function(
            func=_run, name=name, description=fn.get("description", ""),
            args_schema=fn.get("parameters") or {"type": "object", "properties": {}},
        )
    return tools
