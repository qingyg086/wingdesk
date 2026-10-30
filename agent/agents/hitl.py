"""
人工确认（Human-in-the-loop）：写工具执行前用 LangGraph 的 interrupt() 暂停，等用户确认。

为什么不只靠提示词让模型"先问用户"：
- 模型可能忘记问、可能把"好的我看看"误当成确认，提示词约束是概率性的；
- interrupt 是代码层面的硬约束：写工具被包了一层，不经过用户确认就不可能执行。

流程：
1. 模型调用写工具（如 book_reservation）→ 进入包装函数 → interrupt(payload) 暂停整张图，
   payload 里有操作名、参数和一段**由代码生成**的中文摘要（不让模型写，避免摘要和真实参数不一致）；
2. 图的状态由 checkpointer 保存，chat_service 把 payload 作为 confirm 事件推给前端；
3. 用户确认/拒绝后，chat_service 用 Command(resume={"approved": ..., "feedback": ...}) 恢复；
4. 恢复时 LangGraph 会**重新执行**被中断的工具节点，interrupt() 这次直接返回用户的决定：
   确认 → 调用真正的 MCP 工具；拒绝 → 返回"用户未确认"，模型据此调整。

注意：因为恢复时会重新执行整个工具节点，interrupt() 之前不能有副作用（这里之前只有拼摘要，没有副作用）。
"""

import json
from pathlib import Path
from typing import Annotated, Any

from langchain_core.tools import BaseTool, InjectedToolArg, StructuredTool
from langgraph.types import interrupt

from backends.base import WRITE_TOOLS

_ENUMS = json.loads((Path(__file__).resolve().parents[2] / "data" / "airline" / "enums_zh.json")
                    .read_text(encoding="utf-8"))
YES_WORDS = {"yes", "y", "ok", "确认", "确定", "是", "是的", "好", "好的", "同意", "可以", "没问题"}


def _zh(category: str, value: Any) -> str:
    return _ENUMS.get(category, {}).get(value, str(value))


def _flights(flights) -> str:
    return "、".join(f"{f.get('flight_number')}（{f.get('date')}）" for f in flights or [])


def _people(passengers) -> str:
    return "、".join(f"{p.get('first_name')} {p.get('last_name')}（{p.get('dob')}）" for p in passengers or [])


def summarize(action: str, a: dict) -> str:
    """把写工具的参数翻译成给用户看的中文摘要。"""
    if action == "book_reservation":
        pays = "；".join(f"{p.get('payment_id')} 支付 ${p.get('amount')}" for p in a.get("payment_methods", []))
        total = sum(p.get("amount", 0) for p in a.get("payment_methods", []))
        return (f"预订机票：{a.get('origin')} → {a.get('destination')}，{_zh('flight_type', a.get('flight_type'))}，"
                f"{_zh('cabin', a.get('cabin'))}\n航班：{_flights(a.get('flights'))}\n乘客：{_people(a.get('passengers'))}\n"
                f"托运行李：共 {a.get('total_baggages')} 件，其中付费 {a.get('nonfree_baggages')} 件；"
                f"旅行保险：{_zh('insurance', a.get('insurance'))}\n支付：{pays}（合计 ${total}）")
    if action == "update_reservation_flights":
        return (f"修改订单 {a.get('reservation_id')} 的航班/舱位\n新航班：{_flights(a.get('flights'))}\n"
                f"舱位：{_zh('cabin', a.get('cabin'))}\n差价支付/退款方式：{a.get('payment_id')}")
    if action == "update_reservation_passengers":
        return f"修改订单 {a.get('reservation_id')} 的乘客信息为：{_people(a.get('passengers'))}"
    if action == "update_reservation_baggages":
        return (f"修改订单 {a.get('reservation_id')} 的托运行李：共 {a.get('total_baggages')} 件，"
                f"其中付费 {a.get('nonfree_baggages')} 件\n支付方式：{a.get('payment_id')}")
    if action == "cancel_reservation":
        return f"取消订单 {a.get('reservation_id')}（整单取消，款项按原支付方式退回）"
    if action == "send_certificate":
        return f"向您的账户发放旅行代金券 ${a.get('amount')}"
    return f"{action}：{json.dumps(a, ensure_ascii=False)}"


def parse_decision(decision: Any) -> tuple[bool, str]:
    """resume 的值可能是 dict / bool / 字符串，统一成 (是否确认, 用户反馈)。"""
    if isinstance(decision, dict):
        return bool(decision.get("approved")), str(decision.get("feedback") or "")
    if isinstance(decision, bool):
        return decision, ""
    text = str(decision or "").strip()
    return text.lower().rstrip("。.!！") in YES_WORDS, text


def with_confirmation(tool: BaseTool) -> BaseTool:
    """返回一个同名、同参数的新工具：先 interrupt 等确认，确认后再调用原工具。"""
    original = tool

    async def _run(runtime: Annotated[object | None, InjectedToolArg()] = None, **kwargs):
        payload = {
            "type": "confirm_action",
            "action": original.name,
            "args": kwargs,
            "summary_zh": summarize(original.name, kwargs),
        }
        approved, feedback = parse_decision(interrupt(payload))
        if not approved:
            msg = "用户未确认该操作，操作没有执行。" + (f"用户的反馈：{feedback}" if feedback else "")
            return msg, None
        # 原工具是 MCP 适配器生成的，签名同样是 (runtime, **arguments)，返回 (content, artifact)。
        # runtime 必须原样传下去：UserIdInjector 要从 runtime.config 里取当前登录用户。
        return await original.coroutine(runtime=runtime, **kwargs)

    return StructuredTool(
        name=original.name,
        description=original.description,
        args_schema=original.args_schema,
        coroutine=_run,
        response_format="content_and_artifact",
        metadata={**(original.metadata or {}), "requires_confirmation": True},
    )


def wrap_write_tools(tools: list[BaseTool]) -> list[BaseTool]:
    return [with_confirmation(t) if t.name in WRITE_TOOLS else t for t in tools]
