"""
Orchestrator（总控路由）。

1. structured output：用 Pydantic 模型 + Literal 枚举约束输出，模型只能从 6 个合法值里选，解析失败有兜底。
   不让模型输出自由文本再做字符串匹配——模型多说一句话、换个大小写就可能路由错；
2. 粘性路由：把"当前正在处理的 Agent"和最近几轮对话一起给路由器，用户在补充信息、回答追问、
   确认方案时，继续交给同一个 Agent；只有明确换话题才切换。
"""

import logging
import re
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from agents.prompts import NOW_PLACEHOLDER
from core.clock import now_text

log = logging.getLogger(__name__)

Target = Literal["flight_search_agent", "booking_agent", "reservation_agent",
                 "cancel_refund_agent", "policy_agent", "human_handoff"]
TARGETS: tuple[str, ...] = Target.__args__


class RouteDecision(BaseModel):
    """路由决定。"""
    target: Target = Field(description="负责处理这条消息的 Agent")
    reason: str = Field(description="一句话说明为什么这样路由")


ROUTER_PROMPT = f"""你是航空公司客服系统的总控路由。根据对话内容，决定由哪个专业 Agent 处理用户的最新消息。
系统当前时间：{NOW_PLACEHOLDER}。

可选的 Agent：
- flight_search_agent：查航班、查航线、查机场、比较直飞和中转方案（只查询，不下单）。
- booking_agent：预订新机票（新建订单），包括给同行人/朋友订票、取消后重新预订。
- reservation_agent：查询账户信息（会员等级、已保存的支付方式、礼品卡/代金券余额、名下订单列表）和已有订单；
  修改已有订单的航班、舱位（升舱/降舱）、行李、乘客信息。
- cancel_refund_agent：取消订单、退票退款；航班取消或延误后的投诉与补偿。
- policy_agent：只问规则、不办业务的政策咨询（行李额度、保险、改签规则、退票条件、婴儿儿童、宠物、值机等）。
- human_handoff：**只在用户明确要求转人工**（如"转人工""找真人客服""transfer me to a human agent"）时选择。

路由规则（按优先级）：
1. 粘性路由：如果"当前处理中的 Agent"不为空，且用户的最新消息是在继续这件事——补充信息、回答追问、
   选择方案（"订第二个""就这个"）、确认或拒绝（"好的""可以""不用了"）——必须继续交给当前 Agent。
2. 只有当用户明确开始了另一件事时，才切换 Agent。例如查完航班说"帮我订这个"→ 切到 booking_agent。
3. 用户在办理业务的过程中顺带问规则（如订票时问"能带几件行李"），交给正在办理的 Agent，它的提示词里有相关政策。
4. 只问规则、不涉及具体订单或下单，才交给 policy_agent。
5. 以下情况都**不是**转人工的理由，必须交给对应的业务 Agent，由它按政策办理或拒绝：
   用户坚持违反政策的要求、情绪激动、声称紧急、一次要办多个订单、需求复杂、问余额等账户信息。"""


def router_prompt() -> str:
    """路由提示词，当前时间在每次调用时现取。"""
    return ROUTER_PROMPT.replace(NOW_PLACEHOLDER, now_text())



# 用户明确要求人工客服的说法（中英文）。只有命中时才允许路由到 human_handoff。
HUMAN_REQUEST = re.compile(
    r"human|real person|live (agent|person)|representative|supervisor|manager|transfer me|"
    r"人工|真人|主管|经理", re.IGNORECASE)
BUSINESS_TARGETS = tuple(t for t in TARGETS if t != "human_handoff")


class BusinessRouteDecision(BaseModel):
    """用户没有要求人工时的路由决定（不含 human_handoff）。"""
    target: Literal["flight_search_agent", "booking_agent", "reservation_agent",
                    "cancel_refund_agent", "policy_agent"] = Field(description="负责处理这条消息的业务 Agent")
    reason: str = Field(description="一句话说明为什么这样路由")


def _last_user_text(messages) -> str:
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return m.content if isinstance(m.content, str) else str(m.content)
    return ""


def explicit_human_request(text: str) -> bool:
    return bool(HUMAN_REQUEST.search(text or ""))


def _excerpt(messages, max_turns: int = 8) -> str:
    """只取最近几轮的用户消息和助手的文字回复（跳过工具调用和工具结果），给路由器做上下文。"""
    lines = []
    for m in messages:
        if isinstance(m, HumanMessage):
            lines.append(f"用户：{m.content}")
        elif isinstance(m, AIMessage) and m.content and not m.tool_calls:
            text = m.content if isinstance(m.content, str) else str(m.content)
            lines.append(f"客服：{text[:300]}")
    return "\n".join(lines[-max_turns:])


class OrchestratorAgent:
    def __init__(self, llm):
        # method="function_calling"：用工具调用的方式输出结构化结果，qwen-plus 支持；
        # 比让模型自由输出 JSON 再解析可靠得多
        self.router = llm.with_structured_output(RouteDecision, method="function_calling")
        # 路由器选了转人工、但用户并没有要求人工时，用不含 human_handoff 的枚举重新路由。
        # 失败案例分析发现：不少失败是路由器在用户"坚持违规 / 问余额 / 一次办多单"时直接转人工，
        # 本该办理的业务一件没办。提示词约束是概率性的，这里用代码兜底。
        self.business_router = llm.with_structured_output(BusinessRouteDecision, method="function_calling")

    async def route(self, state) -> dict:
        active = state.get("active_agent") or ""
        convo = _excerpt(state.get("messages", []))
        question = HumanMessage(content=f"当前处理中的 Agent：{active or '无'}\n\n最近的对话：\n{convo}\n\n"
                                        "请决定由哪个 Agent 处理用户的最新消息。")
        try:
            decision: RouteDecision = await self.router.ainvoke([SystemMessage(content=router_prompt()), question])
            target, reason = decision.target, decision.reason
            if target == "human_handoff" and not explicit_human_request(_last_user_text(state.get("messages", []))):
                log.info("路由器选了转人工，但用户没有明确要求人工，改为在业务 Agent 中重新路由")
                try:
                    again = await self.business_router.ainvoke([
                        SystemMessage(content=router_prompt() + "\n\n注意：用户没有明确要求人工客服，"
                                                              "只能从 5 个业务 Agent 中选择。"),
                        question])
                    target, reason = again.target, f"{again.reason}（用户未要求人工，不转人工）"
                except Exception as e:
                    target = active if active in BUSINESS_TARGETS else "policy_agent"
                    reason = f"重新路由失败，兜底到 {target}（{type(e).__name__}）"
                    log.warning(reason)
        except Exception as e:   # 结构化输出失败（网络、格式）时的兜底：有进行中的任务就继续，否则走政策咨询
            target = active if active in TARGETS else "policy_agent"
            reason = f"路由失败，兜底到 {target}（{type(e).__name__}）"
            log.warning(reason)
        log.info("路由 → %s（%s）", target, reason)
        # 被选中的 Agent 成为"当前处理中的 Agent"，并由 checkpointer 保存到下一轮，实现粘性路由
        return {"next_agent": target, "active_agent": target, "route_reason": reason}
