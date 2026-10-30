"""
5 个专业 Agent + 转人工节点。

每个 Agent = create_react_agent（ReAct 循环：模型思考 → 调工具 → 看结果 → 再思考……）+ 一份工具白名单 + 一段提示词。
设计原则：
- 最小权限：每个 Agent 只拿到完成本职工作需要的工具。例如 flight_search_agent 没有任何写工具，
  policy_agent 连业务工具都没有。模型就算"想"越界也做不到。
- 政策按需注入：每个 Agent 的提示词只放和它相关的政策章节（来自 data/airline/policy/core），
  上下文更短，模型更不容易被无关条款干扰。
- reservation_agent 也拿到 send_certificate。
  原因：政策规定"航班延误、用户改签后，可按乘客数发 50 美元/人 的代金券"，改签由 reservation_agent 办理，
  如果补偿必须切到 cancel_refund_agent，一件事会被拆到两个 Agent 里。
- 评测模式（τ-bench）：个别 Agent 的职责说明换成 eval_duties，
  · 订票的最后一步改成"先在对话里列出详情、得到 yes 再下单"（评测没有确认卡片，模拟用户只会在对话里确认）；
  · 航班查询不提 Neo4j 航线工具，政策咨询不用 RAG 检索、直接注入全部核心政策——
    评测环境只有 τ-bench 的 14 个工具和原版政策，不能依赖本项目的数据库和扩充的仿真政策。
"""

from dataclasses import dataclass, field

from langchain_core.messages import SystemMessage
from langgraph.prebuilt import create_react_agent

from agents.hitl import wrap_write_tools
from agents.prompts import build_system_prompt
from core.workflow.state import AgentState

SEARCH = ["list_all_airports", "search_direct_flight", "search_onestop_flight"]
COMMON = ["calculate", "think", "transfer_to_human_agents"]


@dataclass
class AgentSpec:
    role: str
    duties: str
    tools: list[str]                          # MCP 工具白名单
    policy: tuple[str, ...] = ()              # 注入的核心政策文件
    local_tools: list[str] = field(default_factory=list)   # 本地工具（Neo4j 航线、Milvus 政策检索）
    eval_duties: str | None = None            # 评测模式下替换 duties（None = 两种模式相同）
    eval_policy: tuple[str, ...] | None = None   # 评测模式下替换 policy（None = 相同）


ALL_CORE_POLICY = ("01_general", "02_booking", "03_checked_baggage", "04_travel_insurance", "05_modify",
                   "06_cancellation", "07_compensation")

_FLIGHT_SEARCH_BASE = """你负责帮用户查询航班和航线，只做查询，不能下单或修改订单。
- 用户说城市名时，用 list_all_airports 找到对应的机场代码（一个城市可能有多个机场，如纽约有 JFK 和 LGA）。
- 查某天能订的航班：用 search_direct_flight（直飞）和 search_onestop_flight（一次中转）。
{routes}- 列出结果时说明航班号、起降时间（美东时间）、各舱位价格和余座。
- 用户想订票时，告诉他可以直接说"帮我订这个航班"。"""
_ROUTES_LINE = "- 只想了解两地之间有哪些走法、经哪里转、要不要两次中转：用 query_flight_routes（不含余座和价格）。\n"

_BOOKING_BASE = """你负责为用户预订新机票。办理步骤：
1. 调用 get_user_details 查看用户的会员等级、已保存的支付方式和常用乘客。
2. 确认行程类型（单程/往返）、出发地、目的地、日期，用搜索工具找到可订航班，和用户确定具体航班和舱位。
3. 收集每位乘客的名、姓、出生日期（最多 5 人，所有人同航班同舱位）。
4. 按会员等级和舱位算出免费托运行李额度，确认托运件数和付费件数；询问是否购买旅行保险。
5. 用 calculate 算出总价（机票 × 人数 + 保险 + 付费行李），确认支付方式和各自金额（必须是档案里已有的支付方式）。
6. {last}"""
_BOOKING_LAST_PRODUCT = "以上信息齐全后直接调用 book_reservation（系统会弹出确认卡片让用户确认，你不用在对话里再问一次）。"
_BOOKING_LAST_EVAL = "以上信息齐全后，先向用户完整列出订单详情（航班、舱位、乘客、行李、保险、支付方式与金额），得到用户明确确认（yes）后再调用 book_reservation。"


SPECS: dict[str, AgentSpec] = {
    "flight_search_agent": AgentSpec(
        role="航班查询专员",
        duties=_FLIGHT_SEARCH_BASE.format(routes=_ROUTES_LINE),
        eval_duties=_FLIGHT_SEARCH_BASE.format(routes=""),
        tools=SEARCH + COMMON,
        policy=("01_general",),
        local_tools=["query_flight_routes"],
    ),
    "booking_agent": AgentSpec(
        role="订票专员",
        duties=_BOOKING_BASE.format(last=_BOOKING_LAST_PRODUCT),
        eval_duties=_BOOKING_BASE.format(last=_BOOKING_LAST_EVAL),
        tools=["get_user_details"] + SEARCH + ["book_reservation"] + COMMON,
        policy=("01_general", "02_booking", "03_checked_baggage", "04_travel_insurance"),
    ),
    "reservation_agent": AgentSpec(
        role="订单管理专员",
        duties="""你负责查询账户信息，以及查询和修改用户已有的订单：改航班、改舱位、加托运行李、修改乘客信息。
账户查询：用户问会员等级、已保存的支付方式、礼品卡或代金券余额、名下有哪些订单时，用 get_user_details 查询；
需要求和（如所有礼品卡余额合计）时用 calculate 算出后回答。
修改订单的办理步骤：
1. 获取订单号（用户不记得时，用 get_user_details 查看订单列表），用 get_reservation_details 查看订单详情。
2. 对照政策判断能否修改（如基础经济舱不能改航班、不能改出发地/目的地/行程类型、乘客人数不能改、行李只能加不能减）。
3. 改航班或舱位前，用搜索工具查新航班的价格，用 calculate 算出差价，确认用于支付或接收退款的礼品卡或信用卡。
4. 调用对应的修改工具。
5. 如果用户投诉航班延误并因此改签，且符合补偿条件、明确要求补偿，在改签完成后可用 send_certificate 发放补偿。""",
        tools=["get_user_details", "get_reservation_details"] + SEARCH +
              ["update_reservation_flights", "update_reservation_passengers", "update_reservation_baggages",
               "send_certificate"] + COMMON,
        policy=("01_general", "03_checked_baggage", "05_modify", "07_compensation"),
    ),
    "cancel_refund_agent": AgentSpec(
        role="退票与补偿专员",
        duties="""你负责取消订单（退票）以及航班取消/延误后的补偿。办理步骤：
1. 获取订单号和取消原因（行程变更、航空公司取消了航班、其他原因），用 get_reservation_details 查看订单。
2. 严格按退票政策判断能否取消：下单 24 小时内、或航空公司取消了航班，可以退；否则经济舱/基础经济舱
   必须买了保险且满足保险条件；商务舱随时可退。有航段已经飞过的，不能办理，需转人工。
3. 不符合条件时明确拒绝，并说明依据。符合条件时调用 cancel_reservation。
4. 补偿：只有用户投诉并明确要求补偿时才处理；先核实会员等级、保险、舱位和航班状态，
   按政策用 calculate 算出金额，再用 send_certificate 发放。不要主动提出补偿。""",
        tools=["get_user_details", "get_reservation_details", "cancel_reservation", "send_certificate"] + COMMON,
        policy=("01_general", "04_travel_insurance", "06_cancellation", "07_compensation"),
    ),
    "policy_agent": AgentSpec(
        role="政策咨询专员",
        duties="""你负责回答航空公司的规则类问题（行李、保险、改签、退票、补偿、特殊旅客、宠物、值机等）。
- 每个问题都先用 query_vector_db 检索政策文档，只根据检索到的内容回答，并说明出自哪份政策。
- 检索结果里没有答案时，直接说政策文档中没有相关规定，不要凭常识补充。
- 回答只能复述检索到的条款，不要添加检索内容里没有的解释、推论或例外情况（宁可少说，不能多说）。
- 你不能办理任何业务；用户想订票、改签、退票时，告诉他可以直接提出办理需求。""",
        tools=["transfer_to_human_agents"],
        local_tools=["query_vector_db"],
        eval_duties="""你负责回答航空公司的规则类问题（行李、保险、改签、退票、补偿等）。
- 只根据下方政策条款回答；条款里没有的内容，直接说政策中没有相关规定，不要凭常识补充。
- 你不能办理任何业务；用户想订票、改签、退票时，告诉他可以直接提出办理需求。""",
        eval_policy=ALL_CORE_POLICY,
    ),
    "human_handoff": AgentSpec(
        role="人工转接",
        duties="""用户要求人工服务，或者问题超出线上客服的处理范围。
请用一两句话总结用户的问题，调用 transfer_to_human_agents（summary 参数填写总结），
然后告诉用户已为其转接人工客服，并简要说明转接原因。""",
        tools=["transfer_to_human_agents"],
    ),
}


def load_local_tools() -> dict:
    """本地（非 MCP）工具。延迟导入：连不上 Neo4j / Milvus 时工具会自己返回降级提示，不影响启动。"""
    from tools.route_tool import query_flight_routes
    from tools.vector_tool import query_vector_db
    return {"query_flight_routes": query_flight_routes, "query_vector_db": query_vector_db}


def build_specialists(mcp_tools: dict, llm, mode: str = "product", local_tools: dict | None = None) -> dict:
    """返回 {Agent 名: 编译好的 ReAct 子图}。
    mcp_tools：{工具名: 工具}。产品模式来自 MCPToolRegistry；评测模式来自 TauBenchBackend。
    产品模式下，写工具会被包上人工确认（interrupt）。"""
    local_tools = local_tools if local_tools is not None else load_local_tools()
    agents = {}
    for name, spec in SPECS.items():
        tools = [mcp_tools[t] for t in spec.tools if t in mcp_tools]
        tools += [local_tools[t] for t in spec.local_tools if t in local_tools]
        if mode == "product":
            tools = wrap_write_tools(tools)
        duties, policy = spec.duties, spec.policy
        if mode == "eval":
            duties = spec.eval_duties or spec.duties
            policy = spec.eval_policy if spec.eval_policy is not None else spec.policy

        def prompt(state, spec=spec, duties=duties, policy=policy):
            system = build_system_prompt(spec.role, duties, policy, mode, state.get("memory_context", ""))
            return [SystemMessage(content=system)] + list(state["messages"])

        agents[name] = create_react_agent(llm, tools, prompt=prompt, state_schema=AgentState, name=name)
    return agents
