"""
多 Agent 编排图。

    START → orchestrator ──(条件边，按 next_agent)──→ flight_search_agent ─┐
                                                   → booking_agent       │
                                                   → reservation_agent   ├→ END
                                                   → cancel_refund_agent │
                                                   → policy_agent        │
                                                   → human_handoff ──────┘

- 每个专业 Agent 是一个 create_react_agent 子图，直接作为节点挂进来，和父图共享 AgentState。
- checkpointer：编译时传入（Web 服务传 SQLite saver；测试/评测默认 MemorySaver）。
  调用时 config 里的 thread_id = user_id:session_id，
  同一个会话的消息、active_agent、以及被 interrupt 暂停时的执行现场，都按 thread_id 保存。
  - 多轮对话不需要把历史拼进提示词，每轮只传新的一句话，checkpointer 会把历史接上；
  - interrupt 依赖 checkpointer：暂停时要把"执行到哪了"存下来，恢复时才能接着跑。
- 单机 Web 服务用 SQLite 跨重启保存状态；多实例部署需要共享的 checkpointer 和跨实例会话锁。
"""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from agents.orchestrator import TARGETS, OrchestratorAgent
from agents.specialists import build_specialists
from core.workflow.state import AgentState


class AgentGraphManager:
    def __init__(self, tools: dict | None = None, llm=None, router_llm=None, mode: str = "product",
                 checkpointer=None, local_tools: dict | None = None):
        """
        tools：{工具名: 工具}。不传时从全局 MCPToolRegistry 取（需要先 await mcp_registry.start()）。
        llm / router_llm：产品模式默认 qwen-plus 专员 + qwen-flash 路由；
        测试或评测时可传入假模型，也可显式指定 router_llm。
        mode："product"（写工具需人工确认）或 "eval"（τ-bench 评测，确认在对话里完成）。
        """
        if tools is None:
            from core.mcp.mcp_manager import mcp_registry
            tools = {t.name: t for t in mcp_registry.get_tools()}
        if llm is None:
            from core.llm import get_llm, get_router_llm
            llm = get_llm(0.0)
            if router_llm is None and mode == "product":
                router_llm = get_router_llm()
        self.orchestrator = OrchestratorAgent(router_llm or llm)
        self.agents = build_specialists(tools, llm, mode=mode, local_tools=local_tools)
        self.checkpointer = checkpointer if checkpointer is not None else MemorySaver()

    @staticmethod
    def _route(state: AgentState) -> str:
        return state.get("next_agent") or "policy_agent"

    def build_graph(self):
        builder = StateGraph(AgentState)
        builder.add_node("orchestrator", self.orchestrator.route)
        for name, agent in self.agents.items():
            builder.add_node(name, agent)
            builder.add_edge(name, END)
        builder.add_edge(START, "orchestrator")
        builder.add_conditional_edges("orchestrator", self._route, {t: t for t in TARGETS})
        return builder.compile(checkpointer=self.checkpointer)
