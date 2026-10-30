"""
多 Agent 编排图。

    START → orchestrator ──(条件边，按 next_agent)──→ flight_search_agent ─┐
                                                   → booking_agent       │
                                                   → reservation_agent   ├→ END
                                                   → cancel_refund_agent │
                                                   → policy_agent        │
                                                   → human_handoff ──────┘

- 每个专业 Agent 是一个 create_react_agent 子图，直接作为节点挂进来，和父图共享 AgentState。
- checkpointer：编译时传入（默认 MemorySaver，进程内存）。调用时 config 里的 thread_id = session_id，
  同一个会话的消息、active_agent、以及被 interrupt 暂停时的执行现场，都按 thread_id 保存。
  - 多轮对话不需要把历史拼进提示词，每轮只传新的一句话，checkpointer 会把历史接上；
  - interrupt 依赖 checkpointer：暂停时要把"执行到哪了"存下来，恢复时才能接着跑。
- 生产环境可把 MemorySaver 换成 Redis / Postgres checkpointer，服务重启后会话不丢，多实例部署也能共享。
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
        llm / router_llm：不传时用 qwen-plus；测试时可传入假模型。
        mode："product"（写工具需人工确认）或 "eval"（τ-bench 评测，确认在对话里完成）。
        """
        if tools is None:
            from core.mcp.mcp_manager import mcp_registry
            tools = {t.name: t for t in mcp_registry.get_tools()}
        if llm is None:
            from core.llm import get_llm
            llm = get_llm(0.0)
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
