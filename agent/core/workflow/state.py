"""
LangGraph 全局状态。

- messages 的 reducer 用 add_messages 而不是 operator.add：
  operator.add 只会无脑拼接；add_messages 按消息 id 去重/覆盖。有了 checkpointer 之后，
  消息会跨轮保存，子 Agent（create_react_agent 子图）返回的是"完整消息列表"，
  用 operator.add 会把历史消息重复追加一遍。
- active_agent：当前正在处理任务的专业 Agent，用于"粘性路由"（多轮对话中同一任务继续交给它）。
- remaining_steps：create_react_agent 要求的托管字段，防止工具调用死循环。
"""

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from langgraph.managed import RemainingSteps


class AgentState(TypedDict, total=False):
    # 对话消息（含工具调用和工具结果），由 checkpointer 按 thread_id=session_id 跨轮保存
    messages: Annotated[list[AnyMessage], add_messages]

    # 登录用户和会话。user_id 只用于展示和记忆；工具里的身份以 config["configurable"]["user_id"] 为准
    user_id: str
    session_id: str

    # 长期记忆（用户偏好等），由 chat_service 每轮写入；多轮对话历史不再拼在这里，而是由 checkpointer 保存
    memory_context: str

    # 路由
    next_agent: str          # 本轮 orchestrator 选中的 Agent
    active_agent: str        # 上一轮处理任务的 Agent（粘性路由用），跨轮保留
    route_reason: str        # 路由理由（日志 / 前端展示用）

    # create_react_agent 需要的托管字段：剩余可执行步数
    remaining_steps: RemainingSteps

    metadata: dict[str, Any]


class AgentOutput(TypedDict):
    """Agent 执行的标准输出格式（保留给 CLI 等调用方使用）。"""
    response: str
    tool_calls: list[dict[str, Any]]
    metadata: dict[str, Any]
