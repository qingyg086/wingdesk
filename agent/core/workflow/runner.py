"""
执行"一轮对话"的公共逻辑：FastAPI 的 chat_service、命令行 main.py、演示脚本和测试都用它。

一轮对话分两种情况：
1. 这个会话有一个被 interrupt 暂停、等待确认的写操作 → 把用户这句话当作确认/拒绝，用 Command(resume=...) 恢复；
2. 没有 → 正常把用户消息送进图（orchestrator 路由 → 专业 Agent）。

thread_id 用 "user_id:session_id"：只用 session_id 的话，两个用户恰好用了同一个 session_id（比如前端默认的
"default_session"）就会读到对方的对话、甚至恢复对方暂停的操作。

流式输出：stream_turn 用 graph.astream(stream_mode=["updates", "messages"], subgraphs=True)
边执行边产出事件，而不是等整张图跑完再一次性返回：
- updates 模式：每个节点执行完的状态更新。子图（专业 Agent）里的事件带命名空间，
  形如 ("booking_agent:<任务id>",)，据此知道"路由到了哪个 Agent"；Agent 的 AIMessage 带 tool_calls 时
  就知道"正在调用哪个工具"；
- messages 模式：大模型逐 token 的输出。只转发 langgraph_node == "agent"（专业 Agent 的模型节点）的文字，
  跳过路由器（orchestrator 用 function calling 输出结构化结果，没有给用户看的文字）和工具结果。
run_turn 只是把 stream_turn 的事件收集起来，所以 CLI、测试和网页走的是同一条代码路径。
"""

from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Optional

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langgraph.types import Command

from agents.hitl import parse_decision

RECURSION_LIMIT = 50


def make_config(user_id: str, session_id: str, callbacks: Optional[list] = None) -> dict:
    config = {"configurable": {"thread_id": f"{user_id}:{session_id}", "user_id": user_id},
              "recursion_limit": RECURSION_LIMIT}
    if callbacks:                               # 评测用：统计 token 用量
        config["callbacks"] = callbacks
    return config


@dataclass
class TurnResult:
    text: str                                   # 给用户看的回复
    route: list[str]                            # 本轮经过的节点，前端画"服务线路"用
    confirm: Optional[dict] = None              # 有值 = 本轮停在一个待确认的写操作上（interrupt 的 payload）
    resumed: bool = False                       # 本轮是否是在恢复一个待确认的操作
    tool_calls: list[dict] = field(default_factory=list)   # 本轮调用过的工具（日志 / 演示用）


async def pending_confirmation(graph, config) -> Optional[dict]:
    """返回该会话当前等待确认的操作（interrupt payload），没有则返回 None。"""
    snap = await graph.aget_state(config)
    for intr in getattr(snap, "interrupts", ()) or ():
        return intr.value
    for task in snap.tasks:                      # 兼容旧版本 LangGraph：中断信息挂在 tasks 上
        for intr in task.interrupts:
            return intr.value
    return None


def _text(content) -> str:
    return content if isinstance(content, str) else "".join(
        b.get("text", "") for b in content if isinstance(b, dict))


def confirm_prompt(payload: dict) -> str:
    return (f"请确认以下操作：\n{payload.get('summary_zh', '')}\n\n"
            "回复“确认”执行；如需调整，直接告诉我要改什么。")


async def stream_turn(graph, user_id: str, session_id: str, text: str, *,
                      confirm: Optional[bool] = None, memory_context: str = "",
                      callbacks: Optional[list] = None) -> AsyncIterator[dict]:
    """边执行边产出事件（dict），type 取值：
        node     {"key": 节点名}                       经过了一个新节点（orchestrator / confirm / 某个专业 Agent）
        tool     {"name": 工具名, "args": 参数}         Agent 决定调用一个工具
        token    {"text": 片段}                         模型输出的一小段文字（真流式）
        confirm  {"payload": interrupt 的内容}           停在一个待确认的写操作上
        done     {"text", "route", "confirm", "resumed", "tool_calls"}   本轮结束，附完整结果
    confirm：前端点了【确认】/【取消】按钮时直接传 True/False；为 None 时从 text 判断（"确认""好的"等）。"""
    config = make_config(user_id, session_id, callbacks)
    pending = await pending_confirmation(graph, config)
    route: list[str] = []
    calls: list[dict] = []
    parts: list[str] = []

    def visit(key: str):
        if key and (not route or route[-1] != key) and key not in route:
            route.append(key)
            return {"type": "node", "key": key}
        return None

    if pending:
        approved, _ = parse_decision(text) if confirm is None else (confirm, "")
        graph_input: Any = Command(resume={"approved": approved, "feedback": "" if approved else text})
        yield visit("confirm")
    else:
        graph_input = {"messages": [HumanMessage(content=text)], "user_id": user_id,
                       "session_id": session_id, "memory_context": memory_context}
        yield visit("orchestrator")

    async for ns, mode, data in graph.astream(graph_input, config, stream_mode=["updates", "messages"],
                                              subgraphs=True):
        agent = ns[0].split(":", 1)[0] if ns else None       # 子图命名空间 → 专业 Agent 名
        if agent and (ev := visit(agent)):
            yield ev
        if mode == "messages":
            msg, meta = data
            if meta.get("langgraph_node") != "agent" or not isinstance(msg, (AIMessage, AIMessageChunk)):
                continue
            if isinstance(msg, AIMessage) and not isinstance(msg, AIMessageChunk) and msg.tool_calls:
                continue
            piece = _text(msg.content)
            if piece:
                parts.append(piece)
                yield {"type": "token", "text": piece}
        elif mode == "updates" and agent:
            for update in data.values():
                for m in (update or {}).get("messages", []) if isinstance(update, dict) else []:
                    if isinstance(m, AIMessage) and m.tool_calls:
                        # 带工具调用的那条 AIMessage 如果夹带了文字，属于"思考过程"，不算最终回复
                        if parts and _text(m.content) and "".join(parts).endswith(_text(m.content)):
                            del parts[:]
                        for tc in m.tool_calls:
                            calls.append({"name": tc["name"], "args": tc["args"]})
                            yield {"type": "tool", "name": tc["name"], "args": tc["args"]}

    payload = await pending_confirmation(graph, config)
    if payload:
        prompt = confirm_prompt(payload)
        yield {"type": "confirm", "payload": payload}
        yield {"type": "token", "text": prompt}
        final = prompt
    else:
        final = "".join(parts)
    yield {"type": "done", "text": final, "route": route, "confirm": payload, "resumed": bool(pending),
           "tool_calls": calls}


async def run_turn(graph, user_id: str, session_id: str, text: str, *,
                   confirm: Optional[bool] = None, memory_context: str = "",
                   callbacks: Optional[list] = None) -> TurnResult:
    """非流式版本：收集 stream_turn 的事件，返回完整结果（CLI、测试、演示脚本、τ-bench 评测用）。"""
    async for ev in stream_turn(graph, user_id, session_id, text, confirm=confirm, memory_context=memory_context,
                                callbacks=callbacks):
        if ev and ev["type"] == "done":
            return TurnResult(ev["text"], ev["route"], confirm=ev["confirm"], resumed=ev["resumed"],
                              tool_calls=ev["tool_calls"])
    raise RuntimeError("stream_turn 没有产出 done 事件")


async def record_cached_turn(graph, user_id: str, session_id: str, question: str, answer: str) -> None:
    """语义缓存命中时，图没有运行，这一问一答不会进 checkpointer 的历史——下一轮 Agent 就不知道上一轮问过什么。
    这里把它补写进去。as_node="policy_agent"：假装这轮是政策咨询 Agent 回答的；
    它后面只连着 END，所以写完之后图不会有"待执行的下一步"。active_agent 保持不变，不影响粘性路由。"""
    await graph.aupdate_state(make_config(user_id, session_id),
                              {"messages": [HumanMessage(content=question), AIMessage(content=answer)]},
                              as_node="policy_agent")
