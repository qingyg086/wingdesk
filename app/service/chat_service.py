"""
聊天服务。

1. 多轮对话：由 LangGraph 的 checkpointer 按 thread_id 保存完整消息（包括工具调用和结果），每轮只传新消息。
2. 人工确认：Agent 调用写工具时图会暂停（interrupt），这里通过 SSE 推一个 confirm 事件给前端；
   下一次请求带 confirm=true/false（前端按钮）或用户直接回复"确认"，就用 Command(resume=...) 恢复。
3. 有待确认的操作时跳过语义缓存：否则用户回复"确认"可能被缓存里某条相似问答截走。
4. Redis 短期记忆仍然保存每轮对话，作为长期记忆（偏好抽取）的原材料；提示词里只放长期记忆。
5. 语义缓存命中时，把这一问一答补写进 checkpointer，下一轮 Agent 才知道上一轮聊过什么；
6. 接通长期记忆：每 EXTRACT_EVERY 轮用 asyncio.create_task 在后台抽取偏好，不阻塞本轮回复。
7. 真流式：边跑边推，路由到哪个 Agent、调了哪个工具、模型吐出的每个 token 都立刻推出去。
   而不是先把整张图跑完、再把完整回复切片推送（那样用户要等全部算完才看到第一个字）。

SSE 事件（每条一行 data: {json}）：
    {"step": {"kind": "node", "key": "booking_agent"}}       经过一个节点（前端画服务线路）
    {"step": {"kind": "tool", "name": "search_direct_flight", "args": {...}}}   Agent 调用工具
    {"content": "片段"}                                        回复文字，逐 token
    {"confirm": {...}}                                         待确认的写操作（前端显示【确认】【取消】卡片）
    {"error": "..."}                                           出错（之后仍会发 meta 和 done）
    {"meta": {"cache", "route", "timing": {"first_token_ms", "total_ms"}}}   本轮汇总，放在最后
    {"done": true}

图在后台任务（_run_graph）里跑，通过 asyncio.Queue 把事件交给 SSE 生成器：
用户中途关闭页面 → SSE 生成器被取消，但后台任务继续把这一轮跑完、写进 checkpointer 和短期记忆，
不会留下"模型发起了工具调用、却没有工具结果"的半截历史（那样下一轮调用模型会直接报错）。

同一会话（user_id:session_id）同一时间只处理一个请求：两个请求同时读写同一份 checkpointer 状态，
可能把同一个确认操作执行两次，或者在图暂停时又插进一条新消息。会话正忙时，新请求直接返回提示，
不排队——重复点击、重复提交排队执行的话，很可能把同一件事办两次。
锁在拿到之后交给后台任务，这一轮真正跑完才释放（浏览器断开不会提前释放）。
这是进程内的锁；多实例部署时要换成分布式锁（例如 Redis SET NX PX）。
"""

import asyncio
import json
import logging
import os
import sys
import time
import weakref

AGENT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "agent")
if AGENT_DIR not in sys.path:
    sys.path.insert(0, AGENT_DIR)

from core.workflow.graph_manager import AgentGraphManager          # noqa: E402
from core.workflow.runner import make_config, pending_confirmation, record_cached_turn, stream_turn   # noqa: E402
from core.memory.memory_manager import MemoryManager               # noqa: E402
from infra.cache import semantic_cache                              # noqa: E402

logger = logging.getLogger(__name__)

graph = None
memory = None
_background_tasks: set = set()     # 持有后台任务的引用：asyncio 只保留弱引用，不持有的话任务可能被垃圾回收掉
# 会话锁：弱引用字典，没有请求持有某个会话的锁时自动回收，不会随会话数无限增长
_session_locks: "weakref.WeakValueDictionary[str, asyncio.Lock]" = weakref.WeakValueDictionary()
BUSY_MESSAGE = "上一条消息还在处理中，请等它完成后再发送。"


def _session_lock(user_id: str, session_id: str) -> asyncio.Lock:
    key = f"{user_id}:{session_id}"
    lock = _session_locks.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _session_locks[key] = lock
    return lock


async def init_agent_system():
    """在 FastAPI lifespan 里、mcp_registry.start() 之后调用（建图时要从注册表拿 MCP 工具）。"""
    global graph, memory
    if graph is None:
        print("初始化 Multi-Agent 图编排（orchestrator + 5 个专业 Agent + 转人工）...")
        graph = AgentGraphManager().build_graph()

        print("初始化 Memory 系统...")
        from config import get_settings
        settings = get_settings()
        memory = MemoryManager(
            redis_url=settings.redis_url,
            redis_ttl=settings.redis_ttl,
            milvus_host=settings.milvus_host,
            milvus_port=settings.milvus_port,
            milvus_api_key=settings.milvus_api_key,
            embedding_api_key=settings.dashscope_api_key,
        )
        await memory.initialize()
        await semantic_cache.initialize()
        print("[OK] Agent 系统初始化完成！")


async def _long_term_context(user_id: str, query: str) -> str:
    """只取长期记忆（用户偏好）。多轮对话历史已经由 checkpointer 保存，不再拼进提示词。"""
    if memory and memory.long_term.available:
        prefs = await memory.load_preferences(user_id, query)
        if prefs:
            return "\n".join(f"- {p}" for p in prefs)
    return ""


def _spawn(coro) -> asyncio.Task:
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return task


def _schedule_extraction(user_id: str, session_id: str) -> None:
    """后台抽取偏好：fire-and-forget，失败只记日志，不影响用户。"""
    from core.llm import get_llm

    _spawn(memory.background_extract(user_id, session_id, get_llm(0.0)))


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


async def _save_turn(user_id: str, session_id: str, query: str, answer: str) -> None:
    if memory and memory.short_term.available and answer:
        turns = await memory.save_conversation(user_id, session_id, [
            {"role": "user", "content": query},
            {"role": "assistant", "content": answer},
        ])
        if memory.should_extract(turns):
            _schedule_extraction(user_id, session_id)


async def _run_graph(queue: asyncio.Queue, user_id: str, session_id: str, query: str,
                     confirm: bool | None, mem_context: str, lock: asyncio.Lock) -> None:
    """生产者：跑一轮图，把事件放进队列；结束时保存短期记忆，释放会话锁，最后放一个 None 表示结束。"""
    answer = ""
    try:
        async for ev in stream_turn(graph, user_id, session_id, query, confirm=confirm, memory_context=mem_context):
            if ev:
                if ev["type"] == "done":
                    answer = ev["text"]
                await queue.put(ev)
    except Exception as exc:                 # 不让异常把 SSE 连接直接断掉：告诉前端出错了，再正常收尾
        logger.exception("stream_turn failed")
        await queue.put({"type": "error", "message": f"服务暂时出错（{type(exc).__name__}），请稍后再试。"})
    finally:
        try:
            await _save_turn(user_id, session_id, query, answer)
        except Exception:
            logger.exception("save_conversation failed")
        lock.release()
        await queue.put(None)


async def stream_chat(query: str, user_id: str, session_id: str, confirm: bool | None = None):
    t0 = time.perf_counter()

    def ms() -> int:
        return round((time.perf_counter() - t0) * 1000)

    lock = _session_lock(user_id, session_id)
    if lock.locked():
        logger.info("session busy, rejected: %s:%s", user_id, session_id)
        yield _sse({"error": BUSY_MESSAGE})
        yield _sse({"meta": {"cache": None, "route": [], "busy": True,
                             "timing": {"first_token_ms": None, "total_ms": ms()}}})
        yield _sse({"done": True})
        return
    await lock.acquire()             # 上面刚检查过没被占用，这里不会等待
    handed_off = False               # 锁是否已交给后台任务（之后由后台任务释放）
    turn = _serve_turn(query, user_id, session_id, confirm, lock, t0)
    try:
        async for chunk in turn:
            if chunk is _HANDED_OFF:
                handed_off = True
            else:
                yield chunk
    finally:
        await turn.aclose()          # 连接断开时也立刻关闭内层生成器，不等垃圾回收
        if not handed_off:
            lock.release()


_HANDED_OFF = object()


async def _serve_turn(query: str, user_id: str, session_id: str, confirm: bool | None,
                      lock: asyncio.Lock, t0: float):
    """持有会话锁时执行的一轮对话。产出 SSE 字符串；把锁交给后台任务时产出 _HANDED_OFF 标记。"""
    first_token_ms = None

    def ms() -> int:
        return round((time.perf_counter() - t0) * 1000)

    pending = await pending_confirmation(graph, make_config(user_id, session_id))
    cache_hit = None if pending else await semantic_cache.get_cache(query, user_id)

    if cache_hit:
        # 缓存命中：答案现成，一次性推出去
        answer = cache_hit["answer"]
        meta = {"cache": cache_hit["level"], "route": ["semantic_cache"],
                "similarity": round(cache_hit["similarity"], 3)}
        print(f"语义缓存命中: 相似度={cache_hit['similarity']:.3f} matched='{cache_hit['matched_question']}'")
        yield _sse({"step": {"kind": "node", "key": "semantic_cache"}})
        first_token_ms = ms()
        yield _sse({"content": answer})
        await record_cached_turn(graph, user_id, session_id, query, answer)
        await _save_turn(user_id, session_id, query, answer)
    else:
        print("恢复待确认的操作..." if pending else "进入 Agent 工作流推理...")
        mem_context = "" if pending else await _long_term_context(user_id, query)
        queue: asyncio.Queue = asyncio.Queue()
        _spawn(_run_graph(queue, user_id, session_id, query, confirm, mem_context, lock))
        yield _HANDED_OFF
        meta = {"cache": None, "route": []}
        # 图在后台任务里跑，这里只负责把事件转成 SSE。用户中途关掉页面时，这个生成器会被取消，
        # 但后台任务会继续把这一轮跑完并保存——不会留下"调了工具却没有结果"的半截状态。
        while (ev := await queue.get()) is not None:
            kind = ev["type"]
            if kind == "node":
                yield _sse({"step": {"kind": "node", "key": ev["key"]}})
            elif kind == "tool":
                yield _sse({"step": {"kind": "tool", "name": ev["name"], "args": ev["args"]}})
            elif kind == "token":
                if first_token_ms is None:
                    first_token_ms = ms()
                yield _sse({"content": ev["text"]})
            elif kind == "confirm":
                yield _sse({"confirm": ev["payload"]})
            elif kind == "error":
                yield _sse({"error": ev["message"]})
            elif kind == "done":
                meta["route"] = ev["route"]

    meta["timing"] = {"first_token_ms": first_token_ms, "total_ms": ms()}
    print(f"耗时:  首字 {first_token_ms} ms，总耗时 {meta['timing']['total_ms']} ms，路线 {meta['route']}")
    yield _sse({"meta": meta})
    yield _sse({"done": True})
