"""统一的记忆管理器：短期记忆（Redis）+ 长期记忆（Milvus）+ 偏好抽取。

三层记忆各管一件事：
- 对话上下文（这一轮之前说过什么）：LangGraph checkpointer，不在这里；
- 短期记忆（Redis List）：记录最近的问答文本，作为偏好抽取的原材料，30 分钟无活动过期；
- 长期记忆（Milvus）：抽取出来的用户偏好，跨会话保留，下次对话时检索出来放进提示词。

写入链路：
    每轮对话 → save_conversation 追加到 Redis，并给轮数计数器 +1
    每 EXTRACT_EVERY 轮 → chat_service 用 asyncio.create_task 在后台调用 background_extract
        → 只取用户的发言交给大模型抽取偏好 → 按类别覆盖写入 Milvus
读取链路：新一轮对话开始时，chat_service 用用户的问题去 Milvus 检索最相关的偏好，放进 memory_context。
"""

import asyncio
import logging
from typing import Any

from .long_term import LongTermMemory
from .preference_extractor import PreferenceExtractor
from .short_term import ShortTermMemory

logger = logging.getLogger(__name__)

EXTRACT_EVERY = 3              # 每 3 轮对话在后台抽取一次偏好
_MAX_HISTORY_MESSAGES = 20


class MemoryManager:
    def __init__(self, redis_url: str = "redis://localhost:6379", redis_ttl: int = 1800,
                 milvus_host: str = "localhost", milvus_port: int = 19530,
                 milvus_api_key: str | None = None, embedding_api_key: str | None = None) -> None:
        self.short_term = ShortTermMemory(redis_url=redis_url, ttl=redis_ttl)
        self.long_term = LongTermMemory(host=milvus_host, port=milvus_port, api_key=milvus_api_key,
                                        embedding_api_key=embedding_api_key)

    async def initialize(self) -> None:
        await asyncio.gather(self.short_term.initialize(), self.long_term.initialize(), return_exceptions=True)
        logger.info("MemoryManager ready – short_term=%s, long_term=%s",
                    "✓" if self.short_term.available else "✗", "✓" if self.long_term.available else "✗")

    async def close(self) -> None:
        await asyncio.gather(self.short_term.close(), self.long_term.close(), return_exceptions=True)

    # ------------------------------------------------------------------ 每轮
    async def save_conversation(self, user_id: str, session_id: str, messages: list[dict[str, Any]]) -> int:
        """原子追加本轮消息，返回这个会话累计的轮数（用于决定什么时候触发偏好抽取）。"""
        msgs = [m for m in messages if m.get("role") != "system"]
        await self.short_term.append_messages(user_id, session_id, msgs)
        return await self.short_term.incr_turns(user_id, session_id)

    @staticmethod
    def should_extract(turns: int) -> bool:
        return turns > 0 and turns % EXTRACT_EVERY == 0

    async def get_recent_messages(self, user_id: str, session_id: str) -> list[dict[str, Any]]:
        return await self.short_term.get_messages(user_id, session_id)

    # ------------------------------------------------------------------ 长期偏好
    async def load_preferences(self, user_id: str, query: str = "用户的出行偏好和习惯", top_k: int = 5) -> list[str]:
        """按问题检索最相关的偏好（读取链路用）。"""
        if not self.long_term.available:
            return []
        return await self.long_term.retrieve_relevant(user_id=user_id, query=query, top_k=top_k)

    async def save_preference(self, user_id: str, category: str, value: str) -> None:
        await self.long_term.upsert_preference(user_id, category, f"{category}: {value}")

    async def background_extract(self, user_id: str, session_id: str, llm: Any) -> list[str]:
        """从这个会话最近的对话里抽取偏好，按类别覆盖写入长期记忆。不清空 Redis，会话可以继续。
        设计成后台任务调用：失败只记日志，绝不影响用户这一轮的回复。"""
        if not (self.long_term.available and user_id and session_id):
            return []
        try:
            messages = (await self.short_term.get_messages(user_id, session_id))[-_MAX_HISTORY_MESSAGES:]
            user_lines = [f"用户：{m['content']}" for m in messages if m.get("role") == "user" and m.get("content")]
            if not user_lines:
                return []
            existing = await self.long_term.list_memories(user_id)         # 拿全部已有偏好去重
            new_items = await PreferenceExtractor(llm=llm).extract("\n".join(user_lines), existing=existing)
            for item in new_items:
                await self.long_term.upsert_preference(user_id, PreferenceExtractor.category_of(item), item)
            if new_items:
                logger.info("[MEMORY] 用户 %s 新增/更新偏好 %d 条：%s", user_id, len(new_items), new_items)
            return new_items
        except Exception as exc:
            logger.warning("[MEMORY] background_extract failed for %s: %s", user_id, exc)
            return []

    async def finalize_session(self, user_id: str, session_id: str, llm: Any) -> list[str]:
        """会话结束：最后抽取一次偏好，然后清空这个会话的短期记忆。"""
        new_items = await self.background_extract(user_id, session_id, llm)
        await self.short_term.clear(user_id, session_id)
        return new_items
