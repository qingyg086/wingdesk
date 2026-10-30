"""由 Redis 支持的短期对话记忆。

为什么不把一个会话存成一个 String（整段 JSON），追加时 GET → 在 Python 里 append → SET：
这是"读-改-写"，不是原子操作。同一会话两个请求并发时：
    A 读到 [m1]        B 读到 [m1]
    A 写回 [m1, a]     B 写回 [m1, b]   ← A 写的 a 丢了（丢失更新 / lost update）

做法：一个会话存成一个 Redis List，每个元素是一条消息的 JSON。追加 = 一个事务里执行三条命令：
    RPUSH key m1 m2 ...      追加到末尾（Redis 单线程执行命令，RPUSH 本身就是原子的）
    LTRIM key -N -1          只保留最近 N 条
    EXPIRE key ttl           刷新过期时间
三条命令用 pipeline(transaction=True) 包成 MULTI/EXEC，要么全部执行，要么都不执行，
也不会被别的客户端的命令插进中间。读取用 LRANGE key 0 -1。

现在多轮对话的上下文由 LangGraph 的 checkpointer 负责；这里的短期记忆主要给长期记忆的偏好抽取用。
"""

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

MAX_MESSAGES = 20           # 每个会话最多保留最近 20 条（约 10 轮）
DEFAULT_TTL = 1800          # 30 分钟无活动后过期
MAX_CONNECTIONS = 50        # Redis 连接池上限


class ShortTermMemory:
    def __init__(self, redis_url: str = "redis://localhost:6379", ttl: int = DEFAULT_TTL,
                 max_messages: int = MAX_MESSAGES) -> None:
        self._redis_url = redis_url
        self._ttl = ttl
        self._max = max_messages
        self._client: Any = None
        self._available: bool = False

    # ------------------------------------------------------------------ 生命周期
    async def initialize(self) -> None:
        """连接 Redis；失败时把 available 置为 False（不抛异常，降级为不记录）。"""
        try:
            import redis.asyncio as aioredis

            # BlockingConnectionPool：连接用满时排队等待（最多 5 秒），而不是直接抛 "Too many connections"。
            # 默认的 ConnectionPool 在并发超过 max_connections 时会立刻报错（并发压测时踩到）。
            pool = aioredis.BlockingConnectionPool.from_url(
                self._redis_url, max_connections=MAX_CONNECTIONS, timeout=5, decode_responses=True,
                socket_connect_timeout=2, socket_timeout=2, health_check_interval=30, retry_on_timeout=True,
            )
            self._client = aioredis.Redis(connection_pool=pool)
            await self._client.ping()
            self._available = True
            logger.info("ShortTermMemory: Redis connected at %s", self._redis_url)
        except Exception as exc:
            logger.warning("ShortTermMemory: Redis unavailable (%s) – short-term memory disabled.", exc)
            self._available = False

    async def close(self) -> None:
        if self._client is not None:
            try:
                await self._client.aclose()
                await self._client.connection_pool.disconnect()
            except Exception:
                pass

    # ------------------------------------------------------------------ 读写
    async def get_messages(self, user_id: str, session_id: str) -> list[dict[str, Any]]:
        if not self._available:
            return []
        key = self._key(user_id, session_id)
        try:
            return [json.loads(x) for x in await self._client.lrange(key, 0, -1)]
        except Exception as exc:
            if "WRONGTYPE" in str(exc):           # key 被其他类型（如 String）占用
                await self._client.delete(key)
                return []
            logger.warning("ShortTermMemory.get_messages failed: %s", exc)
            return []

    async def append_messages(self, user_id: str, session_id: str, messages: list[dict[str, Any]]) -> None:
        """原子地追加若干条消息，并裁剪、刷新过期时间。"""
        if not self._available or not messages:
            return
        key = self._key(user_id, session_id)
        items = [json.dumps(m, ensure_ascii=False) for m in messages]
        for attempt in range(2):
            try:
                async with self._client.pipeline(transaction=True) as pipe:     # MULTI ... EXEC
                    pipe.rpush(key, *items)
                    pipe.ltrim(key, -self._max, -1)
                    pipe.expire(key, self._ttl)
                    await pipe.execute()
                return
            except Exception as exc:
                if attempt == 0 and "WRONGTYPE" in str(exc):
                    # key 被其他类型（如 String）占用时不能 RPUSH，删掉后重试一次
                    await self._client.delete(key)
                    continue
                logger.warning("ShortTermMemory.append_messages failed: %s", exc)
                return

    async def append_message(self, user_id: str, session_id: str, role: str, content: str) -> None:
        await self.append_messages(user_id, session_id, [{"role": role, "content": content}])

    async def incr_turns(self, user_id: str, session_id: str) -> int:
        """会话轮数计数器 +1 并返回新值（INCR 是原子的）。用来决定每隔几轮触发一次偏好抽取。
        为什么不用消息条数算：List 会被 LTRIM 裁剪到固定长度，条数到上限后就不再变化了。"""
        if not self._available:
            return 0
        key = self._key(user_id, session_id) + ":turns"
        try:
            async with self._client.pipeline(transaction=True) as pipe:
                pipe.incr(key)
                pipe.expire(key, self._ttl)
                n, _ = await pipe.execute()
            return int(n)
        except Exception as exc:
            logger.warning("ShortTermMemory.incr_turns failed: %s", exc)
            return 0

    async def clear(self, user_id: str, session_id: str) -> None:
        if not self._available:
            return
        try:
            await self._client.delete(self._key(user_id, session_id), self._key(user_id, session_id) + ":turns")
        except Exception as exc:
            logger.error("ShortTermMemory.clear failed: %s", exc)

    @property
    def available(self) -> bool:
        return self._available

    @staticmethod
    def _key(user_id: str, session_id: str) -> str:
        return f"memory:short:{user_id}:{session_id}"
