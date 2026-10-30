"""由 Milvus 向量数据库支持的长期内存。

用户偏好和关键事实作为密集向量嵌入进行存储。
检索使用余弦相似度搜索，并按 user_id 进行过滤，因此每个用户的记忆保持隔离。
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

COLLECTION_NAME = "long_term_memory"


def _escape(value: str) -> str:
    """Milvus 过滤表达式的字符串转义：user_id 直接拼进 filter 的话，含引号时可改写过滤条件。"""
    return value.replace("\\", "\\\\").replace('"', '\\"')


EMBEDDING_DIM = 1536  


class LongTermMemory:
    """用于用户偏好和事实的基于 Milvus 的长期内存。

    功能：
    - 通过 Milvus 进行密集向量搜索（余弦相似度）
    - 对 ``user_id`` 进行标量过滤，实现每用户隔离
    - 偏好助手：``save_preference(user_id, type, value)``
    - 优雅降级：如果 Milvus 不可用，操作将变为空操作

    用法::

        mem = LongTermMemory(embedding_api_key="sk-...")
        await mem.initialize()

        await mem.save_preference("user1", "language", "Chinese")
        results = await mem.retrieve_relevant("user1", "preferred language")
        await mem.close()
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 19530,
        api_key: str | None = None,
        embedding_api_key: str | None = None,
    ) -> None:
        self._host = host
        self._port = port
        self._api_key = api_key
        self._embedding_api_key = embedding_api_key
        self._client: Any = None
        self._embeddings: Any = None
        self._available: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self, client: Any = None, embeddings: Any = None) -> None:
        """Connect to Milvus and ensure collection exists.

        Sets _available=False on failure (no exception raised).
        client / embeddings 可以注入（测试时用 Milvus Lite 和假向量）。
        """
        try:
            from pymilvus import MilvusClient  # type: ignore[import]
            from langchain_community.embeddings import DashScopeEmbeddings  # type: ignore[import]

            if client is None:
                connect_kwargs: dict[str, Any] = {"uri": f"http://{self._host}:{self._port}"}
                if self._api_key:
                    connect_kwargs["token"] = self._api_key
                client = MilvusClient(**connect_kwargs)
            self._client = client
            self._embeddings = embeddings or DashScopeEmbeddings(
                model="text-embedding-v2",
                dashscope_api_key=self._embedding_api_key,
            )
            self._ensure_collection()
            self._available = True
            logger.info("LongTermMemory: Milvus connected at %s:%s", self._host, self._port)
        except Exception as exc:
            logger.warning(
                "LongTermMemory: Milvus unavailable (%s) – long-term memory disabled.", exc
            )
            self._available = False

    async def close(self) -> None:
        """Close Milvus client."""
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def save_memory(
        self,
        user_id: str,
        content: str,
        memory_type: str = "general",
    ) -> None:
        """Embed and store a memory entry.

        Args:
            user_id: Owner of this memory.
            content: Text to embed and store.
            memory_type: Category label (e.g. "preference", "fact").
        """
        if not self._available:
            return
        try:
            embedding = await self._embeddings.aembed_query(content)
            self._client.insert(
                collection_name=COLLECTION_NAME,
                data=[
                    {
                        "user_id": user_id,
                        "content": content,
                        "memory_type": memory_type,
                        "embedding": embedding,
                    }
                ],
            )
            logger.debug(
                "LongTermMemory: stored %s memory for user %s: %s",
                memory_type, user_id, content[:60],
            )
        except Exception as exc:
            logger.error("LongTermMemory.save_memory failed: %s", exc)

    async def save_preference(
        self, user_id: str, preference_type: str, value: str
    ) -> None:
        """Convenience wrapper for storing a user preference.

        Args:
            user_id: Owner of this preference.
            preference_type: Short label (e.g. "language", "city").
            value: Preference value (e.g. "Chinese", "Beijing").
        """
        content = f"User preference – {preference_type}: {value}"
        await self.save_memory(user_id, content, memory_type="preference")

    async def retrieve_relevant(
        self, user_id: str, query: str, top_k: int = 5
    ) -> list[str]:
        """Return the top-k most relevant memory entries for a query.

        Args:
            user_id: Filter results to this user only.
            query: Natural-language query text.
            top_k: Maximum number of results to return.

        Returns:
            List of content strings ordered by relevance.
        """
        if not self._available:
            return []
        try:
            query_embedding = await self._embeddings.aembed_query(query)
            results = self._client.search(
                collection_name=COLLECTION_NAME,
                data=[query_embedding],
                filter=f'user_id == "{_escape(user_id)}"',
                limit=top_k,
                search_params={"metric_type": "COSINE"},
                # 强一致：Milvus 默认 Bounded 一致性，刚写入的数据可能暂时查不到（端到端测试时踩到：
                # 刚抽取的 3 条偏好只检索回 2 条，缺的正是最后写入的那条）。记忆数据量小，强一致开销可忽略
                consistency_level="Strong",
                output_fields=["content", "memory_type"],
            )
            memories: list[str] = []
            for hits in results:
                for hit in hits:
                    memories.append(hit["entity"]["content"])
            return memories
        except Exception as exc:
            logger.error("LongTermMemory.retrieve_relevant failed: %s", exc)
            return []

    async def list_memories(self, user_id: str, memory_type: str = "preference", limit: int = 100) -> list[str]:
        """列出某用户的全部记忆（去重和覆盖更新用）。只拿相似度前几条比对去重的话，重复条目会越积越多。"""
        if not self._available:
            return []
        try:
            rows = self._client.query(
                collection_name=COLLECTION_NAME,
                filter=f'user_id == "{_escape(user_id)}" and memory_type == "{_escape(memory_type)}"',
                output_fields=["content"], limit=limit,
                consistency_level="Strong",      # 去重/覆盖更新依赖它：必须看到刚删掉、刚写入的结果
            )
            return [r["content"] for r in rows]
        except Exception as exc:
            logger.error("LongTermMemory.list_memories failed: %s", exc)
            return []

    async def upsert_preference(self, user_id: str, category: str, content: str) -> None:
        """按类别覆盖更新：先删掉该用户同一类别的旧偏好，再写入新的。
        例：已有"座位偏好: 靠窗"，用户后来说喜欢靠过道 → 只保留"座位偏好: 靠过道"，不会两条矛盾的并存。"""
        if not self._available:
            return
        try:
            self._client.delete(collection_name=COLLECTION_NAME, filter=(
                f'user_id == "{_escape(user_id)}" and memory_type == "preference" '
                f'and content like "{_escape(category)}:%"'))
        except Exception as exc:
            logger.error("LongTermMemory.upsert_preference delete failed: %s", exc)
            return
        await self.save_memory(user_id, content, memory_type="preference")

    @property
    def available(self) -> bool:
        """True if Milvus is reachable."""
        return self._available

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ensure_collection(self) -> None:
        """Create the Milvus collection and index if they do not exist."""
        from pymilvus import DataType  # type: ignore[import]

        if self._client.has_collection(COLLECTION_NAME):
            return

        schema = self._client.create_schema()
        schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field("user_id", DataType.VARCHAR, max_length=128)
        schema.add_field("content", DataType.VARCHAR, max_length=2048)
        schema.add_field("memory_type", DataType.VARCHAR, max_length=64)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)

        index_params = self._client.prepare_index_params()
        index_params.add_index(
            "embedding",
            index_type="IVF_FLAT",
            metric_type="COSINE",
            params={"nlist": 128},
        )

        self._client.create_collection(
            collection_name=COLLECTION_NAME,
            schema=schema,
            index_params=index_params,
        )
        logger.info("LongTermMemory: created Milvus collection '%s'", COLLECTION_NAME)
