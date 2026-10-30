"""
L1 语义缓存：高频的政策类问题直接返回预置答案，不走 Agent（省一次多轮大模型调用）。

两层匹配：
1. 精确匹配：归一化（去首尾空格、转小写、合并空白）后的问题文本完全相同；
2. 语义匹配：问题向量和缓存里的问题向量做相似度检索，相似度 >= 阈值才算命中。

注意：Milvus 用 COSINE 度量时，返回结果里叫 "distance" 的字段**其实是余弦相似度**（越大越像，最大 1.0），
不是距离（越小越像）。如果按距离来比较（distance 小于阈值才命中），语义匹配会永远不命中。
所以这里是 `similarity >= 阈值` 才命中。阈值由实验确定（agent/scripts/cache_threshold_experiment.py，
结果在 eval/results/20261011_cache_threshold.json），可用环境变量覆盖。

安全原则：缓存里只放"答案对所有人都一样"的政策问答；任何依赖个人数据、或需要办理业务的问题都不应命中。
所以阈值要宁高勿低——误命中（把"帮我退票"当成"退票规则是什么"回答）比漏命中（多花一次大模型调用）代价大得多。
"""

from __future__ import annotations

import os
import re
from typing import Any

COLLECTION_NAME = "qa_semantic_cache"
# 默认配置来自阈值实验（agent/scripts/cache_threshold_experiment.py）：
#   v2 / v3 / v4 三个模型在"多问法入库 + 规则过滤"下召回率都是 90%、零误命中；
#   v2 命中的正例离阈值最远（v3 有 8 条正例挤在阈值上方 0.05 以内，最脆弱），且与 RAG、长期记忆用同一个模型。
#   相似度刻度因模型而异：换模型必须重做实验、重新定阈值，并 preload_cache.py --reset。
EMBEDDING_MODEL = os.getenv("SEMANTIC_CACHE_MODEL", "text-embedding-v2")
EMBEDDING_DIMS = {"text-embedding-v2": 1536, "text-embedding-v3": 1024, "text-embedding-v4": 1024}
# 相似度阈值（COSINE，越大越严格）：零误命中的最低阈值 0.64，再留 0.02 安全余量
SIMILARITY_THRESHOLD = float(os.getenv("SEMANTIC_CACHE_THRESHOLD", "0.66"))

# 规则前置过滤：缓存只回答"对所有人都一样"的通用政策问题。下面这些说明用户在问自己的数据、或者要办业务，
# 一律不查缓存、直接交给 Agent：
#   "我的…""帮我…""给我…""替我…"   → 查个人数据 / 办理业务
#   6 位大写字母数字（如 AIXC49）      → 带了订单号
#   归一化后少于 4 个字                 → "好的""确认""第一个"这类对话中的短句，脱离上下文没有意义
_PERSONAL = re.compile(r"我的|帮我|给我|替我|为我")
_ORDER_ID = re.compile(r"(?<![a-z0-9])(?=[a-z0-9]*\d)(?=[a-z0-9]*[a-z])[a-z0-9]{6}(?![a-z0-9])")
MIN_CHARS = 4


def normalize_query(text: str) -> str:
    """缓存和阈值实验共用的文本归一化：去首尾空格、转小写、合并连续空白。"""
    return " ".join(text.strip().lower().split())


def is_cacheable(text: str) -> bool:
    """是否允许查缓存（规则前置过滤）。"""
    t = normalize_query(text)
    return len(t) >= MIN_CHARS and not _PERSONAL.search(t) and not _ORDER_ID.search(t)


def _escape(value: str) -> str:
    """Milvus 过滤表达式里的字符串字面量转义，防止用户输入里的引号改写过滤条件（表达式注入）。"""
    return value.replace("\\", "\\\\").replace('"', '\\"')


class SemanticCache:
    def __init__(self, threshold: float = SIMILARITY_THRESHOLD, model: str = EMBEDDING_MODEL) -> None:
        self.threshold = threshold
        self.model = model
        self.dim = EMBEDDING_DIMS.get(model, 1536)
        self._client: Any = None
        self._embeddings: Any = None
        self._available: bool = False

    async def initialize(self, client: Any = None, embeddings: Any = None) -> None:
        """client / embeddings 可以注入（测试时用 Milvus Lite 和假向量）；不传时连真实的 Milvus 和百炼。"""
        try:
            if client is None:
                from pymilvus import MilvusClient
                from app_config.settings import settings

                kwargs: dict[str, Any] = {"uri": f"http://{settings.milvus_host}:{settings.milvus_port}"}
                if settings.milvus_api_key:
                    kwargs["token"] = settings.milvus_api_key
                client = MilvusClient(**kwargs)
            if embeddings is None:
                from langchain_community.embeddings import DashScopeEmbeddings
                from app_config.settings import settings

                embeddings = DashScopeEmbeddings(model=self.model, dashscope_api_key=settings.dashscope_api_key)
            self._client, self._embeddings = client, embeddings
            self._ensure_collection()
            self._available = True
        except Exception as exc:
            print(f"SemanticCache init failed: {exc}")
            self._available = False

    # ------------------------------------------------------------------ 写
    async def set_cache(self, query: str, response: str, user_id: str | None = None,
                        scope: str = "public") -> None:
        if not self._available:
            return
        normalized = self._normalize(query)
        owner = user_id or ""
        cache_scope = "user" if owner else scope
        try:
            embedding = await self._embeddings.aembed_query(normalized)
            # 同一个问题重复写入时先删旧的，保证一个问题只有一条（预热脚本可以重复执行）
            self._client.delete(collection_name=COLLECTION_NAME, filter=(
                f'question_norm == "{_escape(normalized)}" and scope == "{_escape(cache_scope)}" '
                f'and user_id == "{_escape(owner)}"'))
            self._client.insert(collection_name=COLLECTION_NAME, data=[{
                "question": query.strip(), "question_norm": normalized, "answer": response,
                "scope": cache_scope, "user_id": owner, "enabled": 1, "embedding": embedding,
            }])
        except Exception as exc:
            print(f"SemanticCache set_cache failed: {exc}")

    # ------------------------------------------------------------------ 读
    async def get_cache(self, query: str, user_id: str) -> dict[str, Any] | None:
        if not self._available:
            return None
        normalized = self._normalize(query)
        if not is_cacheable(normalized):
            return None
        n, u = _escape(normalized), _escape(user_id)

        # 第 1 层：精确匹配（先查用户私有，再查公共）
        for flt in (f'enabled == 1 and question_norm == "{n}" and scope == "user" and user_id == "{u}"',
                    f'enabled == 1 and question_norm == "{n}" and scope == "public"'):
            row = self._query_one(flt)
            if row:
                return {"answer": row["answer"], "matched_question": row["question"],
                        "level": "L1_EXACT", "similarity": 1.0}

        # 第 2 层：语义匹配
        try:
            vec = await self._embeddings.aembed_query(normalized)
            results = self._client.search(
                collection_name=COLLECTION_NAME, data=[vec], limit=1,
                filter=f'enabled == 1 and (scope == "public" or (scope == "user" and user_id == "{u}"))',
                search_params={"metric_type": "COSINE"},
                output_fields=["question", "answer"],
            )
            hit = results[0][0] if results and results[0] else None
            if not hit:
                return None
            similarity = float(hit["distance"])        # COSINE 度量下，这个字段是相似度，越大越像
            if similarity < self.threshold:
                return None
            entity = hit.get("entity", {})
            return {"answer": entity.get("answer", ""), "matched_question": entity.get("question", ""),
                    "level": "L1_SEMANTIC", "similarity": similarity}
        except Exception as exc:
            print(f"SemanticCache get_cache failed: {exc}")
            return None

    @property
    def available(self) -> bool:
        return self._available

    @staticmethod
    def _normalize(text: str) -> str:
        return normalize_query(text)

    def _query_one(self, filter_expr: str) -> dict[str, Any] | None:
        try:
            rows = self._client.query(collection_name=COLLECTION_NAME, filter=filter_expr,
                                      output_fields=["question", "answer"], limit=1)
            return rows[0] if rows else None
        except Exception:
            return None

    def reset(self) -> None:
        """删除并重建集合（预热脚本 --reset 用）。"""
        if self._client.has_collection(COLLECTION_NAME):
            self._client.drop_collection(COLLECTION_NAME)
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        from pymilvus import DataType

        if self._client.has_collection(COLLECTION_NAME):
            return
        schema = self._client.create_schema()
        schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field("question", DataType.VARCHAR, max_length=2048)
        schema.add_field("question_norm", DataType.VARCHAR, max_length=2048)
        schema.add_field("answer", DataType.VARCHAR, max_length=8192)
        schema.add_field("scope", DataType.VARCHAR, max_length=16)
        schema.add_field("user_id", DataType.VARCHAR, max_length=128)
        schema.add_field("enabled", DataType.INT8)
        # 维度跟随 embedding 模型（v2 是 1536 维，v3/v4 是 1024 维）；换模型后要 preload_cache.py --reset 重建集合
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=self.dim)
        index_params = self._client.prepare_index_params()
        # 缓存条目很少（几十到几百条），用 FLAT 暴力检索：结果精确，不存在 IVF 的"漏召回"问题
        index_params.add_index(field_name="embedding", index_type="FLAT", metric_type="COSINE")
        self._client.create_collection(collection_name=COLLECTION_NAME, schema=schema, index_params=index_params)


semantic_cache = SemanticCache()
