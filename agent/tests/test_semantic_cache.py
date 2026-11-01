"""政策问答缓存：默认精确匹配、显式语义模式、过滤与用户隔离。"""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest
import pytest_asyncio

# 本文件用 Milvus Lite（嵌入式 Milvus，数据存在临时文件里）测试，不碰 Docker 里的正式数据
pytest.importorskip("milvus_lite", reason="需要 milvus-lite：pip install milvus-lite")

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "app"))

import infra.cache as cache_module                 # noqa: E402
from infra.cache import SemanticCache              # noqa: E402
from memory_fakes import ControlledEmbeddings    # noqa: E402

Q = "免费托运行李额度是多少？"
A = "免费托运行李额度按会员等级和舱位确定……"


@pytest_asyncio.fixture()
async def cache(tmp_path):
    from pymilvus import MilvusClient
    emb = ControlledEmbeddings()
    c = SemanticCache(threshold=0.90, mode="semantic")
    await c.initialize(client=MilvusClient(str(tmp_path / "cache.db")), embeddings=emb)
    assert c.available
    await c.set_cache(Q, A)
    c.emb = emb
    return c


@pytest_asyncio.fixture()
async def exact_cache(tmp_path, monkeypatch):
    """Use a missing .env so the default mode is tested independently of the developer machine."""
    from pymilvus import MilvusClient

    monkeypatch.delenv(cache_module.MODE_ENV, raising=False)
    monkeypatch.setattr(cache_module, "ENV_FILE", tmp_path / "missing.env")
    emb = ControlledEmbeddings()
    c = SemanticCache(threshold=0.90)
    assert c.mode == "exact"
    await c.initialize(client=MilvusClient(str(tmp_path / "exact_cache.db")), embeddings=emb)
    assert c.available
    await c.set_cache(Q, A)
    c.emb = emb
    return c


@pytest.mark.asyncio
async def test_similarity_threshold_direction(cache):
    """相似度 0.95 ≥ 0.90 → 命中；0.85 < 0.90 → 不命中。
    这个字段是相似度不是"距离"：如果按距离比较（大于阈值就不命中），0.95 这样高度相似的问题反而永远不命中。"""
    norm = cache._normalize
    cache.emb.set(norm("托运行李能免费带几件？"), norm(Q), 0.95)
    cache.emb.set(norm("托运的行李丢了怎么办"), norm(Q), 0.85)
    hit = await cache.get_cache("托运行李能免费带几件？", user_id="u1")
    assert hit and hit["level"] == "L1_SEMANTIC" and hit["answer"] == A
    assert hit["similarity"] == pytest.approx(0.95, abs=1e-3)       # Milvus COSINE 返回的就是相似度
    assert await cache.get_cache("托运的行李丢了怎么办", user_id="u1") is None


@pytest.mark.asyncio
async def test_default_exact_mode_skips_vector_read_even_for_close_paraphrase(exact_cache):
    paraphrase = "托运行李能免费带几件？"
    exact_cache.emb.set(exact_cache._normalize(paraphrase), exact_cache._normalize(Q), 0.99)

    real_client = exact_cache._client
    client = Mock(wraps=real_client)
    embedding = Mock(aembed_query=AsyncMock(side_effect=AssertionError("vector read is disabled")))
    exact_cache._client = client
    exact_cache._embeddings = embedding

    exact = await exact_cache.get_cache(f"   {Q}  ", user_id="u1")
    assert exact and exact["level"] == "L1_EXACT" and exact["answer"] == A
    assert await exact_cache.get_cache(paraphrase, user_id="u1") is None
    embedding.aembed_query.assert_not_awaited()
    client.search.assert_not_called()


def test_mode_environment_overrides_env_file_and_defaults_to_exact(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("SEMANTIC_CACHE_MODE=semantic\n", encoding="utf-8")
    monkeypatch.setattr(cache_module, "ENV_FILE", env_file)
    monkeypatch.delenv(cache_module.MODE_ENV, raising=False)
    assert cache_module.cache_mode() == "semantic"
    assert SemanticCache().mode == "semantic"

    monkeypatch.setenv(cache_module.MODE_ENV, " exact ")
    assert cache_module.cache_mode() == "exact"
    assert SemanticCache().mode == "exact"

    monkeypatch.delenv(cache_module.MODE_ENV)
    env_file.unlink()
    assert cache_module.cache_mode() == "exact"
    assert SemanticCache().mode == "exact"


@pytest.mark.asyncio
async def test_exact_match_after_normalization(cache):
    hit = await cache.get_cache(f"   {Q}  ", user_id="u1")
    assert hit and hit["level"] == "L1_EXACT"


@pytest.mark.asyncio
async def test_unrelated_and_empty_queries_miss(cache):
    assert await cache.get_cache("好的", user_id="u1") is None
    assert await cache.get_cache("   ", user_id="u1") is None


@pytest.mark.asyncio
async def test_filter_injection_is_escaped(cache):
    """用户输入里带引号，不能改写过滤条件（比如把 scope == "user" 的私有缓存捞出来）。"""
    await cache.set_cache("常用卡是哪一张呢", "mia 的私有答案", user_id="mia")
    evil = '常用卡是哪一张呢" or scope == "user'
    assert await cache.get_cache(evil, user_id="lucas") is None
    assert await cache.get_cache("常用卡是哪一张呢", user_id='lucas" or user_id != "x') is None


@pytest.mark.asyncio
async def test_user_scoped_entries_isolated(cache):
    await cache.set_cache("常用卡是哪一张呢", "mia 的私有答案", user_id="mia")
    assert (await cache.get_cache("常用卡是哪一张呢", user_id="mia"))["answer"] == "mia 的私有答案"
    assert await cache.get_cache("常用卡是哪一张呢", user_id="lucas") is None


@pytest.mark.asyncio
async def test_set_cache_is_idempotent(cache):
    await cache.set_cache(Q, "新答案")
    rows = cache._client.query(collection_name="qa_semantic_cache", filter='scope == "public"',
                               output_fields=["answer"], limit=10)
    assert [r["answer"] for r in rows] == ["新答案"]


def test_guard_skips_personal_transactional_and_short_queries():
    from infra.cache import is_cacheable
    for q in ["我的退款为什么还没到账", "帮我把订单 AIXC49 退掉", "查一下订单aixc49", "好的", "第一个", "确认"]:
        assert not is_cacheable(q), q
    for q in ["免费托运行李额度是多少？", "我可以免费托运多少件行李", "最多可以帮几个人一起订票", "充电宝能带吗"]:
        assert is_cacheable(q), q


@pytest.mark.asyncio
async def test_guarded_query_never_hits_even_if_identical(cache):
    """即使和缓存里的问题一字不差，带订单号/个人请求的问句也不走缓存。"""
    await cache.set_cache("帮我退掉订单 AIXC49", "不该返回")
    assert await cache.get_cache("帮我退掉订单 AIXC49", user_id="u1") is None
