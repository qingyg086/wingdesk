"""长期记忆：偏好抽取、只看用户发言、按类别覆盖、敏感信息过滤、跨会话读取、过滤转义。
短期记忆用真实 Redis（15 号库），长期记忆用 Milvus Lite，大模型用记录输入的假模型。"""

import uuid

import pytest
import pytest_asyncio

# 本文件用 Milvus Lite（嵌入式 Milvus，数据存在临时文件里）测试，不碰 Docker 里的正式数据
pytest.importorskip("milvus_lite", reason="需要 milvus-lite：pip install milvus-lite")
from langchain_core.messages import AIMessage

from core.memory.memory_manager import MemoryManager
from core.memory.preference_extractor import PreferenceExtractor
from memory_fakes import ControlledEmbeddings


class RecordingLLM:
    def __init__(self, reply: str):
        self.reply, self.prompts = reply, []

    async def ainvoke(self, messages):
        self.prompts.append(messages[0]["content"])
        return AIMessage(content=self.reply)


@pytest_asyncio.fixture()
async def mm(tmp_path):
    from pymilvus import MilvusClient
    m = MemoryManager(redis_url="redis://localhost:6379/15", redis_ttl=60)
    await m.short_term.initialize()
    if not m.short_term.available:
        pytest.skip("Redis 不可用")
    await m.long_term.initialize(client=MilvusClient(str(tmp_path / "ltm.db")), embeddings=ControlledEmbeddings())
    assert m.long_term.available
    m.uid = f"u_{uuid.uuid4().hex[:6]}"
    yield m
    await m.short_term.clear(m.uid, "s1")
    await m.close()


async def chat(m, session, pairs):
    turns = 0
    for user, bot in pairs:
        turns = await m.save_conversation(m.uid, session, [{"role": "user", "content": user},
                                                          {"role": "assistant", "content": bot}])
    return turns


@pytest.mark.asyncio
async def test_extract_every_three_turns_and_read_in_new_session(mm):
    turns = await chat(mm, "s1", [("我一般都坐靠窗的位置", "好的"), ("我经常飞纽约到西雅图", "明白"),
                                  ("付款习惯用尾号 7447 的信用卡", "收到")])
    assert turns == 3 and MemoryManager.should_extract(turns)
    llm = RecordingLLM("座位偏好: 靠窗\n常飞航线: 纽约 → 西雅图\n常用支付方式: 尾号 7447 的信用卡")
    saved = await mm.background_extract(mm.uid, "s1", llm)
    assert len(saved) == 3
    # 新会话：按问题检索，能拿到上一个会话抽取的偏好
    prefs = await mm.load_preferences(mm.uid, "帮我订去西雅图的机票", top_k=5)
    assert set(prefs) == {"座位偏好: 靠窗", "常飞航线: 纽约 → 西雅图", "常用支付方式: 尾号 7447 的信用卡"}


@pytest.mark.asyncio
async def test_only_user_messages_sent_to_extractor(mm):
    await chat(mm, "s1", [("我喜欢靠过道", "建议您选择商务舱，空间更大")])
    llm = RecordingLLM("无")
    await mm.background_extract(mm.uid, "s1", llm)
    assert "我喜欢靠过道" in llm.prompts[0]
    assert "建议您选择商务舱" not in llm.prompts[0]          # 客服的建议不能被当成用户偏好


@pytest.mark.asyncio
async def test_same_category_is_overwritten_not_duplicated(mm):
    await chat(mm, "s1", [("我一般坐靠窗", "好")])
    await mm.background_extract(mm.uid, "s1", RecordingLLM("座位偏好: 靠窗"))
    await mm.background_extract(mm.uid, "s1", RecordingLLM("座位偏好: 靠过道\n餐食偏好: 素食"))
    prefs = sorted(await mm.long_term.list_memories(mm.uid))
    assert prefs == ["座位偏好: 靠过道", "餐食偏好: 素食"]


@pytest.mark.asyncio
async def test_extraction_failure_never_raises(mm):
    class Broken:
        async def ainvoke(self, _):
            raise TimeoutError("LLM down")
    await chat(mm, "s1", [("我喜欢靠窗", "好")])
    assert await mm.background_extract(mm.uid, "s1", Broken()) == []


@pytest.mark.asyncio
async def test_user_id_filter_is_escaped(mm):
    await mm.save_preference(mm.uid, "座位偏好", "靠窗")
    assert await mm.long_term.list_memories('x" or user_id != "x') == []


def test_parse_whitelist_and_sensitive_filter():
    raw = ("座位偏好：靠窗\n- 常用支付方式: 卡号 6222021234567890\n性格: 和善\n"
           "常飞航线: 纽约到西雅图\n座位偏好: 靠过道\n舱位偏好: 无")
    assert PreferenceExtractor.parse(raw) == ["座位偏好: 靠窗", "常飞航线: 纽约到西雅图"]
