"""Redis 短期记忆：List + MULTI 事务。需要本机 Redis（localhost:6379），连不上则跳过。"""

import asyncio
import uuid

import pytest

from core.memory.short_term import ShortTermMemory


async def make(max_messages=20, ttl=60):
    mem = ShortTermMemory("redis://localhost:6379/15", ttl=ttl, max_messages=max_messages)   # 用 15 号库，不碰业务数据
    await mem.initialize()
    if not mem.available:
        pytest.skip("Redis 不可用")
    return mem, f"test_{uuid.uuid4().hex[:8]}"


@pytest.mark.asyncio
async def test_append_order_trim_and_ttl():
    mem, uid = await make(max_messages=5, ttl=60)
    for i in range(8):
        await mem.append_message(uid, "s", "user", f"m{i}")
    msgs = await mem.get_messages(uid, "s")
    assert [m["content"] for m in msgs] == ["m3", "m4", "m5", "m6", "m7"]      # 只留最近 5 条，顺序不变
    assert 0 < await mem._client.ttl(mem._key(uid, "s")) <= 60
    await mem.clear(uid, "s")
    assert await mem.get_messages(uid, "s") == []
    await mem.close()


@pytest.mark.asyncio
async def test_concurrent_appends_lose_nothing():
    """100 个并发追加，一条都不丢（非原子的 GET→改→SET 写法会丢，见 scripts/bench_redis_race.py）。"""
    mem, uid = await make(max_messages=1000)
    await asyncio.gather(*[mem.append_message(uid, "s", "user", f"m{i}") for i in range(100)])
    msgs = await mem.get_messages(uid, "s")
    assert sorted(m["content"] for m in msgs) == sorted(f"m{i}" for i in range(100))
    await mem.clear(uid, "s")
    await mem.close()


@pytest.mark.asyncio
async def test_migrates_old_string_key():
    """key 被 String 类型占用时，追加不能报错。"""
    mem, uid = await make()
    await mem._client.set(mem._key(uid, "s"), '[{"role": "user", "content": "old"}]')
    await mem.append_message(uid, "s", "user", "new")
    assert [m["content"] for m in await mem.get_messages(uid, "s")] == ["new"]
    await mem.clear(uid, "s")
    await mem.close()
