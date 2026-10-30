"""
对比两种短期记忆写法在并发下的"丢失更新"问题。

用法（在项目根目录，Redis 需已启动）：
    python agent/scripts/bench_redis_race.py            # 默认 200 个并发写
    python agent/scripts/bench_redis_race.py -n 500

非原子：GET 整段 JSON → Python 里 append → SET 回去（读-改-写，非原子）
原子：RPUSH + LTRIM + EXPIRE 放在一个 MULTI/EXEC 事务里（core/memory/short_term.py）
"""

import argparse
import asyncio
import json
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import redis.asyncio as aioredis                       # noqa: E402

from core.memory.short_term import ShortTermMemory     # noqa: E402

URL = "redis://localhost:6379/15"                      # 15 号库，不碰业务数据


async def old_append(r, key, msg):
    data = await r.get(key)                            # ① 读
    msgs = json.loads(data) if data else []
    await asyncio.sleep(0)                             # 让出事件循环：真实服务里这里是请求之间的自然交错
    msgs.append(msg)                                   # ② 改
    await r.set(key, json.dumps(msgs), ex=60)          # ③ 写（覆盖掉别人在 ①~③ 之间写入的内容）


async def main(n):
    r = aioredis.Redis(connection_pool=aioredis.BlockingConnectionPool.from_url(URL, max_connections=50, decode_responses=True))
    key = f"bench_old_{uuid.uuid4().hex[:6]}"
    t0 = time.perf_counter()
    await asyncio.gather(*[old_append(r, key, {"i": i}) for i in range(n)])
    old_ms = (time.perf_counter() - t0) * 1000
    old_kept = len(json.loads(await r.get(key) or "[]"))
    await r.delete(key)
    await r.aclose()

    mem = ShortTermMemory(URL, ttl=60, max_messages=n + 10)
    await mem.initialize()
    uid = f"bench_new_{uuid.uuid4().hex[:6]}"
    t0 = time.perf_counter()
    await asyncio.gather(*[mem.append_message(uid, "s", "user", str(i)) for i in range(n)])
    new_ms = (time.perf_counter() - t0) * 1000
    new_kept = len(await mem.get_messages(uid, "s"))
    await mem.clear(uid, "s")
    await mem.close()

    print(f"\n同一会话 {n} 个并发写入：")
    print(f"非原子（GET→改→SET）  保留 {old_kept:>4} 条，丢失 {n - old_kept:>4} 条（{(n - old_kept) / n:.0%}），耗时 {old_ms:.0f} ms")
    print(f"原子（MULTI: RPUSH+LTRIM+EXPIRE） 保留 {new_kept:>4} 条，丢失 {n - new_kept:>4} 条，耗时 {new_ms:.0f} ms")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=200)
    asyncio.run(main(ap.parse_args().n))
