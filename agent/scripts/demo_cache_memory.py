"""
缓存与记忆的端到端测试（真实环境：百炼向量 + Milvus + Redis + qwen-plus）。

用法（在项目根目录，先执行过 python app/preload_cache.py --reset）：
    python agent/scripts/demo_cache_memory.py

① 语义缓存：近义问法应命中，"像但不同"的问题和办理类请求不应命中（使用当前阈值）
② 长期记忆：会话 A 里说出偏好 → 后台抽取写入 Milvus → 新会话 B 能检索到，并出现在 Agent 的提示词里
"""

import asyncio
import os
import sys
import uuid
import warnings
from datetime import timedelta
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "agent"), str(ROOT / "app")]

import logging                                                      # noqa: E402
logging.basicConfig(level=logging.WARNING)

from config import get_settings                                     # noqa: E402
from core.clock import day_label, today                             # noqa: E402
from core.llm import get_llm                                        # noqa: E402
from core.memory.memory_manager import MemoryManager                # noqa: E402
from infra.cache import semantic_cache                              # noqa: E402

USER = "mia_li_3668"
SHOULD_HIT = ["托运行李能免费带几件", "退的钱几天能到账", "飞机延误了有补偿吗"]
SHOULD_MISS = ["托运的行李丢了怎么办", "帮我把订单 AIXC49 退掉", "好的"]
SESSION_A = [("我一般都坐靠窗的位置，出差的话经常飞纽约到西雅图。", "好的，已了解。"),
             ("付款我习惯用尾号 7447 的那张信用卡。", "明白。"),
             (f"这次我想看看 {day_label(5)}的航班，先不订。", "好的，需要时随时告诉我。")]


async def main():
    results = []
    s = get_settings()

    print(f"\n① 语义缓存（阈值 {semantic_cache.threshold}）")
    await semantic_cache.initialize()
    for q, expect in [(q, True) for q in SHOULD_HIT] + [(q, False) for q in SHOULD_MISS]:
        hit = await semantic_cache.get_cache(q, USER)
        sim = f"{hit['similarity']:.3f} → {hit['matched_question']}" if hit else "未命中"
        ok = bool(hit) == expect
        results.append((f"缓存 {'应命中' if expect else '不应命中'}：{q}", ok))
        print(f"   {'[OK]' if ok else '[FAIL]'} {q:<22} {sim}")

    print("\n② 长期记忆")
    mm = MemoryManager(redis_url=s.redis_url, redis_ttl=s.redis_ttl, milvus_host=s.milvus_host,
                       milvus_port=s.milvus_port, embedding_api_key=s.dashscope_api_key)
    await mm.initialize()
    if not (mm.short_term.available and mm.long_term.available):
        print("   [FAIL] Redis 或 Milvus 不可用"); return
    mm.long_term._client.delete(collection_name="long_term_memory", filter=f'user_id == "{USER}"')
    print(f"   已清空 {USER} 的旧偏好，开始会话 A")
    sa = f"demo-a-{uuid.uuid4().hex[:6]}"
    turns = 0
    for u, b in SESSION_A:
        turns = await mm.save_conversation(USER, sa, [{"role": "user", "content": u}, {"role": "assistant", "content": b}])
        print(f"   用户> {u}")
    print(f"   第 {turns} 轮 → 触发偏好抽取：{MemoryManager.should_extract(turns)}")
    saved = await mm.background_extract(USER, sa, get_llm(0.0))
    print(f"   抽取并写入 Milvus：{saved}")
    results.append(("抽取到座位偏好（靠窗）", any("座位偏好" in x and "窗" in x for x in saved)))
    results.append(("抽取到支付方式（7447）", any("7447" in x for x in saved)))
    trip_day = (today() + timedelta(days=5)).day
    results.append(("没有把'这次某月某日'当成偏好",
                    not any(f"{trip_day} 日" in x or f"{trip_day}日" in x for x in saved)))

    print("\n   新会话 B：用户问「帮我订一张去西雅图的机票」")
    prefs = await mm.load_preferences(USER, "帮我订一张去西雅图的机票")
    print(f"   检索到的偏好（会放进 Agent 提示词的 memory_context）：{prefs}")
    results.append(("新会话能读到上个会话的偏好", any("7447" in p for p in prefs)))
    await mm.short_term.clear(USER, sa)
    await mm.close()

    print("\n================ 结果 ================")
    for name, ok in results:
        print(f"{'[OK]' if ok else '[FAIL]'} {name}")
    print(f"通过 {sum(ok for _, ok in results)}/{len(results)}")


if __name__ == "__main__":
    asyncio.run(main())
