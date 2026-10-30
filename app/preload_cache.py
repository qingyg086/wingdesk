"""
预热 L1 语义缓存：把高频的航空政策问答写进 Milvus 的 qa_semantic_cache 集合。

用法（在项目根目录）：
    python app/preload_cache.py            # 写入/更新预置问答（可重复执行，同一问题不会重复）
    python app/preload_cache.py --reset    # 先删除整个缓存集合再写入（改了索引类型或 embedding 模型后必须执行）

问答来源：data/airline/cache/preset_qa.json，答案严格依据 data/airline/policy 下的政策原文。
只预置"答案对所有用户都一样"的政策问答；涉及个人订单、需要办理业务的问题不放进缓存。
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.append(APP_DIR)

from infra.cache import SIMILARITY_THRESHOLD, semantic_cache   # noqa: E402

PRESET = Path(APP_DIR).parent / "data" / "airline" / "cache" / "preset_qa.json"


async def preload_cache(reset: bool = False):
    items = json.loads(PRESET.read_text(encoding="utf-8"))["items"]
    print(f"预热 L1 语义缓存（{len(items)} 个答案，模型 {semantic_cache.model}，相似度阈值 {SIMILARITY_THRESHOLD}）...")
    await semantic_cache.initialize()
    if not semantic_cache.available:
        print("[FAIL] 连不上 Milvus 或向量模型，退出")
        return
    if reset:
        semantic_cache.reset()
        print(" 已删除并重建缓存集合")
    n = 0
    for item in items:
        # 多问法入库：标准问题 + 几个其他问法，都指向同一个答案（提高召回，见阈值实验）
        for q in [item["question"]] + item.get("aliases", []):
            await semantic_cache.set_cache(q, item["answer"])
            n += 1
        print(f"   [OK] {item['question']} + {len(item.get('aliases', []))} 个问法（来源：{item['source']}）")
    print(f"共写入 {n} 条")

    print("自检：用原问题查一遍，应全部精确命中")
    ok = 0
    for item in items:
        hit = await semantic_cache.get_cache(item["question"], user_id="")
        ok += bool(hit and hit["answer"] == item["answer"])
    print(f"{'[OK]' if ok == len(items) else '[FAIL]'} 自检 {ok}/{len(items)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true", help="先删除整个缓存集合")
    asyncio.run(preload_cache(ap.parse_args().reset))
