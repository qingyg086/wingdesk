"""
把航空政策文档（data/airline/policy/**/*.md，中文）切分后写入 Milvus 集合 airline_policy_docs。

用法（在项目根目录）：
    python agent/database/import_airline_policy.py            # 切分 + 计算向量 + 入库 + 检索测试
    python agent/database/import_airline_policy.py --dry-run  # 只切分并打印统计，不调用大模型、不连 Milvus

同时会清空 qa_semantic_cache：政策更新后，缓存里的旧答案可能过期。后端启动时会自动重建空集合，
之后运行 python app/preload_cache.py 重新预热。

切分策略：
    1. MarkdownHeaderTextSplitter 按 # / ## 切成"小节"，每个小节讲一件事，语义完整；
    2. 小节超过 500 字时，用 RecursiveCharacterTextSplitter 再切，chunk_size=500，overlap=50，
       优先在段落、换行、句号处断开；
    3. 每个 chunk 正文前补上"【文档标题 / 小节标题】"（contextual chunk header），
       孤立的片段也能看出在讲什么，检索更准；
    4. 文档开头的"来源 / 仿真声明"不放进正文（每篇都一样，是噪声），改为 metadata.category。
"""

import argparse
import sys
import warnings
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
POLICY_DIR = ROOT / "data" / "airline" / "policy"
sys.path.insert(0, str(ROOT / "agent"))
# langchain-milvus 0.3 内部仍在用 pymilvus 的 ORM 接口，会刷一屏弃用警告，不影响结果，这里屏蔽
warnings.filterwarnings("ignore", module=r"langchain_milvus.*")
warnings.filterwarnings("ignore", message=r".*was never awaited")      # 复用 tools/vector_tool.py 的配置和 pymilvus 兼容补丁

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

CHUNK_SIZE, CHUNK_OVERLAP = 500, 50
OLD_COLLECTIONS = ["qa_semantic_cache"]
SAMPLE_QUERIES = [
    "金卡会员坐经济舱可以免费托运几件行李？",
    "充电宝可以托运吗？",
    "买了经济舱没买保险，现在想退票可以吗？",
    "航班延误了能拿到什么补偿？",
    "可以带猫上飞机吗？",
]


def load_and_split():
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "title"), ("##", "section")], strip_headers=True)
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "；", "，", " ", ""])

    chunks = []
    for path in sorted(POLICY_DIR.glob("*/*.md")):
        category = path.parent.name                     # core / extended
        lines = path.read_text(encoding="utf-8").splitlines()
        body = "\n".join(ln for ln in lines if not ln.startswith(">"))   # 去掉来源/仿真声明
        for sec in header_splitter.split_text(body):
            title = sec.metadata.get("title", path.stem)
            section = sec.metadata.get("section", "概述")
            for piece in char_splitter.split_text(sec.page_content):
                chunks.append(Document(
                    page_content=f"【{title} / {section}】\n{piece}",
                    # 所有 chunk 的 metadata 键必须一致：langchain-milvus 按第一条数据建字段
                    metadata={"source": f"{category}/{path.name}", "category": category,
                              "title": title, "section": section},
                ))
    return chunks


def print_stats(chunks):
    files = Counter(c.metadata["source"] for c in chunks)
    lengths = [len(c.page_content) for c in chunks]
    by_cat = Counter(c.metadata["category"] for c in chunks)
    n_core = len({f for f in files if f.startswith("core/")})
    print(f"{len(files)} 份文档（核心 {n_core}，扩充 {len(files) - n_core}）"
          f" → {len(chunks)} 个 chunk（核心 {by_cat['core']}，扩充 {by_cat['extended']}）")
    print(f"chunk 长度：最短 {min(lengths)}，最长 {max(lengths)}，平均 {sum(lengths) / len(lengths):.0f} 字")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="只切分，不入库")
    args = ap.parse_args()

    chunks = load_and_split()
    print_stats(chunks)
    if args.dry_run:
        c = chunks[0]
        print(f"\n示例 chunk（{c.metadata}）：\n{c.page_content}")
        return

    from langchain_milvus import Milvus
    from pymilvus import MilvusClient
    from tools.vector_tool import (COLLECTION_NAME, INDEX_PARAMS, SEARCH_PARAMS, get_embeddings,
                                   get_milvus_uri)

    uri = get_milvus_uri()
    client = MilvusClient(uri=uri)
    for name in OLD_COLLECTIONS:
        if client.has_collection(name):
            client.drop_collection(name)
            print(f" 已清空集合 {name}")

    print(f"计算向量并写入 {COLLECTION_NAME}（drop_old=True，可重复执行）…")
    store = Milvus.from_documents(
        chunks, get_embeddings(), connection_args={"uri": uri}, collection_name=COLLECTION_NAME,
        index_params=INDEX_PARAMS, search_params=SEARCH_PARAMS, drop_old=True)
    # 新写入的数据先在内存的 growing segment 里，flush 落盘后统计才准；
    # 用 count(*) 查询（强一致）计数，而不是读 get_collection_stats 的缓存统计
    client.flush(COLLECTION_NAME)
    n = client.query(COLLECTION_NAME, filter="", output_fields=["count(*)"],
                     consistency_level="Strong")[0]["count(*)"]
    print(f"{'[OK]' if n == len(chunks) else '[FAIL]'} 集合 {COLLECTION_NAME} 共 {n} 条 / 期望 {len(chunks)}")

    print("\n检索测试（分数是余弦相似度，越大越相关）：")
    for q in SAMPLE_QUERIES:
        print(f"\n问：{q}")
        for doc, score in store.similarity_search_with_score(q, k=3):
            print(f"   {score:.3f}  {doc.metadata['source']:<34} {doc.metadata['section']}")
    sys.exit(0 if n == len(chunks) else 1)


if __name__ == "__main__":
    main()
