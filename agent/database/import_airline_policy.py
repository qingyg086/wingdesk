"""
把航空政策文档（data/airline/policy/**/*.md，中文）切分后写入按语料内容版本命名的 Milvus 混合检索集合。

用法（在项目根目录）：
    python agent/database/import_airline_policy.py            # 切分 + 计算向量 + 入库 + 检索测试
    python agent/database/import_airline_policy.py --dry-run  # 只切分并打印统计，不调用大模型、不连 Milvus

新集合同时保存向量与中文 BM25 索引，不覆盖旧政策集合。
政策内容有变化时，加 --invalidate-cache 清空旧问答缓存，再更新 preset_qa.json 并重新预热。

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
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
POLICY_DIR = ROOT / "data" / "airline" / "policy"
sys.path.insert(0, str(ROOT / "agent"))

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

CHUNK_SIZE, CHUNK_OVERLAP = 500, 50
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
                    # 文档/小节字段用于引用来源、检索评测和索引版本校验。
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
    ap.add_argument("--invalidate-cache", action="store_true", help="政策更新后清空旧问答缓存")
    args = ap.parse_args()

    chunks = load_and_split()
    print_stats(chunks)
    if args.dry_run:
        c = chunks[0]
        print(f"\n示例 chunk（{c.metadata}）：\n{c.page_content}")
        return

    from core.policy_retrieval import PolicyRetriever, build_index, client, corpus

    rows, fingerprint = corpus()
    db = client()
    try:
        name = build_index(db, rows, fingerprint)
        print(f"[OK] {name}: {len(rows)} 条，语料 {fingerprint}")
        if args.invalidate_cache and db.has_collection("qa_semantic_cache"):
            db.drop_collection("qa_semantic_cache")
            print("已清空旧问答缓存，请更新预置答案后重新预热")
        retriever = PolicyRetriever(db, name)
        for query in SAMPLE_QUERIES:
            print(f"\n问：{query}")
            for hit in retriever.search(query, "hybrid_rerank", limit=3):
                print(f"   {hit['source']} / {hit['section']}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
