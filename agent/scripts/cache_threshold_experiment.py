"""
语义缓存阈值实验：在 0.50 ~ 0.98 之间扫描相似度阈值，看"该命中的命中了多少、不该命中的误命中了多少"，
并对比三项改进：多问法入库、规则前置过滤、换 embedding 模型。

用法（在项目根目录，需要 agent/.env 里的 DASHSCOPE_API_KEY）：
    python agent/scripts/cache_threshold_experiment.py                         # 对比 v2 / v3 / v4 三个模型
    python agent/scripts/cache_threshold_experiment.py --models text-embedding-v2

数据：
- data/airline/cache/preset_qa.json        10 个答案，每个有 1 个标准问题 + 3 个其他问法（aliases）
- data/airline/cache/threshold_eval.json    57 条测试问句（与 aliases 不重复）：
    positive       30 条，近义问法，应命中对应的答案
    hard_negative  17 条，字面接近某个答案，但缓存答案回答不了 / 需要查个人数据 / 实际是要办理业务
    easy_negative  10 条，无关问题和对话中的短句
四种配置：
    A 基线          只有标准问题入库，纯阈值
    B +多问法       标准问题 + aliases 入库（命中任意一个问法都返回同一答案）
    C +规则过滤     app/infra/cache.py 的 is_cacheable：带"我的/帮我"、订单号、过短的问句不查缓存
    D 两者都用
每种配置都在"错配 = 0 且误命中 = 0"的阈值里取召回率最高的，再留 0.02 安全余量作为推荐值。
注意：规则过滤是看过难负例之后设计的，在同一批数据上评估会偏乐观；它的依据是"缓存只服务通用政策问题"这一原则。
"""

import argparse
import json
import os
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "app"))
from infra.cache import is_cacheable, normalize_query     # noqa: E402  与线上缓存用同一套归一化和过滤规则

CACHE_DIR = ROOT / "data" / "airline" / "cache"
OUT_DIR = ROOT / "eval" / "results"
load_dotenv(ROOT / "agent" / ".env")

THRESHOLDS = [round(0.50 + 0.01 * i, 2) for i in range(49)]       # 0.50 ~ 0.98
MARGIN = 0.02
CONFIGS = {"A 基线": (False, False), "B +多问法": (True, False), "C +规则过滤": (False, True), "D 多问法+规则": (True, True)}


def embed_with(model):
    from langchain_community.embeddings import DashScopeEmbeddings
    emb = DashScopeEmbeddings(model=model, dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"))

    def run(texts):
        vecs = np.array(emb.embed_documents([normalize_query(t) for t in texts]), dtype=float)
        return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    return run


def sweep(sims, best_ans, queries, allowed):
    """返回每个阈值下的指标。best_ans[i]：第 i 条问句最相似的那条缓存记录所属的答案 id。"""
    n_pos = sum(q["label"] == "positive" for q in queries)
    rows = []
    for t in THRESHOLDS:
        r = {"threshold": t, "pos_correct": 0, "pos_wrong": 0, "pos_miss": 0, "hard_false": 0, "easy_false": 0}
        for q, s, b, ok in zip(queries, sims, best_ans, allowed):
            hit = ok and s >= t
            if q["label"] == "positive":
                r["pos_correct" if hit and b == q["expected"] else "pos_wrong" if hit else "pos_miss"] += 1
            elif hit:
                r["hard_false" if q["label"] == "hard_negative" else "easy_false"] += 1
        hits = r["pos_correct"] + r["pos_wrong"] + r["hard_false"] + r["easy_false"]
        r["recall"] = r["pos_correct"] / n_pos
        r["precision"] = r["pos_correct"] / hits if hits else 1.0
        r["errors"] = r["pos_wrong"] + r["hard_false"] + r["easy_false"]
        rows.append(r)
    safe = [r for r in rows if r["errors"] == 0]
    if not safe:
        return rows, None, None
    best = max(safe, key=lambda r: (r["recall"], -r["threshold"]))
    rec_t = min(round(best["threshold"] + MARGIN, 2), THRESHOLDS[-1])
    return rows, best, next(r for r in rows if r["threshold"] == rec_t)


def run_model(model, preset, queries):
    entries = [(p["id"], p["question"], False) for p in preset] + \
              [(p["id"], a, True) for p in preset for a in p.get("aliases", [])]
    t0 = time.perf_counter()
    vecs = embed_with(model)([e[1] for e in entries] + [q["text"] for q in queries])
    ev, qv = vecs[:len(entries)], vecs[len(entries):]
    print(f"\n━━━━━━━━ {model}：向量化 {len(vecs)} 条，用时 {time.perf_counter() - t0:.1f}s ━━━━━━━━")
    cos = qv @ ev.T
    out = {}
    for name, (use_alias, use_guard) in CONFIGS.items():
        cols = [i for i, e in enumerate(entries) if use_alias or not e[2]]
        sub = cos[:, cols]
        best_i = sub.argmax(axis=1)
        sims = sub.max(axis=1)
        best_ans = [entries[cols[i]][0] for i in best_i]
        allowed = [is_cacheable(q["text"]) if use_guard else True for q in queries]
        rows, best, rec = sweep(sims, best_ans, queries, allowed)
        out[name] = {"rows": rows, "best_zero_error": best, "recommended": rec,
                     "per_query": [{"text": q["text"], "label": q["label"], "best": b, "similarity": round(float(s), 4),
                                    "cacheable": a} for q, s, b, a in zip(queries, sims, best_ans, allowed)]}
        if rec:
            print(f"  {name:<12} 推荐阈值 {rec['threshold']:.2f} → 召回 {rec['recall']:>4.0%}  精确 {rec['precision']:>4.0%}"
                  f"   （零错误最低阈值 {best['threshold']:.2f}，召回 {best['recall']:.0%}）")
        else:
            print(f"  {name:<12} 没有零错误的阈值")
    # 基线配置下最危险的负例 / 最难的正例
    pq = out["A 基线"]["per_query"]
    neg = sorted([x for x in pq if x["label"] != "positive"], key=lambda x: -x["similarity"])[:4]
    pos = sorted([x for x in pq if x["label"] == "positive"], key=lambda x: x["similarity"])[:4]
    print("  相似度最高的负例：" + "；".join(f"{x['text']} {x['similarity']:.3f}" for x in neg))
    print("  相似度最低的正例：" + "；".join(f"{x['text']} {x['similarity']:.3f}" for x in pos))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="text-embedding-v2,text-embedding-v3,text-embedding-v4")
    args = ap.parse_args()
    preset = json.loads((CACHE_DIR / "preset_qa.json").read_text(encoding="utf-8"))["items"]
    queries = json.loads((CACHE_DIR / "threshold_eval.json").read_text(encoding="utf-8"))["queries"]

    results = {}
    for model in args.models.split(","):
        try:
            results[model] = run_model(model.strip(), preset, queries)
        except Exception as e:                      # 账号没开通某个模型时跳过，不影响其他模型
            print(f"\n[WARN] {model} 调用失败，跳过：{type(e).__name__}: {str(e)[:120]}")

    print("\n================ 汇总（推荐阈值下的召回率；精确率均为 100%）================")
    print(f"{'模型':<20}" + "".join(f"{n:<16}" for n in CONFIGS))
    best_overall = None
    for model, res in results.items():
        cells = []
        for name in CONFIGS:
            rec = res[name]["recommended"]
            cells.append(f"{rec['recall']:.0%} @ {rec['threshold']:.2f}" if rec else "无零错误阈值")
            if rec and (best_overall is None or rec["recall"] > best_overall[2]["recall"]):
                best_overall = (model, name, rec)
        print(f"{model:<20}" + "".join(f"{c:<16}" for c in cells))
    if best_overall:
        m, n, r = best_overall
        print(f"\n最佳：{m} + {n}，阈值 {r['threshold']:.2f}，召回 {r['recall']:.0%}，精确 {r['precision']:.0%}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{date.today():%Y%m%d}_cache_threshold.json"
    out.write_text(json.dumps({"n_answers": len(preset), "n_aliases": sum(len(p.get("aliases", [])) for p in preset),
                               "n_queries": {k: sum(q["label"] == k for q in queries)
                                             for k in ("positive", "hard_negative", "easy_negative")},
                               "results": results}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"结果已保存：{out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
