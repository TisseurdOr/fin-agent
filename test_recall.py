"""
RAG 召回率综合对比测试 — 测试所有方案组合

测试矩阵（5 组）：
  1. 基准: v4 embedding 纯向量检索（无 rerank, 无 rewrite, 无 hyde）
  2. Rewrite: v4 + multi-query rewrite（无 rerank）
  3. HyDE: v4 + HyDE 假设文档检索（无 rerank）
  4. Rewrite + HyDE: v4 + 两者叠加（无 rerank）
  5. 生产模式: v4 + Rewrite + HyDE + Rerank（全开）
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from app.rag import query, get_stats, _rewrite_query, _hyde_generate


# ── 测试用例（同上轮 18 题）──
TEST_CASES = [
    # 直接匹配
    {"id": 1, "category": "直接匹配", "question": "2026年A股投资策略有哪些核心观点？",
     "expected_sources": ["2026A股中期策略.pdf"]},
    {"id": 2, "category": "直接匹配", "question": "金融科技行业在2026年的发展趋势是什么？",
     "expected_sources": ["2026中国金融科技深度分析.pdf"]},
    {"id": 3, "category": "直接匹配", "question": "银行如何进行数字化转型？",
     "expected_sources": ["银行数字化转型报告.pdf"]},
    {"id": 4, "category": "直接匹配", "question": "AI行业2026年投资展望如何？",
     "expected_sources": ["AI行业2026投资展望.pdf"]},
    # 口语化
    {"id": 5, "category": "口语化", "question": "现在炒股应该买什么板块比较好？",
     "expected_sources": ["2026A股中期策略.pdf"]},
    {"id": 6, "category": "口语化", "question": "现在的fintech公司都在搞什么新技术？",
     "expected_sources": ["2026中国金融科技深度分析.pdf"]},
    {"id": 7, "category": "口语化", "question": "AI这块现在能不能投？有什么机会？",
     "expected_sources": ["AI行业2026投资展望.pdf"]},
    {"id": 8, "category": "口语化", "question": "杜邦分析法的公式是啥？ROE怎么拆？",
     "expected_sources": ["CFA一级核心知识点.pdf"]},
    # 术语鸿沟
    {"id": 9, "category": "术语鸿沟", "question": "大模型概念股有哪些，值得关注吗？",
     "expected_sources": ["AI行业2026投资展望.pdf"]},
    {"id": 10, "category": "术语鸿沟", "question": "支付公司和网贷平台现在的监管环境怎么样？",
     "expected_sources": ["2026中国金融科技深度分析.pdf"]},
    {"id": 11, "category": "术语鸿沟", "question": "经济下行期哪些行业更抗跌？",
     "expected_sources": ["2026A股中期策略.pdf"]},
    {"id": 12, "category": "术语鸿沟", "question": "银行怎么利用AI和大数据？",
     "expected_sources": ["银行数字化转型报告.pdf"]},
    # 极短模糊
    {"id": 13, "category": "极短模糊", "question": "下半年股市怎么看？",
     "expected_sources": ["2026A股中期策略.pdf"]},
    {"id": 14, "category": "极短模糊", "question": "fintech前景",
     "expected_sources": ["2026中国金融科技深度分析.pdf"]},
    {"id": 15, "category": "极短模糊", "question": "PE、PB、ROE这些指标怎么用？",
     "expected_sources": ["CFA_Ratio_Sheet.pdf", "CFA一级核心知识点.pdf"]},
    {"id": 16, "category": "极短模糊", "question": "银行数字化",
     "expected_sources": ["银行数字化转型报告.pdf"]},
    # 跨文档
    {"id": 17, "category": "跨文档", "question": "2026年科技板块的风险和挑战有哪些？",
     "expected_sources": ["2026中国金融科技深度分析.pdf", "AI行业2026投资展望.pdf"]},
    {"id": 18, "category": "跨文档", "question": "股票估值常用的财务比率和公式有哪些？",
     "expected_sources": ["CFA_Ratio_Sheet.pdf", "CFA一级核心知识点.pdf", "CFA_Level1_Formulas.pdf"]},
]


def check_hit(passages, expected_sources):
    """检查是否命中至少一个预期来源"""
    for p in passages:
        for expected in expected_sources:
            if expected.lower() in p["source"].lower():
                return True
    return False


def evaluate(label: str, k: int, use_rerank: bool, use_rewrite: bool, use_hyde: bool) -> dict:
    """评估一种配置"""
    flags = []
    if use_rewrite: flags.append("RW")
    if use_hyde: flags.append("HyDE")
    if use_rerank: flags.append("Rerank")
    flag_str = "+".join(flags) if flags else "纯向量"

    print(f"\n{'─'*60}")
    print(f"  {label}: {flag_str}  |  k={k}")
    print(f"{'─'*60}")

    hits = 0
    total_time = 0
    failures = []

    for tc in TEST_CASES:
        start = time.time()
        context, passages = query(
            tc["question"], k=k,
            use_rerank=use_rerank, use_rewrite=use_rewrite, use_hyde=use_hyde,
        )
        elapsed = time.time() - start
        total_time += elapsed

        hit = check_hit(passages, tc["expected_sources"])
        if hit:
            hits += 1
            status = "✓"
        else:
            status = "✗"
            got = sorted(set(p["source"] for p in passages))
            failures.append((tc["id"], tc["question"][:50], tc["expected_sources"], got))

        # 紧凑输出
        got_src = sorted(set(p["source"] for p in passages))
        print(f"  [{status}] #{tc['id']:02d} {tc['category']:6s} | "
              f"{tc['question'][:42]:42s} | → {', '.join(s[:25] for s in got_src[:2])}")

    recall = hits / len(TEST_CASES)
    avg_t = total_time / len(TEST_CASES)

    print(f"  {'─'*58}")
    print(f"  {label}: {hits}/{len(TEST_CASES)} = {recall:.1%}  |  平均 {avg_t:.1f}s  |  总计 {total_time:.1f}s")

    if failures:
        print(f"\n  ❌ 失败 ({len(failures)}):")
        for fid, fq, fexp, fgot in failures:
            print(f"    #{fid:02d} \"{fq}\" → 期望 {fexp}, 实际 {fgot}")

    return {"label": label, "flag_str": flag_str, "recall": recall,
            "hits": hits, "total": len(TEST_CASES), "avg_time": avg_t, "total_time": total_time}


def main():
    print("=" * 60)
    print("  RAG 召回率综合对比 — Embedding v4 + Rewrite + HyDE")
    print("=" * 60)

    # 索引状态
    stats = get_stats()
    print(f"\nEmbedding: text-embedding-v4")
    print(f"索引: {stats['indexed_pdfs']} PDF, {stats['total_chunks']} chunks")

    # ── 测试矩阵 ──
    results = []

    print(f"\n\n{'#'*60}")
    print(f"#  阶段 1: 纯向量检索（无 Rerank）— 对比各方案的原始召回能力")
    print(f"{'#'*60}")

    results.append(evaluate("A1 基准", k=12, use_rerank=False, use_rewrite=False, use_hyde=False))
    results.append(evaluate("A2 Rewrite", k=12, use_rerank=False, use_rewrite=True, use_hyde=False))
    results.append(evaluate("A3 HyDE", k=12, use_rerank=False, use_rewrite=False, use_hyde=True))
    results.append(evaluate("A4 RW+HyDE", k=12, use_rerank=False, use_rewrite=True, use_hyde=True))

    print(f"\n\n{'#'*60}")
    print(f"#  阶段 2: 生产模式（+Rerank）— 全链路最终效果")
    print(f"{'#'*60}")

    results.append(evaluate("B1 Rerank", k=4, use_rerank=True, use_rewrite=False, use_hyde=False))
    results.append(evaluate("B2 RW+Rerank", k=4, use_rerank=True, use_rewrite=True, use_hyde=False))
    results.append(evaluate("B3 HyDE+Rerank", k=4, use_rerank=True, use_rewrite=False, use_hyde=True))
    results.append(evaluate("B4 全开", k=4, use_rerank=True, use_rewrite=True, use_hyde=True))

    # ── 对比汇总 ──
    print(f"\n\n{'='*70}")
    print(f"  对 比 汇 总（Embedding v4）")
    print(f"{'='*70}")
    print(f"  {'配置':<22} {'Recall':>8} {'命中':>8} {'平均耗时':>10}")
    print(f"  {'─'*48}")

    best_recall = 0
    best_label = ""
    for r in results:
        marker = " ★" if r["recall"] > best_recall else ""
        if r["recall"] > best_recall:
            best_recall = r["recall"]
            best_label = r["label"]
        print(f"  {r['label']:<22} {r['recall']:>7.1%} {str(r['hits'])+'/'+str(r['total']):>8} {r['avg_time']:>9.1f}s{marker}")

    # ── 效果分析 ──
    print(f"\n  ── 提升分析 ──")
    baseline = results[0]
    for r in results[1:]:
        delta = r["recall"] - baseline["recall"]
        time_delta = r["avg_time"] - baseline["avg_time"]
        print(f"  {r['label']:<22} vs 基准: recall {delta:+.1%}  |  延迟 +{time_delta:+.1f}s")

    # ── HyDE 样本展示 ──
    print(f"\n\n{'='*60}")
    print(f"  HyDE 假设答案样本展示")
    print(f"{'='*60}")

    sample_questions = [
        "下半年股市怎么看？",
        "fintech前景",
        "经济下行期哪些行业更抗跌？",
    ]
    for q in sample_questions:
        print(f"\n  问题: {q}")
        hyde_answer = _hyde_generate(q)
        # 截断显示
        display = hyde_answer[:200] + "..." if len(hyde_answer) > 200 else hyde_answer
        print(f"  HyDE 生成: {display}")

    print(f"\n  💡 最佳配置: {best_label} ({best_recall:.1%})")
    print(f"  💡 生产推荐: B2 Rewrite+Rerank（{results[5]['recall']:.1%}，{results[5]['avg_time']:.1f}s）\n")


if __name__ == "__main__":
    main()
