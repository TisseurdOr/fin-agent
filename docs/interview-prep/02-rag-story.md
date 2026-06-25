# RAG 迭代故事：背诵要点

> 这是面试「杀手锏」——用数据讲你如何做技术决策，而不是堆名词。

---

## 一句话结论

**在小规模知识库（7 PDF / 542 chunks）上，embedding v2→v4 的收益远大于 Query Rewrite 和 HyDE；生产推荐 v4 + Rerank（B1），Rewrite/HyDE 保留为可选开关。**

---

## 必须记住的数字

| 指标 | 数值 |
|------|------|
| 知识库规模 | 7 PDF，542 chunks |
| Embedding | text-embedding-v4，1024 维 |
| 分块 | chunk_size=800，overlap=150 |
| 测试集 | 18 题，5 类（直接匹配/口语化/术语鸿沟/极短模糊/跨文档） |
| v2 纯向量召回 | **88.9%** |
| v4 纯向量召回 | **100%**，延迟 **0.2s** |
| 生产推荐 B1（+Rerank） | **100%** 召回，**2.3s** |
| 全开 B4 | **100%** 召回，**10.3s** |

---

## 迭代时间线（按面试叙述顺序）

### 阶段 0：初始 — 只有向量检索，无 Rerank

- 问题：top-1 召回率仅 **25%**
- 根因：只取 1 条，稍有噪声就 miss

### 阶段 1：引入 LLM Rerank

- 先召回 top-12，LLM 打分选 top-4
- 召回提升到 **88%**（仍用 v2 embedding）
- 代价：多一次 LLM 调用，延迟约 +2s

### 阶段 2：Chunk 参数调优

- 500/50 → **800/150**（中文约 400 字一块，overlap 防边界切断）
- v2 纯向量：**88.9%**
- 瓶颈：口语化问题、术语鸿沟（如「炒股买什么板块」vs「A股行业配置策略」）

### 阶段 3：Query Rewriting

- LLM 把问题改写成 2-3 个专业检索词，多路召回合并去重
- v2 纯向量拉到 **100%**
- 代价：延迟 **+2s**

### 阶段 4：Embedding v4 升级（关键转折）

- 换 text-embedding-v4，自建 `DashScopeV4Embeddings`（tiktoken 不兼容 + batch 限制）
- **纯向量直接 100%**，延迟仅 0.2s
- Rewrite/HyDE 在 v4 下**无额外召回增益**，但延迟仍增加

### 当前生产配置

- 默认：`use_rerank=True`，`use_rewrite=False`，`use_hyde=False`
- API 保留开关，文档量增大后可再开

---

## 8 组消融实验结果（2026-06-24）

```
配置                        召回率     延迟
─────────────────────────────────────────
A1 纯向量 v4                 100%     0.2s   ← 最佳基准
A2 +Rewrite                 100%     2.1s   (+1.8s, 无增益)
A3 +HyDE                    100%     4.9s   (+4.6s, 无增益)
A4 +Rewrite+HyDE            100%     6.6s
B1 +Rerank                  100%     2.3s   ← 生产推荐
B2 +Rewrite+Rerank          100%     3.8s
B3 +HyDE+Rerank             100%     7.8s
B4 全开                      100%    10.3s
```

复现：`python test_recall.py`（需配置 API key 和已索引的 chroma_db）

---

## 三个增强技术：各解决什么问题

### Query Rewriting

- **问题**：用户口语 vs 研报书面语（术语鸿沟）
- **做法**：LLM 生成 2-3 个检索友好 query → 多路 `similarity_search` → 前 200 字符去重合并
- **例子**：「现在炒股应该买什么板块」→「2026年A股配置方向与策略展望」

### HyDE（Hypothetical Document Embeddings）

- **问题**：极短/模糊 query（如「fintech前景」）embedding 语义稀疏
- **做法**：LLM 生成 200-300 字假设答案 → 用假设文档做向量检索
- **原理**：假答案向量比问题向量更接近真实文档

### LLM Rerank

- **问题**：向量检索 top-k 里有噪声
- **做法**：召回 fetch_k = k×3（默认 12），LLM 对每条打 0-10 分，取 top-4
- **fallback**：Rerank 失败或 JSON 解析失败 → 直接取前 k 条

---

## 面试高频追问的标准答法

### Q: 为什么不用 BM25 混合检索？

暂缓原因：当前 7 PDF 规模下 v4 纯向量已 100%；BM25 实现成本（分词、索引维护）和收益不成比例。文档量到万级、专有名词密集时再上 hybrid。

### Q: 为什么用 LLM Rerank 不用 cross-encoder？

权衡：LLM Rerank 零额外部署、和现有 DeepSeek 复用；缺点是延迟高、成本高、打分不稳定。生产规模化会换 bge-reranker 等专用模型。

### Q: 100% 召回率可信吗？

诚实回答：
1. 只有 **18 题**手工集，不是大规模 benchmark
2. 指标是 **source hit**（top-k 里是否出现预期 PDF 文件名），不是答案正确性
3. 小库不代表生产；换更难的问题集召回可能下降
4. 下一步：RAGAS faithfulness/relevancy + 人工标注

### Q: 你做过最错的假设？

在 v2 时代以为必须上 Rewrite 才能解决术语鸿沟；升级 v4 后发现 embedding 质量才是第一杠杆，Rewrite 成了「可选增强」而非「必须」。

---

## 测试集 18 题分类（展示你有方法论）

| 类别 | 题号 | 考察点 |
|------|------|--------|
| 直接匹配 | 1-4 | 问题和文档表述接近 |
| 口语化 | 5-8 | 非专业表达 |
| 术语鸿沟 | 9-12 | 口语/网络词 vs 研报术语 |
| 极短模糊 | 13-16 | 短 query、信息量少 |
| 跨文档 | 17-18 | 需从多份 PDF 召回 |

命中判定：`check_hit()` — passages 的 source 字段包含预期 PDF 文件名即算 hit。
