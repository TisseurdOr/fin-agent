# 项目介绍：1 分钟 & 5 分钟版本

> 面试口述稿，建议对着镜子练到能脱稿。

---

## 1 分钟版（电梯演讲）

我做了一个**金融研报智能分析系统**，后端是 FastAPI。

它有两条核心能力：

**第一，多 Agent 研报分析。** 用户上传 PDF 后，LangGraph 会跑一条线性流水线：分析师节点提取结构化数据，风控节点做五维风险检查，报告节点生成 Markdown 摘要。三个节点用不同 system prompt 分工，State 在节点间逐步丰富。

**第二，RAG 知识库问答。** PDF 切块后存入 ChromaDB，用阿里 text-embedding-v4 做向量检索，可选 LLM Rerank 精排，再让大模型基于检索结果回答，减少幻觉。

工程上我抽了统一的 LLM Harness：指数退避重试、结构化错误返回、请求 ID 可追踪。RAG 侧我做了完整的消融实验，从 embedding v2 迭代到 v4，有 18 题测试集和延迟数据支撑。

---

## 5 分钟版（技术面试标准开场）

### 1. 背景与目标（30s）

金融研报又长又密，分析师每天要读大量 PDF。我想做一个系统，既能**自动提取关键指标和风险**，又能**基于已入库研报做可追溯的问答**。目标用户是投研助理、风控初审——不是直接给散户荐股，而是提效和信息整理。

### 2. 整体架构（60s）

```
用户 → FastAPI
         ├─ /api/upload      → PDF 切块 → ChromaDB 索引
         ├─ /api/askRAG      → 检索增强问答（RAG 主链路）
         ├─ /api/analyze     → LangGraph 多节点分析
         └─ /api/chat        → 纯 LLM 对话（无检索）
```

技术栈：FastAPI + LangGraph + ChromaDB + DeepSeek/Qwen（OpenAI 兼容接口）。

分层设计：
- `config.py` — 模型注册表，延迟实例化
- `llm_harness.py` — 统一 LLM 调用（重试、日志、LLMResult）
- `rag.py` / `graph.py` — 业务逻辑
- `observability.py` — 日志 + Request ID 中间件

### 3. 多 Agent 分析链路（90s）

`/api/analyze` 调用 `analyze_pdf()`：

1. PyPDF2 提取全文
2. `graph.invoke()` 跑 LangGraph：
   - **analyst**：从研报提取 JSON（公司、指标、趋势、data_quality）
   - **risk**：五维风控（数据合理性、一致性、缺失、行业、集中度）
   - **report**：汇总成 Markdown 摘要 + 综合评级

这是**线性 workflow orchestration**，不是 ReAct 式自主 Agent。优势是流程可控、每步可观测；后续可加 conditional edge（解析失败短路）或并行节点。

诚实定位：当前是「多角色 prompt 分工 + Graph 编排」，不是 tool-calling 自主 agent。

### 4. RAG 链路（90s）

`/api/askRAG` 默认生产配置：**v4 embedding + top-12 召回 + LLM Rerank 到 top-4**。

可选增强（API 开关）：
- **Query Rewrite**：口语问题改写成 2-3 个专业检索词，多路召回合并
- **HyDE**：生成假设答案再 embedding，拉近 query-doc 语义距离

我自己封装了 `DashScopeV4Embeddings`，因为 LangChain 的 OpenAIEmbeddings 用 tiktoken 分词，和 DashScope 不兼容，且 v4 单次最多 10 条要手动 batch。

分块参数：chunk_size=800、overlap=150，针对中文研报调过。

### 5. 工程亮点与数据（60s）

- **LLM Harness**：`max_retries=0` 在 config 层，重试统一在 harness；指数退避 + jitter；永不抛异常，返回 `LLMResult`
- **RAG 迭代有数据**：18 题测试集，v2 纯向量 88.9% → v4 100%；生产推荐 B1（+Rerank，2.3s）
- **可观测性**：Request ID 贯穿 HTTP 响应和 LLM 日志

### 6. 收尾：已知局限 & 下一步（30s）

当前是小规模验证（7 PDF / 542 chunks），没有 generation eval，analyze 和 RAG 各读一遍 PDF。上线前需要：鉴权、async LLM 调用、cross-encoder reranker、pytest CI、更大 eval set。

---

## 架构图口述（配合白板）

面试时边画边说：

```
[Client]
   │
   ▼
[FastAPI] ── RequestID ── CORS
   │
   ├─ upload ──► process_pdf ──► ChromaDB
   │
   ├─ askRAG ──► query() ──► [Rewrite?] ──► [HyDE?] ──► vector search ──► [Rerank?] ──► LLM 生成
   │
   └─ analyze ──► analyze_pdf() ──► LangGraph: analyst → risk → report
```

---

## Demo 路径建议

**路径 A（RAG，推荐）：**
1. `POST /api/upload` 上传一份研报
2. `POST /api/askRAG` 问「2026年A股投资策略有哪些核心观点？」
3. 展示 `sources` 字段和回答中的引用

**路径 B（多 Agent）：**
1. `POST /api/analyze` 传入已有 PDF 文件名
2. 展示 `extracted_data`、`risk_flags`、`final_report` 三段输出
3. `GET /api/graph` 展示 ASCII 图

---

## 英文版（1 分钟，外企/海外岗备用）

I built a financial research report analysis system with FastAPI. It has two main capabilities: first, a LangGraph pipeline with three role-specialized nodes—analyst, risk, and report—that extract structured data, flag risks, and generate summaries from uploaded PDFs. Second, a RAG pipeline using ChromaDB with DashScope text-embedding-v4, optional query rewriting and HyDE, and an LLM reranker for grounded Q&A. I centralized LLM calls in a harness with exponential backoff retries and structured error handling, and I ran ablation experiments on retrieval—improving recall from 88.9% to 100% on an 18-question eval set after upgrading embeddings from v2 to v4.
