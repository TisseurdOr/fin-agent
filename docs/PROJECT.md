# fin-agent 项目进度与功能清单

> 更新：2026-06-24

## 已完成

### 基础层（D1）

- [x] FastAPI + `POST /api/chat`
- [x] 多模型配置（DeepSeek v4-flash / v4-pro、Qwen-plus）
- [x] `.env` + `.gitignore` + `.env.example`

### RAG（D2 + 增强）

- [x] PDF 上传与 ChromaDB 索引（`POST /api/upload`）
- [x] `POST /api/askRAG` 检索增强问答
- [x] `DashScopeV4Embeddings` 自建封装（tiktoken 不兼容 + v4 batch 限制）
- [x] 分块参数：chunk_size=800，overlap=150
- [x] LLM Rerank / Query Rewrite / HyDE（API 可开关）
- [x] `test_recall.py` 18 题测试集 + 8 组消融实验
- [x] `docs/rag-improvement-guide.md`

### LangGraph（D3）

- [x] analyst → risk → report 线性流水线
- [x] `POST /api/analyze`
- [x] `GET /api/graph` ASCII 可视化
- [x] `AnalysisState` 6 字段状态传递

### 工程化（D3.5）

- [x] `app/config.py` — 模型注册表，与重试解耦
- [x] `app/llm_harness.py` — `call_llm` + `LLMResult`
- [x] `app/observability.py` — RequestID 中间件 + `setup_logging`
- [x] `requirements.txt`
- [x] `tests/test_llm_harness.py`、`tests/test_observability.py`（8 用例）

## Harness 设计说明

### 动机

Graph 3 节点 + RAG 3 种增强 = **8+ LLM 调用点**。若各节点直接 `llm.invoke()`，重试策略、异常处理、日志格式会不一致，排查一次 analyze（3 次 LLM 串行）很困难。

### 职责分层

| 模块 | 职责 |
|------|------|
| `config.py` | 实例化 ChatOpenAI（`max_retries=0`）、embedding client、模型缓存 |
| `llm_harness.py` | 重试（指数退避 + jitter）、`LLMResult`、结构化日志 |
| `graph.py` / `rag.py` / `main.py` | 业务 prompt、JSON 解析、业务 fallback |

### 调用链示例

```
HTTP 请求
  → RequestIDMiddleware（写入 ContextVar）
  → ask_rag()
    → rag.query()
      → call_llm(caller="rewrite")   # 可选
      → call_llm(caller="hyde")      # 可选
      → call_llm(caller="rerank")    # 可选
    → call_llm(caller="ask_rag")
```

日志字段：`request_id`、`caller`、`model`、`duration_ms`、`content_len`

### 面试话术（30 秒）

> Agent 链路里 LLM 是最不稳定的一环。我把所有调用收口到 `call_llm`：统一指数退避重试、结构化返回 `LLMResult` 不抛异常、日志带 `caller` 和 `request_id`。config 管模型实例，harness 管调用策略，Graph/RAG 只管 prompt 和结果解析。

## RAG 关键数据

| 配置 | 召回率 | 延迟 |
|------|--------|------|
| A1 纯向量 v4 | 100% | 0.2s |
| B1 +Rerank（生产推荐） | 100% | 2.3s |
| B4 全开 | 100% | 10.3s |

v2 → v4：纯向量召回 88.9% → 100%

## 待完成（对照作战手册）

- [ ] D4a LangGraph tool calling（查股价、行业 PE）
- [ ] D4b MCP Server 暴露同一套 tools
- [ ] D5 docker-compose + 演示视频
- [ ] conditional edge（analyst 失败短路）
- [ ] async LLM（`ainvoke` / `asyncio.to_thread`）
- [ ] RAGAS generation eval
- [ ] ChromaDB 增量索引（同名 PDF 去重）

## API 一览

详见根目录 [README.md](../README.md)。
