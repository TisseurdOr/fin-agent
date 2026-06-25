# fin-agent — 金融研报多 Agent 分析系统

FastAPI + LangGraph + RAG（ChromaDB）+ 统一 LLM Harness

## 功能概览

| 能力 | 接口 | 说明 |
|------|------|------|
| 对话 | `POST /api/chat` | 纯 LLM，无检索 |
| RAG 问答 | `POST /api/askRAG` | 向量检索 + 可选 Rerank / Rewrite / HyDE |
| PDF 入库 | `POST /api/upload` | 切块写入 ChromaDB |
| 多节点分析 | `POST /api/analyze` | LangGraph：分析师 → 风控 → 报告 |
| Graph 可视化 | `GET /api/graph` | ASCII 状态图 |
| 健康检查 | `GET /api/health` | Chroma + API key 配置检测 |
| 索引统计 | `GET /api/stats` | 已索引 PDF 与 chunk 数量 |

## 架构

```
用户 → FastAPI（RequestID 中间件）
         ├─ /api/upload      → PDF 切块 → ChromaDB
         ├─ /api/askRAG      → 检索增强问答（RAG）
         ├─ /api/analyze     → LangGraph 多节点分析
         └─ /api/chat        → 纯 LLM 对话

分层：
  config.py        → 模型注册表（DeepSeek / Qwen）
  llm_harness.py   → 统一 LLM 调用（重试、日志、LLMResult）
  rag.py / graph.py → 业务逻辑
  observability.py → 日志 + Request ID
```

## 工程化

- **config**：多模型注册表，延迟实例化；`max_retries=0`，重试交给 harness
- **llm_harness**：指数退避重试、`LLMResult` 结构化返回、日志含 `caller` + `request_id`
- **observability**：`RequestIDMiddleware` + 第三方库日志降噪
- **测试**：`pytest tests/`（harness + middleware，8 用例）

## RAG 指标（18 题测试集）

- Embedding：阿里 DashScope `text-embedding-v4`（1024 维）
- 分块：chunk_size=800，overlap=150
- 召回率：v2 **88.9%** → v4 **100%**
- 生产推荐：v4 + Rerank，约 **2.3s**/问
- 复现：`python test_recall.py`
- 详见 [docs/rag-improvement-guide.md](docs/rag-improvement-guide.md)

## 快速启动

```bash
git clone <your-repo> fin-agent && cd fin-agent
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # 填入 DEEPSEEK_API_KEY、DASHSCOPE_API_KEY
uvicorn app.main:app --reload
```

交互文档：http://127.0.0.1:8000/docs

## 环境变量

| 变量 | 说明 |
|------|------|
| `DEEPSEEK_API_KEY` | DeepSeek API（chat / analyze / rerank） |
| `DEEPSEEK_BASE_URL` | 默认 `https://api.deepseek.com` |
| `DASHSCOPE_API_KEY` | 通义千问 + text-embedding-v4 |

## 文档

- [docs/PROJECT.md](docs/PROJECT.md) — 功能清单、Harness 设计、待办项
- [docs/rag-improvement-guide.md](docs/rag-improvement-guide.md) — RAG 迭代与消融实验
- [docs/interview-prep/](docs/interview-prep/) — 面试准备材料

## 技术栈

Python · FastAPI · LangGraph · LangChain · ChromaDB · DeepSeek · DashScope Embedding
