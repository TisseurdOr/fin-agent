# 已知弱点 & 改进方案（面试防追问）

> 每个弱点格式：**面试官怎么问 → 当前现状 → 改进方案**

---

## 1. 「多 Agent」名不副实

**问法**：你这算 Agent 吗？和 ReAct / AutoGPT 有什么区别？

**现状**：LangGraph 线性流水线 + 三个不同 system prompt 的节点。无 tool calling、无循环、无 agent 间协商。

**改进**：
- 面试诚实定位：当前是 **workflow orchestration**，「Agent」指角色分工
- 演进：给 analyst 加 tool（查股价 API、计算器）；risk 加 conditional edge；report 可回退 analyst 补数据

---

## 2. FastAPI async 但 LLM 同步阻塞

**问法**：`call_llm` 里 `llm.invoke()` 会阻塞事件循环，高并发怎么办？

**现状**：所有路由是 `async def`，但 LLM 调用同步阻塞；一次 analyze 要串行 3 次 LLM。

**改进**：
```python
# 方案 A：线程池 offload
import asyncio
result = await asyncio.to_thread(call_llm, llm, messages, caller="chat")

# 方案 B：LangChain ainvoke
resp = await llm.ainvoke(prompt)

# 方案 C：analyze 走 Celery/ARQ 异步任务队列，HTTP 返回 task_id
```

---

## 3. 无单元测试 / CI

**问法**：重构怎么保证不挂？

**现状**：`test_recall.py` 是手工脚本，需 API key；核心函数无 pytest coverage。

**改进**（已落地）：
- `tests/test_llm_harness.py` — mock LLM，测重试、LLMResult、request_id 日志（6 个用例）
- `tests/test_observability.py` — 测 RequestIDMiddleware 生成与传播（2 个用例）
- 运行：`pytest tests/`（8 passed）
- CI 建议：单元测试每次 PR；`test_recall.py` 标 integration 夜间跑

---

## 4. JSON 输出靠字符串解析

**问法**：analyst/risk/rerank 都要求 JSON，怎么保证格式？

**现状**：strip markdown fence + `json.loads()`，失败走 fallback。

**改进**：
- 使用模型 JSON mode / `response_format={"type":"json_object"}`
- LangChain `with_structured_output(PydanticModel)`
- 解析失败时 conditional edge 重试或短路

---

## 5. analyst 截断 8000 字符

**问法**：超长研报怎么办？

**现状**：`truncated = text[:8000]`，后面内容丢失。

**改进**：
- Map-Reduce：按章节切块分别提取，再 merge
- 先用 RAG 检索相关段落再送 analyst
- 用长上下文模型（128k）并做 token 计数

---

## 6. 节点失败无短路

**问法**：analyst 解析失败，risk 还跑吗？

**现状**：analyst 写 `{"error": ...}` 进 state，risk 检测到后返回高风险占位，report 仍继续。

**改进**：
```python
def route_after_analyst(state):
    if state.get("extracted_data", {}).get("error"):
        return "error_handler"
    return "risk"

builder.add_conditional_edges("analyst", route_after_analyst, {...})
```

---

## 7. Request ID 未关联日志（已修复）

**问法**：出了事故怎么串日志？

**现状（修复前）**：中间件写响应头，但 `call_llm` 日志只有 `caller`。

**改进（已实现）**：`call_llm` 日志增加 `request_id=%s`，从 `get_request_id()` 读取。

---

## 8. `_get_reranker()` 修改共享实例 temperature

**问法**：reranker 把 temperature 设为 0，会影响 chat 吗？

**现状**：`_reranker` 和 `get_llm("deepseek-v4-flash")` 是**同一个缓存实例**，mutate temperature 会影响后续调用。

**改进**：
- rerank 用独立 model 名或 `llm.bind(temperature=0)` 每次调用
- 或 config 层为 rerank 单独缓存 key：`deepseek-v4-flash-rerank`

---

## 9. ChromaDB 索引更新策略粗糙

**问法**：重复上传同名 PDF 会怎样？

**现状**：`process_pdf` 直接 `add_documents`，不删旧 chunk；同名文件会产生重复向量。

**改进**：
- metadata 加 `doc_id` + `version`
- 上传前先 `collection.delete(where={"source": filename})`
- 或全量 rebuild 脚本 + 增量 CDC

---

## 10. 无鉴权 / CORS 全开 / 无 rate limit

**问法**：怎么上线？

**现状**：`allow_origins=["*"]`，无 API key 校验。

**改进优先级**：
1. API Key 或 JWT 中间件
2. CORS 限制到前端域名
3. slowapi / nginx rate limit
4. 敏感操作（upload）加配额

---

## 11. health check 不 ping LLM

**问法**：`/api/health` 能说明服务真能用吗？

**现状**：只检查 ChromaDB 连通性和 API key 是否配置（含 placeholder 检测），不实际调 LLM。

**改进**：可选 `?deep=1` 发一个最小 embedding/chat 探测；注意成本和频率限制。

---

## 12. 无 generation eval

**问法**：召回 100%，答案就对吗？

**现状**：只测 retrieval source hit，不测答案质量。

**改进**：RAGAS（faithfulness, answer_relevancy）、人工 rubric、LLM-as-judge 对比 golden answer。

---

## 13. Analyze 与 RAG 重复解析 PDF

**问法**：同一份 PDF 读了两遍？

**现状**：`analyze_pdf` 和 `process_pdf` 各自 PyPDF2 提取。

**改进**：统一 `pdf_service.extract(path) -> text`，结果缓存到 SQLite/Redis；或 analyze 直接读已切块内容。

---

## 14. PyPDF2 表格/扫描件差

**问法**：复杂排版怎么办？

**现状**：`page.extract_text()`，表格和双栏效果差。

**改进**：pdfplumber / Unstructured / OCR（PaddleOCR）；金融表格用 camelot。

---

## 15. 无 requirements.txt（已修复）

**问法**：别人怎么复现？

**现状（修复前）**：无依赖清单。

**改进（已实现）**：根目录 `requirements.txt`，`pip install -r requirements.txt`。

---

## 快速对照表

| 弱点 | 严重程度 | 面试话术 |
|------|----------|----------|
| 非真 Agent | 中 | 诚实 + 说清演进路径 |
| sync blocking | 高 | to_thread / ainvoke / 任务队列 |
| 无测试 | 高 | 正在补 pytest + 分层测试策略 |
| JSON 解析脆 | 中 | structured output |
| 8000 截断 | 中 | map-reduce / RAG 先筛 |
| 无鉴权 | 高（上线） | 知道优先级，不是 demo 重点 |
| 无 gen eval | 中 | 知道 RAGAS，retrieval 先达标 |
