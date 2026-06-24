import os
from pathlib import Path
from typing import List
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from openai import OpenAI
import json

DATA_DIR = Path(__file__).parent.parent / "data"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"

# ── Embedding：text-embedding-v4 via OpenAI 兼容接口 ──
# 不用 LangChain 的 OpenAIEmbeddings（tiktoken 分词与 DashScope 不兼容），
# 直接用 OpenAI client 封装，完全可控

_dashscope_client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY", "sk-placeholder"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

class DashScopeV4Embeddings(Embeddings):
    """text-embedding-v4，OpenAI 兼容接口，1024 维"""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        all_embeddings = []
        # v4 单次最多 10 条，分批调用
        for i in range(0, len(texts), 10):
            batch = texts[i : i + 10]
            resp = _dashscope_client.embeddings.create(
                model="text-embedding-v4", input=batch,
            )
            all_embeddings.extend(d.embedding for d in resp.data)
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


embeddings = DashScopeV4Embeddings()

# ── Chunker（改进参数） ──
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,       # 500→800，中文约400字，保全文段
    chunk_overlap=150,    # 50→150，防止关键信息在边界被切断
    separators=["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";", " "],
)

# ── LLM Reranker ──
reranker_llm = ChatOpenAI(
    model="deepseek-v4-flash",
    api_key=os.getenv("DEEPSEEK_API_KEY", "sk-placeholder"),
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    temperature=0,
)

RERANK_PROMPT = """你是一个信息检索专家。评估以下文档片段与用户问题的相关性。

用户问题：{question}

{passages}

请选出最相关的 TOP{top_n} 个片段。严格按以下 JSON 格式输出，不要输出其他内容：
{{"rankings": [{{"index": 数字, "score": 0-10的整数, "reason": "一句话理由"}}]}}"""


def _rerank(question: str, passages: list[dict], top_n: int = 4) -> list[dict]:
    """LLM 重排序：从候选片段中精选最相关的"""
    if len(passages) <= top_n:
        return passages

    candidates = "\n\n".join(
        f"[索引 {i}]\n来源: {p['source']}\n内容: {p['content']}"
        for i, p in enumerate(passages)
    )
    prompt = RERANK_PROMPT.format(
        question=question, passages=candidates, top_n=top_n
    )

    resp = reranker_llm.invoke(prompt)
    try:
        # 提取 JSON（处理 LLM 可能包裹 ```json ... ``` 的情况）
        text = resp.content.strip()
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        rankings = json.loads(text).get("rankings", [])
        rankings.sort(key=lambda x: x.get("score", 0), reverse=True)
        return [passages[r["index"]] for r in rankings[:top_n] if r["index"] < len(passages)]
    except (json.JSONDecodeError, KeyError, IndexError):
        return passages[:top_n]


# ── Vector store helpers ──
def _get_vectorstore():
    if not CHROMA_DIR.exists():
        return _rebuild_index()
    return Chroma(persist_directory=str(CHROMA_DIR), embedding_function=embeddings)


def _rebuild_index():
    docs = []
    for pdf_path in DATA_DIR.glob("*.pdf"):
        reader = PdfReader(str(pdf_path))
        text = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
        chunks = splitter.create_documents(
            texts=[text], metadatas=[{"source": pdf_path.name}] * 9999
        )
        docs.extend(chunks)
    return Chroma.from_documents(docs, embeddings, persist_directory=str(CHROMA_DIR))


def process_pdf(file_path: str) -> int:
    reader = PdfReader(file_path)
    text = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
    chunks = splitter.create_documents(
        texts=[text], metadatas=[{"source": os.path.basename(file_path)}] * 9999
    )
    Chroma(persist_directory=str(CHROMA_DIR), embedding_function=embeddings).add_documents(chunks)
    return len(chunks)
# 改进 Query Rewrite：增加 use_rewrite 参数，支持多查询融合检索
# ── Query Rewriting ──
REWRITE_PROMPT = """你是一个搜索引擎优化专家。用户的自然语言问题可能不适合直接做向量检索。

请将以下问题改写为 2-3 个更适合检索的版本。每个版本应该：
1. 提取核心关键词和概念，去掉口语化表达
2. 用更专业、更贴近金融研报术语的方式表达
3. 如果原问题简短或模糊，补充可能的同义表述和上下文

原问题：{question}

严格按以下 JSON 格式输出，不要输出其他内容：
{{"queries": ["改写版本1", "改写版本2", "改写版本3"]}}"""


def _rewrite_query(question: str) -> list[str]:
    """用 LLM 将用户问题改写为多个检索友好的版本"""
    try:
        resp = reranker_llm.invoke(REWRITE_PROMPT.format(question=question))
        text = resp.content.strip()
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        queries = json.loads(text).get("queries", [question])
    except (json.JSONDecodeError, KeyError, AttributeError):
        return [question]

    # 原问题排第一位，再去重
    seen = {question}
    result = [question]
    for q in queries:
        q = q.strip()
        if q and q not in seen:
            seen.add(q)
            result.append(q)
    return result


# ── HyDE（Hypothetical Document Embeddings）──
HYDE_PROMPT = """你是一个金融研究分析师。请针对以下问题，写一段约200-300字的回答，风格类似于专业的金融研究报告。即使你不确定具体数据，也请基于你的金融知识给出合理的分析和判断。不需要面面俱到，抓住问题的核心给出分析即可。

问题：{question}

研究报告风格的回答："""


def _hyde_generate(question: str) -> str:
    """生成假设文档，用于 HyDE 检索——假答案向量比问题向量更接近真实文档"""
    try:
        resp = reranker_llm.invoke(HYDE_PROMPT.format(question=question))
        return resp.content.strip()
    except Exception:
        return ""


# 主函数
def query(question: str, k: int = 4, use_rerank: bool = True,
          use_rewrite: bool = False, use_hyde: bool = False) -> tuple[str, list[dict]]:
    """检索 + 可选 rerank + 可选 query rewrite + 可选 HyDE

    use_rewrite: 多查询改写融合，解决术语鸿沟（+2s）
    use_hyde:    假设文档嵌入，解决极短/模糊问题（+2s）
    两者可叠加，会合并所有查询的检索结果
    """
    vs = _get_vectorstore()
    fetch_k = k * 3 if use_rerank else k

    # ── 构建搜索查询列表 ──
    search_queries = [question]  # 原始问题始终参与检索

    if use_rewrite:
        rewritten = _rewrite_query(question)
        for q in rewritten:
            if q != question and q not in search_queries:
                search_queries.append(q)

    if use_hyde:
        hyde_answer = _hyde_generate(question)
        if hyde_answer and hyde_answer not in search_queries:
            search_queries.append(hyde_answer)

    # ── 检索 ──
    if len(search_queries) > 1:
        # 多查询融合：均分检索配额，合并去重
        per_query_k = max(fetch_k // len(search_queries), k)
        all_passages = []
        seen_contents = set()

        for q in search_queries:
            docs = vs.similarity_search(q, k=per_query_k)
            for d in docs:
                # 前 200 字符去重（兼顾精度和 chunk 边界差异）
                key = d.page_content[:200].strip()
                if key not in seen_contents:
                    seen_contents.add(key)
                    all_passages.append({
                        "source": d.metadata.get("source", ""),
                        "content": d.page_content,
                    })
    else:
        docs = vs.similarity_search(question, k=fetch_k)
        all_passages = [
            {"source": d.metadata.get("source", ""), "content": d.page_content}
            for d in docs
        ]

    if use_rerank and len(all_passages) > k:
        all_passages = _rerank(question, all_passages, top_n=k)

    context = "\n\n---\n\n".join(
        f"[来源: {p['source']}]\n{p['content']}" for p in all_passages
    )
    return context, all_passages


def get_stats() -> dict:
    if not CHROMA_DIR.exists():
        return {"indexed_pdfs": 0, "total_chunks": 0}
    vs = Chroma(persist_directory=str(CHROMA_DIR), embedding_function=embeddings)
    collection = vs.get()
    pdfs = set(m.get("source") for m in collection.get("metadatas", []) if m.get("source"))
    return {
        "indexed_pdfs": len(pdfs),
        "pdf_names": sorted(pdfs),
        "total_chunks": len(collection.get("ids", [])),
    }
