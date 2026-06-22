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
import requests
import json

DATA_DIR = Path(__file__).parent.parent / "data"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"

# ── Embedding ──
class DashScopeEmbeddings(Embeddings):
    def __init__(self, api_key: str, model: str = "text-embedding-v2"):
        self.api_key = api_key
        self.model = model
        self.url = "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding"

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        all_embeddings = []
        for i in range(0, len(texts), 25):
            batch = texts[i : i + 25]
            resp = requests.post(
                self.url,
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json={"model": self.model, "input": {"texts": batch}},
            )
            resp.raise_for_status()
            data = resp.json()
            all_embeddings.extend(e["embedding"] for e in data["output"]["embeddings"])
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


embeddings = DashScopeEmbeddings(
    api_key=os.getenv("DASHSCOPE_API_KEY", "sk-placeholder"),
)

# ── Chunker（改进参数） ──
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,       # 500→800，中文约400字，保全文段
    chunk_overlap=150,    # 50→150，防止关键信息在边界被切断
    separators=["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";", " "],
)

# ── LLM Reranker ──
reranker_llm = ChatOpenAI(
    model="deepseek-chat",
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


def query(question: str, k: int = 4, use_rerank: bool = True) -> tuple[str, list[dict]]:
    """检索 + 可选 rerank"""
    vs = _get_vectorstore()
    # 先召回 k*3 条候选，给 reranker 足够空间
    fetch_k = k * 3 if use_rerank else k
    docs = vs.similarity_search(question, k=fetch_k)

    passages = [
        {"source": d.metadata.get("source", ""), "content": d.page_content}
        for d in docs
    ]

    if use_rerank and len(passages) > k:
        passages = _rerank(question, passages, top_n=k)

    context = "\n\n---\n\n".join(
        f"[来源: {p['source']}]\n{p['content']}" for p in passages
    )
    return context, passages


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
