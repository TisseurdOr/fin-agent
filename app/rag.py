import os
from pathlib import Path
from typing import List
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import Chroma
import requests

DATA_DIR = Path(__file__).parent.parent / "data"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"


class DashScopeEmbeddings(Embeddings):
    """阿里云 DashScope embedding，兼容 LangChain Embeddings 接口"""

    def __init__(self, api_key: str, model: str = "text-embedding-v2"):
        self.api_key = api_key
        self.model = model
        self.url = "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding"

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # DashScope 一次最多 25 条
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

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";", " "],
)


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


def query(question: str, k: int = 4) -> tuple[str, list[dict]]:
    vs = _get_vectorstore()
    docs = vs.similarity_search(question, k=k)
    context = "\n\n---\n\n".join(
        f"[来源: {d.metadata.get('source', 'unknown')}]\n{d.page_content}" for d in docs
    )
    sources = [
        {"source": d.metadata.get("source", ""), "content": d.page_content[:200]}
        for d in docs
    ]
    return context, sources


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
