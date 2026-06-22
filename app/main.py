import os
import shutil
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from app.rag import process_pdf, query as rag_query, get_stats, DATA_DIR

app = FastAPI(title="金融研报多 Agent 分析系统", version="0.1.0")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── 模型工厂 ──
MODELS = {
    "deepseek-chat": ChatOpenAI(
        model="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY", "sk-placeholder"),
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        temperature=0.3,
    ),
    "qwen-plus": ChatOpenAI(
        model="qwen-plus",
        api_key=os.getenv("DASHSCOPE_API_KEY", "sk-placeholder"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature=0.3,
    ),
}

# ── Schema ──
class ChatRequest(BaseModel):
    message: str
    model: str = "deepseek-chat"

class ChatResponse(BaseModel):
    reply: str

class AskRequest(BaseModel):
    question: str
    use_rag: bool = True
    use_rerank: bool = True
    model: str = "deepseek-chat"

class AskResponse(BaseModel):
    answer: str
    sources: list[dict] = []

# ── Routes ──
@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    llm = MODELS.get(req.model)
    if not llm:
        available = ", ".join(MODELS.keys())
        raise HTTPException(status_code=400, detail=f"不支持的模型: {req.model}，可用模型: {available}")
    messages = [
        SystemMessage(content="你是一个金融分析助手，回答简洁专业。"),
        HumanMessage(content=req.message),
    ]
    result = llm.invoke(messages)
    return ChatResponse(reply=result.content)


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(400, "只支持 PDF 文件")
    file_path = DATA_DIR / file.filename
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    chunk_count = process_pdf(str(file_path))
    return {"filename": file.filename, "chunks": chunk_count, "status": "ok"}


@app.post("/api/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    llm = MODELS.get(req.model)
    if not llm:
        raise HTTPException(400, detail=f"不支持的模型: {req.model}")

    sources = []
    if req.use_rag:
        context, sources = rag_query(req.question, use_rerank=req.use_rerank)
        system_prompt = (
            "你是一个金融分析助手。请严格基于以下参考文档内容回答问题，"
            "不要编造文档中没有的信息。如果文档中没有相关信息，请明确说明。\n\n"
            f"参考文档：\n{context}"
        )
    else:
        system_prompt = "你是一个金融分析助手，回答简洁专业。"

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=req.question),
    ]
    result = llm.invoke(messages)
    return AskResponse(answer=result.content, sources=sources)


@app.get("/api/stats")
async def stats():
    return get_stats()


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "models": list(MODELS.keys()),
    }
