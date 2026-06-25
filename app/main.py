import os
import shutil
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, SystemMessage
from app.rag import process_pdf, query as rag_query, get_stats, DATA_DIR
from app.graph import analyze_pdf, print_graph_ascii
from app.config import get_llm, get_available_models
from app.llm_harness import call_llm
from app.observability import setup_logging, RequestIDMiddleware

setup_logging()

app = FastAPI(title="金融研报多 Agent 分析系统", version="0.1.0")

app.add_middleware(RequestIDMiddleware)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Schema ──
class ChatRequest(BaseModel):
    message: str
    model: str = "deepseek-v4-flash"

class ChatResponse(BaseModel):
    reply: str

class AskRequest(BaseModel):
    question: str
    use_rag: bool = True
    use_rerank: bool = True
    use_rewrite: bool = False
    use_hyde: bool = False
    model: str = "deepseek-v4-flash"

class AskResponse(BaseModel):
    answer: str
    sources: list[dict] = []

# ── Routes ──
@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
        llm = get_llm(req.model)
    except ValueError:
        available = ", ".join(get_available_models())
        raise HTTPException(status_code=400, detail=f"不支持的模型: {req.model}，可用模型: {available}")
    messages = [
        SystemMessage(content="你是一个金融分析助手，回答简洁专业。"),
        HumanMessage(content=req.message),
    ]
    result = call_llm(llm, messages, caller="chat")
    if not result.ok:
        raise HTTPException(status_code=503, detail=f"LLM 服务暂时不可用: {result.error}")
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

class AnalyzeRequest(BaseModel):
    filename: str

class AnalyzeResponse(BaseModel):
    pdf_filename: str
    extracted_data: dict = {}
    risk_flags: list = []
    risk_level: str = ""
    final_report: str = ""


# ── 多 Agent 分析接口 ──
@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    result = analyze_pdf(req.filename)
    if "error" in result:
        raise HTTPException(400, detail=result["error"])
    return result


@app.get("/api/graph")
async def show_graph():
    """返回 LangGraph 可视化 ASCII 图"""
    return {"graph_ascii": print_graph_ascii()}


# RAG 查询接口
@app.post("/api/askRAG", response_model=AskResponse)
async def ask_rag(req: AskRequest):
    try:
        llm = get_llm(req.model)
    except ValueError:
        raise HTTPException(400, detail=f"不支持的模型: {req.model}")

    sources = []
    if req.use_rag:
        context, sources = rag_query(req.question, use_rerank=req.use_rerank, use_rewrite=req.use_rewrite, use_hyde=req.use_hyde)
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
    result = call_llm(llm, messages, caller="ask_rag")
    if not result.ok:
        raise HTTPException(status_code=503, detail=f"LLM 服务暂时不可用: {result.error}")
    return AskResponse(answer=result.content, sources=sources)


@app.get("/api/stats")
async def stats():
    return get_stats()


@app.get("/api/health")
async def health():
    issues = []
    # ChromaDB 连通性检测
    try:
        stats = get_stats()
    except Exception as e:
        issues.append(f"chromadb: {e}")
    # API key 有效性检测（不调用，仅检查是否配置）
    for model_name in get_available_models():
        try:
            llm = get_llm(model_name)
            key = getattr(llm, "openai_api_key", "")
            if not key or "placeholder" in str(key):
                issues.append(f"{model_name}: API key 未配置")
        except Exception as e:
            issues.append(f"{model_name}: {e}")
    return {
        "status": "degraded" if issues else "ok",
        "issues": issues if issues else None,
        "models": get_available_models(),
    }
