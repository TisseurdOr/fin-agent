import os
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

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
    model: str = "deepseek-chat"  # 默认使用 DeepSeek，可切换为 qwen-plus

class ChatResponse(BaseModel):
    reply: str

# ── Routes ──
@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    llm = MODELS.get(req.model)
    if not llm:
        available = ", ".join(MODELS.keys())
        raise HTTPException(
            status_code=400,
            detail=f"不支持的模型: {req.model}，可用模型: {available}",
        )
    messages = [
        SystemMessage(content="你是一个金融分析助手，回答简洁专业。"),
        HumanMessage(content=req.message),
    ]
    result = llm.invoke(messages)
    return ChatResponse(reply=result.content)


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "models": list(MODELS.keys()),
    }
