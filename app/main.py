import os
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

app = FastAPI(title="金融研报多 Agent 分析系统", version="0.1.0")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── LLM ──
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY", "sk-placeholder"),
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    temperature=0.3,
)

# ── Schema ──
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str

# ── Routes ──
@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    messages = [
        SystemMessage(content="你是一个金融分析助手，回答简洁专业。"),
        HumanMessage(content=req.message),
    ]
    result = llm.invoke(messages)
    return ChatResponse(reply=result.content)


@app.get("/api/health")
async def health():
    return {"status": "ok", "model": "deepseek-chat"}
