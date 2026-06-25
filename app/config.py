"""集中化 LLM 配置——延迟实例化，按需创建"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

from langchain_openai import ChatOpenAI
from openai import OpenAI

# ── 模型注册表 ──
_MODEL_REGISTRY = {
    "deepseek-v4-flash": {
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url_env": "DEEPSEEK_BASE_URL",
        "base_url_default": "https://api.deepseek.com",
    },
    "deepseek-v4-pro": {
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url_env": "DEEPSEEK_BASE_URL",
        "base_url_default": "https://api.deepseek.com",
    },
    "qwen-plus": {
        "api_key_env": "DASHSCOPE_API_KEY",
        "base_url_env": None,
        "base_url_default": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    },
}

_llm_cache: dict[str, ChatOpenAI] = {}
_embedding_client: OpenAI | None = None


def get_llm(model_name: str = "deepseek-v4-flash") -> ChatOpenAI:
    """获取或创建 ChatOpenAI 实例（延迟实例化 + 缓存）。

    重试策略交给 llm_harness.call_llm()——这里只负责配置。
    """
    if model_name not in _MODEL_REGISTRY:
        available = ", ".join(_MODEL_REGISTRY.keys())
        raise ValueError(f"不支持的模型: {model_name}，可用模型: {available}")

    if model_name not in _llm_cache:
        cfg = _MODEL_REGISTRY[model_name]
        api_key = os.getenv(cfg["api_key_env"], "sk-placeholder")
        base_url = os.getenv(cfg["base_url_env"] or "", cfg["base_url_default"])

        _llm_cache[model_name] = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            temperature=0.1,
            request_timeout=60,
            max_retries=0,  # 重试交给 llm_harness
        )

    return _llm_cache[model_name]


def get_embedding_client() -> OpenAI:
    """获取或创建 DashScope embedding 的 OpenAI 客户端（延迟实例化 + 缓存）。"""
    global _embedding_client
    if _embedding_client is None:
        _embedding_client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY", "sk-placeholder"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
    return _embedding_client


def get_available_models() -> list[str]:
    """返回所有已注册模型名称"""
    return list(_MODEL_REGISTRY.keys())
