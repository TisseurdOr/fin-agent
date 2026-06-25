"""LLM 执行 Harness——重试、日志、统一错误返回。

每个 call_llm() 调用：
  - 最多 max_retries+1 次尝试，指数退避 + 随机抖动
  - 成功 → INFO 日志 + LLMResult(content=...)
  - 失败 → ERROR 日志 + LLMResult(error=...)  ——永不抛异常
"""
import time
import random
import logging
from dataclasses import dataclass
from typing import Optional, Union

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage

from app.observability import get_request_id

logger = logging.getLogger(__name__)


@dataclass
class LLMResult:
    """LLM 调用结果——成功或失败。调用方检查 .ok 决定后续行为。"""
    content: Optional[str] = None
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None


def call_llm(
    llm: ChatOpenAI,
    prompt: Union[str, list[BaseMessage]],
    *,
    max_retries: int = 3,
    caller: str = "unknown",
) -> LLMResult:
    """调用 LLM，带指数退避重试和结构化日志。

    Args:
        llm: ChatOpenAI 实例（从 app.config.get_llm 获取）
        prompt: 字符串 prompt 或 LangChain 消息列表
        max_retries: 最大重试次数（总尝试次数 = max_retries + 1）
        caller: 调用方标识（如 "analyst" / "risk" / "rerank"），用于日志

    Returns:
        LLMResult——成功时 .content 非空，失败时 .error 非空。永不抛异常。
    """
    last_error = None
    request_id = get_request_id()

    for attempt in range(max_retries + 1):
        try:
            t0 = time.perf_counter()
            resp = llm.invoke(prompt)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            content = resp.content.strip() if resp.content else ""
            logger.info(
                "LLM call ok | request_id=%s caller=%s model=%s duration_ms=%.0f content_len=%d attempt=%d/%d",
                request_id or "-",
                caller,
                getattr(llm, "model_name", "?"),
                elapsed_ms,
                len(content),
                attempt + 1,
                max_retries + 1,
            )
            return LLMResult(content=content)

        except Exception as e:
            last_error = e
            remaining = max_retries - attempt

            if remaining > 0:
                backoff = (2 ** attempt) + random.uniform(0, 1)
                logger.warning(
                    "LLM call retry | request_id=%s caller=%s attempt=%d/%d error=%s backoff=%.1fs",
                    request_id or "-",
                    caller,
                    attempt + 1,
                    max_retries + 1,
                    e,
                    backoff,
                )
                time.sleep(backoff)
            else:
                logger.error(
                    "LLM call exhausted | request_id=%s caller=%s attempts=%d final_error=%s",
                    request_id or "-",
                    caller,
                    max_retries + 1,
                    e,
                )

    return LLMResult(error=str(last_error))
