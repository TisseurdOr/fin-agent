"""可观测性层——日志配置 + 请求ID中间件"""
import logging
import uuid
from contextvars import ContextVar

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

_request_id_var: ContextVar[str] = ContextVar("request_id", default="")


def setup_logging(level: int = logging.INFO) -> None:
    """配置根 logger。

    生产代码 logger 用 INFO 级别；第三方库（httpx/openai/chromadb）抑制到 WARNING。
    在 app = FastAPI(...) 之前调用一次。
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    # 抑制第三方库噪声
    for noisy in ("httpx", "openai", "httpcore", "chromadb", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """注入/传播 X-Request-ID，存入 ContextVar 供任意调用点使用。

    放在 CORS 中间件之前注册，确保最早拿到 request_id。
    """

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        _request_id_var.set(request_id)
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


def get_request_id() -> str:
    """获取当前请求的 request ID。

    在请求处理链路中任意位置调用，用于日志关联。
    无活跃请求时返回空字符串。
    """
    return _request_id_var.get()
