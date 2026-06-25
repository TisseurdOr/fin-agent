"""
MCP Server — 将金融工具通过 Model Context Protocol 暴露

同一套 tool（stock_quote / industry_pe）既服务 LangGraph 内部节点，
也通过 MCP 对外暴露，可被 Cursor / Claude Code 等客户端复用。

启动方式：
  # stdio 模式（供 Claude Code / Cursor 等 MCP 客户端消费）
  python -m app.mcp_server

  # HTTP/SSE 模式（供远程客户端或调试）
  python -m app.mcp_server --transport sse --port 8001
"""
import argparse
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

from mcp.server.fastmcp import FastMCP

from app.tools import get_stock_quote, get_industry_pe

logger = logging.getLogger(__name__)

mcp = FastMCP(
    "fin-agent",
    instructions="金融研报分析工具集 — 提供 A 股实时行情与行业市盈率查询",
)


@mcp.tool(
    description="获取 A 股实时行情。symbol 为 6 位股票代码，如 '600036'（招商银行）、'002230'（科大讯飞）"
)
def stock_quote(symbol: str) -> dict:
    """获取 A 股实时行情"""
    return get_stock_quote(symbol)


@mcp.tool(
    description="获取 A 股行业平均市盈率。sector 为行业名称，如 '银行'、'半导体'、'白酒'、'新能源'"
)
def industry_pe(sector: str) -> dict:
    """获取 A 股行业平均市盈率"""
    return get_industry_pe(sector)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="fin-agent MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="stdio",
        help="传输协议 (default: stdio)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8001,
        help="SSE 模式端口 (default: 8001)",
    )
    args = parser.parse_args()

    if args.transport == "sse":
        logger.info("MCP Server starting on http://0.0.0.0:%d/sse", args.port)
        mcp.run(transport="sse", host="0.0.0.0", port=args.port)
    else:
        mcp.run()
