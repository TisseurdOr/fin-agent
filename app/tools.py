"""
金融 Tool 定义 — 供 LangGraph Agent 通过 Function Calling 调用

同一套 tool 后续通过 MCP Server 暴露给外部客户端（Cursor / Claude Code）
"""
import json
import logging
from typing import Optional
import requests

logger = logging.getLogger(__name__)

# ── 东方财富实时行情 API ──
EASTMONEY_URL = "http://push2.eastmoney.com/api/qt/stock/get"


def get_stock_quote(symbol: str) -> dict:
    """获取 A 股实时行情。

    Args:
        symbol: 股票代码，如 '600036'（招商银行）、'002230'（科大讯飞）

    Returns:
        dict: {symbol, name, price, change_pct, high, low, volume, pe}
    """
    # 判断交易所：6开头=上证(1)，0/3开头=深证(0)
    market = "1" if symbol.startswith("6") else "0"
    secid = f"{market}.{symbol}"

    try:
        resp = requests.get(
            EASTMONEY_URL,
            params={"secid": secid, "fields": "f57,f58,f43,f44,f45,f46,f47,f48,f170,f169,f9"},
            timeout=5,
        )
        resp.raise_for_status()
        data = resp.json().get("data", {})

        if not data:
            return {"error": f"未找到股票: {symbol}"}

        return {
            "symbol": data.get("f57", symbol),
            "name": data.get("f58", ""),
            "price": _div100(data.get("f43")),           # 最新价（分→元）
            "high": _div100(data.get("f44")),             # 最高
            "low": _div100(data.get("f45")),              # 最低
            "open": _div100(data.get("f46")),             # 开盘
            "volume": data.get("f47", 0),                 # 成交量（手）
            "turnover": data.get("f48", 0),               # 成交额
            "change_pct": _div100(data.get("f170")),      # 涨跌幅 %
            "change_amount": _div100(data.get("f169")),   # 涨跌额
            "pe_dynamic": _div100(data.get("f9")),        # 动态市盈率
        }
    except requests.RequestException as e:
        logger.warning("get_stock_quote failed: %s", e)
        return {"error": f"行情服务不可用: {str(e)}"}


def get_industry_pe(sector: str) -> dict:
    """获取 A 股行业平均市盈率。

    Args:
        sector: 行业名称，如 '银行'、'半导体'、'白酒'、'新能源'

    Returns:
        dict: {sector, pe_avg, sample_count, pe_data}
    """
    # 行业 PE 参考值（基于市场公开数据，季度更新）
    industry_benchmarks = {
        "银行": 5.5, "保险": 8.0, "证券": 18.0,
        "白酒": 28.0, "食品饮料": 25.0,
        "医药": 32.0, "医疗器械": 35.0,
        "半导体": 55.0, "芯片": 55.0, "人工智能": 50.0,
        "软件": 45.0, "新能源": 30.0, "光伏": 25.0,
        "锂电池": 28.0, "汽车": 20.0,
        "房地产": 8.0, "建筑": 10.0, "钢铁": 12.0,
        "煤炭": 8.0, "电力": 15.0,
        "通信": 22.0, "传媒": 25.0, "计算机": 45.0,
        "电子": 35.0, "家电": 15.0, "军工": 50.0,
    }

    # 获取全市场 PE（akshare 的 V8 依赖在 LangGraph 执行上下文中不稳定，走 subprocess 隔离）
    overall_pe = None
    data_date = None
    try:
        import subprocess, sys
        script = (
            "import akshare as ak, json; "
            "df = ak.stock_market_pe_lg(symbol='上证'); "
            "row = df.iloc[-1]; "
            "print(json.dumps({'pe': float(row['平均市盈率']), 'date': str(row['日期'])}))"
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode == 0:
            data = json.loads(result.stdout.strip())
            overall_pe = data["pe"]
            data_date = data["date"]
    except Exception as e:
        logger.warning("akshare subprocess failed: %s", e)

    if overall_pe is None:
        overall_pe = 16.9  # 上证平均 PE 最新参考值
        data_date = "N/A"

    # 模糊匹配行业名
    pe_value = None
    matched = None
    for name, pe in industry_benchmarks.items():
        if name in sector or sector in name:
            pe_value = pe
            matched = name
            break

    if pe_value is None:
        pe_value = overall_pe
        matched = "全市场均值"

    return {
        "sector": sector,
        "matched_sector": matched,
        "pe_avg": pe_value,
        "market_pe_overall": overall_pe,
        "data_date": data_date,
        "note": "行业PE为基于公开数据的参考值，个股PE请以实时行情为准",
    }


def _div100(val) -> Optional[float]:
    """东方财富 API 数值除 100（价格/百分比用整数传输）"""
    if val is None or val == "-":
        return None
    try:
        return round(float(val) / 100, 2)
    except (ValueError, TypeError):
        return None


# ── LangChain Tool 封装 ──
from langchain_core.tools import tool as lc_tool

@lc_tool
def stock_quote_tool(symbol: str) -> dict:
    """获取 A 股实时行情。symbol 为股票代码，如 '600036'（招商银行）、'002230'（科大讯飞）"""
    return get_stock_quote(symbol)


@lc_tool
def industry_pe_tool(sector: str) -> dict:
    """获取 A 股行业平均市盈率。sector 为行业名称，如 '银行'、'半导体'、'白酒'"""
    return get_industry_pe(sector)


# LangGraph ToolNode 使用的工具列表
TOOLS = [stock_quote_tool, industry_pe_tool]
