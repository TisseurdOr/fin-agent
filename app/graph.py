"""
LangGraph 多 Agent 金融研报分析系统

3 个 Agent 节点 + Tool Calling：
  START → analyst ⇄ tools（条件循环）→ risk → report → END

analyst 可调用 get_stock_quote / get_industry_pe 获取实时行情，
风控节点用实时数据校验估值合理性，而非仅靠 prompt 推测。
"""
import json
import logging
from pathlib import Path
from typing import TypedDict, Optional, Annotated
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from PyPDF2 import PdfReader

from app.config import get_llm
from app.llm_harness import call_llm
from app.tools import TOOLS

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"

# ── State 定义（增加 messages 用于 tool calling 循环）──
class AnalysisState(TypedDict):
    pdf_filename: str
    raw_text: str
    # tool calling 轮次的消息历史（analyst + tool 结果）
    messages: Annotated[list[BaseMessage], add_messages]
    extracted_data: Optional[dict]
    risk_flags: Optional[list]
    risk_level: Optional[str]
    final_report: Optional[str]


# ── Node 1: 分析师 Agent（带 tool calling）──
ANALYST_PROMPT = """你是一个资深金融分析师。请从研报内容中提取关键数据。

工作流程：
1. 阅读研报内容，找出其中提及的上市公司名称和股票代码
2. 使用 get_stock_quote 工具查询这些公司的实时行情（可一次调用多个）
3. 使用 get_industry_pe 工具查询相关行业的平均市盈率
4. 将研报数据与实时行情对比后，严格输出 JSON（不要输出 markdown 报告，报告由下游节点生成）

如果研报中没有提及具体的 A 股上市公司，直接基于研报内容提取数据即可，不需要调用工具。

最终输出必须是纯 JSON（不要用 ```json 包裹，不要输出 markdown）：
{
  "company_name": "公司名",
  "stock_code": "股票代码",
  "key_metrics": {"营收": "数值", "利润": "数值", "增速": "百分比"},
  "market_data": {"实时股价": "元", "涨跌幅": "%", "动态PE": "倍", "行业PE": "倍"},
  "industry_data": {"行业规模": "数值", "增长率": "百分比"},
  "trends": ["趋势关键词"],
  "investment_advice": "建议方向或 null",
  "data_quality": "high/medium/low"
}"""


def analyst_node(state: AnalysisState) -> dict:
    """提取研报关键数据，主动调用工具查实时行情。

    首次调用：构建 system prompt + 研报内容
    后续调用：继续 messages 历史（含 tool 返回结果），LLM 可再次决定调 tool 或结束
    """
    llm = get_llm("deepseek-v4-flash")
    llm_with_tools = llm.bind_tools(TOOLS)

    existing = state.get("messages", [])
    if not existing:
        # 首轮：构建初始消息
        text = state["raw_text"][:8000]
        messages = [
            SystemMessage(content=ANALYST_PROMPT),
            HumanMessage(content=f"研报内容：\n{text}"),
        ]
    else:
        # 后续轮次：延续完整对话历史
        messages = list(existing)

    result = call_llm(llm_with_tools, messages, caller="analyst")
    if not result.ok:
        return {
            "messages": [AIMessage(content=f"LLM调用失败: {result.error}")],
            "extracted_data": {"error": f"LLM调用失败: {result.error}"},
        }

    ai_msg = AIMessage(content=result.content or "")
    if result.tool_calls:
        ai_msg.tool_calls = result.tool_calls
        logger.info("analyst requested %d tool call(s)", len(result.tool_calls))

    return {"messages": [ai_msg]}


# ── 解析 analyst 最终输出 ──
def parse_analyst_result(state: AnalysisState) -> dict:
    """从 analyst 的最后一条 AIMessage 中提取 JSON 结果（tool calling 结束后的最终回复）"""
    messages = state.get("messages", [])
    last_content = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content and msg.content.strip():
            last_content = msg.content.strip()
            break

    if not last_content:
        return {"extracted_data": {"error": "analyst 未产生有效输出"}}

    data = _extract_json(last_content)

    # JSON 提取失败时，用 LLM json_mode 兜底转换
    if data is None or "error" in data:
        data = _llm_json_extract(last_content)

    return {"extracted_data": data}


def _extract_json(text: str) -> Optional[dict]:
    """尝试多种方式从文本中提取 JSON"""
    # 1. 纯 JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # 2. ```json ... ``` 包裹
    try:
        content = text
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        return json.loads(content.strip())
    except (json.JSONDecodeError, IndexError):
        pass
    # 3. 正则提取含 company_name 的 JSON 对象
    import re
    m = re.search(r'\{[^{}]*"company_name"[^{}]*\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return None


def _llm_json_extract(text: str) -> dict:
    """用 json_mode LLM 从非结构化文本中提取结构化数据"""
    try:
        llm = get_llm("deepseek-v4-flash")
        llm_json = llm.bind(response_format={"type": "json_object"})
        resp = llm_json.invoke([
            SystemMessage(content="""从金融分析文本中提取结构化 JSON。必须包含以下字段：
{
  "company_name": "公司名",
  "stock_code": "股票代码",
  "key_metrics": {"指标名": "数值"},
  "market_data": {"实时股价": "元", "涨跌幅": "%", "动态PE": "倍", "行业PE": "倍"},
  "industry_data": {"指标名": "数值"},
  "trends": ["趋势"],
  "investment_advice": "建议或null",
  "data_quality": "high/medium/low"
}
只输出 JSON，不要任何解释。"""),
            HumanMessage(content=text[:6000]),
        ])
        data = json.loads(resp.content.strip())
        return data
    except Exception as e:
        logger.warning("LLM JSON extraction failed: %s", e)
        return {"error": "JSON解析失败", "raw": text[:500]}


# ── Node 2: 风控 Agent（增加实时数据交叉校验）──
RISK_PROMPT = """你是一个金融风控专家。请检查分析师提取的数据，从多个维度评估风险。

特别关注：
1. 如果 market_data 中有实时股价和行业 PE，对比研报估值是否合理
2. 涨跌幅异常（>9% 或 <-9%）需要标注
3. 个股 PE 与行业 PE 偏差超过 50% 需要标注

检查维度：
1. 数据合理性：增速是否过于极端？估值是否显著偏离行业均值？
2. 数据一致性：不同指标之间是否存在逻辑矛盾？
3. 缺失风险：关键数据缺失（null）是否影响投资决策？
4. 市场风险：实时行情是否有异常波动？
5. 行业风险：行业趋势是否提及政策风险、竞争加剧等？

输出 JSON：
{
  "risk_flags": [{"维度": "..", "等级": "高/中/低", "描述": ".."}],
  "risk_level": "high/medium/low",
  "risk_summary": "一句话总结",
  "suggestions": ["建议1", "建议2"]
}"""


def risk_node(state: AnalysisState) -> dict:
    """检查数据合理性，标注风险（含实时行情交叉验证）"""
    data = state.get("extracted_data", {})
    if not data or "error" in data:
        return {
            "risk_flags": [{"维度": "数据质量", "等级": "高", "描述": "上游未能提取有效数据"}],
            "risk_level": "high",
        }

    result = call_llm(
        get_llm("deepseek-v4-flash"),
        [SystemMessage(content=RISK_PROMPT),
         HumanMessage(content=f"分析师提取数据：\n{json.dumps(data, ensure_ascii=False, indent=2)}")],
        caller="risk",
    )

    if not result.ok:
        return {
            "risk_flags": [{"维度": "系统", "等级": "高", "描述": f"LLM失败: {result.error}"}],
            "risk_level": "high",
        }

    try:
        content = result.content.strip()
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        parsed = json.loads(content)
    except json.JSONDecodeError:
        parsed = {"risk_flags": [], "risk_level": "unknown", "risk_summary": "解析失败"}

    return {
        "risk_flags": parsed.get("risk_flags", []),
        "risk_level": parsed.get("risk_level", "low"),
    }


# ── Node 3: 报告 Agent ──
REPORT_PROMPT = """你是一个金融报告撰写专家。请根据分析数据生成结构化摘要。

格式（Markdown）：
## 研报分析摘要

### 核心数据
- 用表格列出关键指标（含实时行情数据如有的）

### 估值分析（如有实时数据）
- 对比个股 PE 与行业 PE

### 风险提示
- 按风险等级排列，高风险项标 ⚠️

### 投资要点
- 3-5 个要点，基于数据而非猜测

### 综合评级
- 推荐关注 / 中性观望 / 谨慎回避
"""


def report_node(state: AnalysisState) -> dict:
    """汇总生成结构化摘要"""
    data = state.get("extracted_data", {})
    risks = state.get("risk_flags", [])
    risk_level = state.get("risk_level", "unknown")

    input_text = f"""## 提取数据
{json.dumps(data, ensure_ascii=False, indent=2)}

## 风险标注
风险等级：{risk_level}
风险项：{json.dumps(risks, ensure_ascii=False, indent=2)}
"""

    result = call_llm(
        get_llm("deepseek-v4-flash"),
        [SystemMessage(content=REPORT_PROMPT), HumanMessage(content=input_text)],
        caller="report",
    )

    if not result.ok:
        return {"final_report": f"报告生成失败: {result.error}"}

    return {"final_report": result.content}


# ── 条件边路由：检查 analyst 的回复是否需要 tool ──
def route_after_analyst(state: AnalysisState) -> str:
    """检查最后一条消息是否包含 tool_calls"""
    messages = state.get("messages", [])
    if not messages:
        return "parse"
    last_msg = messages[-1]
    if isinstance(last_msg, AIMessage) and getattr(last_msg, "tool_calls", None):
        return "tools"
    return "parse"


# ── 构建 Graph ──
builder = StateGraph(AnalysisState)

tool_node = ToolNode(TOOLS)

builder.add_node("analyst", analyst_node)
builder.add_node("tools", tool_node)
builder.add_node("parse", parse_analyst_result)
builder.add_node("risk", risk_node)
builder.add_node("report", report_node)

# 边：
# START → analyst ⇄ tools（tool calling 循环）
# analyst 无 tool_calls → parse（提取 JSON）→ risk → report → END
builder.add_edge(START, "analyst")
builder.add_conditional_edges("analyst", route_after_analyst, {
    "tools": "tools",
    "parse": "parse",
})
builder.add_edge("tools", "analyst")
builder.add_edge("parse", "risk")
builder.add_edge("risk", "report")
builder.add_edge("report", END)

graph = builder.compile()


# ── 对外接口 ──
def analyze_pdf(filename: str) -> dict:
    """对指定 PDF 执行多 Agent 分析（含实时行情查询），返回完整结果"""
    file_path = DATA_DIR / filename
    if not file_path.exists():
        available = [f.name for f in DATA_DIR.glob("*.pdf")]
        return {"error": f"文件不存在: {filename}", "available_pdfs": available}

    reader = PdfReader(str(file_path))
    raw_text = "\n".join(
        page.extract_text() for page in reader.pages if page.extract_text()
    )

    if not raw_text.strip():
        return {"error": "PDF 无法提取文本内容"}

    logger.info("analyze_pdf start | file=%s text_len=%d", filename, len(raw_text))

    result = graph.invoke({
        "pdf_filename": filename,
        "raw_text": raw_text,
        "messages": [],
        "extracted_data": None,
        "risk_flags": None,
        "risk_level": None,
        "final_report": None,
    })

    logger.info(
        "analyze_pdf done | file=%s risk_level=%s report_len=%d",
        filename,
        result.get("risk_level", "?"),
        len(result.get("final_report", "")),
    )

    return {
        "pdf_filename": result["pdf_filename"],
        "extracted_data": result["extracted_data"],
        "risk_flags": result["risk_flags"],
        "risk_level": result["risk_level"],
        "final_report": result["final_report"],
    }


def print_graph_ascii():
    """打印 Graph 可视化"""
    return graph.get_graph().draw_ascii()
