"""
LangGraph 多 Agent 金融研报分析系统

3 个 Agent 节点组成的线性流水线：
  分析师 Agent → 风控 Agent → 报告 Agent

每个节点由 DeepSeek 驱动，通过不同的 system prompt 实现角色分工。
State 在节点间传递，逐步丰富：raw_text → extracted_data → risk_flags → final_report
"""
import json
import logging
from pathlib import Path
from typing import TypedDict, Optional
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage
from PyPDF2 import PdfReader

from app.config import get_llm
from app.llm_harness import call_llm

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"

# ── State 定义 ──
class AnalysisState(TypedDict):
    pdf_filename: str          # 输入：PDF 文件名
    raw_text: str              # PDF 原始文本
    extracted_data: Optional[dict]   # 分析师输出
    risk_flags: Optional[list]       # 风控输出
    risk_level: Optional[str]        # low / medium / high
    final_report: Optional[str]      # 报告输出


# ── Node 1: 分析师 Agent ──
ANALYST_PROMPT = """你是一个资深金融分析师。请从以下研报内容中提取关键数据，输出严格的 JSON。

提取要求：
1. 核心指标：营收/市场规模、利润、增速（百分比）、市盈率/估值等
2. 行业数据：行业增长率、市场份额、政策影响
3. 公司数据：提及的上市公司名称、股票代码、关键财务数据
4. 趋势判断：行业趋势关键词、投资建议方向
5. 如果某些字段在原文档中没有明确数据，填 null，不要编造

输出 JSON 格式：
{
  "company_name": "公司名",
  "stock_code": "股票代码",
  "key_metrics": {"营收": "数值", "利润": "数值", "增速": "百分比"},
  "industry_data": {"行业规模": "数值", "增长率": "百分比"},
  "trends": ["趋势关键词"],
  "investment_advice": "建议方向或 null",
  "data_quality": "high/medium/low（数据完整度评估）"
}"""


def analyst_node(state: AnalysisState) -> dict:
    """提取研报关键数据"""
    text = state["raw_text"]
    # 截断过长文本（DeepSeek 上下文足够，但控制成本）
    truncated = text[:8000] if len(text) > 8000 else text

    result = call_llm(
        get_llm("deepseek-v4-flash"),
        [SystemMessage(content=ANALYST_PROMPT),
         HumanMessage(content=f"研报内容：\n{truncated}")],
        caller="analyst",
    )

    if not result.ok:
        return {"extracted_data": {"error": f"LLM调用失败: {result.error}"}}

    try:
        content = result.content
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        data = json.loads(content)
    except json.JSONDecodeError:
        data = {"error": "解析失败", "raw_output": result.content[:500]}

    return {"extracted_data": data}


# ── Node 2: 风控 Agent ──
RISK_PROMPT = """你是一个金融风控专家。请检查以下分析师提取的数据，从多个维度评估风险。

检查维度：
1. 数据合理性：增速是否过于极端（>100%或<-50%）？估值是否显著偏离行业均值？
2. 数据一致性：不同指标之间是否存在逻辑矛盾？
3. 缺失风险：关键数据缺失（null）是否影响投资决策？
4. 行业风险：行业趋势是否提及政策风险、竞争加剧、技术替代等？
5. 集中度风险：是否过度依赖单一客户/产品/市场？

输出 JSON 格式：
{
  "risk_flags": [
    {"维度": "数据合理性", "等级": "高/中/低", "描述": "具体问题"},
    {"维度": "数据缺失", "等级": "高/中/低", "描述": "缺失的关键数据"}
  ],
  "risk_level": "high/medium/low",
  "risk_summary": "一句话风险总结",
  "suggestions": ["建议1", "建议2"]
}"""


def risk_node(state: AnalysisState) -> dict:
    """检查数据合理性，标注风险"""
    data = state.get("extracted_data", {})
    if not data or "error" in data:
        return {
            "risk_flags": [{"维度": "数据质量", "等级": "高", "描述": "上游分析师未能提取有效数据"}],
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
            "risk_flags": [{"维度": "系统", "等级": "高", "描述": f"LLM调用失败: {result.error}"}],
            "risk_level": "high",
        }

    try:
        content = result.content
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        parsed = json.loads(content)
    except json.JSONDecodeError:
        parsed = {"risk_flags": [], "risk_level": "unknown", "risk_summary": "风控分析解析失败"}

    return {
        "risk_flags": parsed.get("risk_flags", []),
        "risk_level": parsed.get("risk_level", "low"),
    }


# ── Node 3: 报告 Agent ──
REPORT_PROMPT = """你是一个金融报告撰写专家。请根据分析师提取的数据和风控标注的风险，生成一份结构化的分析摘要。

格式要求（Markdown）：
## 研报分析摘要

### 核心数据
- 用表格列出关键指标和数值

### 风险提示
- 按风险等级排列，高风险项标 ⚠️

### 投资要点
- 3-5 个要点，基于数据而非猜测

### 综合评级
- 基于数据质量和风险等级给出：推荐关注 / 中性观望 / 谨慎回避

如果数据质量低（data_quality: low），在报告开头明确声明数据局限性。"""


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
        [SystemMessage(content=REPORT_PROMPT),
         HumanMessage(content=input_text)],
        caller="report",
    )

    if not result.ok:
        return {"final_report": f"报告生成失败: {result.error}"}

    return {"final_report": result.content}


# ── 构建 Graph ──
builder = StateGraph(AnalysisState)

builder.add_node("analyst", analyst_node)
builder.add_node("risk", risk_node)
builder.add_node("report", report_node)

# 线性流水线：analyst → risk → report
builder.add_edge(START, "analyst")
builder.add_edge("analyst", "risk")
builder.add_edge("risk", "report")
builder.add_edge("report", END)

graph = builder.compile()


# ── 对外接口 ──
def analyze_pdf(filename: str) -> dict:
    """对指定 PDF 执行多 Agent 分析，返回完整结果"""
    file_path = DATA_DIR / filename
    if not file_path.exists():
        available = [f.name for f in DATA_DIR.glob("*.pdf")]
        return {"error": f"文件不存在: {filename}", "available_pdfs": available}

    # 读取 PDF 文本
    reader = PdfReader(str(file_path))
    raw_text = "\n".join(
        page.extract_text() for page in reader.pages if page.extract_text()
    )

    if not raw_text.strip():
        return {"error": "PDF 无法提取文本内容"}

    logger.info("analyze_pdf start | file=%s text_len=%d", filename, len(raw_text))

    # 运行多 Agent 流水线
    result = graph.invoke({
        "pdf_filename": filename,
        "raw_text": raw_text,
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
