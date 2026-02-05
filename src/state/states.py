from typing import TypedDict, List, Dict, Any, Annotated
from langgraph.graph import MessagesState
import operator


class Source(TypedDict):
    """每个来源的信息"""

    title: str  # 来源标题
    url: str  # 来源URL
    content: str  # 来源内容摘要或全文


class Question(MessagesState):
    """每次搜索的状态"""

    query: str  # 用户输入的原始查询
    questions: List[str]  # 分解后的查询列表
    human_feedback: str | None  # 人工反馈
    context: List[Dict[str, Any]]  # 素材
    sources: Annotated[List[Source], operator.add]  # 用于生成搜索结果的来源信息
    summary: str  # 搜索结果的总结


class Paragraph(TypedDict):
    """报告中的每个段落的状态"""

    title: str  # 段落标题
    content: str  # 段落内容
    sources: Annotated[List[Source], operator.add]  # 用于生成该段落的来源信息


class AgentState(TypedDict):
    """整个报告的状态"""

    query: str  # 原始查询
    report_title: str  # 报告标题
    paragraphs: List[Paragraph]
    final_report: str  # 最终报告内容
    is_completed: bool  # 是否完成
