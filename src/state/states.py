from typing import TypedDict, List, Dict, Annotated
from langgraph.graph import MessagesState
import operator


class Source(TypedDict):
    """Information for each source"""

    title: str  # Source title
    url: str  # Source URL
    content: str  # Source content summary or full text


class Search(MessagesState):
    query: str  # Search query
    search_results: Annotated[
        List[Dict[str, str]], operator.add
    ]  # List of search results
    sources: Annotated[
        List[Source], operator.add
    ]  # Source information used to generate search results


class Question(MessagesState):
    query: str  # User's original query
    questions: List[str]  # List of decomposed queries
    next_step_reason: str | None  # Reason for decomposition
    break_questions_iterations_count: int  # Number of iterations for query decomposition
    human_feedback: str | None  # Human feedback
    search_results: Annotated[List[Dict[str, str]], operator.add]  # Search materials
    sources: Annotated[
        List[Source], operator.add
    ]  # Source information used to generate search results
    summary: str  # Summary of search results


class Paragraph(TypedDict):
    """State for each paragraph in the report"""

    title: str  # Paragraph title
    content: str  # Paragraph content
    sources: Annotated[
        List[Source], operator.add
    ]  # Source information used to generate this paragraph


class AgentState(TypedDict):
    """State for the entire report"""

    query: str  # Original query
    report_title: str  # Report title
    paragraphs: List[Paragraph]
    final_report: str  # Final report content
    is_completed: bool  # Whether completed
