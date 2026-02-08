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


class WebSearchState(MessagesState):
    query: str  # User's original query
    questions: List[str]  # List of decomposed queries
    break_questions_iterations_count: int  # Number of iterations for query decomposition
    human_feedback: str | None  # Human feedback
    search_results: Annotated[List[Dict[str, str]], operator.add]  # Search materials
    sources: Annotated[
        List[Source], operator.add
    ]  # Source information used to generate search results
    summary: str  # Summary of search results
    score: int | None  # Overall score for the summary
    strengths: str | None  # Overall positive feedback
    weaknesses: str | None  # Overall negative feedback
    summarise_iterations: int  # Number of iterations for review and feedback


class Plan(MessagesState):
    query: str  # User's original query
    questions: List[str]  # List of decomposed queries
    break_questions_iterations_count: int  # Number of iterations for query decomposition
    human_feedback: str | None  # Human feedback
    score: int | None  # Overall score for the summary
    strengths: str | None  # Overall positive feedback
    weaknesses: str | None  # Overall negative feedback
    summarise_iterations: int  # Number of iterations for review and feedback


class Review(MessagesState):
    query: str  # User's original query
    summary: str  # The summary generated from search results
    score: int | None  # Overall score for the summary
    strengths: str | None  # Overall positive feedback
    weaknesses: str | None  # Overall negative feedback
