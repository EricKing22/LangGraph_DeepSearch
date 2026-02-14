from typing import TypedDict, List, Dict, Annotated
from langgraph.graph import MessagesState
import operator


class Source(TypedDict):
    """Information for each source"""

    title: str  # Source title
    url: str  # Source URL
    content: str  # Source content summary or full text


class LearningState(TypedDict):
    """
    Generic state for learning subgraph - can be reused across different graphs.
    Only contains the minimal fields needed for learning.
    """

    query: str  # The task/query being learned from
    plan_a: str  # Initial plan (before human modification)
    plan_b: str  # Final plan (after human modification)
    human_feedback: str | None  # Human's feedback on the plan
    lesson_learned: str | None  # Extracted lesson from comparison


class RecallState(MessagesState):
    """
    Generic state for recall functionality - can be reused across different graphs.
    """

    query: str  # The task/query to search memory for
    recalled_notes: List[str]  # Notes retrieved from memory store


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
    break_questions_iterations_count: (
        int  # Number of iterations for query decomposition
    )
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

    # Closed-loop Learning System fields
    recalled_notes: List[str]  # Notes retrieved from memory store
    plan_a: str  # Agent's initial plan before human feedback
    plan_b: str  # Human-modified plan (final plan used for execution)
    lesson_learned: str | None  # Distilled lesson from plan comparison


class Plan(MessagesState):
    query: str  # User's original query
    questions: List[str]  # List of decomposed queries
    break_questions_iterations_count: (
        int  # Number of iterations for query decomposition
    )
    human_feedback: str | None  # Human feedback
    score: int | None  # Overall score for the summary
    strengths: str | None  # Overall positive feedback
    weaknesses: str | None  # Overall negative feedback
    summarise_iterations: int  # Number of iterations for review and feedback

    # Closed-loop Learning System fields
    recalled_notes: List[str]  # Notes retrieved from memory store
    plan_a: str  # Agent's initial plan before human feedback
    plan_b: str  # Human-modified plan (final plan used for execution)


class Review(MessagesState):
    query: str  # User's original query
    sources: Annotated[
        List[Source], operator.add
    ]  # Source information used to generate search results
    summary: str  # The summary generated from search results
    score: int | None  # Overall score for the summary
    strengths: str | None  # Overall positive feedback
    weaknesses: str | None  # Overall negative feedback
