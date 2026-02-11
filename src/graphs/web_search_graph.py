# Import config first to ensure logging is configured

from langgraph.graph import StateGraph, START, END
from src.state import WebSearchState
from src.nodes.question_nodes import (
    plan,
    summarise,
    should_break_query,
    human_feedback,
    should_skip_human_feedback,
    is_review_finished,
    is_summarise_finished,
)
from src.nodes.search_nodes import search_web
from src.nodes.review_nodes import review
from langgraph.checkpoint.memory import MemorySaver


# Build the graph
builder = StateGraph(state_schema=WebSearchState)
builder.add_node("plan", plan)
builder.add_node("search_web", search_web)
builder.add_node("summarise", summarise)
builder.add_node("human_feedback", human_feedback)
builder.add_node("review", review)

builder.add_edge(START, "plan")
builder.add_conditional_edges(
    "plan", should_skip_human_feedback, ["human_feedback", "search_web"]
)
builder.add_conditional_edges(
    "human_feedback", should_break_query, ["plan", "search_web"]
)

builder.add_edge("search_web", "summarise")
# Summarise node can lead to review or END based on is_finished router (if review isn't turned off)
builder.add_conditional_edges("summarise", is_summarise_finished, [END, "review"])
# Review node can lead to END or back to plan/summarise based on is_review_finished router and iteration count
builder.add_conditional_edges("review", is_review_finished, [END, "plan", "summarise"])

memory = MemorySaver()
# Compile - LangGraph API provides automatic persistence
graph = builder.compile(checkpointer=memory, interrupt_before=["human_feedback"])
