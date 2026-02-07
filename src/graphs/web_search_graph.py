from langgraph.graph import StateGraph, START, END
from state import Question
from nodes.question_nodes import (
    create_questions,
    summarise,
    should_break_query,
    extract_query,
)
from nodes.search_nodes import search_web


# Build the graph
builder = StateGraph(state_schema=Question)

# Add nodes
builder.add_node("extract_query", extract_query)
builder.add_node("create_questions", create_questions)
builder.add_node("search_web", search_web)
builder.add_node("summarise", summarise)

# Add edges
builder.add_edge(START, "extract_query")
# Start: decide whether to break query or search directly
builder.add_conditional_edges(
    "extract_query", should_break_query, ["search_web", "create_questions"]
)

# After creating questions: decide whether to improve questions or proceed to search
builder.add_conditional_edges(
    "create_questions", should_break_query, ["create_questions", "search_web"]
)

# After search: proceed to summarization
builder.add_edge("search_web", "summarise")

# After summarization: end the workflow
builder.add_edge("summarise", END)

# Compile the graph - LangGraph Cloud will provide checkpointer
graph = builder.compile()
