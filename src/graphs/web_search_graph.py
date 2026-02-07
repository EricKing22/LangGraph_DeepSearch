from langgraph.graph import StateGraph, START, END
from state import Question
from nodes import create_questions, search_web, summarise, should_break_query


# Build the graph
builder = StateGraph(state_schema=Question)

# Add nodes
builder.add_node("create_questions", create_questions)
builder.add_node("search_web", search_web)
builder.add_node("summarise", summarise)

# Add edges
# Start: decide whether to break query or search directly
builder.add_conditional_edges(
    START, should_break_query, ["search_web", "create_questions"]
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
