from langgraph.graph import StateGraph, START, END
from src.state import WebSearchState
from src.nodes.question_nodes import (
    plan,
    summarise,
    should_break_query,
    human_feedback,
    should_skip_human_feedback,
    is_review_finished,
    after_summarise_router,
)
from src.nodes.search_nodes import search_web
from src.nodes.review_nodes import review
from src.graphs.learn_graph import learn_graph
import logging

logger = logging.getLogger("LangGraph_DeepSearch.web_search_graph")

# Build the graph with Closed-loop Learning System
# Memory recall is now done as a skill (tool call) inside the plan node itself,
# using progressive disclosure: search_memories() → get_memory(id).
# Flow: plan -> human_feedback -> search -> summarise [→ async learn] -> review
builder = StateGraph(state_schema=WebSearchState)

# Phase 1: Planning (memory recall embedded as tool calls inside this node)
builder.add_node("plan", plan)

# Phase 2: Human-in-the-loop nodes
builder.add_node("human_feedback", human_feedback)

# Phase 3: Execution nodes
builder.add_node("search_web", search_web)
builder.add_node("summarise", summarise)
builder.add_node("review", review)

# Phase 4: Async learning subgraph
builder.add_node("learn", learn_graph)

# Edge Definitions

# START: always go directly to plan (memory access is a tool inside plan)
builder.add_edge(START, "plan")

# From plan: decide whether to get human feedback or skip
builder.add_conditional_edges(
    "plan", should_skip_human_feedback, ["human_feedback", "search_web"]
)

# Phase 2: Human feedback -> decide next step
builder.add_conditional_edges(
    "human_feedback", should_break_query, ["plan", "search_web"]
)

# Execution phase
builder.add_edge("search_web", "summarise")

# From summarise: use new router (may Send to learn async + continue to review/END)
builder.add_conditional_edges(
    "summarise", after_summarise_router, ["learn", "review", END]
)

# From review: decide whether to loop back or finish
builder.add_conditional_edges("review", is_review_finished, ["plan", "summarise", END])

# Learn subgraph always goes to END (it runs async)
builder.add_edge("learn", END)

# Compile without checkpointer/store — langgraph dev provides these at runtime.
# For CLI usage, compile via builder with checkpointer/store separately.
graph = builder.compile()