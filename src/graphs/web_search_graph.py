from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
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
from src.nodes.learning_nodes import recall_from_memory
from src.graphs.learn_graph import learn_graph
from src import config
import logging

logger = logging.getLogger("LangGraph_DeepSearch.web_search_graph")


def should_start_with_recall(state: WebSearchState):
    """
    Conditional edge from START: decide whether to recall from memory first.
    If ENABLE_LEARNING is True, start with recall to get past experiences.
    Otherwise, skip directly to "plan".
    """
    if config.ENABLE_LEARNING:
        logger.debug("Learning enabled: Starting with recall from memory")
        return "recall"
    else:
        logger.debug("Learning disabled: Skipping directly to plan")
        return "plan"


# Build the graph with Closed-loop Learning System
# Flow: [recall] -> plan -> human_feedback -> search -> summarise [→ async learn] -> review
builder = StateGraph(state_schema=WebSearchState)

# Phase 1: Recall node (beginning only)
builder.add_node("recall", recall_from_memory)

# Phase 2: Planning nodes
builder.add_node("plan", plan)

# Phase 3: Human-in-the-loop nodes
builder.add_node("human_feedback", human_feedback)

# Phase 4: Execution nodes
builder.add_node("search_web", search_web)
builder.add_node("summarise", summarise)
builder.add_node("review", review)

# Phase 5: Async learning subgraph
builder.add_node("learn", learn_graph)

# Edge Definitions

# START: conditional based on ENABLE_LEARNING
builder.add_conditional_edges(START, should_start_with_recall, ["recall", "plan"])

# After recall: always go to plan
builder.add_edge("recall", "plan")

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

# Compile
checkpointer = MemorySaver()
store = InMemoryStore()
graph = builder.compile(
    checkpointer=checkpointer, store=store, interrupt_before=["human_feedback"]
)
