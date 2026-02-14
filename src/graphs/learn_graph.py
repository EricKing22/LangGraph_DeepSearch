from langgraph.graph import StateGraph, START, END
from src.state import LearningState
from src.nodes.learning_nodes import compare_and_learn
import logging

logger = logging.getLogger("LangGraph_DeepSearch.learn_graph")


# Build the learn subgraph (async learning only)
# This subgraph is invoked via Send after summarise completes
learn_builder = StateGraph(state_schema=LearningState)

# Single node: compare and learn (async after summarise)
learn_builder.add_node("compare_and_learn", compare_and_learn)

# Simple flow: START -> compare_and_learn -> END
learn_builder.add_edge(START, "compare_and_learn")
learn_builder.add_edge("compare_and_learn", END)

# Compile the learn subgraph
learn_graph = learn_builder.compile()
