from pydantic import BaseModel, Field
from langchain.messages import SystemMessage, AIMessage
from src.state import LearningState, RecallState
from src.llm import report_llm  # Use same model as summarise
from src.tools.consult_note import recall_notes, save_lesson
from src.prompts import WRITE_NOTES_PROMPT
from langgraph.config import get_store
from langgraph.graph import END
import logging

logger = logging.getLogger("LangGraph_DeepSearch.learning_nodes")


async def recall_from_memory(state: RecallState):
    """
    Search Memory (Store Search)
    Agent searches the LangGraph Store for relevant past experiences.
    Returns recalled notes to help with planning.

    This is a generic function that can be reused in any graph that needs
    to recall from memory.
    """
    query = state.get("query", "")
    if not query:
        # Try to extract from messages
        from langchain.messages import HumanMessage

        for message in reversed(state.get("messages", [])):
            if isinstance(message, HumanMessage):
                query = message.content
                break

    store = get_store()
    recalled_notes = await recall_notes(store, query, limit=3)

    notes_summary = ""
    if recalled_notes:
        notes_summary = "\n".join(f"- {note}" for note in recalled_notes)
        message_content = f"Recalled past experiences from memory:\n{notes_summary}"
    else:
        message_content = (
            "No relevant past experiences found. Planning based on current task."
        )

    logger.debug(f"Recalled {len(recalled_notes)} notes for query: {query[:50]}...")

    return {
        "query": query,
        "recalled_notes": recalled_notes,
        "messages": [AIMessage(content=message_content)],
    }


async def compare_and_learn(state: LearningState):
    """
    Compare plans, distill lesson, and save to memory.
    This runs asynchronously after summarise completes.
    Uses report_llm (same as summarise) for lesson extraction.

    This is a generic function that can be reused in any graph that needs
    to learn from plan comparisons. It only requires:
    - query: The task being learned from
    - plan_a: Initial plan
    - plan_b: Final plan
    - human_feedback: (optional) User's feedback
    """
    plan_a = state.get("plan_a", "")
    plan_b = state.get("plan_b", "")
    query = state.get("query", "")
    human_feedback = state.get("human_feedback", "")

    # If no plans to compare, skip learning
    if not plan_a or not plan_b:
        logger.debug("No plans to compare, skipping learning phase")
        return {"lesson_learned": None}

    # If plans are identical, no lesson to learn
    if plan_a == plan_b:
        logger.debug("Plans are identical, no lesson to learn")
        return {"lesson_learned": None}

    # Use report_llm (same as summarise) to analyze the difference and extract lesson
    class LessonExtraction(BaseModel):
        has_lesson: bool = Field(
            description="Whether there is a meaningful lesson to learn from the difference"
        )
        lesson: str = Field(
            description="A concise, actionable lesson learned from the plan comparison"
        )
        reasoning: str = Field(
            description="Explanation of why this lesson is important"
        )

    structured_llm = report_llm.with_structured_output(LessonExtraction)

    prompt = WRITE_NOTES_PROMPT.format(
        query=query,
        human_feedback=human_feedback if human_feedback else "No feedback provided",
        plan_a=plan_a,
        plan_b=plan_b,
    )

    try:
        result = await structured_llm.ainvoke([SystemMessage(content=prompt)])

        if result.has_lesson and result.lesson:
            # Save the lesson to store
            store = get_store()
            await save_lesson(store, result.lesson, query)

            logger.info(f"[ASYNC LEARNING] Learned new lesson: {result.lesson}")

            return {
                "lesson_learned": result.lesson,
            }
        else:
            logger.debug("[ASYNC LEARNING] No meaningful lesson to record")
            return {"lesson_learned": None}

    except Exception as e:
        logger.error(f"Error during async learning phase: {str(e)}")
        return {"lesson_learned": None}


def should_learn(state: LearningState):
    """
    Conditional edge to decide whether to enter the learning phase.
    Only learn if we have both Plan A and Plan B with differences.

    This is a generic helper function that can be reused in any graph.
    """
    plan_a = state.get("plan_a", "")
    plan_b = state.get("plan_b", "")

    if plan_a and plan_b and plan_a != plan_b:
        return "compare_and_learn"
    else:
        return END
