from langchain_core.tools import tool
from langgraph.store.base import BaseStore
from typing import List, Optional
import logging

logger = logging.getLogger("LangGraph_DeepSearch.consult_note")

# Store namespace for lessons learned
LESSON_NAMESPACE = ("lessons",)


async def recall_notes(store: BaseStore, query: str, limit: int = 3) -> List[str]:
    """
    Search the lesson store for relevant past experiences.
    Used in the "Recall" phase of the closed-loop learning system.

    Args:
        store: LangGraph Store instance
        query: Search query or task description
        limit: Maximum number of notes to retrieve

    Returns:
        List of relevant lesson strings
    """
    if store is None:
        logger.warning("No store provided, returning empty notes")
        return []

    try:
        # Search for similar lessons using semantic search
        results = await store.asearch(LESSON_NAMESPACE, query=query, limit=limit)
        notes = [item.value.get("lesson", "") for item in results if item.value]
        logger.debug(f"Recalled {len(notes)} notes for query: {query[:50]}...")
        return notes
    except Exception as e:
        logger.error(f"Error recalling notes: {str(e)}")
        return []


async def save_lesson(
    store: BaseStore, lesson: str, task_query: str, lesson_id: Optional[str] = None
) -> bool:
    """
    Save a distilled lesson to the store.
    Used in the "Memorize" phase of the closed-loop learning system.

    Args:
        store: LangGraph Store instance
        lesson: The distilled lesson/experience to save
        task_query: The original task query (for context)
        lesson_id: Optional specific ID for the lesson

    Returns:
        True if successful, False otherwise
    """
    if store is None:
        logger.warning("No store provided, cannot save lesson")
        return False

    if not lesson or not lesson.strip():
        logger.warning("Empty lesson provided, skipping save")
        return False

    try:
        import uuid

        key = lesson_id or str(uuid.uuid4())
        await store.aput(
            LESSON_NAMESPACE,
            key,
            {
                "lesson": lesson,
                "task_query": task_query,
                "timestamp": str(__import__("datetime").datetime.now()),
            },
        )
        logger.info(f"Saved lesson: {lesson[:100]}...")
        return True
    except Exception as e:
        logger.error(f"Error saving lesson: {str(e)}")
        return False


@tool
def consult_notebook(query: str, limit: int = 3):
    """
    Consult your personal notebook or past lessons learned.
    Use this tool when you need inspiration, are stuck, or want to reference past experiences.

    Args:
        query: Search keywords or semantic description.
        limit: Number of notes to return (default: 3).
    """
    return f"Searching notes for: '{query}' (limit {limit})...\nPlease use the graph's built-in store for actual operations."
