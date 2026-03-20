"""
File-based persistent memory store for DeepSearch.

Lessons are stored in ~/.deepsearch/memories.json and persist across all
CLI sessions. The LLM accesses them via two progressive-disclosure tools:

  search_memories(query)   → brief previews + IDs (index)
  get_memory(id)           → full lesson text (detail)

This replaces the InMemoryStore / LangGraph Store approach so that the CLI
does not lose learned lessons when the process exits.
"""

from __future__ import annotations

import json
import uuid
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict

from langchain_core.tools import tool
from pydantic import BaseModel, Field

logger = logging.getLogger("LangGraph_DeepSearch.memory_store")

# ── Storage location ──────────────────────────────────────────────────────────
MEMORY_DIR = Path.home() / ".deepsearch"
MEMORY_FILE = MEMORY_DIR / "memories.json"


# ── Low-level file I/O ────────────────────────────────────────────────────────


def _load() -> List[Dict]:
    if not MEMORY_FILE.exists():
        return []
    try:
        with open(MEMORY_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.error(f"Failed to load memory file: {e}")
        return []


def _save(memories: List[Dict]) -> None:
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(memories, f, indent=2, ensure_ascii=False)
    except OSError as e:
        logger.error(f"Failed to save memory file: {e}")


# ── Public write API ──────────────────────────────────────────────────────────


def save_memory(lesson: str, task_query: str) -> str:
    """
    Persist a lesson to the file store.

    Returns:
        The short memory ID assigned to the lesson.
    """
    if not lesson or not lesson.strip():
        return ""
    memories = _load()
    memory_id = str(uuid.uuid4())[:8]
    memories.append(
        {
            "id": memory_id,
            "lesson": lesson.strip(),
            "task_query": task_query,
            "timestamp": datetime.now().isoformat(),
        }
    )
    _save(memories)
    logger.info(f"Saved memory [{memory_id}]: {lesson[:80]}...")
    return memory_id


# ── Search helpers ────────────────────────────────────────────────────────────


def _keyword_score(query: str, memory: Dict) -> int:
    """Simple keyword overlap score (no embedding server required)."""
    query_words = set(query.lower().split())
    text = (memory.get("lesson", "") + " " + memory.get("task_query", "")).lower()
    return sum(1 for w in query_words if w in text)


def search_memories_impl(query: str, limit: int = 8) -> List[Dict]:
    """
    Return brief memory entries relevant to *query*.

    Each entry contains only a short preview (≤150 chars) so the LLM can
    decide which IDs to read in full via get_memory_impl.
    """
    memories = _load()
    if not memories:
        return []

    scored = [(m, _keyword_score(query, m)) for m in memories]
    scored.sort(key=lambda x: x[1], reverse=True)

    # Keep entries with at least one keyword hit; fallback to most recent
    hits = [m for m, s in scored if s > 0]
    if not hits:
        hits = memories[-limit:]  # most recent as fallback

    return [
        {
            "id": m["id"],
            "date": m["timestamp"][:10],
            "task": m["task_query"][:60],
            "preview": m["lesson"][:150],
        }
        for m in hits[:limit]
    ]


def get_memory_impl(memory_id: str) -> str:
    """Return the full lesson text for *memory_id*, or an error string."""
    for m in _load():
        if m["id"] == memory_id:
            return (
                f"[Memory {m['id']} — {m['timestamp'][:10]}]\n"
                f"Original task: {m['task_query']}\n"
                f"Lesson: {m['lesson']}"
            )
    return f"No memory found with ID '{memory_id}'."


def list_all_memories() -> List[Dict]:
    """Return all stored memories (for --show-memory CLI command)."""
    return _load()


# ── LangChain tools (progressive-disclosure interface) ────────────────────────


class SearchMemoriesInput(BaseModel):
    query: str = Field(description="Topic or task description to search memory for")


@tool(args_schema=SearchMemoriesInput)
def search_memories(query: str) -> str:
    """
    Search your long-term memory for past experiences relevant to the current task.

    Returns a short index of matching memories (ID + date + brief preview).
    To read a full lesson, call get_memory(id) with the ID shown here.

    Call this early in planning when you want to benefit from past experience.
    """
    results = search_memories_impl(query)
    if not results:
        return "No relevant past experiences found in memory."

    lines = [
        f"[{r['id']}] {r['date']} | Task: {r['task']} | Preview: {r['preview']}..."
        for r in results
    ]
    header = f"Found {len(results)} relevant past experience(s). Use get_memory(id) for full details:\n"
    return header + "\n".join(lines)


class GetMemoryInput(BaseModel):
    memory_id: str = Field(
        description="The memory ID (8-char string) from search_memories"
    )


@tool(args_schema=GetMemoryInput)
def get_memory(memory_id: str) -> str:
    """
    Retrieve the full lesson text for a specific past experience by its ID.

    Use this after search_memories to read the complete detail of a memory
    you identified as relevant to the current task.
    """
    return get_memory_impl(memory_id)
