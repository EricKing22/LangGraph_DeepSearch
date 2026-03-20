"""
HuggingFace Hub search tool - searches models, datasets, spaces, and papers.
"""

from typing import List, Dict, Literal
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from src import config
import logging

logger = logging.getLogger("LangGraph_DeepSearch.huggingface_tool")

_VALID_TYPES = ["models", "datasets", "spaces", "papers"]


def search_huggingface_impl(
    query: str,
    max_results: int = 5,
    search_types: List[str] | None = None,
) -> List[Dict[str, str]]:
    """
    Search HuggingFace Hub for models, datasets, spaces, and papers.

    Args:
        query: Search query string
        max_results: Total maximum number of results to return
        search_types: Which content types to search. Defaults to config.HUGGINGFACE_SEARCH_TYPES.

    Returns:
        List of dicts with keys: title, url, content
    """
    if search_types is None:
        search_types = config.HUGGINGFACE_SEARCH_TYPES

    results: List[Dict[str, str]] = []
    active_types = [t for t in search_types if t in _VALID_TYPES]
    if not active_types:
        return results

    # Allocate result slots per type
    per_type = max(1, max_results // len(active_types))
    token = config.HUGGINGFACE_TOKEN or None

    # ── Models ──────────────────────────────────────────────────────────────
    if "models" in active_types:
        try:
            from huggingface_hub import list_models

            for model in list_models(
                search=query, limit=per_type, sort="downloads", token=token
            ):
                tags = ", ".join((model.tags or [])[:5]) or "N/A"
                results.append(
                    {
                        "title": f"[HF Model] {model.id}",
                        "url": f"https://huggingface.co/{model.id}",
                        "content": (
                            f"Model: {model.id}\n"
                            f"Author: {model.author or 'Unknown'}\n"
                            f"Tags: {tags}\n"
                            f"Downloads: {model.downloads or 0:,}\n"
                            f"Likes: {model.likes or 0}"
                        ),
                    }
                )
        except Exception as e:
            logger.debug(f"HuggingFace model search error: {e}")

    # ── Datasets ─────────────────────────────────────────────────────────────
    if "datasets" in active_types:
        try:
            from huggingface_hub import list_datasets

            for ds in list_datasets(
                search=query, limit=per_type, sort="downloads", token=token
            ):
                tags = ", ".join((ds.tags or [])[:5]) or "N/A"
                results.append(
                    {
                        "title": f"[HF Dataset] {ds.id}",
                        "url": f"https://huggingface.co/datasets/{ds.id}",
                        "content": (
                            f"Dataset: {ds.id}\n"
                            f"Author: {ds.author or 'Unknown'}\n"
                            f"Tags: {tags}\n"
                            f"Downloads: {ds.downloads or 0:,}\n"
                            f"Likes: {ds.likes or 0}"
                        ),
                    }
                )
        except Exception as e:
            logger.debug(f"HuggingFace dataset search error: {e}")

    # ── Spaces ───────────────────────────────────────────────────────────────
    if "spaces" in active_types:
        try:
            from huggingface_hub import list_spaces

            for space in list_spaces(
                search=query, limit=per_type, sort="likes", token=token
            ):
                tags = ", ".join((space.tags or [])[:5]) or "N/A"
                results.append(
                    {
                        "title": f"[HF Space] {space.id}",
                        "url": f"https://huggingface.co/spaces/{space.id}",
                        "content": (
                            f"Space: {space.id}\n"
                            f"Author: {space.author or 'Unknown'}\n"
                            f"SDK: {space.sdk or 'N/A'}\n"
                            f"Tags: {tags}\n"
                            f"Likes: {space.likes or 0}"
                        ),
                    }
                )
        except Exception as e:
            logger.debug(f"HuggingFace spaces search error: {e}")

    # ── Papers ───────────────────────────────────────────────────────────────
    if "papers" in active_types:
        try:
            import requests

            resp = requests.get(
                "https://huggingface.co/api/papers",
                params={"q": query},
                timeout=config.SEARCH_TIMEOUT,
            )
            if resp.ok:
                for paper in resp.json()[:per_type]:
                    authors = ", ".join(
                        a.get("name", "") for a in paper.get("authors", [])[:3]
                    )
                    abstract = paper.get("summary", "")[:500]
                    results.append(
                        {
                            "title": f"[HF Paper] {paper.get('title', 'N/A')}",
                            "url": f"https://huggingface.co/papers/{paper.get('id', '')}",
                            "content": (
                                f"Title: {paper.get('title', 'N/A')}\n"
                                f"Authors: {authors or 'N/A'}\n"
                                f"Abstract: {abstract}"
                            ),
                        }
                    )
        except Exception as e:
            logger.debug(f"HuggingFace papers search error: {e}")

    logger.debug(
        f"HuggingFace search '{query}': {len(results)} results across {active_types}"
    )
    return results[:max_results]


class HuggingFaceSearchInput(BaseModel):
    query: str = Field(description="Search query for HuggingFace Hub")
    max_results: int = Field(
        default=5, description="Maximum number of results to return"
    )
    search_types: List[Literal["models", "datasets", "spaces", "papers"]] = Field(
        default=["models", "datasets", "spaces", "papers"],
        description="Types of HuggingFace content to search",
    )


@tool(args_schema=HuggingFaceSearchInput)
def search_huggingface(
    query: str,
    max_results: int = 5,
    search_types: List[str] = ["models", "datasets", "spaces", "papers"],
) -> List[Dict[str, str]]:
    """
    Search HuggingFace Hub for ML models, datasets, interactive spaces, and research papers.
    Use this when the query relates to machine learning models, AI research, datasets, or demos.

    Args:
        query: Search query
        max_results: Maximum number of results to return
        search_types: Content types to search (models, datasets, spaces, papers)

    Returns:
        List of search results with title, url, and content
    """
    return search_huggingface_impl(query, max_results, search_types)
