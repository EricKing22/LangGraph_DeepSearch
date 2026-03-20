"""
arXiv academic paper search tool.
Uses the arXiv public API to find research papers.
"""

from typing import List, Dict
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from src import config
import logging

logger = logging.getLogger("LangGraph_DeepSearch.arxiv_tool")

_ARXIV_API_URL = "https://export.arxiv.org/api/query"


def search_arxiv_impl(
    query: str,
    max_results: int = 5,
) -> List[Dict[str, str]]:
    """
    Search arXiv for academic papers using the public API.

    Args:
        query: Search query string
        max_results: Maximum number of papers to return

    Returns:
        List of dicts with keys: title, url, content
    """
    try:
        import requests
        import xml.etree.ElementTree as ET

        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }

        resp = requests.get(
            _ARXIV_API_URL, params=params, timeout=config.SEARCH_TIMEOUT
        )
        resp.raise_for_status()

        ns = {
            "atom": "http://www.w3.org/2005/Atom",
            "arxiv": "http://arxiv.org/schemas/atom",
        }

        root = ET.fromstring(resp.text)
        results: List[Dict[str, str]] = []

        for entry in root.findall("atom:entry", ns):
            title_el = entry.find("atom:title", ns)
            summary_el = entry.find("atom:summary", ns)
            id_el = entry.find("atom:id", ns)

            title = (
                title_el.text.strip().replace("\n", " ")
                if title_el is not None
                else "N/A"
            )
            abstract = (
                summary_el.text.strip().replace("\n", " ")[:600]
                if summary_el is not None
                else ""
            )
            paper_url = id_el.text.strip() if id_el is not None else ""

            # Derive clean arXiv ID for the URL
            if "arxiv.org/abs/" in paper_url:
                arxiv_id = paper_url.split("arxiv.org/abs/")[-1]
            else:
                arxiv_id = paper_url

            authors = [
                a.find("atom:name", ns).text
                for a in entry.findall("atom:author", ns)
                if a.find("atom:name", ns) is not None
            ][:3]
            authors_str = ", ".join(authors) or "N/A"

            published_el = entry.find("atom:published", ns)
            published = published_el.text[:10] if published_el is not None else "N/A"

            # Primary category
            cat_el = entry.find("arxiv:primary_category", ns)
            category = cat_el.get("term", "N/A") if cat_el is not None else "N/A"

            content = (
                f"Title: {title}\n"
                f"Authors: {authors_str}\n"
                f"Published: {published}\n"
                f"Category: {category}\n"
                f"Abstract: {abstract}"
            )

            results.append(
                {
                    "title": f"[arXiv] {title}",
                    "url": f"https://arxiv.org/abs/{arxiv_id}",
                    "content": content,
                }
            )

        logger.debug(f"arXiv search '{query}': {len(results)} results")
        return results

    except Exception as e:
        logger.error(f"arXiv search error for '{query}': {e}")
        return []


class ArxivSearchInput(BaseModel):
    query: str = Field(description="Search query for arXiv papers")
    max_results: int = Field(
        default=5, description="Maximum number of papers to return"
    )


@tool(args_schema=ArxivSearchInput)
def search_arxiv(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """
    Search arXiv for academic research papers.
    Use this when the query relates to academic research, scientific topics, or
    cutting-edge AI/ML papers.

    Args:
        query: Search query
        max_results: Maximum number of papers to return

    Returns:
        List of papers with title, url, and content (abstract + metadata)
    """
    return search_arxiv_impl(query, max_results)
