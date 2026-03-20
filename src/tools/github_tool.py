"""
GitHub search tool — searches repositories, code, and issues via the GitHub REST API.

Rate limits:
  - Without token : 10 search requests/minute
  - With token    : 30 search requests/minute, 5,000 requests/hour
Set GITHUB_TOKEN in .env to increase limits and enable code search.
"""

from typing import List, Dict, Literal
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from src import config
import logging

logger = logging.getLogger("LangGraph_DeepSearch.github_tool")

_GITHUB_API = "https://api.github.com"
_VALID_TYPES = ["repos", "code", "issues"]


def _headers() -> dict:
    """Build request headers, including auth token when available."""
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if config.GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {config.GITHUB_TOKEN}"
    return headers


def _search_repos(query: str, limit: int) -> List[Dict[str, str]]:
    import requests

    params = {
        "q": query,
        "sort": "stars",
        "order": "desc",
        "per_page": min(limit, 30),
    }
    try:
        resp = requests.get(
            f"{_GITHUB_API}/search/repositories",
            headers=_headers(),
            params=params,
            timeout=config.SEARCH_TIMEOUT,
        )
        resp.raise_for_status()
        items = resp.json().get("items", [])
    except Exception as e:
        logger.debug(f"GitHub repo search error: {e}")
        return []

    results = []
    for repo in items:
        topics = ", ".join(repo.get("topics", [])[:6]) or "N/A"
        content = (
            f"Repository: {repo['full_name']}\n"
            f"Description: {repo.get('description') or 'N/A'}\n"
            f"Language: {repo.get('language') or 'N/A'}\n"
            f"Stars: {repo.get('stargazers_count', 0):,}  "
            f"Forks: {repo.get('forks_count', 0):,}\n"
            f"Topics: {topics}\n"
            f"Last updated: {repo.get('updated_at', 'N/A')[:10]}"
        )
        results.append(
            {
                "title": f"[GitHub Repo] {repo['full_name']}",
                "url": repo["html_url"],
                "content": content,
            }
        )
    return results


def _search_code(query: str, limit: int) -> List[Dict[str, str]]:
    """Code search requires a GitHub token."""
    if not config.GITHUB_TOKEN:
        logger.debug("GitHub code search skipped: GITHUB_TOKEN not set")
        return []

    import requests

    params = {"q": query, "per_page": min(limit, 30)}
    try:
        resp = requests.get(
            f"{_GITHUB_API}/search/code",
            headers=_headers(),
            params=params,
            timeout=config.SEARCH_TIMEOUT,
        )
        resp.raise_for_status()
        items = resp.json().get("items", [])
    except Exception as e:
        logger.debug(f"GitHub code search error: {e}")
        return []

    results = []
    for item in items:
        repo_name = item.get("repository", {}).get("full_name", "N/A")
        content = (
            f"File: {item.get('path', 'N/A')}\n"
            f"Repository: {repo_name}\n"
            f"SHA: {item.get('sha', 'N/A')[:8]}"
        )
        results.append(
            {
                "title": f"[GitHub Code] {item.get('name', 'N/A')} in {repo_name}",
                "url": item.get("html_url", ""),
                "content": content,
            }
        )
    return results


def _search_issues(query: str, limit: int) -> List[Dict[str, str]]:
    import requests

    params = {
        "q": query,
        "sort": "relevance",
        "order": "desc",
        "per_page": min(limit, 30),
    }
    try:
        resp = requests.get(
            f"{_GITHUB_API}/search/issues",
            headers=_headers(),
            params=params,
            timeout=config.SEARCH_TIMEOUT,
        )
        resp.raise_for_status()
        items = resp.json().get("items", [])
    except Exception as e:
        logger.debug(f"GitHub issues search error: {e}")
        return []

    results = []
    for issue in items:
        labels = ", ".join(l["name"] for l in issue.get("labels", [])[:5]) or "N/A"
        kind = "Pull Request" if "pull_request" in issue else "Issue"
        body_preview = (issue.get("body") or "")[:300].replace("\n", " ").strip()
        content = (
            f"{kind}: {issue.get('title', 'N/A')}\n"
            f"State: {issue.get('state', 'N/A')}\n"
            f"Labels: {labels}\n"
            f"Comments: {issue.get('comments', 0)}\n"
            f"Preview: {body_preview or 'N/A'}"
        )
        results.append(
            {
                "title": f"[GitHub {kind}] {issue.get('title', 'N/A')}",
                "url": issue.get("html_url", ""),
                "content": content,
            }
        )
    return results


def search_github_impl(
    query: str,
    max_results: int = 5,
    search_types: List[str] | None = None,
) -> List[Dict[str, str]]:
    """
    Search GitHub for repositories, code files, and issues/PRs.

    Args:
        query: Search query string
        max_results: Total maximum number of results to return
        search_types: Which content types to search. Defaults to config.GITHUB_SEARCH_TYPES.

    Returns:
        List of dicts with keys: title, url, content
    """
    if search_types is None:
        search_types = config.GITHUB_SEARCH_TYPES

    active_types = [t for t in search_types if t in _VALID_TYPES]
    if not active_types:
        return []

    per_type = max(1, max_results // len(active_types))
    results: List[Dict[str, str]] = []

    if "repos" in active_types:
        results.extend(_search_repos(query, per_type))

    if "code" in active_types:
        results.extend(_search_code(query, per_type))

    if "issues" in active_types:
        results.extend(_search_issues(query, per_type))

    logger.debug(
        f"GitHub search '{query}': {len(results)} results across {active_types}"
    )
    return results[:max_results]


class GitHubSearchInput(BaseModel):
    query: str = Field(description="Search query for GitHub")
    max_results: int = Field(default=5, description="Maximum number of results to return")
    search_types: List[Literal["repos", "code", "issues"]] = Field(
        default=["repos", "issues"],
        description="Types of GitHub content to search: repos, code (requires token), issues",
    )


@tool(args_schema=GitHubSearchInput)
def search_github(
    query: str,
    max_results: int = 5,
    search_types: List[str] = ["repos", "issues"],
) -> List[Dict[str, str]]:
    """
    Search GitHub for repositories, code files, and issues/pull requests.
    Use this when the query involves open-source projects, code implementations,
    software libraries, or developer discussions.

    Args:
        query: Search query
        max_results: Maximum number of results to return
        search_types: Content types — repos, code (needs GITHUB_TOKEN), issues

    Returns:
        List of search results with title, url, and content
    """
    return search_github_impl(query, max_results, search_types)
