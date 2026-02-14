from typing import List, Dict, Any, Literal, Union
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from langchain_tavily import TavilySearch
from src import config


class TavilySearchInput(BaseModel):
    query: str = Field(description="Keywords or question to search for")
    max_results: int = Field(
        default=5, description="Maximum number of results to return (default: 5)"
    )
    search_depth: Literal["basic", "advanced"] = Field(
        default="advanced",
        description="Search depth: 'basic' is faster, 'advanced' yields higher quality",
    )


def _extract_results(
    response: Union[Dict[str, Any], List[Dict[str, str]]],
) -> List[Dict[str, str]]:
    """Extract and format search results"""
    results = []

    raw_list = response.get("results", []) if isinstance(response, dict) else response

    for i, item in enumerate(raw_list, 1):
        title = item.get("title", "")
        url = item.get("url", "")
        content = item.get("content", "")
        results.append(
            {
                "title": title,
                "url": url,
                "content": content,
            }
        )

    return results


# Create one instance to improve efficiency
api_key = config.TAVILY_API_KEY
if not api_key:
    print("Warning: TAVILY_API_KEY not found in configuration.")
    client = None
else:
    client = TavilySearch(
        max_results=config.MAX_SEARCH_RESULTS, api_key=config.TAVILY_API_KEY
    )


def search_tavily_impl(
    query: str,
    max_results: int = 5,
    search_depth: Literal["basic", "advanced"] = "advanced",
) -> List[Dict[str, str]]:
    """
    Implementation function for Tavily search.
    This is the actual function that performs the search.
    Use this function directly in your nodes.
    """
    try:
        if not client:
            print("Error: Tavily client not initialized. Check TAVILY_API_KEY.")
            return []

        raw_results = client.invoke(
            query,
            max_results=max_results,
            search_depth=search_depth,
            include_answer=True,
            include_raw_content=False,
            include_images=False,
        )

        if isinstance(raw_results, dict) and "results" in raw_results:
            clean_data = raw_results
        elif isinstance(raw_results, list):
            clean_data = {"query": query, "results": raw_results}
        else:
            clean_data = {"query": query, "results": [{"content": str(raw_results)}]}

        return _extract_results(clean_data)

    except Exception as e:
        print(f"Error performing search: {str(e)}")
        return [
            {"title": "", "url": "", "content": f"Error performing search: {str(e)}"}
        ]


@tool(args_schema=TavilySearchInput)
def search_tavily(
    query: str,
    max_results: int = 5,
    search_depth: Literal["basic", "advanced"] = "advanced",
) -> List[Dict[str, str]]:
    """
    Use Tavily search engine to retrieve up-to-date information from the web.
    This tool is designed to fetch the latest news, facts, and data that may not be present in the LLM's training data.

    Args:
        query: Keywords or question to search for
        max_results: Maximum number of results to return (default: 5)
        search_depth: 'basic' is faster, 'advanced' yields higher quality (default: advanced)

    Returns:
        List of search results with title, url, and content
    """
    # Delegate to implementation function
    return search_tavily_impl(query, max_results, search_depth)


@tool
def get_date():
    """Get current date and time"""
    from datetime import datetime

    return datetime.now().isoformat()
