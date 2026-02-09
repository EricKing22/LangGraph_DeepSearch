"""
Tavily search tool module
"""

from typing import List, Dict, Any, Literal
from langchain_tavily import TavilySearch  # updated 1.0
import config

SearchDepth = Literal["basic", "advanced", "fast", "ultra-fast"]


class TavilySearchTool:
    """Tavily search tool wrapper class"""

    def __init__(self, api_key: str, max_results: int):
        """
        Initialize Tavily search tool

        Args:
            api_key: Tavily API Key, read from config if not provided
            max_results: Default maximum number of results to return
        """
        self.api_key = api_key or config.TAVILY_API_KEY
        if not self.api_key:
            raise ValueError(
                "Tavily API key is required. Please set TAVILY_API_KEY in .env"
            )

        self.max_results = max_results or config.MAX_SEARCH_RESULTS
        self.client = TavilySearch(max_results=self.max_results, api_key=self.api_key)

    def search(
        self,
        query: str,
        max_results: int,
        search_depth: SearchDepth = "advanced",
        include_answer: bool = True,
        include_raw_content: bool = False,
        include_images: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute search query

        Args:
            query: Search query string
            max_results: Maximum number of results to return
            search_depth: Search depth ("basic" or "advanced")
            include_answer: Whether to include AI-generated answer
            include_raw_content: Whether to include raw webpage content
            include_images: Whether to include related images

        Returns:
            Dictionary containing search results
        """
        max_results = max_results or self.max_results

        try:
            # TavilySearch.invoke() returns result list
            raw_results = self.client.invoke(
                query,
                max_results=max_results,
                search_depth=search_depth,
                include_answer=include_answer,
                include_raw_content=include_raw_content,
                include_images=include_images,
            )

            # Wrap into format compatible with original TavilyClient
            # Handle different return formats
            if isinstance(raw_results, dict):
                # If already in dictionary format, return directly
                if "results" in raw_results:
                    return raw_results
                # If single result, wrap into list
                return {"query": query, "results": [raw_results]}
            elif isinstance(raw_results, list):
                # If list, wrap into standard format
                return {"query": query, "results": raw_results}
            else:
                # If string or other format, wrap into single result
                return {"query": query, "results": [{"content": str(raw_results)}]}

        except Exception as e:
            raise RuntimeError(f"Tavily search failed: {str(e)}")

    def extract_results(self, response: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Extract structured results from Tavily response

        Args:
            response: Tavily API response

        Returns:
            List of structured search results
        """
        results = []

        for item in response.get("results", []):
            results.append(
                {
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "content": item.get("content", ""),
                }
            )

        return results


# Create global instance (if API key is available)
tavily_search = None
if config.TAVILY_API_KEY:
    try:
        tavily_search = TavilySearchTool(
            api_key=config.TAVILY_API_KEY, max_results=config.MAX_SEARCH_RESULTS
        )
    except Exception as e:
        print(f"Warning: Failed to initialize Tavily search tool: {e}")


def search_tavily(
    query: str, max_results: int = config.MAX_SEARCH_RESULTS
) -> List[Dict[str, str]]:
    """
    Convenience search function

    Args:
        query: Search query
        max_results: Maximum number of results

    Returns:
        List of search results
    """
    if not tavily_search:
        raise RuntimeError("Tavily search tool is not initialized")

    response = tavily_search.search(query, max_results=max_results)
    return tavily_search.extract_results(response)
