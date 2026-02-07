from pydantic import BaseModel, Field
from state import Search
from typing import Dict
from llm import ollama_llm as llm
from langchain.messages import SystemMessage, AIMessage
from tools.search_tool import search_tavily
from prompts import RELEVANCE_CHECK_PROMPT


def judge_relevance(query: str, search_result: Dict[str, str]) -> bool:
    """
    Use LLM to judge whether a single search result is relevant to the query

    Args:
        query: User query
        search_result: Single search result dictionary containing title, url, content

    Returns:
        bool: True if relevant, False if not relevant
    """

    class RelevanceDecision(BaseModel):
        is_relevant: bool = Field(
            description="Whether the content is relevant to the query"
        )
        reason: str = Field(description="Brief explanation for the decision")

    structured_llm = llm.with_structured_output(RelevanceDecision)

    title = search_result.get("title", "")
    content = search_result.get("content", "")

    # If content is empty, directly mark as not relevant
    if not content.strip():
        return False

    prompt = RELEVANCE_CHECK_PROMPT.format(
        query=query,
        title=title,
        content=content[:1000],  # Limit content length to avoid exceeding token limit
    )

    try:
        decision = structured_llm.invoke([SystemMessage(content=prompt)])
        print(
            f"Relevance check for '{title[:50]}...': {decision.is_relevant} - {decision.reason}"
        )
        return decision.is_relevant
    except Exception as e:
        print(f"Error judging relevance: {str(e)}")
        # Conservative handling when error occurs, keep the result
        return True


def search_web(state: Search):
    """
    Execute Tavily search for the query and use LLM to filter irrelevant results
    """
    query = state.get("query")
    search_results = []

    try:
        results = search_tavily(query=query)

        # Use judge_relevance to filter results

        filtered_results = []

        for result in results:
            if judge_relevance(query, result):
                filtered_results.append(result)

        print(
            f"For query {query}, keeping {len(filtered_results)} out of {len(results)} results\n"
        )

        search_results.append({"question": query, "results": filtered_results})
    except Exception as e:
        print(f"Search Failed '{query}': {str(e)}")
        search_results.append({"question": query, "results": [], "error": str(e)})

    # Track search action with summary of what was searched
    search_summary = f"Searched for: **{query}** (Found {len(search_results[0].get('results', []))} relevant results)"
    return {
        "messages": [AIMessage(content=search_summary)],
        "search_results": search_results,
        "sources": [result for result in search_results if "results" in result],
    }
