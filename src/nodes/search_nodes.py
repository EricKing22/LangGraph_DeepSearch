from pydantic import BaseModel, Field
from src.state import Search
from typing import Dict
from src.llm import question_llm as llm
from langchain.messages import SystemMessage, AIMessage
from src.tools.search_tool import search_tavily_impl, search_tavily, get_date
from src.prompts import RELEVANCE_CHECK_PROMPT
from langgraph.prebuilt import ToolNode
import logging

logger = logging.getLogger("LangGraph_DeepSearch.search_nodes")


async def judge_relevance(query: str, search_result: Dict[str, str]) -> bool:
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
        decision = await structured_llm.ainvoke([SystemMessage(content=prompt)])
        logger.debug(
            f"Relevance check for '{title[:50]}...': {decision.is_relevant} - {decision.reason}"
        )
        return decision.is_relevant
    except Exception as e:
        logger.debug(f"Error judging relevance: {str(e)}")
        # Conservative handling when error occurs, keep the result
        return True


async def search_web(state: Search):
    """
    Execute Tavily search for the query and use LLM to filter irrelevant results.
    LLM can decide to use search tools or other tools as needed.
    """
    query = state.get("query")
    search_results = []

    try:
        # Define tools and create ToolNode
        tools = [search_tavily, get_date]
        tool_node = ToolNode(tools)

        # Try to use LLM with tools (if supported)
        results = []
        try:
            llm_with_tools = llm.bind_tools(tools)

            # Invoke LLM with tools
            ai_message = await llm_with_tools.ainvoke(
                [
                    SystemMessage(
                        content=f"Search for information about: {query}\nUse the search_tavily tool to find relevant information."
                    )
                ]
            )

            # Extract results from tool calls using ToolNode
            if hasattr(ai_message, "tool_calls") and ai_message.tool_calls:
                # ToolNode automatically executes all tool calls and returns ToolMessages
                node_result = await tool_node.ainvoke({"messages": [ai_message]})

                # Extract search results from ToolMessages
                for message in node_result.get("messages", []):
                    # ToolMessage.content contains the tool's return value
                    if hasattr(message, "name") and message.name == "search_tavily":
                        tool_result = message.content
                        # search_tavily_impl returns a list of dicts
                        if isinstance(tool_result, list):
                            results.extend(tool_result)
                        elif isinstance(tool_result, str):
                            # If it's a string, it might be an error or serialized result
                            try:
                                import json

                                parsed = json.loads(tool_result)
                                if isinstance(parsed, list):
                                    results.extend(parsed)
                                else:
                                    results.append(parsed)
                            except (json.JSONDecodeError, TypeError, ValueError):
                                # If can't parse, treat as single result
                                results.append({"content": tool_result})
                        else:
                            results.append(tool_result)
        except Exception as tool_error:
            # If LLM doesn't support tools or bind_tools fails, log and continue to fallback
            logger.debug(f"Tool calling not supported or failed: {str(tool_error)}")

        # Fallback: if LLM didn't call search tool or tool calling failed, call it directly
        if not results:
            logger.debug(f"Using direct search implementation for query: {query}")
            results = search_tavily_impl(query=query)

        # Use judge_relevance to filter results
        filtered_results = []
        for result in results:
            # Ensure result is a dict with expected structure
            if isinstance(result, dict):
                if await judge_relevance(query, result):
                    filtered_results.append(result)

        logger.debug(
            f"For query {query}, keeping {len(filtered_results)} out of {len(results)} results\n"
        )

        search_results.append({"question": query, "results": filtered_results})
    except Exception as e:
        logger.error(f"Search Failed '{query}': {str(e)}")
        search_results.append({"question": query, "results": [], "error": str(e)})

    # Track search action with summary of what was searched
    search_summary = f"Search for: **{query}** (Found {len(search_results[0].get('results', []))} relevant results)"
    return {
        "messages": [AIMessage(content=search_summary)],
        "search_results": search_results,
        "sources": [
            item
            for result in search_results
            if "results" in result
            for item in result["results"]
        ],
    }
