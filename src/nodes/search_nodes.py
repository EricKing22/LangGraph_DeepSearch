from pydantic import BaseModel, Field
from src.state import Search
from typing import Dict
from src.llm import question_llm as llm
from langchain.messages import SystemMessage, AIMessage
from src.tools.search_tool import search_tavily_impl, search_tavily, get_date
from src.tools.huggingface_tool import search_huggingface_impl
from src.tools.arxiv_tool import search_arxiv_impl
from src.tools.github_tool import search_github_impl
from src.prompts import RELEVANCE_CHECK_PROMPT
from langgraph.prebuilt import ToolNode
from langchain_core.runnables import RunnableConfig
from src import config as app_config
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


def _get_search_sources(config: RunnableConfig) -> list[str]:
    """
    Determine active search sources from configurable overrides or app config defaults.

    Priority:
      1. 'search_sources' key in configurable (set by CLI --sources flag)
      2. App-level config flags (HUGGINGFACE_SEARCH_ENABLED, etc.)

    Returns a list such as ["web"], ["hf"], or ["web", "hf"].
    """
    configurable = config.get("configurable", {}) if config else {}
    override = configurable.get("search_sources", None)
    if override is not None:
        return list(override)

    # Fall back to environment-level defaults
    sources = ["web"]  # Tavily web search is always the base default
    if app_config.HUGGINGFACE_SEARCH_ENABLED:
        sources.append("hf")
    if app_config.ARXIV_SEARCH_ENABLED:
        sources.append("arxiv")
    if app_config.GITHUB_SEARCH_ENABLED:
        sources.append("github")
    return sources


async def search_web(state: Search, config: RunnableConfig):
    """
    Execute search(es) for the query and use LLM to filter irrelevant results.

    Active search backends are determined by:
      - CLI --sources flag (via RunnableConfig configurable), or
      - environment variables (HUGGINGFACE_SEARCH_ENABLED, ARXIV_SEARCH_ENABLED).

    Supported sources:
      - "web"    : Tavily web search
      - "hf"     : HuggingFace Hub (models, datasets, spaces, papers)
      - "arxiv"  : arXiv academic paper search
      - "github" : GitHub repositories, code, and issues
    """
    query = state.get("query")
    search_sources = _get_search_sources(config)

    logger.debug(f"search_web '{query}' with sources: {search_sources}")

    results = []

    # ── Web search (Tavily) ───────────────────────────────────────────────────
    if "web" in search_sources:
        web_results = []
        try:
            tools = [search_tavily, get_date]
            tool_node = ToolNode(tools)
            llm_with_tools = llm.bind_tools(tools)

            ai_message = await llm_with_tools.ainvoke(
                [
                    SystemMessage(
                        content=f"Search for information about: {query}\nUse the search_tavily tool to find relevant information."
                    )
                ]
            )

            if hasattr(ai_message, "tool_calls") and ai_message.tool_calls:
                node_result = await tool_node.ainvoke({"messages": [ai_message]})
                for message in node_result.get("messages", []):
                    if hasattr(message, "name") and message.name == "search_tavily":
                        tool_result = message.content
                        if isinstance(tool_result, list):
                            web_results.extend(tool_result)
                        elif isinstance(tool_result, str):
                            try:
                                import json

                                parsed = json.loads(tool_result)
                                if isinstance(parsed, list):
                                    web_results.extend(parsed)
                                else:
                                    web_results.append(parsed)
                            except (json.JSONDecodeError, TypeError, ValueError):
                                web_results.append({"content": tool_result})
                        else:
                            web_results.append(tool_result)
        except Exception as tool_error:
            logger.debug(f"Tool calling not supported or failed: {str(tool_error)}")

        # Fallback: direct call if LLM tool calling yielded nothing
        if not web_results:
            logger.debug(f"Using direct Tavily search for query: {query}")
            web_results = search_tavily_impl(query=query)

        results.extend(web_results)

    # ── HuggingFace search ────────────────────────────────────────────────────
    if "hf" in search_sources:
        try:
            hf_results = search_huggingface_impl(
                query=query,
                max_results=app_config.MAX_SEARCH_RESULTS,
            )
            results.extend(hf_results)
            logger.debug(f"HuggingFace search added {len(hf_results)} results")
        except Exception as e:
            logger.error(f"HuggingFace search failed for '{query}': {e}")

    # ── arXiv search ──────────────────────────────────────────────────────────
    if "arxiv" in search_sources:
        try:
            arxiv_results = search_arxiv_impl(
                query=query,
                max_results=app_config.MAX_SEARCH_RESULTS,
            )
            results.extend(arxiv_results)
            logger.debug(f"arXiv search added {len(arxiv_results)} results")
        except Exception as e:
            logger.error(f"arXiv search failed for '{query}': {e}")

    # ── GitHub search ─────────────────────────────────────────────────────────
    if "github" in search_sources:
        try:
            github_results = search_github_impl(
                query=query,
                max_results=app_config.MAX_SEARCH_RESULTS,
            )
            results.extend(github_results)
            logger.debug(f"GitHub search added {len(github_results)} results")
        except Exception as e:
            logger.error(f"GitHub search failed for '{query}': {e}")

    # ── Relevance filtering ───────────────────────────────────────────────────
    filtered_results = []
    for result in results:
        if isinstance(result, dict):
            if await judge_relevance(query, result):
                filtered_results.append(result)

    logger.debug(
        f"For query '{query}': kept {len(filtered_results)}/{len(results)} results"
    )

    search_results = [{"question": query, "results": filtered_results}]
    sources_label = " + ".join(s.upper() for s in search_sources)
    search_summary = (
        f"[{sources_label}] Search for: **{query}** "
        f"(Found {len(filtered_results)} relevant results)"
    )

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
