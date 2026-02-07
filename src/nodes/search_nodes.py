from pydantic import BaseModel, Field
from state import Search
from typing import Dict
from llm import ollama_llm as llm
from langchain.messages import SystemMessage, AIMessage
from tools.search_tool import search_tavily
from prompts import RELEVANCE_CHECK_PROMPT


def judge_relevance(query: str, search_result: Dict[str, str]) -> bool:
    """
    使用LLM判断单个搜索结果是否与查询相关

    Args:
        query: 用户查询
        search_result: 单个搜索结果字典，包含 title, url, content

    Returns:
        bool: True表示相关，False表示不相关
    """

    class RelevanceDecision(BaseModel):
        is_relevant: bool = Field(
            description="Whether the content is relevant to the query"
        )
        reason: str = Field(description="Brief explanation for the decision")

    structured_llm = llm.with_structured_output(RelevanceDecision)

    title = search_result.get("title", "")
    content = search_result.get("content", "")

    # 如果内容为空，直接判定为不相关
    if not content.strip():
        return False

    prompt = RELEVANCE_CHECK_PROMPT.format(
        query=query,
        title=title,
        content=content[:1000],  # 限制内容长度以避免超出token限制
    )

    try:
        decision = structured_llm.invoke([SystemMessage(content=prompt)])
        print(
            f"Relevance check for '{title[:50]}...': {decision.is_relevant} - {decision.reason}"
        )
        return decision.is_relevant
    except Exception as e:
        print(f"Error judging relevance: {str(e)}")
        # 出错时保守处理，保留结果
        return True


def search_web(state: Search):
    """
    对问题query执行 Tavily 搜索，并使用LLM过滤不相关的结果
    """
    query = state.get("query")
    search_results = []

    try:
        results = search_tavily(query=query)

        # 使用judge_relevance过滤结果

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
    search_summary = f"Searched web for query: {query} (found {len(search_results[0].get('results', []))} relevant results)"
    return {
        "messages": [AIMessage(content=search_summary)],
        "search_results": search_results,
        "sources": [result for result in search_results if "results" in result],
    }
