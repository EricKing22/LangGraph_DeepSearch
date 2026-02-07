"""
Tavily 搜索工具模块
"""
from typing import List, Dict, Any, Literal
from langchain_tavily import TavilySearch  # updated 1.0
import config

SearchDepth = Literal["basic", "advanced", "fast", "ultra-fast"]


class TavilySearchTool:
    """Tavily 搜索工具包装类"""

    def __init__(self, api_key: str, max_results: int):
        """
        初始化 Tavily 搜索工具

        Args:
            api_key: Tavily API Key，如果不提供则从配置读取
            max_results: 默认返回的最大结果数
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
        执行搜索查询

        Args:
            query: 搜索查询字符串
            max_results: 返回的最大结果数
            search_depth: 搜索深度 ("basic" 或 "advanced")
            include_answer: 是否包含 AI 生成的答案
            include_raw_content: 是否包含原始网页内容
            include_images: 是否包含相关图片

        Returns:
            包含搜索结果的字典
        """
        max_results = max_results or self.max_results

        try:
            # TavilySearch.invoke() 返回结果列表
            raw_results = self.client.invoke(
                query,
                max_results=max_results,
                search_depth=search_depth,
                include_answer=include_answer,
                include_raw_content=include_raw_content,
                include_images=include_images,
            )

            # 包装成与原 TavilyClient 兼容的格式
            # 处理不同的返回格式
            if isinstance(raw_results, dict):
                # 如果已经是字典格式，直接返回
                if "results" in raw_results:
                    return raw_results
                # 如果是单个结果，包装成列表
                return {"query": query, "results": [raw_results]}
            elif isinstance(raw_results, list):
                # 如果是列表，包装成标准格式
                return {"query": query, "results": raw_results}
            else:
                # 如果是字符串或其他格式，包装成单个结果
                return {"query": query, "results": [{"content": str(raw_results)}]}

        except Exception as e:
            raise RuntimeError(f"Tavily search failed: {str(e)}")

    def extract_results(self, response: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        从 Tavily 响应中提取结构化结果

        Args:
            response: Tavily API 响应

        Returns:
            结构化的搜索结果列表
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


# 创建全局实例（如果 API key 可用）
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
    便捷的搜索函数

    Args:
        query: 搜索查询
        max_results: 最大结果数

    Returns:
        搜索结果列表
    """
    if not tavily_search:
        raise RuntimeError("Tavily search tool is not initialized")

    response = tavily_search.search(query, max_results=max_results)
    return tavily_search.extract_results(response)
