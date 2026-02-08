"""
Tests for search tools
"""
import pytest
from unittest.mock import patch, MagicMock
from tools.search_tool import TavilySearchTool


class TestTavilySearchTool:
    """Test cases for TavilySearchTool"""

    @patch("tools.search_tool.config")
    def test_initialization_with_api_key(self, mock_config):
        """Test tool initialization with valid API key"""
        mock_config.TAVILY_API_KEY = "test_api_key"
        mock_config.MAX_SEARCH_RESULTS = 5

        with patch("tools.search_tool.TavilySearch") as mock_tavily:
            tool = TavilySearchTool(api_key="test_api_key", max_results=5)

            assert tool.api_key == "test_api_key"
            assert tool.max_results == 5
            mock_tavily.assert_called_once_with(max_results=5, api_key="test_api_key")

    @patch("tools.search_tool.config")
    def test_initialization_without_api_key(self, mock_config):
        """Test that initialization fails without API key"""
        mock_config.TAVILY_API_KEY = None

        with pytest.raises(ValueError, match="Tavily API key is required"):
            TavilySearchTool(api_key=None, max_results=5)

    @patch("tools.search_tool.config")
    def test_search_returns_dict_format(self, mock_config):
        """Test search method with dict response"""
        mock_config.TAVILY_API_KEY = "test_api_key"
        mock_config.MAX_SEARCH_RESULTS = 5

        with patch("tools.search_tool.TavilySearch") as mock_tavily:
            mock_client = MagicMock()
            mock_client.invoke.return_value = {
                "results": [
                    {
                        "title": "Test",
                        "url": "http://test.com",
                        "content": "Test content",
                    }
                ]
            }
            mock_tavily.return_value = mock_client

            tool = TavilySearchTool(api_key="test_api_key", max_results=5)
            result = tool.search("test query", max_results=3)

            assert "results" in result
            assert isinstance(result["results"], list)
            mock_client.invoke.assert_called_once()

    @patch("tools.search_tool.config")
    def test_search_returns_list_format(self, mock_config):
        """Test search method with list response"""
        mock_config.TAVILY_API_KEY = "test_api_key"
        mock_config.MAX_SEARCH_RESULTS = 5

        with patch("tools.search_tool.TavilySearch") as mock_tavily:
            mock_client = MagicMock()
            mock_client.invoke.return_value = [
                {"title": "Test", "url": "http://test.com", "content": "Test content"}
            ]
            mock_tavily.return_value = mock_client

            tool = TavilySearchTool(api_key="test_api_key", max_results=5)
            result = tool.search("test query", max_results=3)

            assert "query" in result
            assert "results" in result
            assert result["query"] == "test query"

    @patch("tools.search_tool.config")
    def test_search_error_handling(self, mock_config):
        """Test search error handling"""
        mock_config.TAVILY_API_KEY = "test_api_key"
        mock_config.MAX_SEARCH_RESULTS = 5

        with patch("tools.search_tool.TavilySearch") as mock_tavily:
            mock_client = MagicMock()
            mock_client.invoke.side_effect = Exception("API Error")
            mock_tavily.return_value = mock_client

            tool = TavilySearchTool(api_key="test_api_key", max_results=5)

            with pytest.raises(RuntimeError, match="Tavily search failed"):
                tool.search("test query", max_results=3)

    @patch("tools.search_tool.config")
    def test_extract_results(self, mock_config):
        """Test extract_results method"""
        mock_config.TAVILY_API_KEY = "test_api_key"
        mock_config.MAX_SEARCH_RESULTS = 5

        with patch("tools.search_tool.TavilySearch"):
            tool = TavilySearchTool(api_key="test_api_key", max_results=5)

            response = {
                "results": [
                    {
                        "title": "Title 1",
                        "url": "http://test1.com",
                        "content": "Content 1",
                    },
                    {
                        "title": "Title 2",
                        "url": "http://test2.com",
                        "content": "Content 2",
                    },
                ]
            }

            results = tool.extract_results(response)

            assert len(results) == 2
            assert results[0]["title"] == "Title 1"
            assert results[0]["url"] == "http://test1.com"
            assert results[0]["content"] == "Content 1"

    @patch("tools.search_tool.config")
    def test_extract_results_empty(self, mock_config):
        """Test extract_results with empty response"""
        mock_config.TAVILY_API_KEY = "test_api_key"
        mock_config.MAX_SEARCH_RESULTS = 5

        with patch("tools.search_tool.TavilySearch"):
            tool = TavilySearchTool(api_key="test_api_key", max_results=5)

            response = {"results": []}
            results = tool.extract_results(response)

            assert results == []

    @patch("tools.search_tool.config")
    def test_extract_results_missing_fields(self, mock_config):
        """Test extract_results with missing fields in results"""
        mock_config.TAVILY_API_KEY = "test_api_key"
        mock_config.MAX_SEARCH_RESULTS = 5

        with patch("tools.search_tool.TavilySearch"):
            tool = TavilySearchTool(api_key="test_api_key", max_results=5)

            response = {
                "results": [
                    {"title": "Title 1"},  # Missing url and content
                    {"url": "http://test.com"},  # Missing title and content
                ]
            }

            results = tool.extract_results(response)

            assert len(results) == 2
            assert results[0]["title"] == "Title 1"
            assert results[0]["url"] == ""
            assert results[0]["content"] == ""
            assert results[1]["title"] == ""
            assert results[1]["url"] == "http://test.com"
