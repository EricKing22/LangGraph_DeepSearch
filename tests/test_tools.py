"""
Tests for search tools
"""

from unittest.mock import patch
from src.tools.search_tool import search_tavily, _extract_results


class TestExtractResults:
    """Test cases for _extract_results function"""

    def test_extract_results_from_dict_response(self):
        """Test extract_results with dict response containing results"""
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

        results = _extract_results(response)

        assert len(results) == 2
        assert results[0]["title"] == "Title 1"
        assert results[0]["url"] == "http://test1.com"
        assert results[0]["content"] == "Content 1"
        assert results[1]["title"] == "Title 2"
        assert results[1]["url"] == "http://test2.com"
        assert results[1]["content"] == "Content 2"

    def test_extract_results_from_list_response(self):
        """Test extract_results with list response"""
        response = [
            {"title": "Test", "url": "http://test.com", "content": "Test content"}
        ]

        results = _extract_results(response)

        assert len(results) == 1
        assert results[0]["title"] == "Test"
        assert results[0]["url"] == "http://test.com"
        assert results[0]["content"] == "Test content"

    def test_extract_results_empty(self):
        """Test extract_results with empty response"""
        response = {"results": []}
        results = _extract_results(response)

        assert results == []

    def test_extract_results_missing_fields(self):
        """Test extract_results with missing fields in results"""
        response = {
            "results": [
                {"title": "Title 1"},  # Missing url and content
                {"url": "http://test.com"},  # Missing title and content
            ]
        }

        results = _extract_results(response)

        assert len(results) == 2
        assert results[0]["title"] == "Title 1"
        assert results[0]["url"] == ""
        assert results[0]["content"] == ""
        assert results[1]["title"] == ""
        assert results[1]["url"] == "http://test.com"
        assert results[1]["content"] == ""


class TestSearchTavily:
    """Test cases for search_tavily function"""

    @patch("src.tools.search_tool.client")
    def test_search_returns_list_format(self, mock_client):
        """Test search_tavily with dict response from client"""
        mock_client.invoke.return_value = {
            "results": [
                {
                    "title": "Test",
                    "url": "http://test.com",
                    "content": "Test content",
                }
            ]
        }

        result = search_tavily.invoke({"query": "test query"})

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["title"] == "Test"
        mock_client.invoke.assert_called_once()

    @patch("src.tools.search_tool.client")
    def test_search_handles_list_response(self, mock_client):
        """Test search_tavily when client returns list"""
        mock_client.invoke.return_value = [
            {"title": "Test", "url": "http://test.com", "content": "Test content"}
        ]

        result = search_tavily.invoke({"query": "test query"})

        assert isinstance(result, list)
        assert len(result) == 1

    @patch("src.tools.search_tool.client")
    def test_search_error_handling(self, mock_client):
        """Test search error handling"""
        mock_client.invoke.side_effect = Exception("API Error")

        result = search_tavily.invoke({"query": "test query"})

        assert isinstance(result, list)
        assert len(result) == 1
        assert "Error performing search" in result[0]["content"]

    @patch("src.tools.search_tool.client")
    def test_search_with_custom_parameters(self, mock_client):
        """Test search_tavily with custom parameters"""
        mock_client.invoke.return_value = {"results": []}

        result = search_tavily.invoke(
            {"query": "test query", "max_results": 10, "search_depth": "basic"}
        )

        assert isinstance(result, list)
        mock_client.invoke.assert_called_once()

    @patch("src.tools.search_tool.client")
    def test_search_handles_string_response(self, mock_client):
        """Test search_tavily when client returns unexpected string"""
        mock_client.invoke.return_value = "Some unexpected string response"

        result = search_tavily.invoke({"query": "test query"})

        assert isinstance(result, list)
        assert len(result) == 1
        assert "Some unexpected string response" in result[0]["content"]
