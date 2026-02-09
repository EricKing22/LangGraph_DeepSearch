"""
Tests for graph nodes
"""

import pytest
from unittest.mock import patch, MagicMock
from langchain_core.messages import HumanMessage, AIMessage
from src.nodes.question_nodes import extract_query, plan, should_skip_human_feedback
from src.nodes.review_nodes import review


class TestExtractQuery:
    """Test cases for extract_query function"""

    def test_extract_query_from_state(self):
        """Test extracting query directly from state"""
        state = {"query": "What is AI?"}
        result = extract_query(state)
        assert result == "What is AI?"

    def test_extract_query_from_messages_string(self):
        """Test extracting query from HumanMessage with string content"""
        state = {
            "messages": [
                HumanMessage(content="What is machine learning?"),
                AIMessage(content="Response"),
            ]
        }
        result = extract_query(state)
        assert result == "What is machine learning?"

    def test_extract_query_from_messages_list(self):
        """Test extracting query from HumanMessage with list content"""
        state = {
            "messages": [
                HumanMessage(content=[{"type": "text", "text": "Tell me about Python"}])
            ]
        }
        result = extract_query(state)
        assert result == "Tell me about Python"

    def test_extract_query_no_query_raises_error(self):
        """Test that missing query raises ValueError"""
        state = {"messages": [AIMessage(content="Only AI message")]}
        with pytest.raises(ValueError, match="No query found in state"):
            extract_query(state)

    def test_extract_query_empty_messages(self):
        """Test with empty messages list"""
        state = {"messages": []}
        with pytest.raises(ValueError, match="No query found in state"):
            extract_query(state)


class TestPlan:
    """Test cases for plan node"""

    @patch("src.nodes.question_nodes.llm")
    @patch("src.nodes.question_nodes.config")
    def test_plan_generates_subquestions(self, mock_config, mock_llm):
        """Test that plan generates sub-questions"""
        mock_config.MAX_SUB_QUESTIONS = 5

        # Mock LLM response
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = MagicMock(
            questions=["Question 1?", "Question 2?"],
            reason="These questions cover different aspects",
        )
        mock_llm.with_structured_output.return_value = mock_structured

        state = {
            "query": "What is AI?",
            "messages": [],
            "questions": [],
            "break_questions_iterations_count": 0,
        }

        result = plan(state)

        assert "questions" in result
        assert len(result["questions"]) == 2
        assert "messages" in result
        assert result["break_questions_iterations_count"] == 1

    @patch("src.nodes.question_nodes.llm")
    @patch("src.nodes.question_nodes.config")
    def test_plan_with_human_feedback(self, mock_config, mock_llm):
        """Test plan with human feedback"""
        mock_config.MAX_SUB_QUESTIONS = 5

        mock_structured = MagicMock()
        mock_structured.invoke.return_value = MagicMock(
            questions=["Revised Q1?", "Revised Q2?"], reason="Revised based on feedback"
        )
        mock_llm.with_structured_output.return_value = mock_structured

        state = {
            "query": "What is AI?",
            "messages": [],
            "questions": ["Old Q1?"],
            "human_feedback": "Make questions more specific",
            "break_questions_iterations_count": 1,
        }

        result = plan(state)

        assert "questions" in result
        assert result["break_questions_iterations_count"] == 2


class TestShouldSkipHumanFeedback:
    """Test cases for should_skip_human_feedback router"""

    def test_skip_on_first_iteration(self):
        """Test that human feedback is not skipped on first iteration"""
        state = {"summarise_iterations": 0}
        result = should_skip_human_feedback(state)
        assert result == "human_feedback"

    @patch("src.nodes.question_nodes.map_search")
    def test_skip_on_subsequent_iterations(self, mock_map_search):
        """Test that human feedback is skipped after first iteration"""
        mock_map_search.return_value = "mocked_result"
        state = {"summarise_iterations": 1}
        result = should_skip_human_feedback(state)
        assert result == "mocked_result"


class TestReview:
    """Test cases for review node"""

    @patch("src.nodes.review_nodes.llm")
    def test_review_evaluates_summary(self, mock_llm):
        """Test that review evaluates summary"""
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = MagicMock(
            score=8, strengths="Well structured", weaknesses="Could be more detailed"
        )
        mock_llm.with_structured_output.return_value = mock_structured

        state = {
            "query": "What is AI?",
            "summary": "AI is artificial intelligence",
            "messages": [],
        }

        result = review(state)

        assert "score" in result
        assert "strengths" in result
        assert "weaknesses" in result
        assert "messages" in result
        assert result["score"] == 8


class TestIsFinished:
    """Test cases for is_finished router"""

    @patch("src.nodes.question_nodes.config")
    def test_finished_high_score(self, mock_config):
        """Test that high score leads to END"""
        mock_config.ACCEPTABLE_SCORE = 7
        mock_config.MAX_REVIEW_IMPROVE_ITERATIONS = 3

        state = {"score": 8, "summarise_iterations": 1}

        from src.nodes.question_nodes import is_finished

        result = is_finished(state)

        from langgraph.graph import END

        assert result == END

    @patch("src.nodes.question_nodes.config")
    @patch("src.nodes.question_nodes.llm")
    def test_not_finished_low_score(self, mock_llm, mock_config):
        """Test that low score continues to plan or summarise"""
        mock_config.ACCEPTABLE_SCORE = 7
        mock_config.MAX_REVIEW_IMPROVE_ITERATIONS = 3

        state = {"score": 5, "summarise_iterations": 1}

        # Mock LLM response for router
        mock_router = MagicMock()
        mock_router.invoke.return_value = MagicMock(
            next_step="plan", reason="Need more information"
        )
        mock_llm.with_structured_output.return_value = mock_router

        from src.nodes.question_nodes import is_finished

        result = is_finished(state)

        # Should return plan or summarise, not END
        assert result in ["plan", "summarise"]

    @patch("src.nodes.question_nodes.config")
    def test_max_iterations_reached(self, mock_config):
        """Test that max iterations leads to END"""
        mock_config.ACCEPTABLE_SCORE = 7
        mock_config.MAX_REVIEW_IMPROVE_ITERATIONS = 3

        state = {"score": 5, "summarise_iterations": 3}

        from src.nodes.question_nodes import is_finished

        result = is_finished(state)

        from langgraph.graph import END

        assert result == END
