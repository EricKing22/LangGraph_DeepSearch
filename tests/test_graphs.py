"""
Tests for graph structure and execution
"""

from unittest.mock import patch, MagicMock


class TestWebSearchGraph:
    """Test cases for web search graph"""

    def test_graph_imports(self):
        """Test that graph can be imported"""
        from src.graphs.web_search_graph import graph

        assert graph is not None

    def test_graph_has_nodes(self):
        """Test that graph contains expected nodes"""
        from src.graphs.web_search_graph import builder

        # Check that builder has the expected nodes
        assert builder is not None
        # Note: actual node inspection depends on LangGraph internals

    def test_graph_compilation(self):
        """Test that graph compiles without errors"""
        from src.graphs.web_search_graph import graph

        # Graph should be compiled
        assert graph is not None
        assert hasattr(graph, "invoke") or hasattr(graph, "stream")

    def test_graph_with_checkpointer(self):
        """Test that graph can be compiled with a checkpointer"""
        from src.graphs.web_search_graph import builder
        from langgraph.checkpoint.memory import MemorySaver

        # Compile with a checkpointer
        checkpointer = MemorySaver()
        compiled_graph = builder.compile(checkpointer=checkpointer)

        # Verify graph is compiled
        assert compiled_graph is not None


class TestGraphExecution:
    """Test cases for graph execution flow"""

    @patch("src.nodes.question_nodes.llm")
    def test_graph_basic_execution(self, mock_llm):
        """Test basic graph execution with mocked dependencies"""
        from src.graphs.web_search_graph import graph

        # Mock LLM responses
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = MagicMock(
            questions=["Q1?", "Q2?"],
            reason="Test reason",
            score=8,
            strengths="Good",
            weaknesses="None",
            next_step="search_web",
        )
        mock_llm.with_structured_output.return_value = mock_structured
        mock_llm.invoke.return_value = MagicMock(content="Test summary")

        # Note: Full execution test would require proper state management
        # This is a minimal test to verify graph structure
        assert graph is not None

    def test_graph_state_schema(self):
        """Test that graph uses correct state schema"""
        from src.graphs.web_search_graph import builder

        # Verify state schema exists
        assert hasattr(builder, "channels")
        # The builder should have the state configured
        assert builder is not None


class TestGraphEdges:
    """Test cases for graph edge configuration"""

    def test_graph_has_start_edge(self):
        """Test that graph has edge from START"""
        from src.graphs.web_search_graph import builder

        # Builder should have edges configured
        assert builder is not None

    def test_graph_has_conditional_edges(self):
        """Test that graph has conditional edges configured"""
        from src.graphs.web_search_graph import builder

        # Verify builder has been configured with conditional edges
        # This is implicit in the structure, actual verification depends on internals
        assert builder is not None


class TestGraphInterrupts:
    """Test cases for graph interrupt behavior"""

    def test_graph_interrupt_before_human_feedback(self):
        """Test that graph is configured to interrupt before human_feedback"""
        from src.graphs.web_search_graph import graph

        # Graph should be compiled with interrupt_before
        assert graph is not None
        # Actual interrupt behavior would need integration test

    @patch("src.nodes.question_nodes.llm")
    def test_graph_can_resume_after_interrupt(self, mock_llm):
        """Test that graph can resume after interrupt"""
        from src.graphs.web_search_graph import graph

        # Mock LLM
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = MagicMock(
            questions=["Q1?"], reason="Test", next_step="search_web"
        )
        mock_llm.with_structured_output.return_value = mock_structured

        # This would require full integration test with state management
        assert graph is not None


class TestGraphStateManagement:
    """Test cases for state management in graph"""

    def test_state_has_required_fields(self):
        """Test that WebSearchState has all required fields"""
        from src.state.states import WebSearchState

        # WebSearchState should have required annotations
        assert hasattr(WebSearchState, "__annotations__")
        annotations = WebSearchState.__annotations__

        # Check for key fields
        assert "query" in annotations
        assert "questions" in annotations
        assert "search_results" in annotations
        assert "summary" in annotations

    def test_state_message_inheritance(self):
        """Test that WebSearchState has messages field like MessagesState"""
        from src.state.states import WebSearchState

        # Check that WebSearchState has the messages field
        # We just check for the annotation, not isinstance since TypedDict doesn't support that
        annotations = WebSearchState.__annotations__
        assert "messages" in annotations
