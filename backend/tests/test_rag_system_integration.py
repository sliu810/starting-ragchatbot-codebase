import os
import sys
from unittest.mock import MagicMock, Mock, patch

import pytest

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_system import RAGSystem
from search_tools import ToolManager
from vector_store import SearchResults


class TestRAGSystemIntegration:
    """Integration tests for RAG system end-to-end functionality"""

    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration object"""
        config = Mock()
        config.CHUNK_SIZE = 800
        config.CHUNK_OVERLAP = 100
        config.CHROMA_PATH = "./test_chroma_db"
        config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
        config.MAX_RESULTS = 5
        config.ANTHROPIC_API_KEY = "test-api-key"
        config.ANTHROPIC_MODEL = "claude-3-sonnet-20240229"
        config.MAX_HISTORY = 2
        return config

    @pytest.fixture
    def mock_dependencies(self):
        """Mock all external dependencies"""
        with patch.multiple(
            "rag_system",
            DocumentProcessor=Mock(),
            VectorStore=Mock(),
            AIGenerator=Mock(),
            SessionManager=Mock(),
        ) as mocks:
            yield mocks

    @pytest.fixture
    def rag_system(self, mock_config, mock_dependencies):
        """Create a RAG system with mocked dependencies"""
        system = RAGSystem(mock_config)

        # Setup mock behaviors
        system.vector_store.search.return_value = SearchResults(
            documents=["Sample course content about MCP"],
            metadata=[{"course_title": "MCP Course", "lesson_number": 1}],
            distances=[0.5],
            error=None,
        )

        system.ai_generator.generate_response.return_value = "AI generated response"
        system.session_manager.get_conversation_history.return_value = None

        return system

    def test_rag_system_initialization(self, mock_config, mock_dependencies):
        """Test that RAG system initializes all components correctly"""
        system = RAGSystem(mock_config)

        # Verify all components were initialized
        assert system.document_processor is not None
        assert system.vector_store is not None
        assert system.ai_generator is not None
        assert system.session_manager is not None
        assert system.tool_manager is not None
        assert system.search_tool is not None
        assert system.outline_tool is not None

        # Verify tools were registered
        assert len(system.tool_manager.tools) == 2
        assert "search_course_content" in system.tool_manager.tools
        assert "get_course_outline" in system.tool_manager.tools

    def test_query_without_session(self, rag_system):
        """Test basic query processing without session"""
        response, sources = rag_system.query("What is MCP?")

        # Verify AI generator was called
        rag_system.ai_generator.generate_response.assert_called_once()
        call_args = rag_system.ai_generator.generate_response.call_args

        # Check if called with positional or keyword arguments
        if call_args[0]:  # Has positional arguments
            assert "What is MCP?" in call_args[0][0]  # Query in prompt
        else:  # Called with keyword arguments only
            assert "What is MCP?" in call_args[1]["query"]  # Query in keyword args

        assert call_args[1]["conversation_history"] is None
        assert call_args[1]["tools"] is not None
        assert call_args[1]["tool_manager"] is not None

        assert response == "AI generated response"
        assert isinstance(sources, list)

    def test_query_with_session(self, rag_system):
        """Test query processing with session history"""
        rag_system.session_manager.get_conversation_history.return_value = (
            "Previous conversation"
        )

        response, sources = rag_system.query(
            "Follow up question", session_id="session123"
        )

        # Verify session manager was called
        rag_system.session_manager.get_conversation_history.assert_called_once_with(
            "session123"
        )

        # Verify AI generator received history
        call_args = rag_system.ai_generator.generate_response.call_args
        assert call_args[1]["conversation_history"] == "Previous conversation"

        # Verify session was updated
        rag_system.session_manager.add_exchange.assert_called_once_with(
            "session123", "Follow up question", "AI generated response"
        )

    def test_tools_available_to_ai(self, rag_system):
        """Test that tools are properly provided to AI generator"""
        rag_system.query("Test query")

        call_args = rag_system.ai_generator.generate_response.call_args
        tools = call_args[1]["tools"]

        # Should have 2 tools
        assert len(tools) == 2

        # Extract tool names
        tool_names = [tool["name"] for tool in tools]
        assert "search_course_content" in tool_names
        assert "get_course_outline" in tool_names

        # Verify tool manager was passed
        assert call_args[1]["tool_manager"] == rag_system.tool_manager

    def test_sources_retrieval_and_reset(self, rag_system):
        """Test that sources are retrieved and reset after queries"""
        # Mock tool manager methods since they're real ToolManager instances
        with (
            patch.object(
                rag_system.tool_manager, "get_last_sources"
            ) as mock_get_sources,
            patch.object(
                rag_system.tool_manager, "reset_sources"
            ) as mock_reset_sources,
        ):

            mock_get_sources.return_value = [
                {"text": "MCP Course - Lesson 1", "link": "http://example.com"}
            ]

            response, sources = rag_system.query("Test query")

            # Verify sources were retrieved
            mock_get_sources.assert_called_once()
            assert sources == [
                {"text": "MCP Course - Lesson 1", "link": "http://example.com"}
            ]

            # Verify sources were reset
            mock_reset_sources.assert_called_once()

    def test_course_analytics(self, rag_system):
        """Test course analytics functionality"""
        rag_system.vector_store.get_course_count.return_value = 4
        rag_system.vector_store.get_existing_course_titles.return_value = [
            "MCP Course",
            "Python Basics",
            "Advanced AI",
            "Vector Databases",
        ]

        analytics = rag_system.get_course_analytics()

        assert analytics["total_courses"] == 4
        assert len(analytics["course_titles"]) == 4
        assert "MCP Course" in analytics["course_titles"]

    def test_tool_manager_integration(self, rag_system):
        """Test that tool manager properly integrates with RAG system"""
        # Test tool registration
        assert hasattr(rag_system, "search_tool")
        assert hasattr(rag_system, "outline_tool")
        assert isinstance(rag_system.tool_manager, ToolManager)

        # Test tool definitions are available
        tool_definitions = rag_system.tool_manager.get_tool_definitions()
        assert len(tool_definitions) == 2

        # Test tool execution (mock the execute method)
        rag_system.search_tool.execute = Mock(return_value="Search result")
        result = rag_system.tool_manager.execute_tool(
            "search_course_content", query="test"
        )
        assert result == "Search result"


class TestRAGSystemRealScenarios:
    """Test RAG system with more realistic scenarios"""

    @pytest.fixture
    def mock_config(self):
        config = Mock()
        config.CHUNK_SIZE = 800
        config.CHUNK_OVERLAP = 100
        config.CHROMA_PATH = "./test_chroma_db"
        config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
        config.MAX_RESULTS = 5
        config.ANTHROPIC_API_KEY = "test-api-key"
        config.ANTHROPIC_MODEL = "claude-3-sonnet-20240229"
        config.MAX_HISTORY = 2
        return config

    @pytest.fixture
    def rag_system_with_real_tools(self, mock_config):
        """Create RAG system with real tool implementations but mocked dependencies"""
        with patch.multiple(
            "rag_system",
            DocumentProcessor=Mock(),
            VectorStore=Mock(),
            AIGenerator=Mock(),
            SessionManager=Mock(),
        ):
            system = RAGSystem(mock_config)

            # Keep real tool implementations but mock vector store
            system.vector_store.search = Mock()
            system.vector_store.get_all_courses_metadata = Mock()
            system.vector_store._resolve_course_name = Mock()
            system.ai_generator.generate_response = Mock()

            return system

    def test_search_tool_execution_through_rag_system(self, rag_system_with_real_tools):
        """Test that search tool executes correctly through RAG system"""
        # Setup vector store mock responses
        mock_results = SearchResults(
            documents=["MCP allows you to build AI applications"],
            metadata=[{"course_title": "MCP Course", "lesson_number": 1}],
            distances=[0.3],
            error=None,
        )
        rag_system_with_real_tools.vector_store.search.return_value = mock_results
        rag_system_with_real_tools.vector_store.get_lesson_link.return_value = None

        # Execute search tool directly
        result = rag_system_with_real_tools.search_tool.execute("What is MCP?")

        # Verify result is properly formatted
        assert "[MCP Course - Lesson 1]" in result
        assert "MCP allows you to build AI applications" in result
        assert len(rag_system_with_real_tools.search_tool.last_sources) == 1

    def test_outline_tool_execution_through_rag_system(
        self, rag_system_with_real_tools
    ):
        """Test that outline tool executes correctly through RAG system"""
        # Setup vector store mock responses
        rag_system_with_real_tools.vector_store._resolve_course_name.return_value = (
            "MCP Course"
        )
        rag_system_with_real_tools.vector_store.get_all_courses_metadata.return_value = [
            {
                "title": "MCP Course",
                "course_link": "https://example.com/mcp",
                "lessons": [
                    {"lesson_number": 1, "lesson_title": "Introduction to MCP"},
                    {"lesson_number": 2, "lesson_title": "Building Your First MCP App"},
                ],
            }
        ]

        # Execute outline tool directly
        result = rag_system_with_real_tools.outline_tool.execute("MCP")

        # Verify result is properly formatted
        assert "**MCP Course**" in result
        assert "https://example.com/mcp" in result
        assert "1. Introduction to MCP" in result
        assert "2. Building Your First MCP App" in result

    def test_tool_error_handling_through_rag_system(self, rag_system_with_real_tools):
        """Test error handling in tools through RAG system"""
        # Test search tool with error
        error_results = SearchResults(
            documents=[], metadata=[], distances=[], error="Database connection failed"
        )
        rag_system_with_real_tools.vector_store.search.return_value = error_results

        result = rag_system_with_real_tools.search_tool.execute("test query")
        assert result == "Database connection failed"

        # Test outline tool with no course found
        rag_system_with_real_tools.vector_store._resolve_course_name.return_value = None

        result = rag_system_with_real_tools.outline_tool.execute("NonExistent Course")
        assert "No course found matching 'NonExistent Course'" in result
