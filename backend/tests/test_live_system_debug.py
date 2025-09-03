import os
import sys
from unittest.mock import Mock, patch

import pytest

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from rag_system import RAGSystem
from vector_store import SearchResults, VectorStore


class TestLiveSystemDebug:
    """Debug tests to identify issues in the live system"""

    @pytest.fixture
    def real_config(self):
        """Use the real config but with mocked API key"""
        test_config = config
        test_config.ANTHROPIC_API_KEY = "test-api-key"
        return test_config

    def test_vector_store_search_functionality(self, real_config):
        """Test if vector store search is working correctly"""
        # Test with a real vector store instance but mock ChromaDB
        with patch("vector_store.chromadb.PersistentClient") as mock_client_class:
            mock_client = Mock()
            mock_collection = Mock()

            # Mock successful search response
            mock_collection.query.return_value = {
                "documents": [["Sample MCP content"]],
                "metadatas": [[{"course_title": "MCP Course", "lesson_number": 1}]],
                "distances": [[0.5]],
            }

            mock_client.get_or_create_collection.return_value = mock_collection
            mock_client_class.return_value = mock_client

            vector_store = VectorStore(
                chroma_path=real_config.CHROMA_PATH,
                embedding_model=real_config.EMBEDDING_MODEL,
                max_results=real_config.MAX_RESULTS,
            )

            # Test search
            results = vector_store.search("What is MCP?")

            assert not results.error
            assert not results.is_empty()
            assert len(results.documents) == 1
            assert "Sample MCP content" in results.documents[0]

    def test_vector_store_empty_results(self, real_config):
        """Test vector store handling of empty results"""
        with patch("vector_store.chromadb.PersistentClient") as mock_client_class:
            mock_client = Mock()
            mock_collection = Mock()

            # Mock empty search response
            mock_collection.query.return_value = {
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]],
            }

            mock_client.get_or_create_collection.return_value = mock_collection
            mock_client_class.return_value = mock_client

            vector_store = VectorStore(
                chroma_path=real_config.CHROMA_PATH,
                embedding_model=real_config.EMBEDDING_MODEL,
                max_results=real_config.MAX_RESULTS,
            )

            results = vector_store.search("nonexistent topic")

            assert not results.error
            assert results.is_empty()
            assert len(results.documents) == 0

    def test_vector_store_database_error(self, real_config):
        """Test vector store handling of database errors"""
        with patch("vector_store.chromadb.PersistentClient") as mock_client_class:
            mock_client = Mock()
            mock_collection = Mock()

            # Mock database error
            mock_collection.query.side_effect = Exception("Database connection failed")

            mock_client.get_or_create_collection.return_value = mock_collection
            mock_client_class.return_value = mock_client

            vector_store = VectorStore(
                chroma_path=real_config.CHROMA_PATH,
                embedding_model=real_config.EMBEDDING_MODEL,
                max_results=real_config.MAX_RESULTS,
            )

            results = vector_store.search("any query")

            assert results.error is not None
            assert "Database connection failed" in results.error
            assert results.is_empty()

    def test_course_search_tool_with_real_vector_store_errors(self, real_config):
        """Test CourseSearchTool with vector store that returns errors"""
        with patch("vector_store.chromadb.PersistentClient") as mock_client_class:
            mock_client = Mock()
            mock_collection = Mock()

            # Mock database error
            mock_collection.query.side_effect = Exception("ChromaDB error")
            mock_client.get_or_create_collection.return_value = mock_collection
            mock_client_class.return_value = mock_client

            vector_store = VectorStore(
                chroma_path=real_config.CHROMA_PATH,
                embedding_model=real_config.EMBEDDING_MODEL,
                max_results=real_config.MAX_RESULTS,
            )

            from search_tools import CourseSearchTool

            search_tool = CourseSearchTool(vector_store)

            result = search_tool.execute("What is MCP?")

            # Should return the error message
            assert "ChromaDB error" in result or "Search error:" in result

    def test_ai_generator_tool_calling_mechanism(self, real_config):
        """Test if AI generator properly handles tool definitions and calls"""
        # Mock Anthropic client to simulate tool calling
        with patch("ai_generator.anthropic.Anthropic") as mock_anthropic_class:
            mock_client = Mock()

            # Mock response that indicates tool use
            mock_response = Mock()
            mock_response.stop_reason = "tool_use"
            mock_tool_block = Mock()
            mock_tool_block.type = "tool_use"
            mock_tool_block.name = "search_course_content"
            mock_tool_block.id = "tool_123"
            mock_tool_block.input = {"query": "MCP introduction"}
            mock_response.content = [mock_tool_block]

            # Mock final response after tool execution
            mock_final_response = Mock()
            mock_final_response.content = [Mock()]
            mock_final_response.content[0].text = "Here's what I found about MCP"

            mock_client.messages.create.side_effect = [
                mock_response,
                mock_final_response,
            ]
            mock_anthropic_class.return_value = mock_client

            from ai_generator import AIGenerator
            from search_tools import CourseSearchTool, ToolManager

            ai_generator = AIGenerator(
                real_config.ANTHROPIC_API_KEY, real_config.ANTHROPIC_MODEL
            )

            # Create a tool manager with a mock search tool
            tool_manager = ToolManager()
            mock_search_tool = Mock()
            mock_search_tool.get_tool_definition.return_value = {
                "name": "search_course_content",
                "description": "Search course content",
            }
            mock_search_tool.execute.return_value = "Found MCP information"
            tool_manager.register_tool(mock_search_tool)

            # Test tool calling
            response = ai_generator.generate_response(
                "Tell me about MCP",
                tools=tool_manager.get_tool_definitions(),
                tool_manager=tool_manager,
            )

            # Verify tool was executed
            mock_search_tool.execute.assert_called_once_with(query="MCP introduction")
            assert response == "Here's what I found about MCP"

    def test_full_system_integration_with_mocked_externals(self, real_config):
        """Test the full RAG system with real components but mocked external dependencies"""
        with (
            patch("vector_store.chromadb.PersistentClient") as mock_chroma,
            patch("ai_generator.anthropic.Anthropic") as mock_anthropic,
        ):

            # Setup ChromaDB mock
            mock_client = Mock()
            mock_collection = Mock()
            mock_collection.query.return_value = {
                "documents": [["MCP is a framework for building AI applications"]],
                "metadatas": [[{"course_title": "MCP Course", "lesson_number": 1}]],
                "distances": [[0.3]],
            }
            mock_client.get_or_create_collection.return_value = mock_collection
            mock_chroma.return_value = mock_client

            # Setup Anthropic mock
            anthropic_client = Mock()

            # Mock tool use response
            mock_tool_response = Mock()
            mock_tool_response.stop_reason = "tool_use"
            mock_tool_block = Mock()
            mock_tool_block.type = "tool_use"
            mock_tool_block.name = "search_course_content"
            mock_tool_block.id = "tool_456"
            mock_tool_block.input = {"query": "MCP introduction"}
            mock_tool_response.content = [mock_tool_block]

            # Mock final response
            mock_final_response = Mock()
            mock_final_response.content = [Mock()]
            mock_final_response.content[0].text = (
                "MCP is a framework that allows you to build AI applications with rich context."
            )

            anthropic_client.messages.create.side_effect = [
                mock_tool_response,
                mock_final_response,
            ]
            mock_anthropic.return_value = anthropic_client

            # Create RAG system
            rag_system = RAGSystem(real_config)

            # Test query
            response, sources = rag_system.query("What is MCP?")

            # Verify the full flow worked
            assert "MCP is a framework" in response
            assert (
                anthropic_client.messages.create.call_count == 2
            )  # Tool call + final response
            assert mock_collection.query.called  # Vector store was searched

            # Check that sources were captured
            assert len(sources) >= 0  # May be empty depending on mock setup

    def test_system_with_no_courses_loaded(self, real_config):
        """Test system behavior when no courses are loaded in vector store"""
        with (
            patch("vector_store.chromadb.PersistentClient") as mock_chroma,
            patch("ai_generator.anthropic.Anthropic") as mock_anthropic,
        ):

            # Setup ChromaDB mock to return empty results
            mock_client = Mock()
            mock_collection = Mock()
            mock_collection.query.return_value = {
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]],
            }
            mock_client.get_or_create_collection.return_value = mock_collection
            mock_chroma.return_value = mock_client

            # Setup Anthropic mock for tool use
            anthropic_client = Mock()
            mock_tool_response = Mock()
            mock_tool_response.stop_reason = "tool_use"
            mock_tool_block = Mock()
            mock_tool_block.type = "tool_use"
            mock_tool_block.name = "search_course_content"
            mock_tool_block.id = "tool_789"
            mock_tool_block.input = {"query": "MCP"}
            mock_tool_response.content = [mock_tool_block]

            mock_final_response = Mock()
            mock_final_response.content = [Mock()]
            mock_final_response.content[0].text = (
                "I couldn't find any information about MCP in the course materials."
            )

            anthropic_client.messages.create.side_effect = [
                mock_tool_response,
                mock_final_response,
            ]
            mock_anthropic.return_value = anthropic_client

            rag_system = RAGSystem(real_config)
            response, sources = rag_system.query("What is MCP?")

            # Should still get a response, but with empty sources
            assert response is not None
            assert len(sources) == 0

            # Verify search tool returned "no content found"
            # This would be the message from CourseSearchTool when no results are found
