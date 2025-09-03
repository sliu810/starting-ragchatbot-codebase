import os
import sys
from unittest.mock import patch

import pytest

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from rag_system import RAGSystem


@pytest.mark.skipif(not os.path.exists("../docs"), reason="No docs folder available")
class TestRealSystemIssues:
    """Test the real system to identify actual issues"""

    def test_vector_store_has_data(self):
        """Check if vector store actually has course data"""
        from vector_store import VectorStore

        vector_store = VectorStore(
            chroma_path=config.CHROMA_PATH,
            embedding_model=config.EMBEDDING_MODEL,
            max_results=config.MAX_RESULTS,
        )

        # Check if we have courses
        course_count = vector_store.get_course_count()
        course_titles = vector_store.get_existing_course_titles()

        print(f"Course count: {course_count}")
        print(f"Course titles: {course_titles}")

        # This tells us if the vector store is empty
        assert (
            course_count > 0
        ), "No courses found in vector store - this could be the issue!"
        assert len(course_titles) > 0, "No course titles found"

        # Try a basic search
        results = vector_store.search("MCP")
        print(f"Search results error: {results.error}")
        print(f"Search results empty: {results.is_empty()}")
        print(f"Search results count: {len(results.documents)}")

        if results.error:
            pytest.fail(f"Vector store search failed: {results.error}")

    def test_search_tool_with_real_data(self):
        """Test search tool with real vector store data"""
        from search_tools import CourseSearchTool
        from vector_store import VectorStore

        vector_store = VectorStore(
            chroma_path=config.CHROMA_PATH,
            embedding_model=config.EMBEDDING_MODEL,
            max_results=config.MAX_RESULTS,
        )

        search_tool = CourseSearchTool(vector_store)

        # Test basic search
        result = search_tool.execute("MCP")
        print(f"Search tool result: {result}")
        print(f"Search tool sources: {search_tool.last_sources}")

        # Should not return an error message
        assert "error" not in result.lower(), f"Search tool returned error: {result}"
        assert (
            "no relevant content found" not in result.lower()
            or len(search_tool.last_sources) == 0
        )

    def test_outline_tool_with_real_data(self):
        """Test outline tool with real vector store data"""
        from search_tools import CourseOutlineTool
        from vector_store import VectorStore

        vector_store = VectorStore(
            chroma_path=config.CHROMA_PATH,
            embedding_model=config.EMBEDDING_MODEL,
            max_results=config.MAX_RESULTS,
        )

        # First get a real course title
        course_titles = vector_store.get_existing_course_titles()
        if not course_titles:
            pytest.skip("No courses available to test outline tool")

        first_course = course_titles[0]
        print(f"Testing outline tool with course: {first_course}")

        outline_tool = CourseOutlineTool(vector_store)
        result = outline_tool.execute(first_course)

        print(f"Outline tool result: {result}")

        # Should return course outline, not error
        assert (
            "not found" not in result.lower()
        ), f"Outline tool couldn't find course: {result}"
        assert "**" in result, "Outline should contain formatted course title"

    @pytest.mark.skipif(
        not config.ANTHROPIC_API_KEY or config.ANTHROPIC_API_KEY == "your-api-key",
        reason="No valid Anthropic API key available",
    )
    def test_ai_generator_with_real_api(self):
        """Test AI generator with real Anthropic API (if key is available)"""
        from ai_generator import AIGenerator
        from search_tools import CourseSearchTool, ToolManager
        from vector_store import VectorStore

        # Create real components
        vector_store = VectorStore(
            chroma_path=config.CHROMA_PATH,
            embedding_model=config.EMBEDDING_MODEL,
            max_results=config.MAX_RESULTS,
        )

        ai_generator = AIGenerator(config.ANTHROPIC_API_KEY, config.ANTHROPIC_MODEL)

        tool_manager = ToolManager()
        search_tool = CourseSearchTool(vector_store)
        tool_manager.register_tool(search_tool)

        try:
            response = ai_generator.generate_response(
                "What is MCP?",
                tools=tool_manager.get_tool_definitions(),
                tool_manager=tool_manager,
            )

            print(f"AI Generator response: {response}")
            assert response is not None
            assert len(response.strip()) > 0

        except Exception as e:
            pytest.fail(f"AI Generator failed: {str(e)}")

    def test_system_prompt_format(self):
        """Test that the system prompt is properly formatted"""
        from ai_generator import AIGenerator

        # Check if system prompt contains the expected tool guidance
        system_prompt = AIGenerator.SYSTEM_PROMPT

        print("System prompt:")
        print(system_prompt)
        print("-" * 50)

        assert (
            "get_course_outline" in system_prompt
        ), "System prompt missing outline tool reference"
        assert (
            "search_course_content" in system_prompt
        ), "System prompt missing search tool reference"
        assert (
            "Tool Usage Guidelines" in system_prompt
        ), "System prompt missing tool usage section"

    def test_tool_definitions_format(self):
        """Test that tool definitions are properly formatted"""
        from search_tools import CourseOutlineTool, CourseSearchTool, ToolManager
        from vector_store import VectorStore

        vector_store = VectorStore(
            chroma_path=config.CHROMA_PATH,
            embedding_model=config.EMBEDDING_MODEL,
            max_results=config.MAX_RESULTS,
        )

        tool_manager = ToolManager()
        search_tool = CourseSearchTool(vector_store)
        outline_tool = CourseOutlineTool(vector_store)

        tool_manager.register_tool(search_tool)
        tool_manager.register_tool(outline_tool)

        definitions = tool_manager.get_tool_definitions()

        print("Tool definitions:")
        for definition in definitions:
            print(f"- {definition['name']}: {definition['description']}")
            print(f"  Required params: {definition['input_schema']['required']}")

        assert len(definitions) == 2, "Should have exactly 2 tools registered"

        tool_names = [d["name"] for d in definitions]
        assert "search_course_content" in tool_names
        assert "get_course_outline" in tool_names


class TestEnvironmentSetup:
    """Test environment and configuration issues"""

    def test_api_key_configured(self):
        """Check if Anthropic API key is properly configured"""
        from config import config

        print(
            f"API Key configured: {'Yes' if config.ANTHROPIC_API_KEY and config.ANTHROPIC_API_KEY != 'your-api-key' else 'No'}"
        )
        print(
            f"API Key starts with: {config.ANTHROPIC_API_KEY[:10] if config.ANTHROPIC_API_KEY else 'None'}..."
        )

        if not config.ANTHROPIC_API_KEY or config.ANTHROPIC_API_KEY == "your-api-key":
            pytest.skip(
                "No valid API key configured - this could be the issue with 'query failed'"
            )

    def test_chroma_db_accessible(self):
        """Check if ChromaDB is accessible"""
        import chromadb
        from chromadb.config import Settings

        try:
            client = chromadb.PersistentClient(
                path=config.CHROMA_PATH, settings=Settings(anonymized_telemetry=False)
            )

            # Try to list collections
            collections = client.list_collections()
            print(f"ChromaDB collections: {[c.name for c in collections]}")

            if len(collections) == 0:
                print("WARNING: No collections found in ChromaDB!")

        except Exception as e:
            pytest.fail(f"ChromaDB not accessible: {str(e)}")

    def test_embedding_model_loading(self):
        """Check if embedding model can be loaded"""
        from sentence_transformers import SentenceTransformer

        try:
            model = SentenceTransformer(config.EMBEDDING_MODEL)
            print(f"Embedding model loaded successfully: {config.EMBEDDING_MODEL}")

            # Try encoding a test sentence
            embedding = model.encode("test sentence")
            print(f"Embedding dimension: {len(embedding)}")

        except Exception as e:
            pytest.fail(f"Embedding model failed to load: {str(e)}")

    def test_config_values(self):
        """Print all config values for debugging"""
        print("Configuration values:")
        print(f"CHUNK_SIZE: {config.CHUNK_SIZE}")
        print(f"CHUNK_OVERLAP: {config.CHUNK_OVERLAP}")
        print(f"CHROMA_PATH: {config.CHROMA_PATH}")
        print(f"EMBEDDING_MODEL: {config.EMBEDDING_MODEL}")
        print(f"MAX_RESULTS: {config.MAX_RESULTS}")
        print(f"ANTHROPIC_MODEL: {config.ANTHROPIC_MODEL}")
        print(f"MAX_HISTORY: {config.MAX_HISTORY}")
        print(
            f"Has API key: {'Yes' if config.ANTHROPIC_API_KEY and len(config.ANTHROPIC_API_KEY) > 10 else 'No'}"
        )
