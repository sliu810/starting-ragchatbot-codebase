import pytest
from unittest.mock import Mock, MagicMock
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from search_tools import CourseSearchTool
from vector_store import SearchResults

class TestCourseSearchTool:
    """Test suite for CourseSearchTool"""
    
    @pytest.fixture
    def mock_vector_store(self):
        """Create a mock vector store for testing"""
        mock_store = Mock()
        return mock_store
    
    @pytest.fixture
    def search_tool(self, mock_vector_store):
        """Create a CourseSearchTool with mocked vector store"""
        return CourseSearchTool(mock_vector_store)
    
    def test_execute_successful_search(self, search_tool, mock_vector_store):
        """Test successful search execution with results"""
        # Setup mock response
        mock_results = SearchResults(
            documents=["This is course content about MCP"],
            metadata=[{"course_title": "MCP Course", "lesson_number": 1}],
            distances=[0.5],
            error=None
        )
        mock_vector_store.search.return_value = mock_results
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson1"
        
        # Execute search
        result = search_tool.execute("MCP introduction")
        
        # Verify vector store was called correctly
        mock_vector_store.search.assert_called_once_with(
            query="MCP introduction",
            course_name=None,
            lesson_number=None
        )
        
        # Check result format
        assert "MCP Course - Lesson 1" in result
        assert "This is course content about MCP" in result
        assert len(search_tool.last_sources) == 1
        assert search_tool.last_sources[0]["text"] == "MCP Course - Lesson 1"
        assert "link" in search_tool.last_sources[0]
    
    def test_execute_with_course_filter(self, search_tool, mock_vector_store):
        """Test search with course name filter"""
        mock_results = SearchResults(
            documents=["Course specific content"],
            metadata=[{"course_title": "Python Basics", "lesson_number": 2}],
            distances=[0.3],
            error=None
        )
        mock_vector_store.search.return_value = mock_results
        
        result = search_tool.execute("variables", course_name="Python")
        
        mock_vector_store.search.assert_called_once_with(
            query="variables",
            course_name="Python",
            lesson_number=None
        )
    
    def test_execute_with_lesson_filter(self, search_tool, mock_vector_store):
        """Test search with lesson number filter"""
        mock_results = SearchResults(
            documents=["Lesson specific content"],
            metadata=[{"course_title": "Advanced Topics", "lesson_number": 3}],
            distances=[0.2],
            error=None
        )
        mock_vector_store.search.return_value = mock_results
        
        result = search_tool.execute("algorithms", lesson_number=3)
        
        mock_vector_store.search.assert_called_once_with(
            query="algorithms",
            course_name=None,
            lesson_number=3
        )
    
    def test_execute_empty_results(self, search_tool, mock_vector_store):
        """Test handling of empty search results"""
        mock_results = SearchResults(
            documents=[],
            metadata=[],
            distances=[],
            error=None
        )
        mock_vector_store.search.return_value = mock_results
        
        result = search_tool.execute("nonexistent topic")
        
        assert "No relevant content found" in result
        assert len(search_tool.last_sources) == 0
    
    def test_execute_empty_results_with_filters(self, search_tool, mock_vector_store):
        """Test empty results with filters shows filter information"""
        mock_results = SearchResults(
            documents=[],
            metadata=[],
            distances=[],
            error=None
        )
        mock_vector_store.search.return_value = mock_results
        
        result = search_tool.execute("topic", course_name="NonExistent", lesson_number=99)
        
        assert "No relevant content found in course 'NonExistent' in lesson 99" in result
    
    def test_execute_search_error(self, search_tool, mock_vector_store):
        """Test handling of search errors"""
        mock_results = SearchResults(
            documents=[],
            metadata=[],
            distances=[],
            error="Database connection failed"
        )
        mock_vector_store.search.return_value = mock_results
        
        result = search_tool.execute("any query")
        
        assert result == "Database connection failed"
        assert len(search_tool.last_sources) == 0
    
    def test_format_results_multiple_documents(self, search_tool, mock_vector_store):
        """Test formatting of multiple search results"""
        mock_results = SearchResults(
            documents=[
                "First document content",
                "Second document content"
            ],
            metadata=[
                {"course_title": "Course A", "lesson_number": 1},
                {"course_title": "Course B", "lesson_number": 2}
            ],
            distances=[0.1, 0.2],
            error=None
        )
        mock_vector_store.search.return_value = mock_results
        mock_vector_store.get_lesson_link.side_effect = [
            "https://example.com/courseA/lesson1",
            "https://example.com/courseB/lesson2"
        ]
        
        result = search_tool.execute("test query")
        
        assert "[Course A - Lesson 1]" in result
        assert "[Course B - Lesson 2]" in result
        assert "First document content" in result
        assert "Second document content" in result
        assert len(search_tool.last_sources) == 2
    
    def test_format_results_no_lesson_number(self, search_tool, mock_vector_store):
        """Test formatting when lesson number is missing"""
        mock_results = SearchResults(
            documents=["Content without lesson"],
            metadata=[{"course_title": "General Course"}],  # No lesson_number
            distances=[0.3],
            error=None
        )
        mock_vector_store.search.return_value = mock_results
        
        result = search_tool.execute("general topic")
        
        assert "[General Course]" in result  # No lesson number in header
        assert "General Course" in search_tool.last_sources[0]["text"]
        assert "link" not in search_tool.last_sources[0]  # No lesson link without lesson number
    
    def test_get_tool_definition(self, search_tool):
        """Test that tool definition is properly formatted"""
        definition = search_tool.get_tool_definition()
        
        assert definition["name"] == "search_course_content"
        assert "search course materials" in definition["description"].lower()
        assert "query" in definition["input_schema"]["properties"]
        assert "course_name" in definition["input_schema"]["properties"]
        assert "lesson_number" in definition["input_schema"]["properties"]
        assert definition["input_schema"]["required"] == ["query"]
    
    def test_sources_tracking(self, search_tool, mock_vector_store):
        """Test that sources are properly tracked and reset"""
        # First search
        mock_results1 = SearchResults(
            documents=["First content"],
            metadata=[{"course_title": "Course 1", "lesson_number": 1}],
            distances=[0.1],
            error=None
        )
        mock_vector_store.search.return_value = mock_results1
        
        search_tool.execute("first query")
        assert len(search_tool.last_sources) == 1
        
        # Second search should replace sources
        mock_results2 = SearchResults(
            documents=["Second content", "Third content"],
            metadata=[
                {"course_title": "Course 2", "lesson_number": 2},
                {"course_title": "Course 3", "lesson_number": 3}
            ],
            distances=[0.2, 0.3],
            error=None
        )
        mock_vector_store.search.return_value = mock_results2
        
        search_tool.execute("second query")
        assert len(search_tool.last_sources) == 2
        assert search_tool.last_sources[0]["text"] == "Course 2 - Lesson 2"