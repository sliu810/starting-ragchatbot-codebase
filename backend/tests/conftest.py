import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil
from pathlib import Path
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config


@pytest.fixture
def test_config():
    """Test configuration with isolated paths and test settings"""
    return Config(
        ANTHROPIC_API_KEY="test-key-12345",
        ANTHROPIC_MODEL="claude-sonnet-4-20250514",
        EMBEDDING_MODEL="all-MiniLM-L6-v2",
        CHUNK_SIZE=200,  # Smaller for tests
        CHUNK_OVERLAP=50,  # Smaller for tests
        MAX_RESULTS=3,  # Fewer for tests
        MAX_HISTORY=2,
        CHROMA_PATH=":memory:"  # Use in-memory database for tests
    )


@pytest.fixture
def temp_directory():
    """Temporary directory for test files"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic client for testing"""
    mock_client = Mock()
    
    # Mock successful response
    mock_response = Mock()
    mock_response.content = [Mock(text="Test response from Claude")]
    mock_response.stop_reason = "end_turn"
    mock_response.usage = Mock(input_tokens=10, output_tokens=20)
    
    mock_client.messages.create.return_value = mock_response
    return mock_client


@pytest.fixture
def mock_sentence_transformer():
    """Mock sentence transformer model"""
    mock_model = Mock()
    # Return simple embeddings for testing
    mock_model.encode.return_value = [[0.1, 0.2, 0.3, 0.4] for _ in range(5)]
    return mock_model


@pytest.fixture
def mock_chroma_collection():
    """Mock ChromaDB collection"""
    mock_collection = Mock()
    mock_collection.count.return_value = 0
    mock_collection.query.return_value = {
        'documents': [['Sample document chunk for testing']],
        'metadatas': [[{'course_title': 'Test Course', 'filename': 'test.txt'}]],
        'distances': [[0.5]]
    }
    mock_collection.add.return_value = None
    return mock_collection


@pytest.fixture
def mock_chroma_client():
    """Mock ChromaDB client"""
    mock_client = Mock()
    mock_collection = Mock()
    mock_collection.count.return_value = 0
    mock_collection.query.return_value = {
        'documents': [['Sample document chunk']],
        'metadatas': [[{'course_title': 'Test Course'}]],
        'distances': [[0.5]]
    }
    mock_client.get_or_create_collection.return_value = mock_collection
    mock_client.list_collections.return_value = []
    return mock_client


@pytest.fixture
def sample_course_documents():
    """Sample course documents for testing"""
    return [
        {
            "title": "Introduction to Python",
            "filename": "python_basics.txt",
            "content": "Python is a high-level programming language. Variables in Python are dynamically typed."
        },
        {
            "title": "Advanced Python Concepts",
            "filename": "python_advanced.txt", 
            "content": "Decorators are a powerful feature in Python. List comprehensions provide concise syntax."
        }
    ]


@pytest.fixture
def sample_query_response():
    """Sample query response for testing"""
    return {
        "answer": "Python is a high-level programming language that is widely used for web development, data science, and automation.",
        "sources": [
            {
                "content": "Python is a high-level programming language.",
                "metadata": {"course_title": "Introduction to Python", "filename": "python_basics.txt"}
            }
        ],
        "session_id": "test_session_123"
    }


@pytest.fixture
def test_app():
    """FastAPI test app without static file mounting to avoid filesystem dependencies"""
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.middleware.trustedhost import TrustedHostMiddleware
    from unittest.mock import Mock
    
    # Create a test app without the problematic static file mounting
    app = FastAPI(title="Course Materials RAG System - Test", root_path="")
    
    # Add middleware
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )
    
    # Mock the RAG system
    mock_rag_system = Mock()
    mock_rag_system.query.return_value = (
        "Test response from RAG system",
        [{"content": "Test source", "metadata": {"course_title": "Test Course"}}]
    )
    mock_rag_system.get_course_analytics.return_value = {
        "total_courses": 2,
        "course_titles": ["Test Course 1", "Test Course 2"]
    }
    mock_rag_system.session_manager.create_session.return_value = "test_session_123"
    
    # Import and define the endpoints inline to avoid app.py imports
    from pydantic import BaseModel
    from typing import List, Optional, Dict, Any
    from fastapi import HTTPException
    
    class QueryRequest(BaseModel):
        query: str
        session_id: Optional[str] = None

    class QueryResponse(BaseModel):
        answer: str
        sources: List[Dict[str, Any]]
        session_id: str

    class CourseStats(BaseModel):
        total_courses: int
        course_titles: List[str]
    
    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            session_id = request.session_id
            if not session_id:
                session_id = mock_rag_system.session_manager.create_session()
            
            answer, sources = mock_rag_system.query(request.query, session_id)
            
            return QueryResponse(
                answer=answer,
                sources=sources,
                session_id=session_id
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            analytics = mock_rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/")
    async def read_root():
        return {"message": "RAG System API - Test Environment"}
    
    # Store mock for test access
    app.state.mock_rag_system = mock_rag_system
    
    return app


@pytest.fixture
def test_client(test_app):
    """Test client for FastAPI app"""
    return TestClient(test_app)


@pytest.fixture
def mock_vector_store():
    """Mock vector store for testing"""
    mock_store = Mock()
    mock_store.search.return_value = [
        {
            'content': 'Sample content from vector store',
            'metadata': {'course_title': 'Test Course', 'filename': 'test.txt'}
        }
    ]
    mock_store.add_documents.return_value = 5
    mock_store.get_course_analytics.return_value = {
        'total_courses': 1,
        'course_titles': ['Test Course']
    }
    return mock_store


@pytest.fixture
def mock_session_manager():
    """Mock session manager for testing"""
    mock_manager = Mock()
    mock_manager.create_session.return_value = "test_session_123"
    mock_manager.get_history.return_value = []
    mock_manager.add_exchange.return_value = None
    return mock_manager


@pytest.fixture(autouse=True)
def suppress_warnings():
    """Suppress common warnings during testing"""
    import warnings
    warnings.filterwarnings("ignore", message="resource_tracker: There appear to be.*")
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=PendingDeprecationWarning)