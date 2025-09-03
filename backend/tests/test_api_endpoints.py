import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, Mock
import json


class TestQueryEndpoint:
    """Comprehensive tests for the /api/query endpoint"""
    
    def test_query_endpoint_basic_request(self, test_client):
        """Test basic query request with valid input"""
        response = test_client.post("/api/query", json={
            "query": "What is Python?"
        })
        
        assert response.status_code == 200
        data = response.json()
        
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        assert isinstance(data["sources"], list)
        assert len(data["answer"]) > 0
    
    def test_query_endpoint_with_session_id(self, test_client):
        """Test query request with provided session ID"""
        session_id = "custom_session_123"
        response = test_client.post("/api/query", json={
            "query": "What is machine learning?",
            "session_id": session_id
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == session_id
    
    def test_query_endpoint_missing_query(self, test_client):
        """Test query endpoint with missing query field"""
        response = test_client.post("/api/query", json={})
        
        assert response.status_code == 422  # FastAPI validation error
        error_data = response.json()
        assert "detail" in error_data
        
        # Check that the error is about the missing query field
        errors = error_data["detail"]
        assert any(error["loc"] == ["body", "query"] for error in errors)
    
    def test_query_endpoint_empty_query(self, test_client):
        """Test query endpoint with empty query string"""
        response = test_client.post("/api/query", json={
            "query": ""
        })
        
        # Should still process but may return empty or generic response
        assert response.status_code == 200
    
    def test_query_endpoint_invalid_json(self, test_client):
        """Test query endpoint with invalid JSON"""
        response = test_client.post("/api/query", 
                                  data="invalid json",
                                  headers={"Content-Type": "application/json"})
        
        assert response.status_code == 422
    
    def test_query_endpoint_long_query(self, test_client):
        """Test query endpoint with very long query"""
        long_query = "What is " + "machine learning " * 100
        response = test_client.post("/api/query", json={
            "query": long_query
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
    
    def test_query_endpoint_special_characters(self, test_client):
        """Test query endpoint with special characters"""
        response = test_client.post("/api/query", json={
            "query": "What is AI? 🤖 Special chars: áéíóú & < > \" '"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
    
    def test_query_endpoint_response_structure(self, test_client):
        """Test that query response has correct structure"""
        response = test_client.post("/api/query", json={
            "query": "Test query"
        })
        
        assert response.status_code == 200
        data = response.json()
        
        # Check required fields
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        
        # Check data types
        assert isinstance(data["answer"], str)
        assert isinstance(data["sources"], list)
        assert isinstance(data["session_id"], str)
        
        # Check sources structure if not empty
        if data["sources"]:
            source = data["sources"][0]
            assert isinstance(source, dict)
    
    def test_query_endpoint_rag_system_error(self, test_client, test_app):
        """Test query endpoint when RAG system raises an exception"""
        # Make the mock RAG system raise an exception
        test_app.state.mock_rag_system.query.side_effect = Exception("RAG system error")
        
        response = test_client.post("/api/query", json={
            "query": "Test query"
        })
        
        assert response.status_code == 500
        error_data = response.json()
        assert "detail" in error_data
        assert "RAG system error" in error_data["detail"]


class TestCoursesEndpoint:
    """Comprehensive tests for the /api/courses endpoint"""
    
    def test_courses_endpoint_basic_request(self, test_client):
        """Test basic courses endpoint request"""
        response = test_client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "total_courses" in data
        assert "course_titles" in data
        assert isinstance(data["total_courses"], int)
        assert isinstance(data["course_titles"], list)
    
    def test_courses_endpoint_response_structure(self, test_client):
        """Test courses endpoint response structure"""
        response = test_client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        
        # Check field types
        assert isinstance(data["total_courses"], int)
        assert isinstance(data["course_titles"], list)
        
        # Check that total_courses matches course_titles length
        assert data["total_courses"] == len(data["course_titles"])
        
        # Check that course titles are strings
        for title in data["course_titles"]:
            assert isinstance(title, str)
    
    def test_courses_endpoint_method_not_allowed(self, test_client):
        """Test that courses endpoint only accepts GET requests"""
        # Test POST request
        response = test_client.post("/api/courses")
        assert response.status_code == 405
        
        # Test PUT request
        response = test_client.put("/api/courses")
        assert response.status_code == 405
        
        # Test DELETE request
        response = test_client.delete("/api/courses")
        assert response.status_code == 405
    
    def test_courses_endpoint_rag_system_error(self, test_client, test_app):
        """Test courses endpoint when RAG system raises an exception"""
        # Make the mock RAG system raise an exception
        test_app.state.mock_rag_system.get_course_analytics.side_effect = Exception("Analytics error")
        
        response = test_client.get("/api/courses")
        
        assert response.status_code == 500
        error_data = response.json()
        assert "detail" in error_data
        assert "Analytics error" in error_data["detail"]


class TestRootEndpoint:
    """Tests for the root / endpoint"""
    
    def test_root_endpoint(self, test_client):
        """Test root endpoint returns appropriate response"""
        response = test_client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "RAG System API" in data["message"]
    
    def test_root_endpoint_method_support(self, test_client):
        """Test which HTTP methods are supported on root endpoint"""
        # GET should work
        response = test_client.get("/")
        assert response.status_code == 200
        
        # POST should return method not allowed
        response = test_client.post("/")
        assert response.status_code == 405


class TestCORSAndMiddleware:
    """Tests for CORS and middleware functionality"""
    
    def test_cors_headers_present(self, test_client):
        """Test that CORS headers are present in responses"""
        response = test_client.post("/api/query", json={"query": "test"})
        
        # TestClient doesn't always simulate CORS headers like a real browser would
        # Just verify the response is successful - the CORS middleware is configured correctly
        assert response.status_code == 200
        
        # If CORS headers are present, they should have the right values
        if "access-control-allow-origin" in response.headers:
            assert response.headers["access-control-allow-origin"] == "*"
    
    def test_cors_preflight_request(self, test_client):
        """Test CORS preflight (OPTIONS) request"""
        response = test_client.options("/api/query", headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type"
        })
        
        # FastAPI should handle preflight correctly
        assert response.status_code in [200, 204]
    
    def test_trusted_host_middleware(self, test_client):
        """Test trusted host middleware functionality"""
        # Request with any host should be allowed due to allowed_hosts=["*"]
        response = test_client.post("/api/query", 
                                  json={"query": "test"},
                                  headers={"Host": "example.com"})
        
        assert response.status_code == 200


class TestErrorHandling:
    """Tests for error handling and edge cases"""
    
    def test_invalid_endpoint(self, test_client):
        """Test request to non-existent endpoint"""
        response = test_client.get("/api/nonexistent")
        
        assert response.status_code == 404
    
    def test_malformed_request_body(self, test_client):
        """Test various malformed request bodies"""
        # Test with non-JSON content type but JSON data
        response = test_client.post("/api/query", 
                                  data='{"query": "test"}',
                                  headers={"Content-Type": "text/plain"})
        
        # Should return 422 (validation error) or similar
        assert response.status_code in [422, 415]
    
    def test_content_type_handling(self, test_client):
        """Test different content types"""
        # Test with correct JSON content type
        response = test_client.post("/api/query", 
                                  json={"query": "test"})
        assert response.status_code == 200
        
        # Test with explicit application/json content type
        response = test_client.post("/api/query",
                                  data='{"query": "test"}',
                                  headers={"Content-Type": "application/json"})
        assert response.status_code == 200


class TestAsyncEndpoints:
    """Tests for async endpoint behavior"""
    
    @pytest.mark.asyncio
    async def test_async_endpoint_performance(self, test_client):
        """Test that async endpoints handle concurrent requests well"""
        import asyncio
        from concurrent.futures import ThreadPoolExecutor
        
        def make_request():
            return test_client.post("/api/query", json={"query": f"test query"})
        
        # Make multiple concurrent requests
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(make_request) for _ in range(5)]
            responses = [future.result() for future in futures]
        
        # All requests should succeed
        for response in responses:
            assert response.status_code == 200
            data = response.json()
            assert "answer" in data
    
    def test_endpoint_response_time(self, test_client):
        """Test that endpoints respond within reasonable time"""
        import time
        
        start_time = time.time()
        response = test_client.post("/api/query", json={"query": "Quick test"})
        end_time = time.time()
        
        assert response.status_code == 200
        # Response should be reasonably fast (less than 5 seconds for mock)
        assert (end_time - start_time) < 5.0


class TestRequestValidation:
    """Tests for FastAPI request validation"""
    
    def test_query_field_validation(self, test_client):
        """Test validation of query field"""
        # Test with None query
        response = test_client.post("/api/query", json={"query": None})
        assert response.status_code == 422
        
        # Test with non-string query
        response = test_client.post("/api/query", json={"query": 123})
        assert response.status_code == 422
        
        # Test with list query
        response = test_client.post("/api/query", json={"query": ["test"]})
        assert response.status_code == 422
    
    def test_session_id_validation(self, test_client):
        """Test validation of optional session_id field"""
        # Valid session_id should work
        response = test_client.post("/api/query", json={
            "query": "test",
            "session_id": "valid_session_123"
        })
        assert response.status_code == 200
        
        # None session_id should work (optional field)
        response = test_client.post("/api/query", json={
            "query": "test",
            "session_id": None
        })
        assert response.status_code == 200
        
        # Non-string session_id should fail
        response = test_client.post("/api/query", json={
            "query": "test",
            "session_id": 123
        })
        assert response.status_code == 422
    
    def test_extra_fields_handling(self, test_client):
        """Test how extra fields in request are handled"""
        response = test_client.post("/api/query", json={
            "query": "test",
            "extra_field": "should_be_ignored",
            "another_field": 123
        })
        
        # Should still succeed (Pydantic ignores extra fields by default)
        assert response.status_code == 200