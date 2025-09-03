import os
import sys
from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app


class TestFastAPIEndpoints:
    """Test FastAPI endpoints to identify where 'query failed' might be coming from"""

    @pytest.fixture
    def client(self):
        """Create a test client for FastAPI"""
        return TestClient(app)

    def test_query_endpoint_exists(self, client):
        """Test that the query endpoint exists and accepts POST requests"""
        # Test with a simple request
        response = client.post("/api/query", json={"query": "test query"})

        # Should not return 404 or 405 (method not allowed)
        assert response.status_code != 404, "Query endpoint not found"
        assert response.status_code != 405, "Query endpoint doesn't accept POST"

        print(f"Response status: {response.status_code}")
        print(f"Response body: {response.text}")

    def test_query_endpoint_with_real_system(self, client):
        """Test query endpoint with actual system (may take a while due to AI call)"""
        response = client.post(
            "/api/query",
            json={"query": "What is MCP?", "session_id": "test_session_123"},
        )

        print(f"Query response status: {response.status_code}")
        print(
            f"Query response body: {response.json() if response.status_code == 200 else response.text}"
        )

        if response.status_code != 200:
            pytest.fail(
                f"Query endpoint failed with status {response.status_code}: {response.text}"
            )

        # Check response structure
        json_response = response.json()
        assert "answer" in json_response, "Response missing 'answer' field"
        assert "sources" in json_response, "Response missing 'sources' field"
        assert "session_id" in json_response, "Response missing 'session_id' field"

        # Check that we got a real answer, not "query failed"
        assert json_response["answer"] != "query failed", "Got 'query failed' response!"
        assert len(json_response["answer"].strip()) > 0, "Got empty answer"

    def test_query_endpoint_error_handling(self, client):
        """Test query endpoint error handling with invalid input"""
        # Test missing query field
        response = client.post("/api/query", json={})
        print(f"Missing query response: {response.status_code} - {response.text}")

        # Test invalid JSON
        response = client.post("/api/query", data="invalid json")
        print(f"Invalid JSON response: {response.status_code} - {response.text}")

        # Test empty query
        response = client.post("/api/query", json={"query": ""})
        print(f"Empty query response: {response.status_code} - {response.text}")

    def test_courses_endpoint(self, client):
        """Test the courses analytics endpoint"""
        response = client.get("/api/courses")

        print(f"Courses response status: {response.status_code}")
        print(
            f"Courses response: {response.json() if response.status_code == 200 else response.text}"
        )

        assert response.status_code == 200, f"Courses endpoint failed: {response.text}"

        json_response = response.json()
        assert "total_courses" in json_response
        assert "course_titles" in json_response
        assert json_response["total_courses"] > 0, "No courses found in analytics"

    def test_static_file_serving(self, client):
        """Test that static files are served correctly"""
        # Test main index page
        response = client.get("/")
        print(f"Root response status: {response.status_code}")

        # Should serve the frontend HTML
        assert response.status_code == 200, "Frontend not served correctly"
        assert "html" in response.headers.get("content-type", "").lower()

    @patch("app.rag_system")
    def test_query_endpoint_with_mocked_rag_system_error(self, mock_rag_system, client):
        """Test how the endpoint handles RAG system errors"""
        # Mock RAG system to raise an exception
        mock_rag_system.query.side_effect = Exception("Simulated RAG system error")

        response = client.post("/api/query", json={"query": "test"})

        print(f"Mocked error response: {response.status_code} - {response.text}")

        # Should return 500 internal server error
        assert response.status_code == 500

        # Check if this could be returning "query failed" somehow
        if "query failed" in response.text.lower():
            pytest.fail(
                "Found 'query failed' in error response - this might be the issue!"
            )

    @patch("app.rag_system")
    def test_query_endpoint_with_mocked_empty_response(self, mock_rag_system, client):
        """Test how the endpoint handles empty RAG responses"""
        # Mock RAG system to return empty response
        mock_rag_system.query.return_value = ("", [])
        mock_rag_system.session_manager.create_session.return_value = "test_session"

        response = client.post("/api/query", json={"query": "test"})

        print(f"Empty response: {response.status_code} - {response.json()}")

        # Should still return 200 with empty answer
        assert response.status_code == 200
        json_response = response.json()
        assert json_response["answer"] == ""

    def test_cors_headers(self, client):
        """Test that CORS headers are properly set"""
        # Test preflight request
        response = client.options("/api/query")
        print(f"CORS preflight response: {response.status_code}")
        print(f"CORS headers: {dict(response.headers)}")

        # Test actual request for CORS headers
        response = client.post("/api/query", json={"query": "test"})
        print(
            f"CORS headers in response: {response.headers.get('access-control-allow-origin')}"
        )

    def test_request_validation(self, client):
        """Test FastAPI request validation"""
        # Test various invalid requests
        test_cases = [
            {"data": None, "description": "None data"},
            {"data": {"query": None}, "description": "None query"},
            {"data": {"query": 123}, "description": "Non-string query"},
            {
                "data": {"query": "test", "session_id": 123},
                "description": "Non-string session_id",
            },
        ]

        for case in test_cases:
            response = client.post("/api/query", json=case["data"])
            print(
                f"{case['description']} - Status: {response.status_code}, Response: {response.text}"
            )

            # FastAPI should return 422 for validation errors
            if response.status_code == 422:
                continue  # Expected validation error
            elif "query failed" in response.text.lower():
                pytest.fail(
                    f"Got 'query failed' for {case['description']} - validation issue!"
                )


class TestFrontendBackendIntegration:
    """Test potential issues between frontend and backend"""

    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_javascript_can_call_api(self, client):
        """Simulate a call that might come from JavaScript frontend"""
        # This simulates what the frontend JavaScript might send
        response = client.post(
            "/api/query",
            json={"query": "What is MCP?"},
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Origin": "http://localhost:8000",  # Simulate frontend origin
            },
        )

        print(f"Frontend-style request status: {response.status_code}")
        print(
            f"Frontend-style request response: {response.json() if response.status_code == 200 else response.text}"
        )

        if response.status_code != 200:
            pytest.fail(f"Frontend-style request failed: {response.text}")

        # Check for the specific "query failed" error
        json_response = response.json()
        if json_response.get("answer") == "query failed":
            pytest.fail("Found the 'query failed' issue!")

    def test_concurrent_requests(self, client):
        """Test multiple concurrent requests to see if there are race conditions"""
        import concurrent.futures

        def make_request(query_num):
            response = client.post(
                "/api/query", json={"query": f"What is lesson {query_num}?"}
            )
            return response.status_code, (
                response.json() if response.status_code == 200 else response.text
            )

        # Make 5 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request, i) for i in range(5)]
            results = [
                future.result() for future in concurrent.futures.as_completed(futures)
            ]

        print("Concurrent request results:")
        for i, (status, response) in enumerate(results):
            print(f"Request {i}: Status {status}")
            if status == 200 and isinstance(response, dict):
                answer = response.get("answer", "")
                if answer == "query failed":
                    pytest.fail(f"Concurrent request {i} got 'query failed'!")
                print(f"  Answer length: {len(answer)} chars")
            else:
                print(f"  Error: {response}")
