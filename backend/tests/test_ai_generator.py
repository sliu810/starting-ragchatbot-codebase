import os
import sys
from unittest.mock import MagicMock, Mock, patch

import pytest

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_generator import AIGenerator


class TestAIGenerator:
    """Test suite for AIGenerator"""

    @pytest.fixture
    def mock_anthropic_client(self):
        """Create a mock Anthropic client"""
        with patch("ai_generator.anthropic.Anthropic") as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            yield mock_client

    @pytest.fixture
    def ai_generator(self, mock_anthropic_client):
        """Create an AIGenerator with mocked Anthropic client"""
        generator = AIGenerator("fake-api-key", "claude-3-sonnet-20240229")
        generator.client = mock_anthropic_client
        return generator

    def test_generate_response_without_tools(self, ai_generator, mock_anthropic_client):
        """Test basic response generation without tools"""
        # Setup mock response
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "This is a general knowledge answer"
        mock_response.stop_reason = "end_turn"
        mock_anthropic_client.messages.create.return_value = mock_response

        result = ai_generator.generate_response("What is Python?")

        # Verify API was called correctly
        mock_anthropic_client.messages.create.assert_called_once()
        call_args = mock_anthropic_client.messages.create.call_args

        assert call_args[1]["model"] == "claude-3-sonnet-20240229"
        assert call_args[1]["messages"][0]["content"] == "What is Python?"
        assert "tools" not in call_args[1]
        assert result == "This is a general knowledge answer"

    def test_generate_response_with_conversation_history(
        self, ai_generator, mock_anthropic_client
    ):
        """Test response generation with conversation history"""
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "Follow-up answer"
        mock_response.stop_reason = "end_turn"
        mock_anthropic_client.messages.create.return_value = mock_response

        history = "User: Previous question\nAssistant: Previous answer"
        result = ai_generator.generate_response(
            "Follow-up question", conversation_history=history
        )

        call_args = mock_anthropic_client.messages.create.call_args
        assert "Previous conversation:" in call_args[1]["system"]
        assert history in call_args[1]["system"]

    def test_generate_response_with_tools_no_tool_use(
        self, ai_generator, mock_anthropic_client
    ):
        """Test response with tools available but not used"""
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "Direct answer without tool use"
        mock_response.stop_reason = "end_turn"
        mock_anthropic_client.messages.create.return_value = mock_response

        tools = [{"name": "test_tool", "description": "A test tool"}]
        result = ai_generator.generate_response("General question", tools=tools)

        call_args = mock_anthropic_client.messages.create.call_args
        assert call_args[1]["tools"] == tools
        assert call_args[1]["tool_choice"] == {"type": "auto"}
        assert result == "Direct answer without tool use"

    def test_generate_response_with_tool_use(self, ai_generator, mock_anthropic_client):
        """Test response generation when Claude uses a tool"""
        # Mock initial response with tool use
        mock_initial_response = Mock()
        mock_initial_response.stop_reason = "tool_use"
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "search_course_content"
        mock_tool_block.id = "tool_123"
        mock_tool_block.input = {"query": "MCP introduction"}
        mock_initial_response.content = [mock_tool_block]

        # Mock final response after tool execution
        mock_final_response = Mock()
        mock_final_response.content = [Mock()]
        mock_final_response.content[0].text = "Here's what I found about MCP..."

        # Setup client to return different responses for each call
        mock_anthropic_client.messages.create.side_effect = [
            mock_initial_response,
            mock_final_response,
        ]

        # Mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "MCP is a framework for..."

        tools = [
            {"name": "search_course_content", "description": "Search course content"}
        ]
        result = ai_generator.generate_response(
            "Tell me about MCP", tools=tools, tool_manager=mock_tool_manager
        )

        # Verify tool was executed
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content", query="MCP introduction"
        )

        # Verify second API call was made with tool results
        assert mock_anthropic_client.messages.create.call_count == 2

        # Check that tool results were included in final call
        final_call_args = mock_anthropic_client.messages.create.call_args_list[1]
        messages = final_call_args[1]["messages"]

        # Should have 3 messages: user query, assistant tool use, tool results
        assert len(messages) == 3
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"

        # Tool result should be in the final user message
        tool_result = messages[2]["content"][0]
        assert tool_result["type"] == "tool_result"
        assert tool_result["tool_use_id"] == "tool_123"
        assert tool_result["content"] == "MCP is a framework for..."

        assert result == "Here's what I found about MCP..."

    def test_generate_response_multiple_tool_calls(
        self, ai_generator, mock_anthropic_client
    ):
        """Test handling multiple tool calls in one response"""
        # Mock response with multiple tool uses
        mock_initial_response = Mock()
        mock_initial_response.stop_reason = "tool_use"

        mock_tool_block1 = Mock()
        mock_tool_block1.type = "tool_use"
        mock_tool_block1.name = "search_course_content"
        mock_tool_block1.id = "tool_1"
        mock_tool_block1.input = {"query": "first query"}

        mock_tool_block2 = Mock()
        mock_tool_block2.type = "tool_use"
        mock_tool_block2.name = "get_course_outline"
        mock_tool_block2.id = "tool_2"
        mock_tool_block2.input = {"course_title": "MCP Course"}

        mock_initial_response.content = [mock_tool_block1, mock_tool_block2]

        mock_final_response = Mock()
        mock_final_response.content = [Mock()]
        mock_final_response.content[0].text = "Combined response from both tools"

        mock_anthropic_client.messages.create.side_effect = [
            mock_initial_response,
            mock_final_response,
        ]

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = [
            "Search result content",
            "Course outline content",
        ]

        tools = [
            {"name": "search_course_content", "description": "Search content"},
            {"name": "get_course_outline", "description": "Get outline"},
        ]

        result = ai_generator.generate_response(
            "Tell me about MCP course and its outline",
            tools=tools,
            tool_manager=mock_tool_manager,
        )

        # Verify both tools were executed
        assert mock_tool_manager.execute_tool.call_count == 2
        mock_tool_manager.execute_tool.assert_any_call(
            "search_course_content", query="first query"
        )
        mock_tool_manager.execute_tool.assert_any_call(
            "get_course_outline", course_title="MCP Course"
        )

        # Verify tool results were included
        final_call_args = mock_anthropic_client.messages.create.call_args_list[1]
        tool_results = final_call_args[1]["messages"][2]["content"]
        assert len(tool_results) == 2

        assert result == "Combined response from both tools"

    def test_system_prompt_contains_tool_guidance(self, ai_generator):
        """Test that system prompt contains proper tool usage guidance"""
        system_prompt = ai_generator.SYSTEM_PROMPT

        # Check for tool-related instructions
        assert "Course outline requests" in system_prompt
        assert "get_course_outline" in system_prompt
        assert "search_course_content" in system_prompt
        assert "One tool use per query maximum" in system_prompt

    def test_base_params_configuration(self, ai_generator):
        """Test that base parameters are configured correctly"""
        assert ai_generator.base_params["model"] == "claude-3-sonnet-20240229"
        assert ai_generator.base_params["temperature"] == 0
        assert ai_generator.base_params["max_tokens"] == 800

    def test_handle_tool_execution_error(self, ai_generator, mock_anthropic_client):
        """Test handling of tool execution errors"""
        mock_initial_response = Mock()
        mock_initial_response.stop_reason = "tool_use"
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "search_course_content"
        mock_tool_block.id = "tool_123"
        mock_tool_block.input = {"query": "test"}
        mock_initial_response.content = [mock_tool_block]

        mock_final_response = Mock()
        mock_final_response.content = [Mock()]
        mock_final_response.content[0].text = "Error handling response"

        mock_anthropic_client.messages.create.side_effect = [
            mock_initial_response,
            mock_final_response,
        ]

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Tool execution failed"

        tools = [{"name": "search_course_content", "description": "Search content"}]
        result = ai_generator.generate_response(
            "Test query", tools=tools, tool_manager=mock_tool_manager
        )

        # Even if tool returns an error message, should still get final response
        assert result == "Error handling response"

        # Verify error message was passed to Claude
        final_call_args = mock_anthropic_client.messages.create.call_args_list[1]
        tool_result_content = final_call_args[1]["messages"][2]["content"][0]["content"]
        assert tool_result_content == "Tool execution failed"
