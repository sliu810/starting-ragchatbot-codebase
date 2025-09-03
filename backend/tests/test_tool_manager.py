import os
import sys
from unittest.mock import Mock

import pytest

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from search_tools import Tool, ToolManager


class MockTool(Tool):
    """Mock tool for testing"""

    def __init__(self, name="test_tool"):
        self.name = name
        self.last_sources = []

    def get_tool_definition(self):
        return {
            "name": self.name,
            "description": f"A test tool named {self.name}",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Test query"}
                },
                "required": ["query"],
            },
        }

    def execute(self, **kwargs):
        return f"Executed {self.name} with {kwargs}"


class TestToolManager:
    """Test suite for ToolManager"""

    @pytest.fixture
    def tool_manager(self):
        """Create a fresh ToolManager for each test"""
        return ToolManager()

    @pytest.fixture
    def mock_tool(self):
        """Create a mock tool for testing"""
        return MockTool("test_tool")

    def test_register_tool(self, tool_manager, mock_tool):
        """Test tool registration"""
        tool_manager.register_tool(mock_tool)

        assert "test_tool" in tool_manager.tools
        assert tool_manager.tools["test_tool"] == mock_tool

    def test_register_multiple_tools(self, tool_manager):
        """Test registering multiple tools"""
        tool1 = MockTool("tool_one")
        tool2 = MockTool("tool_two")

        tool_manager.register_tool(tool1)
        tool_manager.register_tool(tool2)

        assert len(tool_manager.tools) == 2
        assert "tool_one" in tool_manager.tools
        assert "tool_two" in tool_manager.tools

    def test_register_tool_without_name(self, tool_manager):
        """Test error handling when tool has no name"""
        bad_tool = Mock()
        bad_tool.get_tool_definition.return_value = {"description": "No name"}

        with pytest.raises(ValueError, match="Tool must have a 'name'"):
            tool_manager.register_tool(bad_tool)

    def test_get_tool_definitions(self, tool_manager):
        """Test getting all tool definitions"""
        tool1 = MockTool("tool_one")
        tool2 = MockTool("tool_two")

        tool_manager.register_tool(tool1)
        tool_manager.register_tool(tool2)

        definitions = tool_manager.get_tool_definitions()

        assert len(definitions) == 2
        tool_names = [def_["name"] for def_ in definitions]
        assert "tool_one" in tool_names
        assert "tool_two" in tool_names

    def test_execute_tool(self, tool_manager, mock_tool):
        """Test tool execution"""
        tool_manager.register_tool(mock_tool)

        result = tool_manager.execute_tool("test_tool", query="hello", param="world")

        assert result == "Executed test_tool with {'query': 'hello', 'param': 'world'}"

    def test_execute_nonexistent_tool(self, tool_manager):
        """Test execution of nonexistent tool"""
        result = tool_manager.execute_tool("nonexistent_tool", query="test")

        assert result == "Tool 'nonexistent_tool' not found"

    def test_get_last_sources_empty(self, tool_manager):
        """Test getting sources when no tools have sources"""
        tool1 = MockTool("tool_one")
        tool_manager.register_tool(tool1)

        sources = tool_manager.get_last_sources()
        assert sources == []

    def test_get_last_sources_with_sources(self, tool_manager):
        """Test getting sources from tools that have them"""
        tool1 = MockTool("tool_one")
        tool1.last_sources = [{"text": "Source 1"}, {"text": "Source 2"}]

        tool2 = MockTool("tool_two")
        tool2.last_sources = []

        tool_manager.register_tool(tool1)
        tool_manager.register_tool(tool2)

        sources = tool_manager.get_last_sources()
        assert sources == [{"text": "Source 1"}, {"text": "Source 2"}]

    def test_get_last_sources_multiple_tools_with_sources(self, tool_manager):
        """Test that only the first tool with sources is returned"""
        tool1 = MockTool("tool_one")
        tool1.last_sources = [{"text": "Source from tool1"}]

        tool2 = MockTool("tool_two")
        tool2.last_sources = [{"text": "Source from tool2"}]

        tool_manager.register_tool(tool1)
        tool_manager.register_tool(tool2)

        sources = tool_manager.get_last_sources()
        # Should return sources from first tool that has them
        # (order depends on dict iteration, but should be consistent)
        assert len(sources) == 1
        assert sources[0]["text"] in ["Source from tool1", "Source from tool2"]

    def test_reset_sources(self, tool_manager):
        """Test resetting sources from all tools"""
        tool1 = MockTool("tool_one")
        tool1.last_sources = [{"text": "Source 1"}]

        tool2 = MockTool("tool_two")
        tool2.last_sources = [{"text": "Source 2"}]

        tool_manager.register_tool(tool1)
        tool_manager.register_tool(tool2)

        # Verify sources exist
        assert len(tool1.last_sources) > 0
        assert len(tool2.last_sources) > 0

        # Reset sources
        tool_manager.reset_sources()

        # Verify sources are cleared
        assert tool1.last_sources == []
        assert tool2.last_sources == []

    def test_reset_sources_tools_without_sources_attribute(self, tool_manager):
        """Test reset_sources doesn't fail on tools without last_sources"""
        # Create a tool without last_sources attribute
        tool_without_sources = Mock()
        tool_without_sources.get_tool_definition.return_value = {
            "name": "no_sources_tool"
        }

        tool_with_sources = MockTool("with_sources")
        tool_with_sources.last_sources = [{"text": "test"}]

        tool_manager.register_tool(tool_without_sources)
        tool_manager.register_tool(tool_with_sources)

        # Should not raise an error
        tool_manager.reset_sources()

        # Tool with sources should be cleared
        assert tool_with_sources.last_sources == []

    def test_tool_manager_empty_state(self, tool_manager):
        """Test tool manager in empty state"""
        assert len(tool_manager.tools) == 0
        assert tool_manager.get_tool_definitions() == []
        assert tool_manager.get_last_sources() == []

        # Reset on empty should not fail
        tool_manager.reset_sources()

    def test_tool_replacement(self, tool_manager):
        """Test that registering a tool with the same name replaces the old one"""
        tool1 = MockTool("same_name")
        tool2 = MockTool("same_name")

        tool_manager.register_tool(tool1)
        assert tool_manager.tools["same_name"] == tool1

        tool_manager.register_tool(tool2)
        assert tool_manager.tools["same_name"] == tool2
        assert len(tool_manager.tools) == 1
