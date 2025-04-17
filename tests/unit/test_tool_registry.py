"""tests for the ToolRegistry classes."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Union
from unittest.mock import MagicMock, patch

import mcp
import pytest
from mcp import ClientSession
from pydantic import BaseModel
from pydantic_core import PydanticUndefined

from portia.errors import DuplicateToolError, ToolNotFoundError
from portia.model import GenerativeModel
from portia.open_source_tools.llm_tool import LLMTool
from portia.open_source_tools.registry import open_source_tool_registry
from portia.tool_registry import (
    InMemoryToolRegistry,
    McpToolRegistry,
    PortiaToolRegistry,
    ToolRegistry,
    generate_pydantic_model_from_json_schema,
)
from tests.utils import MockMcpSessionWrapper, MockTool, get_test_tool_context

if TYPE_CHECKING:
    from collections.abc import Iterator

    from pytest_mock import MockerFixture


MOCK_TOOL_ID = "mock_tool"
OTHER_MOCK_TOOL_ID = "other_mock_tool"


def test_tool_registry_register_tool() -> None:
    """Test registering tools in the ToolRegistry."""
    tool_registry = ToolRegistry()
    tool_registry.with_tool(MockTool(id=MOCK_TOOL_ID))
    tool1 = tool_registry.get_tool(MOCK_TOOL_ID)
    assert tool1.id == MOCK_TOOL_ID

    with pytest.raises(ToolNotFoundError):
        tool_registry.get_tool("tool3")

    with pytest.raises(DuplicateToolError):
        tool_registry.with_tool(MockTool(id=MOCK_TOOL_ID))

    tool_registry.replace_tool(
        MockTool(
            id=MOCK_TOOL_ID,
            name="New Mock Tool",
        ),
    )
    tool2 = tool_registry.get_tool(MOCK_TOOL_ID)
    assert tool2.id == MOCK_TOOL_ID
    assert tool2.name == "New Mock Tool"


def test_tool_registry_get_and_plan_run() -> None:
    """Test getting and running tools in the InMemoryToolRegistry."""
    tool_registry = ToolRegistry()
    tool_registry.with_tool(MockTool(id=MOCK_TOOL_ID))
    tool1 = tool_registry.get_tool(MOCK_TOOL_ID)
    ctx = get_test_tool_context()
    tool1.run(ctx)


def test_tool_registry_get_tools() -> None:
    """Test the get_tools method of InMemoryToolRegistry."""
    tool_registry = ToolRegistry(
        [MockTool(id=MOCK_TOOL_ID), MockTool(id=OTHER_MOCK_TOOL_ID)],
    )
    tools = tool_registry.get_tools()
    assert len(tools) == 2
    assert any(tool.id == MOCK_TOOL_ID for tool in tools)
    assert any(tool.id == OTHER_MOCK_TOOL_ID for tool in tools)


def test_tool_registry_match_tools() -> None:
    """Test matching tools in the InMemoryToolRegistry."""
    tool_registry = ToolRegistry(
        [MockTool(id=MOCK_TOOL_ID), MockTool(id=OTHER_MOCK_TOOL_ID)],
    )

    # Test matching specific tool ID
    matched_tools = tool_registry.match_tools(tool_ids=[MOCK_TOOL_ID])
    assert len(matched_tools) == 1
    assert matched_tools[0].id == MOCK_TOOL_ID

    # Test matching multiple tool IDs
    matched_tools = tool_registry.match_tools(
        tool_ids=[MOCK_TOOL_ID, OTHER_MOCK_TOOL_ID],
    )
    assert len(matched_tools) == 2
    assert {tool.id for tool in matched_tools} == {MOCK_TOOL_ID, OTHER_MOCK_TOOL_ID}

    # Test matching non-existent tool ID
    matched_tools = tool_registry.match_tools(tool_ids=["non_existent_tool"])
    assert len(matched_tools) == 0

    # Test with no tool_ids (should return all tools)
    matched_tools = tool_registry.match_tools()
    assert len(matched_tools) == 2
    assert {tool.id for tool in matched_tools} == {MOCK_TOOL_ID, OTHER_MOCK_TOOL_ID}


def test_combined_tool_registry_duplicate_tool() -> None:
    """Test searching across multiple registries in ToolRegistry."""
    tool_registry = ToolRegistry([MockTool(id=MOCK_TOOL_ID)])
    other_tool_registry = ToolRegistry(
        [MockTool(id=MOCK_TOOL_ID)],
    )
    combined_tool_registry = tool_registry + other_tool_registry

    tool1 = combined_tool_registry.get_tool(MOCK_TOOL_ID)
    assert tool1.id == MOCK_TOOL_ID


def test_combined_tool_registry_get_tool() -> None:
    """Test searching across multiple registries in ToolRegistry."""
    tool_registry = ToolRegistry([MockTool(id=MOCK_TOOL_ID)])
    other_tool_registry = ToolRegistry(
        [MockTool(id=OTHER_MOCK_TOOL_ID)],
    )
    combined_tool_registry = tool_registry + other_tool_registry

    tool1 = combined_tool_registry.get_tool(MOCK_TOOL_ID)
    assert tool1.id == MOCK_TOOL_ID

    with pytest.raises(ToolNotFoundError):
        combined_tool_registry.get_tool("tool_not_found")


def test_combined_tool_registry_get_tools() -> None:
    """Test getting all tools from an ToolRegistry."""
    tool_registry = ToolRegistry([MockTool(id=MOCK_TOOL_ID)])
    other_tool_registry = ToolRegistry(
        [MockTool(id=OTHER_MOCK_TOOL_ID)],
    )
    combined_tool_registry = tool_registry + other_tool_registry

    tools = combined_tool_registry.get_tools()
    assert len(tools) == 2
    assert any(tool.id == MOCK_TOOL_ID for tool in tools)


def test_combined_tool_registry_match_tools() -> None:
    """Test matching tools across multiple registries in ToolRegistry."""
    tool_registry = ToolRegistry([MockTool(id=MOCK_TOOL_ID)])
    other_tool_registry = ToolRegistry(
        [MockTool(id=OTHER_MOCK_TOOL_ID)],
    )
    combined_tool_registry = tool_registry + other_tool_registry

    # Test matching specific tool IDs
    matched_tools = combined_tool_registry.match_tools(tool_ids=[MOCK_TOOL_ID])
    assert len(matched_tools) == 1
    assert matched_tools[0].id == MOCK_TOOL_ID

    # Test matching multiple tool IDs
    matched_tools = combined_tool_registry.match_tools(
        tool_ids=[MOCK_TOOL_ID, OTHER_MOCK_TOOL_ID],
    )
    assert len(matched_tools) == 2
    assert {tool.id for tool in matched_tools} == {MOCK_TOOL_ID, OTHER_MOCK_TOOL_ID}

    # Test matching non-existent tool IDs
    matched_tools = combined_tool_registry.match_tools(tool_ids=["non_existent_tool"])
    assert len(matched_tools) == 0


def test_tool_registry_add_operators(mocker: MockerFixture) -> None:
    """Test the __add__ and __radd__ operators for ToolRegistry."""
    # Mock the logger
    mock_logger = mocker.Mock()
    mocker.patch("portia.tool_registry.logger", return_value=mock_logger)

    # Create registries and tools
    registry1 = ToolRegistry([MockTool(id=MOCK_TOOL_ID)])
    registry2 = ToolRegistry([MockTool(id=OTHER_MOCK_TOOL_ID)])
    tool_list = [MockTool(id="tool3")]

    # Test registry + registry
    combined = registry1 + registry2
    assert isinstance(combined, ToolRegistry)
    assert len(combined.get_tools()) == 2
    assert {tool.id for tool in combined.get_tools()} == {MOCK_TOOL_ID, OTHER_MOCK_TOOL_ID}

    # Test registry + list
    combined = registry1 + tool_list  # type: ignore reportOperatorIssue
    assert isinstance(combined, ToolRegistry)
    assert len(combined.get_tools()) == 2
    assert {tool.id for tool in combined.get_tools()} == {MOCK_TOOL_ID, "tool3"}

    # Test list + registry (radd)
    combined = tool_list + registry1  # type: ignore reportOperatorIssue
    assert isinstance(combined, ToolRegistry)
    assert len(combined.get_tools()) == 2
    assert {tool.id for tool in combined.get_tools()} == {MOCK_TOOL_ID, "tool3"}

    # Test warning on duplicate tools
    duplicate_registry = ToolRegistry([MockTool(id=MOCK_TOOL_ID)])
    combined = registry1 + duplicate_registry
    mock_logger.warning.assert_called_once_with(
        f"Duplicate tool ID found: {MOCK_TOOL_ID}. Unintended behavior may occur.",
    )


def test_in_memory_tool_registry_from_local_tools() -> None:
    """Test creating an InMemoryToolRegistry from a list of local tools."""
    tool_registry = InMemoryToolRegistry.from_local_tools([MockTool(id=MOCK_TOOL_ID)])
    assert isinstance(tool_registry, InMemoryToolRegistry)
    assert len(tool_registry.get_tools()) == 1
    assert tool_registry.get_tool(MOCK_TOOL_ID).id == MOCK_TOOL_ID


def test_tool_registry_filter_tools() -> None:
    """Test filtering tools in a ToolRegistry."""
    tool_registry = ToolRegistry([MockTool(id=MOCK_TOOL_ID), MockTool(id=OTHER_MOCK_TOOL_ID)])
    filtered_registry = tool_registry.filter_tools(lambda tool: tool.id == MOCK_TOOL_ID)
    filtered_tools = filtered_registry.get_tools()
    assert len(filtered_tools) == 1
    assert filtered_tools[0].id == MOCK_TOOL_ID


def test_portia_tool_registry_missing_required_args() -> None:
    """Test that PortiaToolRegistry raises an error if required args are missing."""
    with pytest.raises(ValueError, match="Either config, client or tools must be provided"):
        PortiaToolRegistry()


def test_tool_registry_reconfigure_llm_tool() -> None:
    """Test replacing the LLMTool with a new LLMTool."""
    registry = ToolRegistry(open_source_tool_registry.get_tools())
    llm_tool = registry.get_tool("llm_tool")

    assert llm_tool is not None
    assert getattr(llm_tool, "model", None) is None

    registry.replace_tool(LLMTool(model=MagicMock(spec=GenerativeModel)))

    llm_tool = registry.get_tool("llm_tool")
    assert llm_tool is not None
    assert getattr(llm_tool, "model", None) is not None


@pytest.fixture
def mock_get_mcp_session() -> Iterator[None]:
    """Fixture to mock the get_mcp_session function."""
    mock_session = MagicMock(spec=ClientSession)
    mock_session.list_tools.return_value = mcp.ListToolsResult(
        tools=[
            mcp.Tool(
                name="test_tool",
                description="I am a tool",
                inputSchema={"type": "object", "properties": {"input": {"type": "string"}}},
            ),
            mcp.Tool(
                name="test_tool_2",
                description="I am another tool",
                inputSchema={"type": "object", "properties": {"input": {"type": "number"}}},
            ),
        ],
    )

    with patch(
        "portia.tool_registry.get_mcp_session",
        new=MockMcpSessionWrapper(mock_session).mock_mcp_session,
    ):
        yield


@pytest.fixture
def mcp_tool_registry(mock_get_mcp_session: None) -> McpToolRegistry:  # noqa: ARG001
    """Fixture for a McpToolRegistry."""
    return McpToolRegistry.from_stdio_connection(
        server_name="mock_mcp",
        command="test",
        args=["test"],
    )


@pytest.fixture
@pytest.mark.asyncio
async def mcp_tool_registry_in_async(mock_get_mcp_session: None) -> McpToolRegistry:  # noqa: ARG001
    """Fixture for a McpToolRegistry."""
    loop = asyncio.get_running_loop()
    assert loop.is_running()
    return McpToolRegistry.from_stdio_connection(
        server_name="mock_mcp",
        command="test",
        args=["test"],
    )


@pytest.mark.usefixtures("mock_get_mcp_session")
def test_mcp_tool_registry_from_sse_connection() -> None:
    """Test constructing a McpToolRegistry from an SSE connection."""
    mcp_registry_sse = McpToolRegistry.from_sse_connection(
        server_name="mock_mcp",
        url="http://localhost:8000",
    )
    assert isinstance(mcp_registry_sse, McpToolRegistry)


def test_mcp_tool_registry_get_tools(mcp_tool_registry: McpToolRegistry) -> None:
    """Test getting tools from the MCPToolRegistry."""
    tools = mcp_tool_registry.get_tools()
    assert len(tools) == 2
    assert tools[0].id == "mcp:mock_mcp:test_tool"
    assert tools[0].name == "test_tool"
    assert tools[0].description == "I am a tool"
    assert issubclass(tools[0].args_schema, BaseModel)
    assert tools[1].id == "mcp:mock_mcp:test_tool_2"
    assert tools[1].name == "test_tool_2"
    assert tools[1].description == "I am another tool"
    assert issubclass(tools[1].args_schema, BaseModel)


def test_mcp_tool_registry_get_tool(mcp_tool_registry: McpToolRegistry) -> None:
    """Test getting a tool from the MCPToolRegistry."""
    tool = mcp_tool_registry.get_tool("mcp:mock_mcp:test_tool")
    assert tool.id == "mcp:mock_mcp:test_tool"
    assert tool.name == "test_tool"
    assert tool.description == "I am a tool"
    assert issubclass(tool.args_schema, BaseModel)


def test_generate_pydantic_model_from_json_schema() -> None:
    """Test generating a Pydantic model from a JSON schema."""
    json_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "The name of the user"},
            "age": {"type": "integer", "description": "The age of the user"},
            "height": {"type": "number", "description": "The height of the user", "default": 185.2},
            "is_active": {"type": "boolean", "description": "Whether the user is active"},
            "pets": {
                "type": "array",
                "items": {"type": "string"},
                "description": "The pets of the user",
            },
            "address": {
                "type": "object",
                "properties": {
                    "street": {"type": "string", "description": "The street of the user"},
                    "city": {"type": "string", "description": "The city of the user"},
                    "zip": {"type": "string", "description": "The zip of the user"},
                },
                "description": "The address of the user",
                "required": ["city", "zip"],
            },
        },
        "required": ["name", "age"],
    }
    model = generate_pydantic_model_from_json_schema("TestModel", json_schema)
    assert model.model_fields["name"].annotation is str
    assert model.model_fields["name"].default is PydanticUndefined
    assert model.model_fields["name"].description == "The name of the user"
    assert model.model_fields["age"].annotation is int
    assert model.model_fields["age"].default is PydanticUndefined
    assert model.model_fields["age"].description == "The age of the user"
    assert model.model_fields["height"].annotation is float
    assert model.model_fields["height"].default == 185.2
    assert model.model_fields["height"].description == "The height of the user"
    assert model.model_fields["is_active"].annotation is bool
    assert model.model_fields["is_active"].default is None
    assert model.model_fields["is_active"].description == "Whether the user is active"
    assert model.model_fields["pets"].annotation == list[str]
    assert model.model_fields["pets"].default is None
    assert model.model_fields["pets"].description == "The pets of the user"
    address_type = model.model_fields["address"].annotation
    assert isinstance(address_type, type)
    assert issubclass(address_type, BaseModel)
    assert address_type.model_fields["street"].annotation is str
    assert address_type.model_fields["street"].default is None
    assert address_type.model_fields["street"].description == "The street of the user"
    assert address_type.model_fields["city"].annotation is str
    assert address_type.model_fields["city"].default is PydanticUndefined
    assert address_type.model_fields["city"].description == "The city of the user"
    assert address_type.model_fields["zip"].annotation is str
    assert address_type.model_fields["zip"].default is PydanticUndefined
    assert address_type.model_fields["zip"].description == "The zip of the user"
    assert model.model_fields["address"].default is None
    assert model.model_fields["address"].description == "The address of the user"


def test_generate_pydantic_model_from_json_schema_union_types() -> None:
    """Test generating a Pydantic model from a JSON schema with union types."""
    json_schema = {
        "type": "object",
        "properties": {
            "collaborators": {
                "anyOf": [
                    {"items": {"type": "integer"}, "type": "array"},
                    {"type": "null"},
                ],
                "default": None,
                "description": "Array of user IDs to CC on the ticket",
                "title": "Collaborator Ids",
            },
            "company_number": {
                "anyOf": [
                    {"type": "string"},
                    {"type": "integer"},
                ],
                "description": "Company number to search",
                "title": "Company Number",
            },
            "additional_company_numbers": {
                "type": "array",
                "items": {"oneOf": [{"type": "string"}, {"type": "integer"}]},
                "description": "Additional company numbers to search",
                "title": "Additional Company Numbers",
            },
        },
        "required": ["company_number"],
    }
    model = generate_pydantic_model_from_json_schema("TestUnionModel", json_schema)
    assert model.model_fields["collaborators"].annotation == Union[list[int], None]
    assert model.model_fields["collaborators"].default is None
    assert (
        model.model_fields["collaborators"].description == "Array of user IDs to CC on the ticket"
    )
    assert model.model_fields["company_number"].annotation == Union[str, int]
    assert model.model_fields["company_number"].default is PydanticUndefined
    assert model.model_fields["company_number"].description == "Company number to search"
    assert model.model_fields["additional_company_numbers"].annotation == list[Union[str, int]]
    assert model.model_fields["additional_company_numbers"].default is None
    assert (
        model.model_fields["additional_company_numbers"].description
        == "Additional company numbers to search"
    )


def test_generate_pydantic_model_from_json_schema_doesnt_handle_none_for_non_union_fields() -> None:
    """Test for generate_pydantic_model_from_json_schema.

    Test that generate_pydantic_model_from_json_schema maps 'null' to Any for non-union fields.
    """
    json_schema = {
        "type": "object",
        "properties": {
            "name": {
                "type": "null",
                "default": None,
                "description": "Array of user IDs to CC on the ticket",
            },
            "unknown_field_type": {
                "type": "random_type",
                "default": None,
                "description": "Array of user IDs to CC on the ticket",
            },
        },
    }
    model = generate_pydantic_model_from_json_schema("TestNullSchema", json_schema)
    assert model.model_fields["name"].annotation is Any
    assert model.model_fields["unknown_field_type"].annotation is Any


def test_generate_pydantic_model_from_json_schema_not_single_type_or_union_field() -> None:
    """Test for generate_pydantic_model_from_json_schema.

    Check it represents fields that are neither single type or union fields as Any type.
    """
    json_schema = {
        "type": "object",
        "properties": {
            "unknown": {
                "default": None,
                "description": "Array of user IDs to CC on the ticket",
            },
        },
    }
    model = generate_pydantic_model_from_json_schema("TestNullSchema", json_schema)
    assert model.model_fields["unknown"].annotation is Any
