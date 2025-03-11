"""tests for the ToolRegistry classes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pydantic import BaseModel
from pydantic_core import PydanticUndefined

from portia.errors import DuplicateToolError, ToolNotFoundError
from portia.tool import Tool
from portia.tool_registry import (
    AggregatedToolRegistry,
    InMemoryToolRegistry,
    ToolRegistry,
    generate_pydantic_model_from_json_schema,
)
from tests.utils import AdditionTool, MockTool, get_test_tool_context

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

    from portia.tool import Tool


MOCK_TOOL_ID = "mock_tool"
OTHER_MOCK_TOOL_ID = "other_mock_tool"


def test_registry_base_classes() -> None:
    """Test registry raises."""

    class MyRegistry(ToolRegistry):
        """Override to test base."""

        def get_tools(self) -> list[Tool]:
            return super().get_tools()  # type: ignore  # noqa: PGH003

        def get_tool(self, tool_id: str) -> Tool:
            return super().get_tool(tool_id)  # type: ignore  # noqa: PGH003

        def register_tool(self, tool: Tool) -> None:
            return super().register_tool(tool)  # type: ignore  # noqa: PGH003

        def match_tools(
            self,
            query: str | None = None,
            tool_ids: list[str] | None = None,
        ) -> list[Tool]:
            return super().match_tools(query, tool_ids)

    registry = MyRegistry()

    with pytest.raises(NotImplementedError):
        registry.get_tools()

    with pytest.raises(NotImplementedError):
        registry.get_tool("1")

    with pytest.raises(NotImplementedError):
        registry.register_tool(AdditionTool())

    with pytest.raises(NotImplementedError):
        registry.match_tools("match")

    agg_registry = AggregatedToolRegistry(registries=[registry])
    with pytest.raises(NotImplementedError):
        agg_registry.register_tool(AdditionTool())


def test_local_tool_registry_register_tool() -> None:
    """Test registering tools in the InMemoryToolRegistry."""
    local_tool_registry = InMemoryToolRegistry()
    local_tool_registry.register_tool(MockTool(id=MOCK_TOOL_ID))
    tool1 = local_tool_registry.get_tool(MOCK_TOOL_ID)
    assert tool1.id == MOCK_TOOL_ID

    with pytest.raises(ToolNotFoundError):
        local_tool_registry.get_tool("tool3")

    with pytest.raises(DuplicateToolError):
        local_tool_registry.register_tool(MockTool(id=MOCK_TOOL_ID))


def test_local_tool_registry_get_and_plan_run() -> None:
    """Test getting and running tools in the InMemoryToolRegistry."""
    local_tool_registry = InMemoryToolRegistry()
    local_tool_registry.register_tool(MockTool(id=MOCK_TOOL_ID))
    tool1 = local_tool_registry.get_tool(MOCK_TOOL_ID)
    ctx = get_test_tool_context()
    tool1.run(ctx)


def test_local_tool_registry_get_tools() -> None:
    """Test the get_tools method of InMemoryToolRegistry."""
    local_tool_registry = InMemoryToolRegistry.from_local_tools(
        [MockTool(id=MOCK_TOOL_ID), MockTool(id=OTHER_MOCK_TOOL_ID)],
    )
    tools = local_tool_registry.get_tools()
    assert len(tools) == 2
    assert any(tool.id == MOCK_TOOL_ID for tool in tools)
    assert any(tool.id == OTHER_MOCK_TOOL_ID for tool in tools)


def test_local_tool_registry_match_tools() -> None:
    """Test matching tools in the InMemoryToolRegistry."""
    local_tool_registry = InMemoryToolRegistry.from_local_tools(
        [MockTool(id=MOCK_TOOL_ID), MockTool(id=OTHER_MOCK_TOOL_ID)],
    )

    # Test matching specific tool ID
    matched_tools = local_tool_registry.match_tools(tool_ids=[MOCK_TOOL_ID])
    assert len(matched_tools) == 1
    assert matched_tools[0].id == MOCK_TOOL_ID

    # Test matching multiple tool IDs
    matched_tools = local_tool_registry.match_tools(
        tool_ids=[MOCK_TOOL_ID, OTHER_MOCK_TOOL_ID],
    )
    assert len(matched_tools) == 2
    assert {tool.id for tool in matched_tools} == {MOCK_TOOL_ID, OTHER_MOCK_TOOL_ID}

    # Test matching non-existent tool ID
    matched_tools = local_tool_registry.match_tools(tool_ids=["non_existent_tool"])
    assert len(matched_tools) == 0

    # Test with no tool_ids (should return all tools)
    matched_tools = local_tool_registry.match_tools()
    assert len(matched_tools) == 2
    assert {tool.id for tool in matched_tools} == {MOCK_TOOL_ID, OTHER_MOCK_TOOL_ID}


def test_aggregated_tool_registry_duplicate_tool() -> None:
    """Test searching across multiple registries in AggregatedToolRegistry."""
    local_tool_registry = InMemoryToolRegistry.from_local_tools([MockTool(id=MOCK_TOOL_ID)])
    other_tool_registry = InMemoryToolRegistry.from_local_tools(
        [MockTool(id=MOCK_TOOL_ID)],
    )
    aggregated_tool_registry = local_tool_registry + other_tool_registry

    tool1 = aggregated_tool_registry.get_tool(MOCK_TOOL_ID)
    assert tool1.id == MOCK_TOOL_ID


def test_aggregated_tool_registry_get_tool() -> None:
    """Test searching across multiple registries in AggregatedToolRegistry."""
    local_tool_registry = InMemoryToolRegistry.from_local_tools([MockTool(id=MOCK_TOOL_ID)])
    other_tool_registry = InMemoryToolRegistry.from_local_tools(
        [MockTool(id=OTHER_MOCK_TOOL_ID)],
    )
    aggregated_tool_registry = local_tool_registry + other_tool_registry

    tool1 = aggregated_tool_registry.get_tool(MOCK_TOOL_ID)
    assert tool1.id == MOCK_TOOL_ID

    with pytest.raises(ToolNotFoundError):
        aggregated_tool_registry.get_tool("tool_not_found")


def test_aggregated_tool_registry_get_tools() -> None:
    """Test getting all tools from an AggregatedToolRegistry."""
    local_tool_registry = InMemoryToolRegistry.from_local_tools([MockTool(id=MOCK_TOOL_ID)])
    other_tool_registry = InMemoryToolRegistry.from_local_tools(
        [MockTool(id=OTHER_MOCK_TOOL_ID)],
    )
    aggregated_tool_registry = local_tool_registry + other_tool_registry

    tools = aggregated_tool_registry.get_tools()
    assert len(tools) == 2
    assert any(tool.id == MOCK_TOOL_ID for tool in tools)


def test_aggregated_tool_registry_match_tools() -> None:
    """Test matching tools across multiple registries in AggregatedToolRegistry."""
    local_tool_registry = InMemoryToolRegistry.from_local_tools([MockTool(id=MOCK_TOOL_ID)])
    other_tool_registry = InMemoryToolRegistry.from_local_tools(
        [MockTool(id=OTHER_MOCK_TOOL_ID)],
    )
    aggregated_tool_registry = local_tool_registry + other_tool_registry

    # Test matching specific tool IDs
    matched_tools = aggregated_tool_registry.match_tools(tool_ids=[MOCK_TOOL_ID])
    assert len(matched_tools) == 1
    assert matched_tools[0].id == MOCK_TOOL_ID

    # Test matching multiple tool IDs
    matched_tools = aggregated_tool_registry.match_tools(
        tool_ids=[MOCK_TOOL_ID, OTHER_MOCK_TOOL_ID],
    )
    assert len(matched_tools) == 2
    assert {tool.id for tool in matched_tools} == {MOCK_TOOL_ID, OTHER_MOCK_TOOL_ID}

    # Test matching non-existent tool IDs
    matched_tools = aggregated_tool_registry.match_tools(tool_ids=["non_existent_tool"])
    assert len(matched_tools) == 0


def test_tool_registry_add_operators(mocker: MockerFixture) -> None:
    """Test the __add__ and __radd__ operators for ToolRegistry."""
    # Mock the logger
    mock_logger = mocker.Mock()
    mocker.patch("portia.tool_registry.logger", return_value=mock_logger)

    # Create registries and tools
    registry1 = InMemoryToolRegistry.from_local_tools([MockTool(id=MOCK_TOOL_ID)])
    registry2 = InMemoryToolRegistry.from_local_tools([MockTool(id=OTHER_MOCK_TOOL_ID)])
    tool_list = [MockTool(id="tool3")]

    # Test registry + registry
    combined = registry1 + registry2
    assert isinstance(combined, AggregatedToolRegistry)
    assert len(combined.get_tools()) == 2
    assert {tool.id for tool in combined.get_tools()} == {MOCK_TOOL_ID, OTHER_MOCK_TOOL_ID}

    # Test registry + list
    combined = registry1 + tool_list  # type: ignore reportOperatorIssue
    assert isinstance(combined, AggregatedToolRegistry)
    assert len(combined.get_tools()) == 2
    assert {tool.id for tool in combined.get_tools()} == {MOCK_TOOL_ID, "tool3"}

    # Test list + registry (radd)
    combined = tool_list + registry1  # type: ignore reportOperatorIssue
    assert isinstance(combined, AggregatedToolRegistry)
    assert len(combined.get_tools()) == 2
    assert {tool.id for tool in combined.get_tools()} == {MOCK_TOOL_ID, "tool3"}

    # Test warning on duplicate tools
    duplicate_registry = InMemoryToolRegistry.from_local_tools([MockTool(id=MOCK_TOOL_ID)])
    combined = registry1 + duplicate_registry
    mock_logger.warning.assert_called_once_with(
        f"Duplicate tool ID found: {MOCK_TOOL_ID}. Unintended behavior may occur.",
    )


def test_generate_pydantic_model_from_json_schema() -> None:
    """Test generating a Pydantic model from a JSON schema."""
    json_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "The name of the user"},
            "age": {"type": "integer", "description": "The age of the user"},
            "height": {"type": "number", "description": "The height of the user"},
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
    assert model.model_fields["height"].default is None
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
