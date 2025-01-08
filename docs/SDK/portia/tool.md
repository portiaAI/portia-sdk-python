---
sidebar_label: tool
title: portia.tool
---

Tools module.

This module defines an abstract base class for tools that can be extended to create custom tools
Each tool has a unique ID and a name, and child classes should implement the `run` method
with their specific logic.

## Tool Objects

```python
class Tool(BaseModel, Generic[SERIALIZABLE_TYPE_VAR])
```

Abstract base class for a tool.

This class serves as the blueprint for all tools. Child classes must implement the `run` method.

Attributes:
    id (str): A unique identifier for the tool.
    name (str): The name of the tool.
    description (str): Purpose of the tool and usage.

#### run

```python
@abstractmethod
def run(*args: Any, **kwargs: Any) -> SERIALIZABLE_TYPE_VAR | Clarification
```

Run the tool.

This method must be implemented by subclasses to define the tool&#x27;s specific behavior.

Args:
    args (Any): The arguments passed to the tool for execution.
    kwargs (Any): The keyword arguments passed to the tool for execution.

Returns:
    Any: The result of the tool&#x27;s execution.

#### check\_description\_length

```python
@model_validator(mode="after")
def check_description_length() -> Tool
```

Check that the description is less than 1024 characters.

#### to\_langchain

```python
def to_langchain(return_artifact: bool = False) -> StructuredTool
```

Return a LangChain representation of this tool.

Langchain agent needs to use the &quot;content&quot; response format, but Langgraph
prefers the other.

#### args\_json\_schema

```python
def args_json_schema() -> dict[str, Any]
```

Return the json_schema for the tool args.

## PortiaRemoteTool Objects

```python
class PortiaRemoteTool(Tool, Generic[SERIALIZABLE_TYPE_VAR])
```

Tool that passes run execution to Portia Cloud.

#### run

```python
def run(*args: Any, **kwargs: Any) -> SERIALIZABLE_TYPE_VAR | Clarification
```

Invoke the run endpoint and handle response.

