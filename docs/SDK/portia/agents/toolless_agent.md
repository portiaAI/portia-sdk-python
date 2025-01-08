---
sidebar_label: toolless_agent
title: portia.agents.toolless_agent
---

Agent designed when no tool is needed.

## ToolLessModel Objects

```python
class ToolLessModel()
```

Model to call the toolless agent.

#### invoke

```python
def invoke(_: MessagesState) -> dict[str, Any]
```

Invoke the model with the given message state.

## ToolLessAgent Objects

```python
class ToolLessAgent(BaseAgent)
```

Agent responsible for achieving a task by using langgraph.

#### execute\_sync

```python
def execute_sync(llm: BaseChatModel, step_outputs: dict[str,
                                                        Output]) -> Output
```

Run the core execution logic of the task.

