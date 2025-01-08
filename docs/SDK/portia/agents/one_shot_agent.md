---
sidebar_label: one_shot_agent
title: portia.agents.one_shot_agent
---

One Shot agent.

## OneShotToolCallingModel Objects

```python
class OneShotToolCallingModel()
```

Model to call the tool with unverified arguments.

#### invoke

```python
def invoke(state: MessagesState) -> dict[str, Any]
```

Invoke the model with the given message state.

## OneShotAgent Objects

```python
class OneShotAgent(BaseAgent)
```

Agent responsible for achieving a task by using langgraph.

This agent does the following things:
1. Calls the tool with unverified arguments.
2. Retries tool calls up to 4 times.

#### retry\_tool\_or\_finish

```python
@staticmethod
def retry_tool_or_finish(state: MessagesState) -> Literal["tool_agent", END]
```

Determine if we should retry calling the tool if there was an error.

#### call\_tool\_or\_return

```python
@staticmethod
def call_tool_or_return(state: MessagesState) -> Literal["tools", END]
```

Determine if we should continue or not.

This is only to catch issues when the agent does not figure out how to use the tool
to achieve the goal.

#### process\_output

```python
def process_output(last_message: BaseMessage) -> Output
```

Process the output of the agent.

#### execute\_sync

```python
def execute_sync(llm: BaseChatModel, step_outputs: dict[str,
                                                        Output]) -> Output
```

Run the core execution logic of the task.

