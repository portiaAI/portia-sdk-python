---
sidebar_label: verifier_agent
title: portia.agents.verifier_agent
---

Verifier Agent for hardest problems.

## ToolArgument Objects

```python
class ToolArgument(BaseModel)
```

Represents an argument for a tool as extracted from the goal and context.

## ToolInputs Objects

```python
class ToolInputs(BaseModel)
```

Represents the inputs for a tool.

## VerifiedToolArgument Objects

```python
class VerifiedToolArgument(BaseModel)
```

Represents an argument for a tool after being verified by an agent.

## VerifiedToolInputs Objects

```python
class VerifiedToolInputs(BaseModel)
```

Represents the inputs for a tool.

## ParserModel Objects

```python
class ParserModel()
```

Model to parse the arguments for a tool.

#### invoke

```python
def invoke(_: MessagesState) -> dict[str, Any]
```

Invoke the model with the given message state.

## VerifierModel Objects

```python
class VerifierModel()
```

Model to verify the arguments for a tool.

#### invoke

```python
def invoke(state: MessagesState) -> dict[str, Any]
```

Invoke the model with the given message state.

## ToolCallingModel Objects

```python
class ToolCallingModel()
```

Model to call the tool with the verified arguments.

#### invoke

```python
def invoke(state: MessagesState) -> dict[str, Any]
```

Invoke the model with the given message state.

## VerifierAgent Objects

```python
class VerifierAgent(BaseAgent)
```

Agent responsible for achieving a task by using verification.

This agent does the following things:
 1. It uses an LLM to make sure that we have the right arguments for the tool, with
    explanations of the values and where they come from.
 2. It uses an LLM to make sure that the arguments are correct, and that they are labeled
    as provided, inferred or assumed.
 3. If any of the arguments are assumed, it will request a clarification.
 4. If the arguments are correct, it will call the tool and return the result to the user.
 5. If the tool fails, it will try again at least 3 times.

Also, if the agent is being called a second time, it will just jump to step 4.

Possible improvements:
 1. This approach (as well as the other agents) could be improved for arguments that are lists

#### retry\_tool\_or\_finish

```python
@staticmethod
def retry_tool_or_finish(state: MessagesState) -> Literal["tool_agent", END]
```

Determine if we should retry calling the tool if there was an error.

#### clarifications\_or\_continue

```python
def clarifications_or_continue(
        state: MessagesState) -> Literal["tool_agent", END]
```

Determine if we should continue with the tool call or request clarifications instead.

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
def execute_sync() -> Output
```

Run the core execution logic of the task.

