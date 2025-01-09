---
sidebar_label: base_agent
title: portia.agents.base_agent
---

Agents are responsible for executing steps of a workflow.

The BaseAgent class is the base class all agents must extend.

## BaseAgent Objects

```python
class BaseAgent()
```

An Agent is responsible for carrying out the task defined in the given Step.

This Base agent is the class all agents must extend. Critically agents must implement the
execute_sync function which is responsible for actually carrying out the task as given in
the step. They have access to copies of the step, workflow and config but changes to those
objects will not be respected by the runner.

Optionally agents may also override the get_context function which is responsible for setting
the system_context for the agent.

#### execute\_sync

```python
@abstractmethod
def execute_sync() -> Output
```

Run the core execution logic of the task synchronously.

Implementation of this function is deferred to individual agent implementations
making it simple to write new ones.

#### get\_system\_context

```python
def get_system_context() -> str
```

Build a generic system context string from the step and workflow provided.

