---
sidebar_label: context
title: portia.agents.context
---

Context builders.

#### generate\_main\_system\_context

```python
def generate_main_system_context(
        system_context_extensions: list[str] | None = None) -> list[str]
```

Generate the main system context.

#### generate\_input\_context

```python
def generate_input_context(inputs: list[Variable],
                           previous_outputs: dict[str, Output]) -> list[str]
```

Generate context for the inputs returning the context and which inputs were used.

#### generate\_clarification\_context

```python
def generate_clarification_context(
        clarifications: list[Clarification]) -> list[str]
```

Generate context from clarifications.

#### build\_context

```python
def build_context(inputs: list[Variable],
                  previous_outputs: dict[str, Output],
                  clarifications: list[Clarification],
                  system_context_extensions: list[str] | None = None) -> str
```

Turn inputs and past outputs into a context string for the agent.

