---
sidebar_label: context
title: portia.agents.context
---

Context builders.

#### build\_context

```python
def build_context(inputs: list[Variable],
                  previous_outputs: dict[str, Output],
                  clarifications: list[Clarification] | None = None,
                  system_context: list[str] | None = None) -> str
```

Turn inputs and past outputs into a context string for the agent.

