---
sidebar_label: planner
title: portia.planner
---

Planner module creates plans from queries.

## PlanOrError Objects

```python
class PlanOrError(BaseModel)
```

A plan or an error.

## Planner Objects

```python
class Planner()
```

planner class.

#### generate\_plan\_or\_error

```python
def generate_plan_or_error(query: str,
                           tool_list: list[Tool],
                           system_context_extension: list[str] | None = None,
                           examples: list[Plan] | None = None) -> PlanOrError
```

Generate a plan or error using an LLM from a query and a list of tools.

