---
sidebar_label: workflow
title: portia.workflow
---

Workflow primitives.

## WorkflowState Objects

```python
class WorkflowState(str, Enum)
```

Progress of the Workflow.

## Workflow Objects

```python
class Workflow(BaseModel)
```

A workflow represent a running instance of a Plan.

#### get\_outstanding\_clarifications

```python
def get_outstanding_clarifications() -> list[Clarification]
```

Return all outstanding clarifications.

