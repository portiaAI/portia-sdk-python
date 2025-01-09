---
sidebar_label: addition
title: portia.example_tools.addition
---

Simple Addition Tool.

## AdditionToolSchema Objects

```python
class AdditionToolSchema(BaseModel)
```

Input for AdditionTool.

## AdditionTool Objects

```python
class AdditionTool(Tool[float])
```

Adds two numbers.

#### run

```python
def run(a: float, b: float) -> float
```

Add the numbers.

