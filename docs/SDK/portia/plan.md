---
sidebar_label: plan
title: portia.plan
---

Plan primitives.

## Variable Objects

```python
class Variable(BaseModel)
```

A variable in the plan.

A variable is a way of referencing other parts of the plan usually either another steps output
or a constant input variable.

## Output Objects

```python
class Output(BaseModel, Generic[SERIALIZABLE_TYPE_VAR])
```

Output of a tool with wrapper for data, summaries and LLM interpretation.

Contains a generic value T bound to Serializable.

## Step Objects

```python
class Step(BaseModel)
```

A step in a workflow.

## Plan Objects

```python
class Plan(BaseModel)
```

A plan represent a series of steps that an agent should follow to execute the query.

#### validate\_uuid

```python
@model_validator(mode="before")
@classmethod
def validate_uuid(cls, values: dict[str, Any]) -> dict[str, Any]
```

Validate a given uuid is valid else assign a new one.

