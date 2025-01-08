---
sidebar_label: errors
title: portia.errors
---

Central definition of error classes.

## InvalidStorageError Objects

```python
class InvalidStorageError(Exception)
```

Raised when an invalid storage is provided.

## InvalidLLMProviderError Objects

```python
class InvalidLLMProviderError(Exception)
```

Raised when a provider is invalid.

## ConfigNotFoundError Objects

```python
class ConfigNotFoundError(Exception)
```

Raised when a needed config value is not present.

## InvalidConfigError Objects

```python
class InvalidConfigError(Exception)
```

Raised when a needed config value is invalid.

## PlanError Objects

```python
class PlanError(Exception)
```

Base class for exceptions in the query planner module. Indicates an error in planning.

## PlanNotFoundError Objects

```python
class PlanNotFoundError(Exception)
```

Indicate a plan was not found.

## WorkflowNotFoundError Objects

```python
class WorkflowNotFoundError(Exception)
```

Indicate a workflow was not found.

## ToolNotFoundError Objects

```python
class ToolNotFoundError(Exception)
```

Custom error class when tools aren&#x27;t found.

## InvalidToolDescriptionError Objects

```python
class InvalidToolDescriptionError(Exception)
```

Raised when the tool description is invalid.

## ToolRetryError Objects

```python
class ToolRetryError(Exception)
```

Raised when a tool fails on a retry.

## ToolFailedError Objects

```python
class ToolFailedError(Exception)
```

Raised when a tool fails with a hard error.

## InvalidWorkflowStateError Objects

```python
class InvalidWorkflowStateError(Exception)
```

The given workflow is in an invalid state.

## InvalidAgentOutputError Objects

```python
class InvalidAgentOutputError(Exception)
```

The agent returned output that could not be processed.

## ToolHardError Objects

```python
class ToolHardError(Exception)
```

Raised when a tool hits an error it can&#x27;t retry.

## ToolSoftError Objects

```python
class ToolSoftError(Exception)
```

Raised when a tool hits an error it can retry.

