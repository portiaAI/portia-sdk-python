---
sidebar_label: clarification
title: portia.clarification
---

Clarification Primitives.

## Clarification Objects

```python
class Clarification(BaseModel, Generic[SERIALIZABLE_TYPE_VAR])
```

Base Model for Clarifications.

A Clarification represents some question that requires user input to resolve.
For example it could be:
- That authentication via OAuth needs to happen and the user needs to go through an OAuth flow.
- That one argument provided for a tool is missing and the user needs to provide it.
- That the user has given an input that is not allowed and needs to choose from a list.

#### resolve

```python
def resolve(response: SERIALIZABLE_TYPE_VAR | None) -> None
```

Resolve the clarification with the given response.

## ArgumentClarification Objects

```python
class ArgumentClarification(Clarification[SERIALIZABLE_TYPE_VAR])
```

A clarification about a specific argument for a tool.

The name of the argument should be given within the clarification.

## ActionClarification Objects

```python
class ActionClarification(Clarification[bool])
```

An action based clarification.

Represents a clarification where the user needs to click on a link. Set the response to true
once the user has clicked on the link and done the associated action.

#### serialize\_action\_url

```python
@field_serializer("action_url")
def serialize_action_url(action_url: HttpUrl) -> str
```

Serialize the action URL to a string.

## InputClarification Objects

```python
class InputClarification(ArgumentClarification[str])
```

An input based clarification.

Represents a clarification where the user needs to provide a value for a specific argument.

## MultiChoiceClarification Objects

```python
class MultiChoiceClarification(ArgumentClarification[str])
```

A multiple choice based clarification.

Represents a clarification where the user needs to select an option for a specific argument.

