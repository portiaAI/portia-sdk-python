---
sidebar_label: config
title: portia.config
---

Configuration for the SDK.

## StorageClass Objects

```python
class StorageClass(Enum)
```

Represent locations plans and workflows are written to.

## LLMProvider Objects

```python
class LLMProvider(Enum)
```

Enum of LLM providers.

## LLMModel Objects

```python
class LLMModel(Enum)
```

Supported Models.

## AgentType Objects

```python
class AgentType(Enum)
```

Type of agent to use for executing a step.

## LogLevel Objects

```python
class LogLevel(Enum)
```

Available Log Levels.

#### is\_greater\_than\_zero

```python
def is_greater_than_zero(value: int) -> int
```

Validate greater than zero.

## Config Objects

```python
class Config(BaseModel)
```

General configuration for the library.

#### check\_config

```python
@model_validator(mode="after")
def check_config() -> Config
```

Validate Config is consistent.

#### from\_file

```python
@classmethod
def from_file(cls, file_path: Path) -> Config
```

Load configuration from a JSON file.

#### from\_default

```python
@classmethod
def from_default(cls, **kwargs) -> Config
```

Create a Config instance with default values, allowing overrides.

#### has\_api\_key

```python
def has_api_key(name: str) -> bool
```

Check if the given API Key is available.

#### must\_get\_api\_key

```python
def must_get_api_key(name: str) -> SecretStr
```

Get an api key as a SecretStr or error if not set.

#### must\_get\_raw\_api\_key

```python
def must_get_raw_api_key(name: str) -> str
```

Get a raw api key as a string or errors if not set.

#### must\_get

```python
def must_get(name: str, expected_type: type[T]) -> T
```

Get a given value in the config ensuring a type match.

#### default\_config

```python
def default_config() -> Config
```

Return default config.

