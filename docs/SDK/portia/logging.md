---
sidebar_label: logging
title: portia.logging
---

Logging functions.

## LoggerInterface Objects

```python
class LoggerInterface(Protocol)
```

General Interface for loggers.

## LoggerManager Objects

```python
class LoggerManager()
```

Manages package level logger.

#### logger

```python
@property
def logger() -> LoggerInterface
```

Get the current logger.

#### set\_logger

```python
def set_logger(custom_logger: LoggerInterface) -> None
```

Set a custom logger.

#### configure\_from\_config

```python
def configure_from_config(config: Config) -> None
```

Configure the global logger based on the library&#x27;s configuration.

## LoggerProxy Objects

```python
class LoggerProxy()
```

Wrap the logging property to ensure dynamic resolution.

#### logger

```python
@property
def logger() -> LoggerInterface
```

Return current logger.

