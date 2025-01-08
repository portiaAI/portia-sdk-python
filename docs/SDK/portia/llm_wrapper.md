---
sidebar_label: llm_wrapper
title: portia.llm_wrapper
---

Wrapper around different LLM providers allowing us to treat them the same.

## BaseLLMWrapper Objects

```python
class BaseLLMWrapper(ABC)
```

Abstract base class for LLM wrappers.

#### to\_langchain

```python
@abstractmethod
def to_langchain() -> BaseChatModel
```

Convert to a LangChain-compatible model.

#### to\_instructor

```python
@abstractmethod
def to_instructor(response_model: type[T],
                  messages: list[ChatCompletionMessageParam]) -> T
```

Generate a response using instructor.

## LLMWrapper Objects

```python
class LLMWrapper(BaseLLMWrapper)
```

LLMWrapper class for different LLMs.

#### to\_langchain

```python
def to_langchain() -> BaseChatModel
```

Return a langchain chat model.

#### to\_instructor

```python
def to_instructor(response_model: type[T],
                  messages: list[ChatCompletionMessageParam]) -> T
```

Use instructor to generate an object of response_model type.

