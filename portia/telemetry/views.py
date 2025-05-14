from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class BaseTelemetryEvent(ABC):
	@property
	@abstractmethod
	def name(self) -> str:
		pass

	@property
	def properties(self) -> dict[str, Any]:
		return {k: v for k, v in asdict(self).items() if k != 'name'}


@dataclass
class RegisteredFunction:
	name: str
	params: dict[str, Any]


@dataclass
class ControllerRegisteredFunctionsTelemetryEvent(BaseTelemetryEvent):
	registered_functions: list[RegisteredFunction]
	name: str = 'controller_registered_functions'
	
@dataclass
class TestTelemetryEvent(BaseTelemetryEvent):
	something: str
	name: str = 'test'

@dataclass
class PortiaFunctionCallTelemetryEvent(BaseTelemetryEvent):
	function_name: str
	function_args: dict[str, Any]
	name: str = 'portia_function_call'

@dataclass
class ToolCallTelemetryEvent(BaseTelemetryEvent):
	tool_id: str | None
	name: str = 'tool_call'