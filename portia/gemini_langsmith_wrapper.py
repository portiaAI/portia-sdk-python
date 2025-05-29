"""Custom LangSmith wrapper for Google Generative AI (Gemini)."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any, Literal

from langsmith import run_helpers

from portia.logger import logger

if TYPE_CHECKING:
    import google.generativeai as genai
    from google.generativeai.types import content_types, generation_types


def _get_ls_params(model_name: str, _: dict) -> dict[str, str]:
    """Get LangSmith parameters for tracing."""
    return {
        "ls_provider": "google_genai",
        "ls_model_name": model_name,
        "ls_model_type": "chat",
    }


def _process_outputs(
    outputs: generation_types.GenerateContentResponse,
) -> dict[str, list[dict[str, str]]]:
    """Process outputs for tracing."""
    return {
        "messages": [
            {
                "role": "ai",
                "content": outputs.candidates[0].content.parts[0].text,
            },
        ]
    }


def _process_inputs(
    inputs: dict[Literal["contents"], content_types.ContentsType],
) -> dict[str, list[dict[str, str]]]:
    """Process inputs for tracing."""
    return {
        "messages": [
            {
                "role": "system",
                "content": inputs["contents"][0]["parts"][0],  # pyright: ignore[reportIndexIssue,reportOptionalSubscript,reportGeneralTypeIssues,reportArgumentType]
            },
            {
                "role": "user",
                "content": inputs["contents"][0]["parts"][1],  # pyright: ignore[reportIndexIssue,reportOptionalSubscript,reportGeneralTypeIssues,reportArgumentType]
            },
        ]
    }


def wrap_gemini(client: genai.GenerativeModel) -> genai.GenerativeModel:  # pyright: ignore[reportPrivateImportUsage]
    """Wrap a Google Generative AI model to enable LangSmith tracing."""
    original_generate_content = client.generate_content

    @functools.wraps(original_generate_content)
    def traced_generate_content(
        *args: Any, **kwargs: Any
    ) -> generation_types.GenerateContentResponse:
        """Traced version of generate_content."""
        decorator = run_helpers.traceable(
            name="GoogleGenerativeAI",
            run_type="llm",
            process_outputs=_process_outputs,
            process_inputs=_process_inputs,
            _invocation_params_fn=functools.partial(_get_ls_params, client.model_name),
        )
        try:
            return decorator(original_generate_content)(*args, **kwargs)
        except Exception as e:  # noqa: BLE001
            # We should never fail because of tracing, so fall backk to calling the original method
            logger().error(f"Error tracing Google Generative AI: {e}")
            return original_generate_content(*args, **kwargs)

    client.generate_content = traced_generate_content
    return client
