"""Custom LangSmith wrapper for Google Generative AI (Gemini)."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Literal

from google.genai import types
from langsmith import run_helpers

from portia.logger import logger

if TYPE_CHECKING:
    from google import genai


def _get_ls_params(model_name: str, _: dict) -> dict[str, str]:
    """Get LangSmith parameters for tracing."""
    return {
        "ls_provider": "google_genai",
        "ls_model_name": model_name,
        "ls_model_type": "chat",
    }


def _process_outputs(
    outputs: types.GenerateContentResponse,
) -> dict[str, list[dict[str, str]]]:
    """Process outputs for tracing."""
    try:
        if outputs.candidates and len(outputs.candidates) > 0:
            return {
                "messages": [
                    {
                        "role": "ai",
                        "content": _extract_parts(outputs.candidates[0].content)[0],
                    },
                ]
            }
    except (IndexError, AttributeError):  # pragma: no cover
        return {"messages": []}  # pragma: no cover
    else:
        return {"messages": []}  # pragma: no cover


def _extract_parts(content_item: types.ContentUnion | types.ContentUnionDict) -> list[str]:  # noqa: C901, PLR0911
    """Handle extracting content from response."""
    # Case 1: list of parts (Part, PartDict-like dict, or str)
    if isinstance(content_item, list):
        result = []
        for p in content_item:
            if isinstance(p, str):
                result.append(p)
            elif isinstance(p, types.Part):
                result.append(p.text or "")
            elif isinstance(p, dict) and "text" in p:
                result.append(str(p["text"]) or "")
        return result

    # Case 2: Content object with .parts
    if isinstance(content_item, types.Content):
        result = []
        if not content_item.parts:
            return []
        for p in content_item.parts:
            if isinstance(p, types.Part):
                result.append(p.text or "")
        return result

    # Case 3: single Part or dict
    if isinstance(content_item, types.Part):
        return [content_item.text or ""]
    if isinstance(content_item, dict) and "parts" in content_item:
        return [str(content_item["parts"]) or ""]

    # Case 4: Fallback to string if nothing else matches
    if isinstance(content_item, str):
        return [content_item]

    return []


def _process_inputs(
    inputs: dict[Literal["contents"], types.ContentListUnion | types.ContentListUnionDict],
) -> dict[str, list[dict[str, str]]]:
    """Process inputs for tracing compatible with the genai package."""
    try:
        contents = inputs["contents"]
        if not isinstance(contents, list):
            contents = [contents]

        first = contents[0]
        parts = _extract_parts(first)

        if len(parts) == 2:  # noqa: PLR2004
            return {
                "messages": [
                    {"role": "system", "content": parts[0]},
                    {"role": "user", "content": parts[1]},
                ]
            }

        return {"messages": [{"content": part} for part in parts]}

    except Exception:  # pragma: no cover  # noqa: BLE001
        return {"messages": []}  # pragma: no cover


def wrap_gemini(client: genai.Client) -> genai.Client:  # pyright: ignore[reportPrivateImportUsage]
    """Wrap a Google Generative AI model to enable LangSmith tracing."""
    original_generate_content = client.models.generate_content

    @functools.wraps(original_generate_content)
    def traced_generate_content(
        *,
        model: str,
        contents: types.ContentListUnion | types.ContentListUnionDict,
        config: types.GenerateContentConfigOrDict | None = None,
    ) -> types.GenerateContentResponse:
        """Traced version of generate_content."""
        decorator = run_helpers.traceable(
            name="GoogleGenAI",
            run_type="llm",
            process_outputs=_process_outputs,
            process_inputs=_process_inputs,
            _invocation_params_fn=functools.partial(_get_ls_params, model),
        )
        try:
            return decorator(original_generate_content)(model, contents, config)
        except Exception as e:  # noqa: BLE001
            # We should never fail because of tracing, so fall back to calling the original method
            logger().error(f"Error tracing Google Generative AI: {e}")
            return original_generate_content(model, contents, config)  # type: ignore  # noqa: PGH003

    client.models.generate_content = traced_generate_content
    return client
