"""test summarizer."""

import pytest
from langchain_core.messages import AIMessage, ToolCall, ToolMessage
from langgraph.graph import MessagesState

from portia import LLMProvider
from portia.config import Config
from portia.execution_agents.output import LocalDataValue
from portia.execution_agents.utils.step_summarizer import StepSummarizer
from portia.open_source_tools.llm_tool import LLMTool
from portia.plan import Step
from tests.integration.test_e2e import PROVIDER_MODELS


@pytest.mark.parametrize(("llm_provider", "default_model_name"), PROVIDER_MODELS)
def test_summarizer_with_large_outputs(
    llm_provider: LLMProvider,
    default_model_name: str,
) -> None:
    """Test summary with large output."""
    config = Config.from_default(
        llm_provider=llm_provider,
        default_model=default_model_name,
        large_output_threshold_tokens=1,
    )

    summarizer = StepSummarizer(
        config,
        config.get_execution_model(),
        tool=LLMTool(),
        step=Step(task="Return the name of a specific character from a cartoon.", output=""),
    )

    summary = summarizer.invoke(
        MessagesState(
            messages=[
                AIMessage(
                    content="",
                    tool_calls=[ToolCall(id="123", name="llm", args={})],
                ),
                ToolMessage(
                    content="",
                    tool_call_id="123",
                    artifact=LocalDataValue(
                        value="this is the output which will be truncates to 10 characters. Therefore the summary will not include Mickey Mouse."  # noqa: E501
                    ),
                ),
            ]
        )
    )

    assert "Mickey Mouse" not in summary["messages"][0].artifact.summary
    assert "large value" in summary["messages"][0].artifact.summary
