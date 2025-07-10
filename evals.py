from abc import ABC, abstractmethod
from collections import Counter
from enum import Enum
import re
from typing import Any

import numpy as np
import openai
import pandas as pd
from pydantic import BaseModel, Field

from portia import Config, Portia
from portia.clarification import Clarification
from portia.common import combine_args_kwargs
from portia.config import LogLevel, StorageClass
from portia.plan import Plan
from portia.plan_run import PlanRun
from portia.prefixed_uuid import PlanRunUUID, PlanUUID
from portia.storage import InMemoryStorage
from portia.tool import ReadyResponse, Tool, ToolRunContext
from portia.tool_call import ToolCallRecord, ToolCallStatus
from portia.tool_registry import DefaultToolRegistry, ToolRegistry

from concurrent.futures import ThreadPoolExecutor, as_completed


class DataSetType(Enum):
    PLAN_RUN = "PLAN_RUN"
    PLAN = "PLAN"


class PlanRunDataSetItem(BaseModel):
    id: str
    plan_run: PlanRun | PlanRunUUID
    skip_tool_stubs: list[str] = Field(default=[])


class PlanRunDataSet(BaseModel):
    name: str
    items: list[PlanRunDataSetItem]


class PlanDataSetItem(BaseModel):
    id: str
    plan: Plan | PlanUUID


class PlanDataSet(BaseModel):
    name: str
    items: list[PlanDataSetItem]


class ToolStub(Tool):
    """A tool stub that returns pre-canned data and record tool calls."""

    return_data: list[Any] = Field(
        description="The data for the tool to return. Call N will return the Nth item in the list."
    )
    tool_calls: list[ToolCallRecord] = Field(
        description="A list of all the tool calls this tool has seen."
    )

    def ready(self, ctx: ToolRunContext) -> ReadyResponse:
        return ReadyResponse(ready=True, clarifications=[])

    def run(
        self,
        ctx: ToolRunContext,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        # if only one return data is provided always use it
        if len(self.return_data) == 1:
            tool_output = self.return_data[0]
        else:
            tool_output = self.return_data[len(self.tool_calls)]

        # Set the plan_run_id correctly
        if isinstance(tool_output, Clarification):
            tool_output.plan_run_id = ctx.plan_run.id

        tc = ToolCallRecord(
            tool_name=self.name,
            plan_run_id=ctx.plan_run.id,
            step=ctx.plan_run.current_step_index,
            end_user_id=ctx.end_user.external_id,
            status=ToolCallStatus.SUCCESS,
            input=combine_args_kwargs(*args, **kwargs),
            output=tool_output,
            latency_seconds=0,
        )
        self.tool_calls.append(tc)
        return tool_output


class ToolStubRegistry(ToolRegistry):
    """A registry that allows setting tool stubs whilst allowing other tools to work correctly."""

    def __init__(self, registry: ToolRegistry, stubs: dict[str, Any]) -> None:
        super().__init__(registry.get_tools())
        self.stubs = stubs
        self.stubbed_tools: dict[str, ToolStub] = {}

    def get_tool_calls(self, tool_id: str) -> list[ToolCallRecord]:
        if tool_id in self.stubbed_tools:
            return self.stubbed_tools[tool_id].tool_calls
        return []

    def get_tool(self, tool_id: str) -> Tool:
        tool = super().get_tool(tool_id)

        if tool.id in self.stubbed_tools:
            return self.stubbed_tools[tool.id]

        if tool.id not in self.stubs:
            return tool

        tool_stub = ToolStub(
            id=tool.id,
            name=tool.name,
            description=tool.description,
            args_schema=tool.args_schema,
            output_schema=tool.output_schema,
            should_summarize=tool.should_summarize,
            return_data=self.stubs[tool.id],
            tool_calls=[],
        )
        self.stubbed_tools[tool_id] = tool_stub
        return tool_stub

    def get_tools(self) -> list[Tool]:
        tools = super().get_tools()
        stubbed_tools = []
        for tool in tools:
            stubbed_tools.append(self.get_tool(tool.id))
        return stubbed_tools


class MetricScoreResult(BaseModel):
    item_id: str
    name: str
    score: float


class PlanMetric(ABC):
    @abstractmethod
    def compare_plan(self, example_plan: Plan, actual_plan: Plan) -> float:
        raise NotImplementedError("compare_plan is not implemented")

    @abstractmethod
    def get_name(self) -> str:
        raise NotImplementedError("name is not implemented")


class PlanRunMetric(ABC):
    def get_plan_run_outputs(
        self, example_plan_run: PlanRun, actual_plan_run: PlanRun
    ) -> tuple[Any, Any]:
        output_1 = (
            example_plan_run.outputs.final_output.get_value()
            if example_plan_run.outputs.final_output
            else None
        )
        output_2 = (
            actual_plan_run.outputs.final_output.get_value()
            if actual_plan_run.outputs.final_output
            else None
        )
        return output_1, output_2

    @abstractmethod
    def compare_plan_run(self, example_plan_run: PlanRun, actual_plan_run: PlanRun) -> float:
        raise NotImplementedError("compare_plan_run is not implemented")

    @abstractmethod
    def get_name(self) -> str:
        raise NotImplementedError("name is not implemented")


class FinalStatusMetric(PlanRunMetric):
    def get_name(self) -> str:
        return "final_status_match"

    def compare_plan_run(self, example_plan_run: PlanRun, actual_plan_run: PlanRun) -> float:
        if example_plan_run.state != actual_plan_run.state:
            print(
                example_plan_run,
                actual_plan_run,
                actual_plan_run.outputs,
                actual_plan_run.outputs.final_output,
            )
        return 1 if example_plan_run.state == actual_plan_run.state else 0


class OutputExactMatchMetric(PlanRunMetric):
    def get_name(self) -> str:
        return "output_exact_match"

    def compare_plan_run(self, example_plan_run: PlanRun, actual_plan_run: PlanRun) -> float:
        output_1, output_2 = self.get_plan_run_outputs(example_plan_run, actual_plan_run)

        return 1 if str(output_1) == str(output_2) else 0


class OutputSimilarityMetric(PlanRunMetric):
    def get_name(self) -> str:
        return "output_similarity"

    def _embed(self, text: str) -> list[float]:
        response = openai.embeddings.create(model="text-embedding-3-small", input=text)
        return response.data[0].embedding

    def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        np_1 = np.array(vec1)
        np_2 = np.array(vec2)
        return np.dot(np_1, np_2) / (np.linalg.norm(np_1) * np.linalg.norm(np_2))

    def compare_plan_run(self, example_plan_run: PlanRun, actual_plan_run: PlanRun) -> float:
        output_1, output_2 = self.get_plan_run_outputs(example_plan_run, actual_plan_run)

        if output_1 == output_2:
            return 1

        if not output_1 or not output_2:
            return 0

        emb_1 = self._embed(output_1)
        emb_2 = self._embed(output_2)

        return self._cosine_similarity(emb_1, emb_2)


class CorrectClarificationsMetrics(PlanRunMetric):
    def get_name(self) -> str:
        return "correct_clarifications"

    def compare_plan_run(self, example_plan_run: PlanRun, actual_plan_run: PlanRun) -> float:
        expected = example_plan_run.outputs.clarifications or []
        actual = actual_plan_run.outputs.clarifications or []

        if len(expected) == 0 and len(actual) == 0:
            return 1

        matched = 0
        total = max(len(expected), 1)

        for exp in expected:
            for act in actual:
                if exp.category == act.category and exp.source == act.source:
                    matched += 1
                    break

        return matched / total


DEFAULT_PLAN_RUN_METRICS = [
    FinalStatusMetric(),
    # OutputExactMatchMetric(),
    OutputSimilarityMetric(),
    CorrectClarificationsMetrics(),
]


class SameNumberSteps(PlanMetric):
    def compare_plan(self, example_plan: Plan, actual_plan: Plan) -> float:
        len_expected = len(example_plan.steps)
        len_actual = len(actual_plan.steps)

        max_len = max(len_expected, len_actual, 1)  # avoid div-by-zero
        diff = abs(len_expected - len_actual)

        # Similarity score: 1.0 means identical length, 0.0 means totally off
        return 1 - (diff / max_len)

    def get_name(self) -> str:
        return "same_number_steps"


class SameToolIdMetric(PlanMetric):
    def compare_plan(self, example_plan: Plan, actual_plan: Plan) -> float:
        expected_ids = [step.tool_id for step in example_plan.steps]
        actual_ids = [step.tool_id for step in actual_plan.steps]

        expected_counter = Counter(expected_ids)
        actual_counter = Counter(actual_ids)

        # If exact match (same count of same tools): full score
        if expected_counter == actual_counter:
            score = 1.0
        else:
            # Partial match score: count how many tool_ids match exactly
            all_keys = set(expected_counter.keys()) | set(actual_counter.keys())
            matched = sum(min(expected_counter[k], actual_counter[k]) for k in all_keys)
            total = sum(expected_counter.values())

            score = matched / total if total else 0.0

        return score

    def get_name(self) -> str:
        return "same_tool_ids"


DEFAULT_PLAN_METRICS = [
    SameNumberSteps(),
    SameToolIdMetric(),
    # plan inputs
]


class PlanEvalExecutor:
    def __init__(self, portia: Portia) -> None:
        self.portia = portia
        self.iterations = 3

    def _mount_stubs(
        self,
        portia: Portia,
    ) -> Portia:
        stubbed_portia = Portia(
            config=portia.config.model_copy(update={"storage_class": StorageClass.MEMORY}),
            tools=portia.tool_registry,
        )
        # we need to override this to disable bring auth forward, so that our tool stubs get called.
        stubbed_portia._check_remaining_tool_readiness = lambda plan, plan_run, start_index=None: []  # noqa: ARG005, SLF001

        return stubbed_portia

    def _assert_results(
        self,
        item_id: str,
        input_plan: Plan,
        final_plan: Plan,
        metrics: list[PlanMetric],
    ) -> list[MetricScoreResult]:
        result_metrics: list[MetricScoreResult] = [
            MetricScoreResult(
                item_id=item_id,
                name=metric.get_name(),
                score=metric.compare_plan(input_plan, final_plan),
            )
            for metric in metrics
        ]
        return result_metrics

    def run_plan_evals(
        self,
        data_set: PlanDataSet,
        metrics: list[PlanMetric],
    ):
        portia = self._mount_stubs(self.portia)
        all_results = []

        def run_eval(item: PlanDataSetItem, iteration: int):
            if not isinstance(item, PlanDataSetItem):
                raise TypeError(f"plan data set contains item of type {type(item)}")

            if isinstance(item.plan, PlanUUID):
                item.plan = self.portia.storage.get_plan(item.plan)

            final_plan = portia.plan(
                query=item.plan.plan_context.query,
                tools=item.plan.plan_context.tool_ids,
                plan_inputs=item.plan.plan_inputs,
            )

            return self._assert_results(
                f"{item.id} - {iteration}",
                item.plan,
                final_plan,
                metrics,
            )

        futures = []
        with ThreadPoolExecutor() as executor:
            for item in data_set.items:
                for iteration in range(self.iterations):
                    futures.append(executor.submit(run_eval, item, iteration))

            for future in as_completed(futures):
                all_results.append(future.result())

        # Flatten and compute averages
        flat = [vars(score) for row in all_results for score in row]
        df = pd.DataFrame(flat)
        average_df = df.groupby("name")["score"].mean().reset_index()

        print(data_set.name)
        print(average_df)


class PlanRunEvalExecutor:
    def __init__(self, portia: Portia) -> None:
        self.portia = portia
        self.iterations = 3

    def _mount_stubs(
        self, portia: Portia, plan: Plan, plan_run: PlanRun, skip_tool_stubs: list[str]
    ) -> Portia:
        tool_stubs = {}
        for index, step in enumerate(plan.steps):
            new_calls = []

            if plan_run.get_clarifications_for_step(index):
                new_calls.append(*plan_run.get_clarifications_for_step(index))

            if step.output in plan_run.outputs.step_outputs:
                new_calls.append(plan_run.outputs.step_outputs[step.output].get_value())

            # only include if the tool isn't in the disable list
            if step.tool_id not in skip_tool_stubs:
                if step.tool_id not in tool_stubs:
                    tool_stubs[step.tool_id] = new_calls
                else:
                    tool_stubs[step.tool_id] = [*tool_stubs[step.tool_id], *new_calls]

        stubbed_portia = Portia(
            config=portia.config.model_copy(update={"storage_class": StorageClass.MEMORY}),
            tools=ToolStubRegistry(portia.tool_registry, tool_stubs),
        )
        # we need to override this to disable bring auth forward, so that our tool stubs get called.
        stubbed_portia._check_remaining_tool_readiness = lambda plan, plan_run, start_index=None: []  # noqa: ARG005, SLF001
        return stubbed_portia

    def _assert_results(
        self,
        item_id: str,
        input_plan_run: PlanRun,
        final_plan_run: PlanRun,
        metrics: list[PlanRunMetric],
    ) -> list[MetricScoreResult]:
        result_metrics: list[MetricScoreResult] = [
            MetricScoreResult(
                item_id=item_id,
                name=metric.get_name(),
                score=metric.compare_plan_run(input_plan_run, final_plan_run),
            )
            for metric in metrics
        ]
        return result_metrics

    def run_plan_run_evals(
        self,
        data_set: PlanRunDataSet,
        metrics: list[PlanRunMetric],
    ):
        all_results = []

        def run_eval(item: PlanRunDataSetItem, iteration: int):
            if not isinstance(item, PlanRunDataSetItem):
                raise TypeError(f"plan run data set contains item of type {type(item)}")

            # Resolve UUID to actual plan_run
            if isinstance(item.plan_run, PlanRunUUID):
                item.plan_run = self.portia.storage.get_plan_run(item.plan_run)

            # Fetch plan for the plan_run
            plan = self.portia.storage.get_plan(item.plan_run.plan_id)

            # Stub tools if needed
            portia = self._mount_stubs(self.portia, plan, item.plan_run, item.skip_tool_stubs)

            # Run plan
            final_plan_run = portia.run_plan(
                plan,
                plan_run_inputs=item.plan_run.plan_run_inputs,
            )

            return self._assert_results(
                f"{item.id} - {iteration}",
                item.plan_run,
                final_plan_run,
                metrics,
            )

        futures = []
        with ThreadPoolExecutor() as executor:
            for item in data_set.items:
                for iteration in range(self.iterations):
                    futures.append(executor.submit(run_eval, item, iteration))

            for future in as_completed(futures):
                all_results.append(future.result())

        # Flatten and aggregate
        flat = [vars(score) for row in all_results for score in row]
        df = pd.DataFrame(flat)
        average_df = df.groupby("name")["score"].mean().reset_index()

        print(data_set.name)
        print(average_df)


class SteelThread:
    def __init__(self, portia: Portia) -> None:
        self.portia = portia

    def _load_plan_dataset(self, data_set: PlanDataSet | str) -> PlanDataSet:
        if isinstance(data_set, str):
            raise NotImplementedError("loading remote data sets is not implemented")
        return data_set

    def _load_plan_run_dataset(self, data_set: PlanRunDataSet | str) -> PlanRunDataSet:
        if isinstance(data_set, str):
            # this would load from cloud
            if data_set == "search_tool_evals":
                return PlanRunDataSet(
                    name=data_set,
                    items=[
                        PlanRunDataSetItem(
                            id="search_tool_stubbed",
                            plan_run=PlanRunUUID.from_string(
                                "prun-52d91798-9b56-4274-902b-74d6a57b1d1d"
                            ),
                        ),
                        PlanRunDataSetItem(
                            id="search_tool_llm_stubbed",
                            plan_run=PlanRunUUID.from_string(
                                "prun-52d91798-9b56-4274-902b-74d6a57b1d1d"
                            ),
                            skip_tool_stubs=["llm_tool"],
                        ),
                        PlanRunDataSetItem(
                            id="search_tool_search_stubbed",
                            plan_run=PlanRunUUID.from_string(
                                "prun-52d91798-9b56-4274-902b-74d6a57b1d1d"
                            ),
                            skip_tool_stubs=["search_tool"],
                        ),
                        PlanRunDataSetItem(
                            id="search_tool_no_stubs",
                            plan_run=PlanRunUUID.from_string(
                                "prun-52d91798-9b56-4274-902b-74d6a57b1d1d"
                            ),
                            skip_tool_stubs=["search_tool", "llm_tool"],
                        ),
                    ],
                )
            raise NotImplementedError("loading remote data sets is not implemented")
        return data_set

    def run_plan_evals(
        self,
        data_set: PlanDataSet | str,
        metrics: list[PlanMetric] | None = None,
    ):
        return PlanEvalExecutor(self.portia).run_plan_evals(
            self._load_plan_dataset(data_set),
            metrics or DEFAULT_PLAN_METRICS,
        )

    def run_plan_run_evals(
        self,
        data_set: PlanRunDataSet | str,
        metrics: list[PlanRunMetric] | None = None,
    ):
        return PlanRunEvalExecutor(self.portia).run_plan_run_evals(
            self._load_plan_run_dataset(data_set),
            metrics or DEFAULT_PLAN_RUN_METRICS,
        )


# below this line is user implementation

config = Config.from_default(default_log_level=LogLevel.CRITICAL)
portia = Portia(config, tools=DefaultToolRegistry(config))

# inline
st = SteelThread(portia)
st.run_plan_run_evals(
    data_set=PlanRunDataSet(
        name="weather_email_tool",
        items=[
            PlanRunDataSetItem(
                id="weather_only_real_tool",
                plan_run=PlanRunUUID.from_string("prun-79d72167-7a6c-4b71-8ea6-696cf2467144"),
                skip_tool_stubs=["weather_tool"],
            ),
            PlanRunDataSetItem(
                id="weather_email_clarification",
                plan_run=PlanRunUUID.from_string("prun-de36eaf0-f9a9-48d7-9bf7-a25e02941642"),
            ),
            PlanRunDataSetItem(
                id="weather_only",
                plan_run=PlanRunUUID.from_string("prun-e8ebfa2c-dbaf-4ff7-9c9f-fd857bea67b8"),
            ),
            PlanRunDataSetItem(
                id="weather_error",
                plan_run=PlanRunUUID.from_string("prun-517fcff0-ddeb-41e4-8899-471b4e7a5877"),
            ),
        ],
    )
)


# or fetch remote
st = SteelThread(portia)
st.run_plan_run_evals(data_set="search_tool_evals")


# plan evals
st = SteelThread(portia)
st.run_plan_evals(
    data_set=PlanDataSet(
        name="search_planning",
        items=[
            PlanDataSetItem(
                id="search_correct",
                plan=PlanUUID.from_string("plan-22901fed-9a7d-431e-a226-ca2a1b78e62a"),
            ),
        ],
    )
)
