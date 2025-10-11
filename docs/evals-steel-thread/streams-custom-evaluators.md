* [Evals and SteelThread](/evals-steel-thread)
* [🌊 Streams](/streams)
* Custom Stream evaluators

# Custom Stream evaluators

Evaluators are responsible for the calculation of metrics. To help you get started quickly, Steel Thread provides a built-in `LLMJudgeEvaluator` for stream based evaluation using LLM-as-Judge. This explained in the previous section on [basic usage](/streams-overview).

You can add your own custom Stream evaluators, be it LLM-as-Judge or deterministic ones. `StreamEvaluator` can implement two methods:

```
from steelthread.streams import PlanStreamItem, PlanRunStreamItem, StreamMetric, StreamEvaluator  
class MyStreamEvaluator(StreamEvaluator):  
    def process_plan(self, stream_item: PlanStreamItem) -> list[StreamMetric] | StreamMetric:  
        ...  
  
    def process_plan_run(self, stream_item: PlanRunStreamItem) -> list[StreamMetric] | StreamMetric | None:  
        ...
```

Below are two examples of custom evaluators, both using LLM-as-Judge or deterministic

* LLM-as-judge
* Deterministic

You can use an LLM to score plan runs automatically. When you subclass `StreamEvaluator`, you first initialise your `LLMScorer` and then define how you want to process plans / plan runs with this evaluator.

```
from portia import Config  
  
from steelthread.steelthread import SteelThread  
  
from steelthread.streams import (  
    StreamConfig,   
    PlanRunStreamItem,  
    StreamEvaluator,  
    StreamMetric  
)  
from steelthread.utils.llm import LLMScorer, MetricOnly  
  
  
class LLMVerbosityJudge(StreamEvaluator):  
    def __init__(self, config):  
        self.scorer = LLMScorer(config)  
  
    def process_plan_run(self, stream_item: PlanRunStreamItem):  
        # The stream_item object holds the underlying plan / plan run being evaluated.  
        task_data = stream_item.plan_run.model_dump_json()  
        # The description is used to inform the LLM on how to score the metric.  
        metrics = self.scorer.score(  
            task_data=[task_data],  
            metrics_to_score=[  
                MetricOnly(  
                    name="verbosity",   
                    description="Scores 0 if the answer is too verbose. 0 otherwise."),   
            ],  
        )  
  
        return [  
            StreamMetric.from_stream_item(  
                stream_item=stream_item,  
                score=m.score,  
                name=m.name,  
                description=m.description,  
                explanation=m.explanation,  
            )  
            for m in metrics  
        ]  
  
# Setup config + Steel Thread  
config = Config.from_default()  
  
# To use your evaluator, pass it to the runner  
SteelThread().process_stream(  
    StreamConfig(  
        stream_name="your-stream-name-here",  
        config=config,  
        evaluators=[LLMVerbosityJudge(config)],  
    ),  
)
```

You can score plan runs using your own code by subclassing `StreamEvaluator` and writing your own implementation of `process_plan` or `process_plan_run`.

```
from portia import Config  
  
from steelthread.steelthread import (  
    SteelThread,  
)  
from steelthread.streams import (  
    StreamConfig,   
    PlanRunStreamItem,  
    StreamEvaluator,  
    StreamMetric  
)  
  
from dotenv import load_dotenv  
load_dotenv(override=True)  
  
class JudgeDread(StreamEvaluator):  
    def process_plan_run(self, stream_item: PlanRunStreamItem):  
        # The stream_item object holds the underlying plan / plan run being evaluated.  
        # In this example we're just returning a static score and explanation.  
        return StreamMetric.from_stream_item(  
            stream_item=stream_item,  
            name="dread_score",  
            score=1,  
            description="Dreadful stuff",  
            explanation="The dread was palpable",  
        )  
  
# Setup config + Steel Thread  
config = Config.from_default()  
  
# Process stream  
SteelThread().process_stream(  
    StreamConfig(  
        stream_name="your-stream-name-here",  
        config=config,   
        evaluators=[JudgeDread(config)])  
)
```

Last updated on **Sep 9, 2025** by **robbie-portia**