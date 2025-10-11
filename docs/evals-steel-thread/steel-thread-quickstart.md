* [Evals and SteelThread](/evals-steel-thread)
* Install and quickstart

On this page

# Install and quickstart

SteelThread relies on access to agent activity in Portia cloud (queries, plans, plan runs). You will need a `PORTIA_API_KEY` to get started. Head over to ([**app.portialabs.ai ↗**](https://app.portialabs.ai)) and navigate to the `Manage API keys` tab from the left hand nav. There you can generate a new API key.

For a deeper dive

Below takes you through install and two end-to-end examples. If you wanted to get a deeper understanding, head over to:

* [Streams page](/streams).
* [Evals page](/evals).

## Install using your framework of choice[​](#install-using-your-framework-of-choice "Direct link to Install using your framework of choice")

* pip
* poetry
* uv

```
pip install steel-thread
```

```
poetry add steel-thread
```

```
uv add steel-thread
```

## Create a dataset[​](#create-a-dataset "Direct link to Create a dataset")

If you're new to Portia you may not have agent runs in the cloud just yet so let's start by creating those. Run the query `Read the user feedback notes in local file {path}, and call out recurring themes in their feedback. Use lots of ⚠️ emojis when highlighting areas of concern.` where `path` is a local file you can put a couple of lines of fictitious user feedback in. Here's the script to save you same time:

```
from portia import Portia  
  
path = "./uxr/calorify.txt" # TODO: change to your desired path  
query =f"Read the user feedback notes in local file {path}, \  
            and call out recurring themes in their feedback. \  
                Use lots of ⚠️ emojis when highlighting areas of concern."  
  
Portia().run(query=query)
```

## Basic example Streams[​](#basic-example-streams "Direct link to Basic example Streams")

Below is example code to process a stream. Before running it make sure you set up your stream from the Portia dashboard's Observability tab, **paying special attention to the name you gave your stream** so you can pass it to the `process_stream` method per below. This method will use the built-in set of Stream evaluators to give you data out of the box.

```
from portia import Config  
from steelthread.steelthread import SteelThread, StreamConfig  
from dotenv import load_dotenv  
  
  
load_dotenv(override=True)  
  
config = Config.from_default()  
  
# Setup SteelThread instance and process stream  
st = SteelThread()  
st.process_stream(  
    StreamConfig(  
        # The stream name is the name of the stream we created in the dashboard.  
        stream_name="your-stream-name-here",  
        config=config,  
    )  
)
```

## End-to-end example with Evals[​](#end-to-end-example-with-evals "Direct link to End-to-end example with Evals")

Let's push the envelope with some more advanced usage for Evals. Create an Eval dataset in the dashboard from the plan run we made in the **Create a dataset** section. Navigate to the "Evaluations" tab of the dashboard, create a new eval set from existing data and select the relevant plan run. Record the name you bestowed upon your Eval dataset as you will need to pass it to the evaluators in the code below, which you are now ready to run. This code:

* Uses a custom evaluator to count ⚠️ emojis in the output. We will do this by subclassing the `Evaluator` class.
* Stubs the `file_reader_tool` with static text. We will point our `Portia` client to a `ToolStubRegistry` to do this.
* Run the evals for the dataset you create to compute the emoji count metric over it.

Feel free to mess around with the output from the tool stub and re-run these Evals a few times to see the progression in scoring.

```
from portia import Portia, Config, DefaultToolRegistry  
from steelthread.steelthread import SteelThread, EvalConfig  
from steelthread.evals import Evaluator, EvalMetric  
from steelthread.portia.tools import ToolStubRegistry, ToolStubContext  
  
  
# Define custom evaluator  
class EmojiEvaluator(Evaluator):  
    def eval_test_case(self, test_case,plan, plan_run, metadata):  
        out = plan_run.outputs.final_output.get_value() or ""  
        count = out.count("⚠️")  
        return EvalMetric.from_test_case(  
            test_case=test_case,  
            name="emoji_score",  
            score=min(count / 2, 1.0),  
            description="Emoji usage",  
            explanation=f"Found {count} ⚠️ emojis in the output.",  
            actual_value=str(count),  
            expectation="2"  
        )  
  
# Define stub behavior  
def file_reader_stub_response(  
    ctx: ToolStubContext,  
) -> str:  
    """Stub response for file reader tool to return static file content."""  
    filename = ctx.kwargs.get("filename", "").lower()  
  
    return f"Feedback from file: {filename} suggests \  
        ⚠️ 'One does not simply Calorify' \  
        and ⚠️ 'Calorify is not a diet' \  
        and ⚠️ 'Calorify is not a weight loss program' \  
        and ⚠️ 'Calorify is not a fitness program' \  
        and ⚠️ 'Calorify is not a health program' \  
        and ⚠️ 'Calorify is not a nutrition program' \  
        and ⚠️ 'Calorify is not a meal delivery service' \  
        and ⚠️ 'Calorify is not a meal kit service' "  
  
  
config = Config.from_default()  
  
# Add the tool stub definition to your Portia client using a ToolStubRegistry  
portia = Portia(  
    config,  
    tools=ToolStubRegistry(  
        DefaultToolRegistry(config),  
        stubs={  
            "file_reader_tool": file_reader_stub_response,  
        },  
    ),  
)  
  
# Run evals with stubs   
SteelThread().run_evals(  
    portia,  
    EvalConfig(  
        eval_dataset_name="your-dataset-name-here", #TODO: replace with your dataset name  
        config=config,  
        iterations=5,  
        evaluators=[EmojiEvaluator(config)]  
    ),  
)
```

Last updated on **Sep 9, 2025** by **robbie-portia**