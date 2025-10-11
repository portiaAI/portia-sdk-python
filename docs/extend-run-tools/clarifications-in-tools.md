* [Extend and run tools](/extend-run-tools)
* Use clarifications in custom tools

On this page

# Use clarifications in custom tools

TL;DR

You can raise a `Clarification` in any custom tool definition to prompt a plan run to interrupt itself and solicit input ([**SDK reference ↗**](/SDK/portia/clarification)).

## Add a clarification to your custom tool[​](#add-a-clarification-to-your-custom-tool "Direct link to Add a clarification to your custom tool")

Let's pick up the custom tool example we looked at previously ([**Add custom tools ↗**](/add-custom-tools)). We will now examine the code that defines a clarification in a tool explicitly. We're going to add a clarification to the `FileReaderTool` custom tool to handle cases where a file is not found. Instead of throwing an error directly, we will attempt to find the file in other folders in the project directory. We do that by adding the highlighted lines in the `FileReaderTool` class definition as shown below.

my\_custom\_tools/file\_reader\_tool.py

```
from pathlib import Path  
import pandas as pd  
import json  
from pydantic import BaseModel, Field  
from portia import (  
    MultipleChoiceClarification,  
    Tool,  
    ToolHardError,  
    ToolRunContext,  
)  
  
  
class FileReaderToolSchema(BaseModel):  
    """Schema defining the inputs for the FileReaderTool."""  
  
    filename: str = Field(...,   
        description="The location where the file should be read from",  
    )  
  
  
class FileReaderTool(Tool[str]):  
    """Finds and reads content from a local file on Disk."""  
  
    id: str = "file_reader_tool"  
    name: str = "File reader tool"  
    description: str = "Finds and reads content from a local file on Disk"  
    args_schema: type[BaseModel] = FileReaderToolSchema  
    output_schema: tuple[str, str] = ("str", "A string dump or JSON of the file content")  
  
    def run(self, ctx: ToolRunContext, filename: str) -> str | dict[str,any] | MultipleChoiceClarification:  
        """Run the FileReaderTool."""  
          
        file_path = Path(filename)  
        suffix = file_path.suffix.lower()  
  
        if file_path.is_file():  
            if suffix == '.csv':  
                return pd.read_csv(file_path).to_string()  
            elif suffix == '.json':  
                with file_path.open('r', encoding='utf-8') as json_file:  
                    data = json.load(json_file)  
                    return data  
            elif suffix in ['.xls', '.xlsx']:  
                return pd.read_excel(file_path).to_string  
            elif suffix in ['.txt', '.log']:  
                return file_path.read_text(encoding="utf-8")  
            else:  
               raise ToolHardError(f"Unsupported file format: {suffix}. Supported formats are .txt, .log, .csv, .json, .xls, .xlsx.")  
          
        alt_file_paths = self.find_file(filename)  
        if alt_file_paths:  
            return MultipleChoiceClarification(  
                plan_run_id=ctx.plan_run.id,  
                argument_name="filename",  
                user_guidance=f"Found {filename} in these location(s). Pick one to continue:\n{alt_file_paths}",  
                options=alt_file_paths,  
            )  
  
        raise ToolHardError(f"No file found on disk with the path {filename}.")  
  
    def find_file(self, filename: str) -> list[Path]:  
        """Returns a full file path or None."""  
  
        search_path = Path("../")  
        filepaths = []  
  
        for filepath in search_path.rglob(filename):  
            if filepath.is_file():  
                filepaths.append(str(filepath))  
        if filepaths:  
            return filepaths  
        return None
```

The block below results in the tool using the `find_file` method to look for alternative locations and raising this clarification if multiple paths are found in the project directory. Here we're using `MultipleChoiceClarification` specifically, which takes a `options` property where the paths found are enumerated. You can explore the other types a `Clarification` object can take in our documentation ([**SDK reference ↗**](/SDK/portia/clarification)).

```
alt_file_paths = self.find_file(filename)  
if alt_file_paths:  
    return MultipleChoiceClarification(  
        plan_run_id=ctx.plan_run.id,  
        argument_name="filename",  
        user_guidance=f"Found {filename} in these location(s). Pick one to continue:\n{alt_file_paths}",  
        options=alt_file_paths,  
    )
```

## Testing your tool with clarifications[​](#testing-your-tool-with-clarifications "Direct link to Testing your tool with clarifications")

We're now ready to put our clarification to the test. We won't revisit how clarifications work and are handled in detail here, For that you can check out the section dedicated to clarifications ([**Understand clarifications↗**](/understand-clarifications)).

Make a `weather.txt` file for this section

In this example, our custom tool `FileReaderTool` will attempt to open a non-existent local file `weather.txt`. This should trigger the tool to search for the file across the rest of the project directory and return all matches. Make sure to sprinkle a few copies of a `weather.txt` file around in the project directory.
Note: Our `weather.txt` file contains "The current weather in Llanfairpwllgwyngyllgogerychwyrndrobwllllantysiliogogogoch is broken clouds with a temperature of 6.76°C."

main.py

```
from portia import Portia  
from portia.config import default_config  
from portia.open_source_tools.registry import example_tool_registry  
from my_custom_tools.registry import custom_tool_registry  
from portia.clarification import MultipleChoiceClarification  
from portia.plan_run import PlanRunState  
  
# Load example and custom tool registries into a single one  
complete_tool_registry = example_tool_registry + custom_tool_registry  
# Instantiate a Portia instance. Load it with the default config and with the tools above  
portia = Portia(tools=complete_tool_registry)  
  
# Execute the plan from the user query  
plan_run = portia.run('Read the contents of the file "weather.txt".')  
  
# Check if the plan run was paused due to raised clarifications  
while plan_run.state == PlanRunState.NEED_CLARIFICATION:  
    # If clarifications are needed, resolve them before resuming the plan run  
    for clarification in plan_run.get_outstanding_clarifications():  
        # For each clarification, prompt the user for input  
        print(f"{clarification.user_guidance}")  
        user_input = input("Please enter a value:\n"   
                        + (("\n".join(clarification.options) + "\n") if "options" in clarification else ""))  
        # Resolve the clarification with the user input  
        plan_run = portia.resolve_clarification(clarification, user_input, plan_run)  
  
    # Once clarifications are resolved, resume the plan run  
    plan_run = portia.resume(plan_run)  
  
# Serialise into JSON and print the output  
print(plan_run.model_dump_json(indent=2))
```

For the example query above `Read the contents of the file "weather.txt".`, where the user resolves the clarification by entering one of the options offered by the clarification (in this particular case `demo_runs/weather.txt` in our project directory `momo_sdk_tests`), you should see the following plan run state and notice:

* The multiple choice clarification where the `user_guidance` was generated by Portia based on your clarification definition in the `FileReaderTool` class,
* The `response` in the second plan run snapshot reflecting the user input, and the change in `resolved` to `true` as a result
* The plan run `state` will appear to `NEED_CLARIFICATION` if you look at the logs at the point when the clarification is raised. It then progresses to `COMPLETE` once you respond to the clarification and the plan run is able to resume:

run\_state.json

```
{  
  "id": "prun-54d157fe-4b99-4dbb-a917-8fd8852df63d",  
  "plan_id": "plan-b87de5ac-41d9-4722-8baa-8015327511db",  
  "current_step_index": 0,  
  "state": "COMPLETE",  
  "outputs": {  
    "clarifications": [  
      {  
        "id": "clar-216c13a1-8342-41ca-99e5-59394cbc7008",  
        "category": "Multiple Choice",  
        "response": "../momo_sdk_tests/demo_runs/weather.txt",  
        "step": 0,  
        "user_guidance": "Found weather.txt in these location(s). Pick one to continue:\n['../momo_sdk_tests/demo_runs/weather.txt', '../momo_sdk_tests/my_custom_tools/__pycache__/weather.txt']",  
        "resolved": true,  
        "argument_name": "filename",  
        "options": [  
          "../momo_sdk_tests/demo_runs/weather.txt",  
          "../momo_sdk_tests/my_custom_tools/__pycache__/weather.txt"  
        ]  
      }  
    ],  
    "step_outputs": {  
      "$file_contents": {  
        "value": "The current weather in Llanfairpwllgwyngyllgogerychwyrndrobwllllantysiliogogogoch is broken clouds with a temperature of 6.76°C.",  
        "summary": null  
      }  
    },  
    "final_output": {  
      "value": "The current weather in Llanfairpwllgwyngyllgogerychwyrndrobwllllantysiliogogogoch is broken clouds with a temperature of 6.76°C.",  
      "summary": null  
    }  
  }  
}
```

## Accessing clarifications in your custom tool[​](#accessing-clarifications-in-your-custom-tool "Direct link to Accessing clarifications in your custom tool")

The above example showed how you can access a clarification in your custom tool when it relates directly to the tool's arguments. If however you wanted to access a clarification from your tool that is not related to the tool's arguments, you can do so by using the `ToolRunContext` object that is passed to the `run` method of your tool.

```
from portia import ToolRunContext, MultipleChoiceClarification  
  
def run(self, ctx: ToolRunContext, filename: str) -> str | dict[str,any] | MultipleChoiceClarification:  
    """Run the FileReaderTool."""  
    clarifications = ctx.clarifications
```

This allows you to return more complex clarifications from your tool and access them once they have been resolved by the user.

Last updated on **Sep 9, 2025** by **github-actions[bot]**