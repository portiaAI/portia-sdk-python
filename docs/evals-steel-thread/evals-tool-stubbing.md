* [Evals and SteelThread](/evals-steel-thread)
* [📈 Evals](/evals)
* Tool stubbing

On this page

# Tool stubbing

When running evals, your agent may call tools like `weather_lookup`, `search`, or `send_email`. If those tools hit live systems, you'll get non-deterministic results — which can make evaluation noisy and inconsistent. There's also of course undesirable real-world effects (e.g. emails sent) and cost to these tool calls when you're simply trying to run evals!

To solve this, Steel Thread supports stubbing tools. This makes your tests:

* **Deterministic** — the same input always produces the same output
* **Isolated** — no external API calls or flaky systems
* **Repeatable** — easy to track regressions across changes

## When to stub[​](#when-to-stub "Direct link to When to stub")

Use tool stubs when:

* You're writing **evals**
* The tool response **affects the plan or output**
* You want **consistent scoring** across iterations

## How Tool stubbing works[​](#how-tool-stubbing-works "Direct link to How Tool stubbing works")

Steel Thread provides a `ToolStubRegistry` — a drop-in replacement for Portia’s default registry. You can wrap your existing tools and selectively override individual tools by ID. Tool stubs are simple Python functions and the `ToolStubContext` contains all the original tool's context to help you generate realistic stubs. Below is an example where we use use a tool stub for the open source `weather_tool` [available in the Portia SDK](https://docs.portialabs.ai/portia-tools/open-source/weather).

```
from portia import Portia, Config, DefaultToolRegistry  
from steelthread.steelthread import SteelThread, EvalConfig  
from steelthread.portia.tools import ToolStubRegistry, ToolStubContext  
from dotenv import load_dotenv  
  
load_dotenv(override=True)  
  
  
# Define stub behavior  
def weather_stub_response(  
    ctx: ToolStubContext,  
) -> str:  
    """Stub for weather tool to return deterministic weather."""  
    city = ctx.kwargs.get("city", "").lower()  
    if city == "sydney":  
        return "33.28"  
    if city == "london":  
        return "2.00"  
  
    return f"Unknown city: {city}"  
  
  
config = Config.from_default()  
  
# Run evals with stubs   
portia = Portia(  
    config,  
    tools=ToolStubRegistry(  
        DefaultToolRegistry(config),  
        stubs={  
            "weather_tool": weather_stub_response,  
        },  
    ),  
)  
  
SteelThread().run_evals(  
    portia,  
    EvalConfig(  
        eval_dataset_name="your-dataset-name-here",  
        config=config,  
        iterations=5  
    ),  
)
```

With the stubbed tool in place, your evals will be clean, fast, and reproducible. The rest of your tools still work as normal with only the stubbed one being overridden.

Best Practices for tool stubbing

* Stub only the tools that matter for evaluation
* Use consistent return types (e.g. same as real tool)
* Use `tool_call_index` if you want per-run variance
* Combine stubbing with assertions to detect misuse (e.g. tool called too many times)

---

Last updated on **Sep 9, 2025** by **robbie-portia**