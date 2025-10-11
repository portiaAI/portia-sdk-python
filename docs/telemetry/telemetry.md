* Telemetry

On this page

tip

By default, the telemetry data collected by the library is not personally identifiable information under GDPR and other privacy regulations as the data is deliberately anonymized.

The anonymized data collected is described in detail below. Collecting this data helps us understand how the library is being used and to improve the user experience. It also helps us fix bugs faster and prioritize feature development.

# Data collection

We use **PostHog** for telemetry collection. The data is completely anonymized and contains no personally identifiable information.

We track the following events.

### Portia function calls[​](#portia-function-calls "Direct link to Portia function calls")

We track the following Portia function calls:

* `Portia.run`
* `Portia.plan`
* `Portia.run_plan`
* `Portia.resume`
* `Portia.execute_plan_run_and_handle_clarifications`
* `Portia.resolve_clarification`
* `Portia.wait_for_ready`
* `Portia.create_plan_run`

For each of these, we track usage of features and tool IDs, e.g for `Portia.run`:

```
self.telemetry.capture(PortiaFunctionCallTelemetryEvent(  
    function_name='portia_run', function_args={  
        'tools': ",".join([tool.id if isinstance(tool, Tool) else tool for tool in tools]) if tools else None,  
        'example_plans_provided': example_plans != None, # Whether examples plan were provided.  
        'end_user_provided': end_user != None, # Whether an end user was used.  
        'plan_run_inputs_provided': plan_run_inputs != None # Whether plan inputs were used.  
    }  
))
```

### Portia tool calls[​](#portia-tool-calls "Direct link to Portia tool calls")

We also track when a tool call happens and the name of the tool that was executed. None of the arguments to the tool are tracked.

# Opting out

You can disable telemetry by setting the environment variable:

```
ANONYMIZED_TELEMETRY=false
```

Last updated on **Sep 9, 2025** by **github-actions[bot]**