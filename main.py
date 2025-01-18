from portia.config import Config, StorageClass
from portia.example_tools.registry import example_tool_registry
from portia.runner import Runner

# Load the default config and override the storage class to point to the Portia cloud
#myConfig = Config.from_default(storage_class=StorageClass.DISK, storage_dir="./portia_storage")
myConfig = Config.from_default(storage_class=StorageClass.CLOUD)


# Instantiate a Portia runner. Load it with the default config and with the simple tool above.
runner = Runner(config=myConfig, tool_registry=example_tool_registry)

plan = runner.generate_plan('What is the weather in London?')

# Execute a workflow from the user query
workflow = runner.create_workflow(plan)
output = runner.execute_workflow(workflow)

# Serialise into JSON an print the output
print(output.model_dump_json(indent=2))