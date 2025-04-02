import dotenv

from portia import Portia
from portia.config import Config, StorageClass
from portia.tool_decorator import tool
from portia.tool_registry import DefaultToolRegistry, InMemoryToolRegistry

dotenv.load_dotenv()


config = Config.from_default(
    default_log_level="DEBUG",
    storage_class=StorageClass.MEMORY,
)


@tool
def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b


tools = InMemoryToolRegistry.from_local_tools([add]) + DefaultToolRegistry(config)

# Instantiate a Portia runner. Load it with the config and with the example tools.
portia = Portia(config=config, tools=tools)
plan = portia.plan("Use the add tool to calculate 1 + 1")
print(plan.model_dump_json(indent=2))
portia.run_plan(plan)
