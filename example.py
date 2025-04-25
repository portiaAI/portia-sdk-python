"""Simple Example."""

from portia import (
    Config,
    LogLevel,
    PlanRunState,
    Portia,
    example_tool_registry,
    execution_context,
    open_source_tool_registry,
)
from portia.cli import CLIExecutionHooks
from portia.config import StorageClass
from portia.end_user import EndUser
from dotenv import load_dotenv

from portia.open_source_tools.browser_tool import BrowserTool, BrowserToolForUrl
from portia.tool_registry import PortiaToolRegistry

load_dotenv(override=True)

config = Config.from_default(default_log_level=LogLevel.DEBUG, storage_class=StorageClass.MEMORY)
portia = Portia(
    config,
    tools= PortiaToolRegistry(config=config).get_tools() + open_source_tool_registry.get_tools() + [BrowserTool()],
    
    execution_hooks=CLIExecutionHooks(),
)


query = """
Search for Food Receipt in my google drive (only png).
Create a list of urls as (e.g https://drive.usercontent.google.com/download?id=ID_FROM_GOOGLE_DRIVE) only for shared images.
Then, Analyse each url image. 
Finally, extract how much each person is to be paid and make sure you divide extra charges like service charge, vat, etc.
"""

# query = """
# Analyze the reciept image and extract how much each person is to be paid from https://drive.usercontent.google.com/download?id=1NI3Y9GHRLgzkKltWCvSjbz70itSfjBLO
# Each item is to be paid by the person named in the reciept, make sure you divide extra charges like service charge, vat, etc.
# """

plan = portia.plan(query=query)
print(plan.pretty_print())
input("Press Enter to continue...")




# input("Press Enter to continue...")


plan_run = portia.run_plan(plan)
