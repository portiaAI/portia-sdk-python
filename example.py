"""Simple Example."""

from dotenv import load_dotenv

from portia import (
    Config,
    LogLevel,
    Portia,
    PortiaToolRegistry,
    example_tool_registry,
)
from portia.cli import CLIExecutionHooks
from portia.config import StorageClass

load_dotenv(override=True)
config = Config.from_default(default_log_level=LogLevel.DEBUG, storage_class=StorageClass.MEMORY)
tools = example_tool_registry + PortiaToolRegistry(config)
portia = Portia(
    config,
    tools=tools,
    execution_hooks=CLIExecutionHooks(),
)

query = """Get my (omar@portialabs.ai) availability from Google Calendar tomorrow between 10:00 and 17:00
- Schedule a 30 minute meeting with target@portialabs.ai at a time that works for me with the title "Portia AI Demo" and a description of the meeting as "Test demo".
- Send an email to omar@portialabs.ai with the details of the meeting you scheduled."""

# query = "Email nathan@portialabs.ai and emma@portialabs.ai via Outlook, with the title \"Meeting\" and the body \"Let's meet to discuss the project\""
# query = "Find me the most recent report from Google Drive and its content."
# query = "Who are all the people involved in Zendesk ticket 123456789?"

# query = "Send an email to Emma (burrowse0@gmail.com) with a short history of corruption in US elections. Then retrieve the weather in London."

# query = "Transfer Zendesk ticket 1234567890 to Many (ID: 567891234)"

# query = "Use Google Sheets to open the file https://docs.google.com/spreadsheets/d/1vO6aZD466tHlhfzgSRW_W4HtcKy6AFNtqpyPUBxsxJhQ/edit?gid=0#gid=0 and extract the total revenue."

# query  = "Find everything in my Outlook from robbie@portialabs.ai"

# query = "If the weather in Milton Keynes is sunny or clear, then sum it with it the weather in Cairo"


# query = "Search for Expenses sheet in my drive, and send a summary of car costs to omar@portialabs.ai."

# Simple Example
plan = portia.plan(
    query,
)

print(plan.model_dump_json(indent=2))

input("Press Enter to continue...")

# Execute the plan
output = portia.run_plan(plan)

# print(output.model_dump_json(indent=2))
