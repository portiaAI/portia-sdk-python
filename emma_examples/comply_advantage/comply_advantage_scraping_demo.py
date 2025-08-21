from dotenv import load_dotenv
from pydantic import BaseModel

from portia import (
    Config,
    LLMProvider,
    Portia,
    StorageClass,
)
from portia.open_source_tools.browser_tool import BrowserInfrastructureOption, BrowserTool

task = """Get the mayor, deputy mayor, and city councillors for
        Sydney City Council (https://meetings.cityofsydney.nsw.gov.au/).
        Include titles, and post-nominal qualifications.  Capture the picture of the entity
        as picture_url.  Capture the link to the councillor's detail page as related_url.
        Write the output to a file called captured.json.
        """


class PepExtraction(BaseModel):
    title: str | None
    name: str
    post_nominal_qualifications: str | None
    picture_url: str | None
    related_url: str | None
    additional_information: str | None


class PepExtractionList(BaseModel):
    people: list[PepExtraction]


class ExtractedDataModel(BaseModel):
    people: list[PepExtraction]


load_dotenv(override=True)

import pandas as pd

# Read the CSV file
df = pd.read_csv("emma_examples/comply_advantage/pep_sources.csv")

# Read the new instructions CSV file
new_instructions_df = pd.read_csv("emma_examples/comply_advantage/new_instructions.csv")

# Print the first row
print("First row of the CSV file:")
print(df.iloc[0])

# Also print the column names for context
print("\nColumn names:")
print(df.columns.tolist())

row = 1

# Extract Source Domain and Instruction fields from the specified row into a tuple
source_domain = df.iloc[row]["Source Domain"]
instruction = df.iloc[row]["Instruction"]

# Check if there's a new instruction for this row
new_instruction_match = new_instructions_df[new_instructions_df["row"] == row]
if not new_instruction_match.empty:
    instruction = new_instruction_match.iloc[0]["new_instruction"]
    print(f"Using new instruction for row {row}: {instruction}")

extracted_fields = (source_domain, instruction)

browser_tool = BrowserTool(
    structured_output_schema=PepExtractionList,
    infrastructure_option=BrowserInfrastructureOption.LOCAL,
)

task = f"From the url {source_domain}, navigate using the {instruction} to a page containing a list of people.  Extract the information for each person in the list."

tools = [browser_tool]

portia = Portia(
    Config(storage_class=StorageClass.DISK, llm_provider=LLMProvider.GOOGLE_GENERATIVE_AI),
    tools=[browser_tool],
)

plan = portia.plan(task)
plan_run = portia.run_plan(plan)
