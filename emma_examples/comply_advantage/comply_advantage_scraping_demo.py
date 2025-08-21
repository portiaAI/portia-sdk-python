import os
import json
from datetime import datetime
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

def run_scraping_demo(start_row: int, end_row: int):
    # Create timestamp directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"emma_examples/comply_advantage/run_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created output directory: {output_dir}")

    # Read the CSV file
    df = pd.read_csv("emma_examples/comply_advantage/pep_sources.csv")

    # Read the new instructions CSV file
    new_instructions_df = pd.read_csv("emma_examples/comply_advantage/new_instructions.csv")

    # Print basic info
    print("First row of the CSV file:")
    print(df.iloc[0])
    print("\nColumn names:")
    print(df.columns.tolist())

    # Validate row range
    if start_row < 0 or end_row >= len(df) or start_row > end_row:
        print(f"Invalid row range. CSV has {len(df)} rows (0-{len(df)-1})")
        return

    print(f"Processing rows {start_row} to {end_row}")

    # Process each row in the range
    for row in range(start_row, end_row + 1):
        print(f"\n--- Processing row {row} ---")

        # Extract Source Domain and Instruction fields from the specified row
        source_domain = df.iloc[row]["Source Domain"]
        instruction = df.iloc[row]["Instruction"]

        # Check if there's a new instruction for this row
        new_instruction_match = new_instructions_df[new_instructions_df["row"] == row]
        if not new_instruction_match.empty:
            instruction = new_instruction_match.iloc[0]["new_instruction"]
            print(f"Using new instruction for row {row}: {instruction}")

        browser_tool = BrowserTool(
            structured_output_schema=PepExtractionList,
            infrastructure_option=BrowserInfrastructureOption.LOCAL,
        )

        task = f"From the url {source_domain}, navigate using the {instruction} to a page containing a list of people.  Extract the information for each person in the list."

        portia = Portia(
            Config(storage_class=StorageClass.DISK, llm_provider=LLMProvider.GOOGLE_GENERATIVE_AI),
            tools=[browser_tool],
        )

        try:
            plan = portia.plan(task)
            plan_run = portia.run_plan(plan)

            # Save the output as JSON
            output_file = f"{output_dir}/row_{row}_output.json"

            # Extract the result data
            result_data = {
                "row": row,
                "source_domain": source_domain,
                "instruction": instruction,
                "task": task,
                "result": plan_run.dict() if hasattr(plan_run, 'dict') else str(plan_run)
            }

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, indent=2, ensure_ascii=False)

            print(f"Saved output for row {row} to {output_file}")

        except Exception as e:
            print(f"Error processing row {row}: {e}")
            # Save error info as well
            error_file = f"{output_dir}/row_{row}_error.json"
            error_data = {
                "row": row,
                "source_domain": source_domain,
                "instruction": instruction,
                "task": task,
                "error": str(e)
            }
            with open(error_file, 'w', encoding='utf-8') as f:
                json.dump(error_data, f, indent=2, ensure_ascii=False)

# Example usage - modify these parameters as needed
if __name__ == "__main__":
    start_row = 1
    end_row = 3
    run_scraping_demo(start_row, end_row)
