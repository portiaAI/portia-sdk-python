from typing import Tuple, Type

import pandas as pd
from pydantic import BaseModel, Field

from portia.tool import Tool


class FakeToolSchema(BaseModel):
    """Schema defining the inputs for the FakeTool."""

    arg_description: str = Field(
        ...,
        description="Any input string that will be processed by the fake tool",
    )


class FakeTool(Tool[str]):
    """A configurable fake tool for testing purposes."""

    args_schema: Type[BaseModel] = FakeToolSchema
    output_schema: Tuple[str, str] = ("str", "A predefined fake output string")
    args_description: str = Field(..., description="The description of the arguments for the tool")

    def __init__(
        self,
        tool_id: str,
        description: str,
        args_description: str,
    ):
        super().__init__(
            id=tool_id,
            name=tool_id,
            description=description,
            args_description=args_description,
        )

    def run(self, input: str) -> str:
        return "this was a fake tool"

    @classmethod
    def get_from_row(cls, row: pd.Series) -> str:
        return cls(
            tool_id=row["tool_id"],
            description=row["description"],
            args_description=row["args_description"],
        )


def create_fake_tools(csv_path: str) -> list[FakeTool]:
    """Creates a list of FakeTools from a CSV file.

    Args:
        csv_path: Path to the CSV file containing tool data

    Returns:
        List of FakeTool instances created from the CSV data

    """
    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Create a FakeTool for each row using the get_from_row classmethod
    fake_tools = []
    for _, row in df.iterrows():
        # Create tool using the action_id as tool_id and action_description as description
        tool = FakeTool.get_from_row(
            pd.Series(
                {
                    "tool_id": row["action_id"],
                    "description": row["action_description"],
                    "args_description": row["action_inputs"],
                }
            )
        )
        fake_tools.append(tool)

    return fake_tools


if __name__ == "__main__":
    fake_tools = create_fake_tools("portia/tool_filtering_exploration/fake_tools.csv")
    [print(f"{tool.id}: {tool.description}\n") for tool in fake_tools]
