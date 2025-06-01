import sys
from daytona_sdk import Daytona, DaytonaConfig, SandboxResources, SessionExecuteRequest, CreateSandboxParams
import os
from dotenv import load_dotenv

load_dotenv()

daytona = Daytona(
    DaytonaConfig(api_key=os.getenv("DAYTONA_API_KEY"), target="eu")
)
sandbox = daytona.get_current_sandbox(sys.argv[1])

print(sandbox.process.get_session_command_logs(
    "python-app-session", sys.argv[2],
))