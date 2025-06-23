import asyncio
from time import sleep
from daytona_sdk import Daytona, DaytonaConfig, SessionExecuteRequest, CreateSandboxBaseParams
import os
from dotenv import load_dotenv

load_dotenv()

daytona = Daytona(
    DaytonaConfig(api_key=os.getenv("DAYTONA_API_KEY"), target="eu")
)
IMG_NAME = "harbor-transient.internal.daytona.app/daytona/portia-gui-amd:0.3"
sandbox = daytona.create(params=CreateSandboxBaseParams(image=IMG_NAME), timeout=10 * 60)
exec_session_id = "python-app-session"
sandbox.process.create_session(exec_session_id)

preview_url = sandbox.get_preview_link(8000)
command = sandbox.process.execute_session_command(exec_session_id, SessionExecuteRequest(
    command=f"textual serve --dev -h 0.0.0.0 -u {preview_url} -p 8000 /app/portia_gui.py",
    run_async=True
))

# Get the preview link for the Flask app

print(f"App is available at: {preview_url}?fontsize=12")
print(f"\n----------------------------------------------\n")
# print("To view logs, run:")
# print(f"  `uv run python daytona_logs.py {sandbox} {command.cmd_id}`")
