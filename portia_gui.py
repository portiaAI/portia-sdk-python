"""Portia Terminal GUI using textual library.

This module provides a Terminal GUI for visualizing Portia execution with:
- Left panel: Query display
- Middle panel: Plan execution status
- Right panel: Logs (stdout/stderr capture)
"""

from __future__ import annotations

import contextlib
import threading
from typing import Any

import dotenv
from textual import on
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.reactive import reactive, var
from textual.widget import Widget
from textual.widgets import Button, Footer, Header, Label, RichLog, Static

from portia import PlanRun, PlanRunState, Portia
from portia.clarification import Clarification
from portia.clarification_handler import ClarificationHandler
from portia.config import Config
from portia.execution_agents.output import Output
from portia.execution_hooks import BeforeStepExecutionOutcome, ExecutionHooks
from portia.logger import default_logger
from portia.plan import Plan, Step
from portia.tool import Tool


class QueryText(Static):
    """Panel to display the query being executed."""

    query: reactive[str] = reactive("")

    def render(self) -> str:
        """Render the query panel."""
        return f"Query: {self.query}"

    def set_query(self, query: str) -> None:
        """Set the query to display."""
        self.query = query


class StatusText(Static):
    """Panel to display the status of the execution."""

    def compose(self) -> ComposeResult:
        yield Label("Status")
        yield Static("", id="status-text", markup=True)

    def set_status(self, message: str) -> None:
        """Write a message to the status panel."""
        self.query_one("#status-text", Static).update(message)


class PlanStepWidget(Static):
    """Widget to display a single plan step."""

    status: reactive[str] = reactive("pending")

    def __init__(self, step_index: int, step_task: str) -> None:
        super().__init__()
        self.step_index = step_index
        self.step_task = step_task

    def watch_status(self, status: str) -> None:
        """Render the step with status indicator."""
        status_icons = {
            "pending": "⏳",
            "running": "🔄",
            "completed": "✅",
            "failed": "❌",
            "skipped": "⏭️",
        }
        icon = status_icons.get(self.status, "❓")
        self.update(f"{icon} Step {self.step_index + 1}: {self.step_task}")


class PlanPanel(Vertical):
    """Panel to display the plan execution status."""

    def compose(self) -> ComposeResult:
        """Compose the plan panel."""
        yield Static("Plan Execution:", classes="panel-title")
        yield Container(id="plan-steps")

    def set_plan(self, plan) -> None:
        """Set the plan to display."""
        # Clear existing widgets
        steps_container = self.query_one("#plan-steps", Container)
        steps_container.remove_children()

        # Add new step widgets
        if plan and plan.steps:
            for i, step in enumerate(plan.steps):
                step_widget = PlanStepWidget(i, step.task)
                steps_container.mount(step_widget)

    def update_step_status(self, step_index: int, status: str) -> None:
        """Update the status of a specific step."""
        steps_container = self.query_one("#plan-steps", Container)
        if 0 <= step_index < len(steps_container.children):
            child = steps_container.children[step_index]
            child.status = status


class LogCapture:
    """Captures stdout, stderr, and logging output from loguru."""

    def __init__(self, log_widget: RichLog, level: str = "DEBUG") -> None:
        self.log_widget = log_widget
        self.handler_id = None
        self.level = level

    def __enter__(self):
        """Enter context manager."""
        # Add our custom handler to loguru
        self.handler_id = default_logger.add(
            self._log_sink,
            level=self.level,
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {name}:{function}:{line} - {message}",
            catch=True,
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        # Remove our handler from loguru
        if self.handler_id is not None:
            default_logger.remove(self.handler_id)

    def _log_sink(self, message):
        """Custom sink for loguru that sends logs to the RichLog widget."""
        try:
            # Extract log record information from the loguru record
            record = message.record
            level = record["level"].name
            formatted_message = str(message).rstrip()

            # Use call_from_thread to safely update from logging thread
            if hasattr(self.log_widget, "app") and self.log_widget.app:
                self.log_widget.app.call_from_thread(self._add_log, formatted_message, level)
        except Exception:
            # Prevent logging errors from crashing the app
            pass

    def _add_log(self, message: str, level: str) -> None:
        """Add log message to widget (called from main thread)."""
        # Color code by log level
        if level == "ERROR":
            self.log_widget.write(f"[red]{message}[/red]")
        elif level == "WARNING":
            self.log_widget.write(f"[yellow]{message}[/yellow]")
        elif level == "INFO":
            self.log_widget.write(f"[blue]{message}[/blue]")
        elif level == "DEBUG":
            self.log_widget.write(f"[dim]{message}[/dim]")
        else:
            self.log_widget.write(message)


class LogHandler:
    """Legacy LogHandler class - no longer used but kept for compatibility."""

    def __init__(self, log_widget: RichLog) -> None:
        self.log_widget = log_widget


class LogPanel(Vertical):
    """Panel to display logs."""

    def compose(self) -> ComposeResult:
        """Compose the log panel."""
        yield Static("Logs:", classes="panel-title")
        yield RichLog(id="log-output", auto_scroll=True, wrap=True, highlight=True, markup=True)


class PortiaExit(Exception):
    """Exception to signal that the Portia execution should be stopped."""


class PortiaGUI(App):
    """Main Portia GUI application."""

    CSS = """
    Horizontal {
        height: 100%;
    }
    
    QueryText {
        width: 1fr;
        min-width: 30;
        padding: 1;
        border: solid $primary;
    }
    
    #plan-panel {
        width: 2fr;
        min-width: 40;
        border: solid blue;
    }
    
    #log-panel {
        width: 2fr;
        min-width: 40;
        border: solid red;
    }
    
    .panel-title {
        background: $primary;
        color: $text;
        padding: 0 1;
        margin-bottom: 1;
        text-style: bold;
    }
    
    #plan-steps {
        padding: 1;
        height: auto;
    }
    
    #log-output {
        height: 1fr;
        border: solid $primary;
        margin: 1;
    }
    
    #controls {
        height: 3;
        dock: bottom;
    }
    
    Button {
        margin: 0 1;
    }
    
    PlanStepWidget {
        height: auto;
        margin: 0 0 1 0;
        padding: 0 1;
    }
    """

    # Reactive variables for state management
    current_plan_run: var[PlanRun | None] = var(None)
    execution_complete: var[bool] = var(False)

    def __init__(
        self,
        query: str = "",
        # queue: mp.Queue = None,
        **kwargs,
    ) -> None:
        """Initialize the GUI.

        Args:
            portia: The Portia instance to use for execution
            query: The query to execute
            **kwargs: Additional arguments passed to App
        """
        super().__init__(**kwargs)

        self.query = query
        self.execution_thread: threading.Thread | None = None
        self.plan = None
        self._stop_requested = False
        # self.queue = queue

    def compose(self) -> ComposeResult:
        """Compose the main application layout."""
        yield Header()
        yield Horizontal(
            Vertical(
                Container(QueryText()),
                Container(StatusText())
            ),
            Container(PlanPanel(), id="plan-panel"),
            Container(LogPanel(), id="log-panel"),
        )
        yield Container(
            Button("Start Execution", id="start-btn", variant="success"),
            Button("Stop Execution", id="stop-btn", variant="error", disabled=True),
            Button("Quit", id="quit-btn", variant="warning"),
            id="controls",
        )
        yield Footer()

    def on_mount(self) -> None:
        """Called when the app is mounted."""
        self.title = "Portia Terminal GUI"
        query_panel = self.query_one(QueryText)
        query_panel.set_query(self.query)

    def _run_portia(self) -> None:
        self.query_one(StatusText).set_status("Setting up Portia...")
        def before_step_execution(plan: Plan, plan_run: PlanRun, step: Step) -> BeforeStepExecutionOutcome:
            self.current_plan_run = plan_run
            self.set_status(f"Running step {plan_run.current_step_index + 1} of {len(plan.steps)}")
            self.call_from_thread(
                self.query_one("#plan-panel PlanPanel", PlanPanel).update_step_status,
                plan_run.current_step_index,
                "running"
            )
            if self._stop_requested:
                raise PortiaExit()
            return BeforeStepExecutionOutcome.CONTINUE
        
        def after_step_execution(plan: Plan, plan_run: PlanRun, step: Step, output: Output) -> None:
            self.current_plan_run = None
            self.call_from_thread(
                self.query_one("#plan-panel PlanPanel", PlanPanel).update_step_status,
                plan_run.current_step_index,
                "completed"
            )
            if self._stop_requested:
                raise PortiaExit()

        def before_plan_run(plan: Plan, plan_run: PlanRun) -> None:
            self.current_plan_run = plan_run
            if self._stop_requested:
                raise PortiaExit()

        def after_plan_run(plan: Plan, plan_run: PlanRun, output: Output) -> None:
            self.current_plan_run = plan_run
            if self._stop_requested:
                raise PortiaExit()

        def before_tool_call(tool: Tool, args: dict[str, Any], plan_run: PlanRun, step: Step) -> Clarification | None:
            self.current_plan_run = plan_run
            if self._stop_requested:
                raise PortiaExit()
            return None

        def after_tool_call(tool: Tool, output: Any, plan_run: PlanRun, step: Step) -> Clarification | None:
            self.current_plan_run = plan_run
            if self._stop_requested:
                raise PortiaExit()
            return None

        exec_hooks = ExecutionHooks(
            before_step_execution=before_step_execution,
            after_step_execution=after_step_execution,
            before_plan_run=before_plan_run,
            after_plan_run=after_plan_run,
            before_tool_call=before_tool_call,
            after_tool_call=after_tool_call,
            clarification_handler=None,  # TODO
        )

        portia = Portia(
            Config.from_default(),
            execution_hooks=exec_hooks,
        )

        try:
            with LogCapture(
                self.query_one("#log-output", RichLog),
                level=portia.config.default_log_level.value
            ):
                self.set_status("Planning...")
                self.plan = portia.plan(self.query)
                self.set_status("Planning complete")

                if self._stop_requested:
                    return

                # Update the plan panel
                plan_panel = self.query_one("#plan-panel PlanPanel", PlanPanel)
                self.call_from_thread(plan_panel.set_plan, self.plan)

                self.set_status("Executing...")
                portia.run_plan(self.plan)
        except PortiaExit:
            self.set_status("[red]Execution stop requested by user[/red]")
        except Exception as e:
            self.set_status(f"[red]Execution failed: {e!s}[/red]")
        finally:
            # Re-enable start button
            start_btn = self.query_one("#start-btn", Button)
            stop_btn = self.query_one("#stop-btn", Button)
            self.call_from_thread(self._reset_buttons, start_btn, stop_btn)

    @on(Button.Pressed, "#start-btn")
    def start_execution(self) -> None:
        """Start the execution."""
        if self.execution_thread and self.execution_thread.is_alive():
            return

        self._stop_requested = False

        # Update button states
        start_btn = self.query_one("#start-btn", Button)
        stop_btn = self.query_one("#stop-btn", Button)
        start_btn.disabled = True
        stop_btn.disabled = False

        # Clear the log
        log_widget = self.query_one("#log-output", RichLog)
        log_widget.clear()

        # Start execution in a separate thread
        self.execution_thread = threading.Thread(target=self._run_portia)
        self.execution_thread.daemon = True
        self.execution_thread.start()

    @on(Button.Pressed, "#stop-btn")
    def stop_execution(self) -> None:
        """Stop the execution."""
        self._stop_requested = True

        stop_btn = self.query_one("#stop-btn", Button)
        stop_btn.disabled = True

        self.set_status("[red]Execution stop requested by user[/red]")

    def set_status(self, message: str) -> None:
        """Set the status of the execution."""
        self.log("hello")
        status_text = self.query_one(StatusText)
        self.log(f"Setting status: {message} on {status_text}")
        self.call_from_thread(self.query_one(StatusText).set_status, message)

    @on(Button.Pressed, "#quit-btn")
    def quit_app(self) -> None:
        """Quit the application."""
        self._stop_requested = True
        self.exit()

    # def _run_execution(self) -> None:
    #     """Run the Portia execution in a separate thread."""
    #     log_widget = None

    #     try:
    #         # Get the log widget for capturing output
    #         log_widget = self.query_one("#log-output", RichLog)

    #         # Setup log capture
    #         with LogCapture(log_widget, level=self.portia.config.default_log_level.value):
    #             if self._stop_requested:
    #                 return

    #             # First, create the plan
    #             self.plan = self.portia.plan(self.query)

    #             if self._stop_requested:
    #                 return

    #             # Update the plan panel
    #             plan_panel = self.query_one("#plan-panel PlanPanel", PlanPanel)
    #             self.call_from_thread(plan_panel.set_plan, self.plan)

    #             if self._stop_requested:
    #                 return

    #             # Execute the plan step by step with updates
    #             self._execute_plan_with_updates()

    #     except Exception as e:
    #         self.set_status(f"[red]Execution failed: {e!s}[/red]")

    #     finally:
    #         # Re-enable start button
    #         start_btn = self.query_one("#start-btn", Button)
    #         stop_btn = self.query_one("#stop-btn", Button)
    #         self.call_from_thread(self._reset_buttons, start_btn, stop_btn)

    # def _execute_plan_with_updates(self) -> None:
    #     """Execute the plan while providing real-time updates."""
    #     plan_panel = self.query_one("#plan-panel PlanPanel", PlanPanel)

    #     try:
    #         if self._stop_requested:
    #             return

    #         # Run the plan directly and monitor via polling
    #         # This is simpler than trying to step through manually

    #         # Start execution in the background and monitor progress
    #         execution_thread = threading.Thread(target=self._execute_plan_run)
    #         execution_thread.daemon = True
    #         execution_thread.start()

    #         # Monitor progress
    #         last_observed_step_index = -1
    #         while execution_thread.is_alive() and not self._stop_requested:
    #             if (
    #                 self.plan_run_id is not None
    #                 (plan_run := self.portia.storage.get_plan_run(self.plan_run_id))
    #                 and plan_run.current_step_index != last_observed_step_index
    #             ):
    #                 current_step_index = plan_run.current_step_index

    #                 # Mark previous step as completed
    #                 if last_observed_step_index >= 0:
    #                     self.call_from_thread(
    #                         plan_panel.update_step_status, last_observed_step_index, "completed"
    #                     )

    #                 # Mark current step as running
    #                 if current_step_index < len(self.plan.steps):
    #                     self.call_from_thread(
    #                         plan_panel.update_step_status, current_step_index, "running"
    #                     )

    #                 last_observed_step_index = current_step_index

    #             # Small delay to avoid busy waiting
    #             threading.Event().wait(0.1)

    #         # Wait for execution to complete
    #         execution_thread.join(timeout=1.0)

    #         # Final status updates
    #         if self.plan_run_id is not None:
    #             plan_run = self.portia.storage.get_plan_run(self.plan_run_id)
    #             if plan_run.state == PlanRunState.COMPLETE:
    #                 # Mark last step as completed
    #                 for step_index in range(max(last_observed_step_index, 0), len(self.plan.steps)):
    #                     self.call_from_thread(
    #                         plan_panel.update_step_status, step_index, "completed"
    #                     )

    #                 self.set_status("[green]Execution completed successfully![/green]")
    #                 if plan_run.outputs.final_output:
    #                     self.set_status(f"[green]Final output: {plan_run.outputs.final_output.get_summary()}[/green]")
    #             elif plan_run.state == PlanRunState.FAILED:
    #                 # Mark current step as failed
    #                 if last_observed_step_index >= 0:
    #                     self.call_from_thread(
    #                         plan_panel.update_step_status, last_observed_step_index, "failed"
    #                     )

    #                 self.set_status("[red]Execution failed![/red]")
    #             elif plan_run.state == PlanRunState.NEED_CLARIFICATION:
    #                 clarifications = plan_run.get_outstanding_clarifications()
    #                 clarifications_str = "\n".join(
    #                     clarification.model_dump_json(indent=2) for clarification in clarifications
    #                 )
    #                 self.set_status(
    #                     f"[yellow]Clarifications:\n{clarifications_str}[/yellow]"
    #                 )

    #     except Exception as e:
    #         import traceback
    #         stack_trace = traceback.format_exc()
    #         self.set_status(f"[red]Error during execution: {e!s} {stack_trace}[/red]")
    #     finally:
    #         self.plan_run_id = None

    # def _execute_plan_run(self) -> None:
    #     """Execute the plan run (called in separate thread)."""
    #     with contextlib.suppress(Exception):
    #         plan_run = self.portia.run_plan(self.plan)
    #         self.plan_run_id = plan_run.id

    def _reset_buttons(self, start_btn: Button, stop_btn: Button) -> None:
        """Reset button states after execution."""
        start_btn.disabled = False
        stop_btn.disabled = True


# import multiprocessing as mp


# def portia_server(queue: mp.Queue):
#     while True:
#         queue.get()


dotenv.load_dotenv()
query = "Check my calendar for a free time on Monday then schedule a meeting with sam+test@portialabs.ai for 1 hour in a free slot"
print("🚀 Starting Portia GUI...")
print(f"📝 Query: {query}")
print("🖥️  Press Ctrl+C in the GUI to quit")
# queue = mp.Queue()
# server_process = mp.Process(target=portia_server, args=(queue,))

app = PortiaGUI(query=query)
app.run()