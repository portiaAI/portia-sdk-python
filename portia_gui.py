"""Portia Terminal GUI using textual library.

This module provides a Terminal GUI for visualizing Portia execution with:
- Left panel: Query display
- Middle panel: Plan execution status
- Right panel: Logs (stdout/stderr capture)
"""

from __future__ import annotations

import threading
from typing import Any, ClassVar

import dotenv
from textual import events, log, on
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Container, Horizontal, Vertical
from textual.message import Message
from textual.reactive import reactive, var
from textual.widgets import Button, Footer, Header, Label, RichLog, Static, TextArea

from portia import PlanRun, Portia
from portia.clarification import Clarification
from portia.config import Config
from portia.execution_agents.output import Output
from portia.execution_hooks import BeforeStepExecutionOutcome, ExecutionHooks
from portia.logger import default_logger
from portia.plan import Plan, Step
from portia.tool import Tool


class PanelTitle(Static):
    """Panel title."""

    def __init__(self, title: str) -> None:
        super().__init__()
        self.title = title

    def compose(self) -> ComposeResult:
        yield Label(self.title, classes="panel-title")

class QueryText(Static):
    """Panel to display the query being executed."""

    query: reactive[str] = reactive("")

    class QueryUpdate(Message):
        """Message to update the query."""

        def __init__(self, query: str) -> None:
            super().__init__()
            self.query = query

    def compose(self) -> ComposeResult:
        yield PanelTitle("Query")
        yield TextArea(id="query-input")

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        """Handle the query update."""
        self.post_message(self.QueryUpdate(event.control.text))

class StatusText(Static):
    """Panel to display the status of the execution."""

    def compose(self) -> ComposeResult:
        yield PanelTitle("Status")
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
        yield PanelTitle("Plan Execution")
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
        yield PanelTitle("Logs")
        yield RichLog(id="log-output", auto_scroll=True, wrap=True, highlight=True, markup=True)


class PortiaExit(Exception):
    """Exception to signal that the Portia execution should be stopped."""


class PortiaGUI(App):
    """Main Portia GUI application."""

    theme = "nord"
    CSS_PATH = "portia_gui.tcss"
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("ctrl+r", "start_execution", "Start execution", priority=True, show=True),
    ]

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
        with Container(id="main-container"):
            yield Header()
            yield Horizontal(
                Vertical(
                    Container(QueryText(), classes="app-panel"),
                    Container(StatusText(), classes="app-panel")
                ),
                Container(PlanPanel(), id="plan-panel", classes="app-panel"),
                Container(LogPanel(), id="log-panel", classes="app-panel"),
                id="query-status-panel",
            )
            yield Horizontal(
                Button("Start Execution", id="start-btn", variant="success"),
                Button("Stop Execution", id="stop-btn", variant="error", disabled=True),
                Button("Quit", id="quit-btn", variant="warning"),
                id="controls-panel",
                classes="app-panel",
            )
            yield Footer()

    def on_mount(self) -> None:
        """Called when the app is mounted."""
        self.title = "Portia Terminal GUI"

    @on(QueryText.QueryUpdate)
    def update_query(self, event: QueryText.QueryUpdate) -> None:
        """Update the query."""
        self.query = event.query

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
            self.set_status("[red]Portia exited after execution stop requested by user[/red]")
        except Exception as e:
            self.set_status(f"[red]Execution failed: {e!s}[/red]")
        finally:
            # Re-enable start button
            start_btn = self.query_one("#start-btn", Button)
            stop_btn = self.query_one("#stop-btn", Button)
            self.call_from_thread(self._reset_buttons, start_btn, stop_btn)

    def action_start_execution(self) -> None:
        """Start the execution."""
        self.start_execution()

    @on(Button.Pressed, "#start-btn")
    def handle_press_start_btn(self) -> None:
        """Handle the press of the start button."""
        self.start_execution()

    def start_execution(self) -> None:
        """Start the execution."""
        if self.execution_thread and self.execution_thread.is_alive():
            return
        if self.execution_thread and not self.execution_thread.is_alive():
            self.execution_thread = None

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

        self.query_one(StatusText).set_status("[red]Execution stop requested by user[/red]")

    def set_status(self, message: str) -> None:
        """Set the status of the execution."""
        status_text = self.query_one(StatusText)
        self.call_from_thread(status_text.set_status, message)

    @on(Button.Pressed, "#quit-btn")
    def quit_app(self) -> None:
        """Quit the application."""
        self._stop_requested = True
        self.exit()

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