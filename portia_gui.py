"""Portia Terminal GUI using textual library.

This module provides a Terminal GUI for visualizing Portia execution with:
- Left panel: Query display
- Middle panel: Plan execution status
- Right panel: Logs (stdout/stderr capture)
"""

from __future__ import annotations

import threading
from typing import Any, ClassVar, cast

import dotenv
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.message import Message
from textual.reactive import reactive, var
from textual.widgets import Button, Footer, Header, Label, RichLog, Static, TextArea, ListView, ListItem, Link

from portia import PlanRun, Portia
from portia.clarification import ActionClarification, Clarification
from portia.config import Config
from portia.execution_agents.output import LocalDataValue, Output
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

    def set_status(self, status: str) -> None:
        """Set the status."""
        full_title = f"{self.title} - {status}" if status else self.title
        self.query_one(Label).update(full_title)

class QueryText(Static):
    """Panel to display the query being executed."""

    query_text: reactive[str] = reactive("")

    class QueryUpdate(Message):
        """Message to update the query."""

        def __init__(self, query_text: str) -> None:
            super().__init__()
            self.query_text = query_text

    def compose(self) -> ComposeResult:
        yield PanelTitle("Query: Write your query below")
        yield TextArea(id="query-input")

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        """Handle the query update."""
        self.post_message(self.QueryUpdate(event.control.text))

class ToolsList(Static):
    """Panel to display the tools list."""

    def compose(self) -> ComposeResult:
        yield PanelTitle("Tools List")
        yield ListView(id="tools-list")

    def on_mount(self) -> None:
        """Handle the mount event."""
        portia = Portia(Config.from_default())
        tools = portia.tool_registry
        for tool in tools:
            name = tool.name.replace("Portia ", "")
            list_item = ListItem(Label(name))
            list_item.tooltip = tool.description
            self.query_one("#tools-list", ListView).append(list_item)

class PlanStepWidget(Static):
    """Widget to display a single plan step."""

    status: reactive[str] = reactive("pending")

    def __init__(self, step_index: int, step_task: str) -> None:
        super().__init__()
        self.step_index = step_index
        self.step_task = step_task
        self.output: str | None = None

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
        if self.output:
            self.update(f"{icon} Step {self.step_index + 1}: {self.step_task}\n{self.output}")
        else:
            self.update(f"{icon} Step {self.step_index + 1}: {self.step_task}")


class PlanPanel(Vertical):
    """Panel to display the plan execution status."""

    def compose(self) -> ComposeResult:
        """Compose the plan panel."""
        yield PanelTitle("Plan Viewer")
        yield ScrollableContainer(id="plan-steps")
        yield ScrollableContainer(id="clarification-container")
        yield ScrollableContainer(id="plan-output")

    def set_plan(self, plan: Plan) -> None:
        """Set the plan to display."""
        # Clear existing widgets
        steps_container = self.query_one("#plan-steps", ScrollableContainer)
        steps_container.remove_children()

        # Add new step widgets
        if plan and plan.steps:
            for i, step in enumerate(plan.steps):
                step_widget = PlanStepWidget(i, step.task)
                steps_container.mount(step_widget)

    def update_step_status(self, step_index: int, status: str) -> None:
        """Update the status of a specific step."""
        steps_container = self.query_one("#plan-steps", ScrollableContainer)
        if 0 <= step_index < len(steps_container.children):
            child: PlanStepWidget = cast(PlanStepWidget, steps_container.children[step_index])
            child.status = status

    def update_final_output(self, output: Output) -> None:
        """Update the final output."""
        output_container = self.query_one("#plan-output", ScrollableContainer)
        output_container.remove_children()
        output_container.mount(Label("[b][green]Final output[/green][/b]"))
        rich_log = RichLog(id="plan-output-rich-log", auto_scroll=True, wrap=True)
        output_container.mount(rich_log)
        if isinstance(output, LocalDataValue):
            rich_log.write(str(output.value))
        else:
            rich_log.write(output.summary)

    def set_clarifications(self, clarifications: list[Clarification]) -> None:
        """Set the clarifications."""
        clarification_container = self.query_one("#clarification-container", ScrollableContainer)
        self.clear_clarifications()
        if not clarifications:
            return
        for clarification in clarifications:
            match clarification:
                case ActionClarification():
                    clarification_container.mount(Link("A Tool needs authorization", url=str(clarification.action_url)))
                case _:
                    clarification_container.mount(Label("Sorry, other clarification types are not supported here yet"))

    def clear_clarifications(self) -> None:
        """Clear the clarifications."""
        clarification_container = self.query_one("#clarification-container", ScrollableContainer)
        clarification_container.remove_children()

    def clear_plan(self) -> None:
        """Clear the plan."""
        plan_container = self.query_one("#plan-steps", ScrollableContainer)
        plan_container.remove_children()

    def clear_output(self) -> None:
        """Clear the output."""
        output_container = self.query_one("#plan-output", ScrollableContainer)
        output_container.remove_children()

    def set_status(self, status: str) -> None:
        """Set the status."""
        self.query_one(PanelTitle).set_status(status)

    def reset_panel(self) -> None:
        """Reset the panel."""
        self.clear_clarifications()
        self.clear_plan()
        self.clear_output()
        self.set_status("Ready")

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

    def clear(self) -> None:
        """Clear the log."""
        self.query_one("#log-output", RichLog).clear()

class PortiaGUI(App):
    """Main Portia GUI application."""

    theme = "nord"
    CSS_PATH = "portia_gui.tcss"
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("ctrl+r", "run_plan", "Run plan", priority=True, show=True),
    ]

    # Reactive variables for state management
    current_plan_run: var[PlanRun | None] = var(None)
    execution_complete: var[bool] = var(False)

    def __init__(
        self,
        **kwargs,
    ) -> None:
        """Initialize the GUI.

        Args:
            portia: The Portia instance to use for execution
            **kwargs: Additional arguments passed to App
        """
        super().__init__(**kwargs)

        self.query_text = "What is the weather in Tokyo?"
        self.execution_thread: threading.Thread | None = None
        self.plan: Plan | None = None
        self.plan_run: PlanRun | None = None

    def compose(self) -> ComposeResult:
        """Compose the main application layout."""
        with Container(id="main-container"):
            yield Header()
            yield Horizontal(
                Vertical(
                    Container(QueryText(), classes="app-panel"),
                    Container(ToolsList(), classes="app-panel")
                ),
                Container(PlanPanel(), id="plan-panel", classes="app-panel"),
                Container(LogPanel(), id="log-panel", classes="app-panel"),
                id="query-status-panel",
            )
            yield Horizontal(
                Button("Plan Query", id="plan-btn", variant="success"),
                Button("Run Plan", id="run-btn", variant="error", disabled=True),
                Button("Reset", id="reset-btn", variant="warning"),
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
        self.query_text = event.query_text

    def _plan_query(self) -> None:
        """Plan the query."""
        self.query_one(PlanPanel).set_status("Planning...")
        portia = Portia(Config.from_default())
        try:
            with LogCapture(
                self.query_one("#log-output", RichLog),
                level=portia.config.default_log_level.value
            ):
                self.plan = portia.plan(self.query_text)
                # Update the plan panel
                plan_panel = self.query_one("#plan-panel PlanPanel", PlanPanel)
                self.call_from_thread(plan_panel.set_plan, self.plan)
                self.set_status("Planning complete")
        except Exception as e:
            self.set_status(f"[red]Planning failed: {e!s}[/red]")
        finally:
            self.call_from_thread(self._reset_buttons, self.query_one("#plan-btn", Button), self.query_one("#run-btn", Button))

    def _run_plan(self) -> None:
        if not self.plan:
            return
        self.query_one(PlanPanel).set_status("Setting up Portia...")
        def before_step_execution(plan: Plan, plan_run: PlanRun, step: Step) -> BeforeStepExecutionOutcome:
            self.current_plan_run = plan_run
            self.set_status(f"Running step {plan_run.current_step_index + 1} of {len(plan.steps)}")
            self.call_from_thread(
                self.query_one("#plan-panel PlanPanel", PlanPanel).update_step_status,
                plan_run.current_step_index,
                "running"
            )
            return BeforeStepExecutionOutcome.CONTINUE

        def after_step_execution(plan: Plan, plan_run: PlanRun, step: Step, output: Output) -> None:
            self.current_plan_run = None
            self.call_from_thread(
                self.query_one("#plan-panel PlanPanel", PlanPanel).update_step_status,
                plan_run.current_step_index,
                "completed",
            )

        exec_hooks = ExecutionHooks(
            before_step_execution=before_step_execution,
            after_step_execution=after_step_execution,
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
                self.call_from_thread(
                    self.query_one("#plan-panel PlanPanel", PlanPanel).clear_clarifications
                )
                if self.plan_run:
                    self.set_status("Resuming execution...")
                    plan_run = portia.resume(self.plan_run)
                else:
                    self.set_status("Beginning execution...")
                    plan_run = portia.run_plan(self.plan)
                self.plan_run = plan_run
                if clarifications := plan_run.get_outstanding_clarifications():
                    self.set_status("Waiting for clarifications...")
                    self.call_from_thread(
                        self.query_one("#plan-panel PlanPanel", PlanPanel).set_clarifications,
                        clarifications
                    )
                elif plan_run.outputs.final_output:
                    self.call_from_thread(
                        self.query_one("#plan-panel PlanPanel", PlanPanel).update_final_output,
                        plan_run.outputs.final_output
                    )
                    self.set_status("Execution complete")
                    self.plan_run = None
                else:
                    self.set_status("Something went wrong")
        except Exception as e:
            self.set_status(f"[red]Execution failed: {e!s}[/red]")
        finally:
            plan_btn = self.query_one("#plan-btn", Button)
            run_btn = self.query_one("#run-btn", Button)
            self.call_from_thread(self._reset_buttons, plan_btn, run_btn)

    @on(Button.Pressed, "#plan-btn")
    def handle_press_plan_btn(self) -> None:
        """Handle the press of the start button."""
        self.start_thread(execute=False)

    def start_thread(self, execute: bool = False) -> None:
        """Start the execution."""
        if self.execution_thread and self.execution_thread.is_alive():
            return
        if self.execution_thread and not self.execution_thread.is_alive():
            self.execution_thread = None

        # Update button states
        plan_btn = self.query_one("#plan-btn", Button)
        run_btn = self.query_one("#run-btn", Button)
        plan_btn.disabled = True
        run_btn.disabled = True

        # Clear the log
        log_widget = self.query_one("#log-output", RichLog)
        log_widget.clear()
        target = self._run_plan if execute else self._plan_query
        # Start execution in a separate thread
        self.execution_thread = threading.Thread(target=target)
        self.execution_thread.daemon = True
        self.execution_thread.start()

    @on(Button.Pressed, "#run-btn")
    def run_plan(self) -> None:
        """Stop the execution."""
        if not self.plan or self.plan_run:
            return
        self.start_thread(execute=True)

    def set_status(self, message: str) -> None:
        """Set the status of the execution."""
        plan_panel = self.query_one(PlanPanel)
        self.call_from_thread(plan_panel.set_status, message)

    @on(Button.Pressed, "#reset-btn")
    def quit_app(self) -> None:
        """Quit the application."""
        self.plan_run = None
        self.plan = None
        self.execution_thread = None
        self.query_one(PlanPanel).reset_panel()
        self.query_one(LogPanel).clear()
        self._reset_buttons(self.query_one("#plan-btn", Button), self.query_one("#run-btn", Button))

    def _reset_buttons(self, plan_btn: Button, run_btn: Button) -> None:
        """Reset button states after execution."""
        plan_btn.disabled = False
        run_btn.disabled = not self.plan and not self.plan_run


dotenv.load_dotenv()
app = PortiaGUI()
app.run()