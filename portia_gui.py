"""Portia Terminal GUI using textual library.

This module provides a Terminal GUI for visualizing Portia execution with:
- Left panel: Query display
- Middle panel: Plan execution status
- Right panel: Logs (stdout/stderr capture)
"""

from __future__ import annotations

import io
import threading
from contextlib import redirect_stderr, redirect_stdout
from typing import Any

from textual import on
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.reactive import var
from textual.widgets import Button, Footer, Header, RichLog, Static

from portia import Portia, PlanRun, PlanRunState
from portia.logger import default_logger


class QueryPanel(Static):
    """Panel to display the query being executed."""

    def __init__(self, query: str = "") -> None:
        super().__init__()
        self.query = query

    def compose(self) -> ComposeResult:
        """Compose the query panel."""
        yield Static(f"Query: {self.query}", id="query-text")


class PlanStepWidget(Static):
    """Widget to display a single plan step."""

    status: var[str] = var("pending")

    def __init__(self, step_index: int, step_task: str, status: str = "pending") -> None:
        super().__init__()
        self.step_index = step_index
        self.step_task = step_task
        self.status = status

    def render(self) -> str:
        """Render the step with status indicator."""
        status_icons = {
            "pending": "⏳",
            "running": "🔄",
            "completed": "✅",
            "failed": "❌",
            "skipped": "⏭️",
        }
        icon = status_icons.get(self.status, "❓")
        return f"{icon} Step {self.step_index + 1}: {self.step_task}"


class PlanPanel(Vertical):
    """Panel to display the plan execution status."""

    def __init__(self) -> None:
        super().__init__()
        self.step_widgets: list[PlanStepWidget] = []

    def compose(self) -> ComposeResult:
        """Compose the plan panel."""
        yield Static("Plan Execution:", classes="panel-title")
        yield Container(id="plan-steps")

    def set_plan(self, plan) -> None:
        """Set the plan to display."""
        # Clear existing widgets
        steps_container = self.query_one("#plan-steps", Container)
        steps_container.remove_children()
        self.step_widgets.clear()

        # Add new step widgets
        if plan and plan.steps:
            for i, step in enumerate(plan.steps):
                step_widget = PlanStepWidget(i, step.task)
                self.step_widgets.append(step_widget)
                steps_container.mount(step_widget)

    def update_step_status(self, step_index: int, status: str) -> None:
        """Update the status of a specific step."""
        if 0 <= step_index < len(self.step_widgets):
            self.step_widgets[step_index].status = status


class LogCapture:
    """Captures stdout, stderr, and logging output from loguru."""

    def __init__(self, log_widget: RichLog) -> None:
        self.log_widget = log_widget
        self.handler_id = None

    def __enter__(self):
        """Enter context manager."""
        # Add our custom handler to loguru
        self.handler_id = default_logger.add(
            self._log_sink,
            level="DEBUG",
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {name}:{function}:{line} - {message}",
            catch=True
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
        yield RichLog(id="log-output", auto_scroll=True)


class PortiaGUI(App):
    """Main Portia GUI application."""

    CSS = """
    Horizontal {
        height: 100%;
    }
    
    #query-panel {
        width: 1fr;
        min-width: 30;
        border: solid green;
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
    
    #query-text {
        padding: 1;
        border: solid $primary;
        margin: 1;
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

    def __init__(self, portia: Portia, query: str = "", **kwargs) -> None:
        """Initialize the GUI.

        Args:
            portia: The Portia instance to use for execution
            query: The query to execute
            **kwargs: Additional arguments passed to App
        """
        super().__init__(**kwargs)
        self.portia = portia
        self.query = query
        self.execution_thread: threading.Thread | None = None
        self.plan = None
        self.plan_run = None
        self._stop_requested = False

    def compose(self) -> ComposeResult:
        """Compose the main application layout."""
        yield Header()
        yield Horizontal(
            Container(QueryPanel(self.query), id="query-panel"),
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
        self.sub_title = f"Query: {self.query[:50]}{'...' if len(self.query) > 50 else ''}"

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
        self.execution_thread = threading.Thread(target=self._run_execution)
        self.execution_thread.daemon = True
        self.execution_thread.start()

    @on(Button.Pressed, "#stop-btn")
    def stop_execution(self) -> None:
        """Stop the execution."""
        self._stop_requested = True

        stop_btn = self.query_one("#stop-btn", Button)
        stop_btn.disabled = True

        log_widget = self.query_one("#log-output", RichLog)
        log_widget.write("[red]Execution stop requested by user[/red]")

    @on(Button.Pressed, "#quit-btn")
    def quit_app(self) -> None:
        """Quit the application."""
        self._stop_requested = True
        self.exit()

    def _run_execution(self) -> None:
        """Run the Portia execution in a separate thread."""
        log_widget = None

        try:
            # Get the log widget for capturing output
            log_widget = self.query_one("#log-output", RichLog)

            # Setup log capture
            with LogCapture(log_widget):
                # Capture stdout and stderr as well
                stdout_capture = io.StringIO()
                stderr_capture = io.StringIO()

                with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                    if self._stop_requested:
                        return

                    # First, create the plan
                    self.call_from_thread(log_widget.write, "[green]Creating plan...[/green]")
                    self.plan = self.portia.plan(self.query)

                    if self._stop_requested:
                        return

                    # Update the plan panel
                    plan_panel = self.query_one("#plan-panel PlanPanel", PlanPanel)
                    self.call_from_thread(plan_panel.set_plan, self.plan)

                    self.call_from_thread(
                        log_widget.write,
                        f"[green]Plan created with {len(self.plan.steps)} steps[/green]",
                    )

                    if self._stop_requested:
                        return

                    # Create a plan run
                    self.call_from_thread(log_widget.write, "[green]Starting execution...[/green]")

                    # Execute the plan step by step with updates
                    self._execute_plan_with_updates()

                # Capture any stdout/stderr output
                stdout_content = stdout_capture.getvalue()
                stderr_content = stderr_capture.getvalue()

                if stdout_content:
                    self.call_from_thread(log_widget.write, f"[dim]STDOUT: {stdout_content}[/dim]")
                if stderr_content:
                    self.call_from_thread(log_widget.write, f"[red]STDERR: {stderr_content}[/red]")

        except Exception as e:
            if log_widget:
                self.call_from_thread(log_widget.write, f"[red]Execution failed: {str(e)}[/red]")

        finally:
            # Re-enable start button
            start_btn = self.query_one("#start-btn", Button)
            stop_btn = self.query_one("#stop-btn", Button)
            self.call_from_thread(self._reset_buttons, start_btn, stop_btn)

    def _execute_plan_with_updates(self) -> None:
        """Execute the plan while providing real-time updates."""
        plan_panel = self.query_one("#plan-panel PlanPanel", PlanPanel)
        log_widget = self.query_one("#log-output", RichLog)

        try:
            if self._stop_requested:
                return

            # Run the plan directly and monitor via polling
            # This is simpler than trying to step through manually
            self.plan_run = self.portia.create_plan_run(self.plan)

            # Start execution in the background and monitor progress
            execution_thread = threading.Thread(target=self._execute_plan_run)
            execution_thread.daemon = True
            execution_thread.start()

            # Monitor progress
            last_step_index = -1
            while execution_thread.is_alive() and not self._stop_requested:
                if self.plan_run and self.plan_run.current_step_index != last_step_index:
                    current_step = self.plan_run.current_step_index

                    # Mark previous step as completed
                    if last_step_index >= 0:
                        self.call_from_thread(
                            plan_panel.update_step_status, last_step_index, "completed"
                        )

                    # Mark current step as running
                    if current_step < len(self.plan.steps):
                        self.call_from_thread(
                            plan_panel.update_step_status, current_step, "running"
                        )

                        step = self.plan.steps[current_step]
                        self.call_from_thread(
                            log_widget.write,
                            f"[blue]Executing step {current_step + 1}: {step.task}[/blue]",
                        )

                    last_step_index = current_step

                # Small delay to avoid busy waiting
                threading.Event().wait(0.1)

            # Wait for execution to complete
            execution_thread.join(timeout=1.0)

            # Final status updates
            if self.plan_run:
                if self.plan_run.state == PlanRunState.COMPLETE:
                    # Mark last step as completed
                    if last_step_index >= 0:
                        self.call_from_thread(
                            plan_panel.update_step_status, last_step_index, "completed"
                        )

                    self.call_from_thread(
                        log_widget.write, "[green]Execution completed successfully![/green]"
                    )
                    if self.plan_run.outputs.final_output:
                        self.call_from_thread(
                            log_widget.write,
                            f"[green]Final output: {self.plan_run.outputs.final_output.get_summary()}[/green]",
                        )
                elif self.plan_run.state == PlanRunState.FAILED:
                    # Mark current step as failed
                    if last_step_index >= 0:
                        self.call_from_thread(
                            plan_panel.update_step_status, last_step_index, "failed"
                        )

                    self.call_from_thread(log_widget.write, "[red]Execution failed![/red]")
                elif self.plan_run.state == PlanRunState.NEED_CLARIFICATION:
                    clarifications = self.plan_run.get_outstanding_clarifications()
                    for clarification in clarifications:
                        self.call_from_thread(
                            log_widget.write,
                            f"[yellow]Clarification needed: {clarification.prompt}[/yellow]",
                        )

        except Exception as e:
            self.call_from_thread(log_widget.write, f"[red]Error during execution: {str(e)}[/red]")

    def _execute_plan_run(self) -> None:
        """Execute the plan run (called in separate thread)."""
        try:
            self.plan_run = self.portia.run_plan(self.plan)
        except Exception as e:
            # Log will be captured by the main execution handler
            pass

    def _reset_buttons(self, start_btn: Button, stop_btn: Button) -> None:
        """Reset button states after execution."""
        start_btn.disabled = False
        stop_btn.disabled = True


def run_portia_gui(portia: Portia, query: str) -> None:
    """Run the Portia GUI with the given Portia instance and query.

    Args:
        portia: The Portia instance to use for execution
        query: The query to execute
    """
    app = PortiaGUI(portia=portia, query=query)
    app.run()


if __name__ == "__main__":
    # Example usage with error handling
    import dotenv
    dotenv.load_dotenv()

    try:
        from portia import Config

        portia = Portia(
            Config.from_default(),
        )

        query = "Check my calendar for a free time on Monday then schedule a meeting with sam+test@portialabs.ai for 1 hour in a free slot"
        print("🚀 Starting Portia GUI...")
        print(f"📝 Query: {query}")
        print("🖥️  Press Ctrl+C in the GUI to quit")
        run_portia_gui(portia, query)

    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("Make sure you have installed the Portia SDK and its dependencies.")
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
