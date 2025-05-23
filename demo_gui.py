#!/usr/bin/env python3
"""Demo version of Portia Terminal GUI.

This script demonstrates the GUI interface without requiring API keys
by simulating plan execution with mock data.
"""

import time
import threading
from typing import Any

from textual import on
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.reactive import var
from textual.widgets import Button, Footer, Header, RichLog, Static


class MockStep:
    """Mock step for demo purposes."""
    def __init__(self, task: str):
        self.task = task


class MockPlan:
    """Mock plan for demo purposes."""
    def __init__(self, query: str):
        self.query = query
        self.steps = [
            MockStep("Analyze the query and identify required operations"),
            MockStep("Break down the task into smaller components"),
            MockStep("Execute the first calculation step"),
            MockStep("Process intermediate results"),
            MockStep("Execute the second calculation step"),
            MockStep("Combine results and format output"),
            MockStep("Validate the final answer"),
        ]


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
            "skipped": "⏭️"
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


class LogPanel(Vertical):
    """Panel to display logs."""
    
    def compose(self) -> ComposeResult:
        """Compose the log panel."""
        yield Static("Logs:", classes="panel-title")
        yield RichLog(id="log-output", auto_scroll=True)


class PortiaDemoGUI(App):
    """Demo Portia GUI application."""
    
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
    
    def __init__(self, query: str = "", **kwargs) -> None:
        """Initialize the demo GUI.
        
        Args:
            query: The query to execute
            **kwargs: Additional arguments passed to App
        """
        super().__init__(**kwargs)
        self.query = query
        self.execution_thread: threading.Thread | None = None
        self.plan = None
        self._stop_requested = False
    
    def compose(self) -> ComposeResult:
        """Compose the main application layout."""
        yield Header()
        yield Horizontal(
            Container(
                QueryPanel(self.query),
                id="query-panel"
            ),
            Container(
                PlanPanel(),
                id="plan-panel"
            ),
            Container(
                LogPanel(),
                id="log-panel"
            ),
        )
        yield Container(
            Button("Start Demo", id="start-btn", variant="success"),
            Button("Stop Demo", id="stop-btn", variant="error", disabled=True),
            Button("Quit", id="quit-btn", variant="warning"),
            id="controls"
        )
        yield Footer()
    
    def on_mount(self) -> None:
        """Called when the app is mounted."""
        self.title = "Portia Terminal GUI - Demo Mode"
        self.sub_title = f"Query: {self.query[:50]}{'...' if len(self.query) > 50 else ''}"
        
        # Add demo notice
        log_widget = self.query_one("#log-output", RichLog)
        log_widget.write("[yellow]🎭 DEMO MODE - This is a simulation of Portia execution[/yellow]")
        log_widget.write("[dim]No API keys required for this demo[/dim]")
        log_widget.write("[green]Click 'Start Demo' to see simulated execution[/green]")
    
    @on(Button.Pressed, "#start-btn")
    def start_execution(self) -> None:
        """Start the demo execution."""
        if self.execution_thread and self.execution_thread.is_alive():
            return
        
        self._stop_requested = False
        
        # Update button states
        start_btn = self.query_one("#start-btn", Button)
        stop_btn = self.query_one("#stop-btn", Button)
        start_btn.disabled = True
        stop_btn.disabled = False
        
        # Clear previous logs except demo notice
        log_widget = self.query_one("#log-output", RichLog)
        log_widget.clear()
        log_widget.write("[yellow]🎭 DEMO MODE - This is a simulation of Portia execution[/yellow]")
        log_widget.write("[dim]No API keys required for this demo[/dim]")
        
        # Start demo execution in a separate thread
        self.execution_thread = threading.Thread(target=self._run_demo_execution)
        self.execution_thread.daemon = True
        self.execution_thread.start()
    
    @on(Button.Pressed, "#stop-btn")
    def stop_execution(self) -> None:
        """Stop the demo execution."""
        self._stop_requested = True
        
        stop_btn = self.query_one("#stop-btn", Button)
        stop_btn.disabled = True
        
        log_widget = self.query_one("#log-output", RichLog)
        log_widget.write("[red]Demo execution stopped by user[/red]")
    
    @on(Button.Pressed, "#quit-btn")
    def quit_app(self) -> None:
        """Quit the application."""
        self._stop_requested = True
        self.exit()
    
    def _run_demo_execution(self) -> None:
        """Run the demo execution simulation."""
        log_widget = self.query_one("#log-output", RichLog)
        plan_panel = self.query_one("#plan-panel PlanPanel", PlanPanel)
        
        try:
            if self._stop_requested:
                return
            
            # Create mock plan
            self.call_from_thread(log_widget.write, "[green]Creating plan...[/green]")
            time.sleep(1)
            
            self.plan = MockPlan(self.query)
            self.call_from_thread(plan_panel.set_plan, self.plan)
            
            self.call_from_thread(log_widget.write, f"[green]Plan created with {len(self.plan.steps)} steps[/green]")
            time.sleep(0.5)
            
            if self._stop_requested:
                return
            
            self.call_from_thread(log_widget.write, "[green]Starting execution...[/green]")
            time.sleep(0.5)
            
            # Simulate step execution
            for i, step in enumerate(self.plan.steps):
                if self._stop_requested:
                    return
                
                # Mark step as running
                self.call_from_thread(plan_panel.update_step_status, i, "running")
                self.call_from_thread(
                    log_widget.write, 
                    f"[blue]Executing step {i + 1}: {step.task}[/blue]"
                )
                
                # Simulate some processing time
                time.sleep(1.5 + (i * 0.3))  # Gradually slower steps
                
                if self._stop_requested:
                    return
                
                # Mark step as completed
                self.call_from_thread(plan_panel.update_step_status, i, "completed")
                self.call_from_thread(
                    log_widget.write, 
                    f"[green]Step {i + 1} completed successfully[/green]"
                )
                
                # Add some sample log output
                if i == 2:
                    self.call_from_thread(log_widget.write, "[dim]Tool execution output: calculation_result = 42[/dim]")
                elif i == 4:
                    self.call_from_thread(log_widget.write, "[dim]Tool execution output: final_calculation = 84[/dim]")
                
                time.sleep(0.3)
            
            # Final completion
            if not self._stop_requested:
                self.call_from_thread(log_widget.write, "[green]🎉 Demo execution completed successfully![/green]")
                self.call_from_thread(log_widget.write, "[green]Final output: The result is 84 (this is a simulated result)[/green]")
                self.call_from_thread(log_widget.write, "[yellow]In real usage, this would show actual Portia execution results[/yellow]")
        
        except Exception as e:
            self.call_from_thread(log_widget.write, f"[red]Demo error: {str(e)}[/red]")
        
        finally:
            # Re-enable start button
            start_btn = self.query_one("#start-btn", Button)
            stop_btn = self.query_one("#stop-btn", Button)
            self.call_from_thread(self._reset_buttons, start_btn, stop_btn)
    
    def _reset_buttons(self, start_btn: Button, stop_btn: Button) -> None:
        """Reset button states after execution."""
        start_btn.disabled = False
        stop_btn.disabled = True


def run_demo_gui(query: str) -> None:
    """Run the demo Portia GUI.
    
    Args:
        query: The query to simulate
    """
    app = PortiaDemoGUI(query=query)
    app.run()


if __name__ == "__main__":
    query = "Calculate 42 + 58, then multiply by 3, and tell me if the result is greater than 200"
    
    print("🎭 Portia Terminal GUI - Demo Mode")
    print("=" * 50)
    print("This demo shows the GUI interface without requiring API keys.")
    print(f"📝 Simulating query: {query}")
    print("🖥️  Press Ctrl+C in the GUI to quit")
    print()
    
    run_demo_gui(query) 