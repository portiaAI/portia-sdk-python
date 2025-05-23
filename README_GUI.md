# Portia Terminal GUI

A Terminal GUI for visualizing Portia execution using the `textual` library. This provides a real-time, interactive interface for monitoring Portia query execution with three main panels showing the query, plan execution status, and logs.

## Features

### Three-Panel Layout
- **Left Panel (Query)**: Displays the query being executed
- **Middle Panel (Plan)**: Shows the execution plan with real-time step status updates
- **Right Panel (Logs)**: Captures and displays logs, stdout, and stderr output

### Real-time Monitoring
- ✅ Live step execution status with visual indicators
- 📊 Progress tracking with emojis (⏳ pending, 🔄 running, ✅ completed, ❌ failed)
- 📝 Color-coded log output (errors in red, warnings in yellow, info in blue, debug dimmed)
- 🔄 Real-time updates during execution

### User Controls
- **Start Execution**: Begin running the Portia query
- **Stop Execution**: Request cancellation of running execution
- **Quit**: Exit the application

## Installation

The GUI uses the `textual` library which is already included in the Portia SDK dependencies:

```bash
# textual is already in pyproject.toml dependencies
pip install portia-sdk-python
```

## Usage

### Basic Usage

```python
from portia_gui import run_portia_gui
from portia import Config, Portia, example_tool_registry

# Create a Portia instance
portia = Portia(
    Config.from_default(),
    tools=example_tool_registry,
)

# Define your query
query = "Get the temperature in London and Sydney and then add the two temperatures"

# Run the GUI
run_portia_gui(portia, query)
```

### Using the Example Script

```bash
python example_gui.py
```

This will present you with several example queries to choose from:
1. Simple math calculation
2. Weather information query
3. Complex multi-step query
4. Custom query (enter your own)

### Advanced Usage

```python
from portia_gui import PortiaGUI
from portia import Config, LogLevel, Portia, example_tool_registry

# Create Portia with custom configuration
portia = Portia(
    Config.from_default(default_log_level=LogLevel.DEBUG),
    tools=example_tool_registry,
)

# Create and run the GUI app directly
app = PortiaGUI(portia=portia, query="Your custom query here")
app.run()
```

## GUI Components

### QueryPanel
Displays the query text in a formatted, scrollable panel with word wrapping for long queries.

### PlanPanel
Shows the execution plan as a list of steps with status indicators:
- ⏳ **Pending**: Step has not started yet
- 🔄 **Running**: Step is currently executing
- ✅ **Completed**: Step completed successfully
- ❌ **Failed**: Step failed with an error
- ⏭️ **Skipped**: Step was skipped

### LogPanel
Captures and displays:
- **Portia logs**: System logs with appropriate color coding
- **stdout/stderr**: Any output from tool execution
- **Execution status**: Progress messages and final results

### Controls
- **Start Execution**: Begins execution of the query
- **Stop Execution**: Requests cancellation (graceful shutdown)
- **Quit**: Exits the application

## Technical Implementation

### Architecture
- **Main App**: `PortiaGUI` extends `textual.app.App`
- **Panels**: Custom widgets for each section
- **Threading**: Execution runs in background thread to keep UI responsive
- **Log Capture**: Custom logging handler to capture Portia logs
- **Stdout/Stderr**: Context managers capture console output

### Thread Safety
The GUI uses `call_from_thread()` to safely update the UI from background threads, ensuring thread-safe updates to the interface.

### Logging Integration
A custom `LogHandler` captures logs from Portia and routes them to the GUI with appropriate formatting and color coding.

## Customization

### Styling
The GUI uses CSS-like styling. You can customize colors, borders, and layout by modifying the `CSS` property in the `PortiaGUI` class:

```python
class PortiaGUI(App):
    CSS = """
    #query-panel {
        border: solid green;
    }
    
    #plan-panel {
        border: solid blue;
    }
    
    #log-panel {
        border: solid red;
    }
    """
```

### Adding Custom Panels
You can extend the GUI by creating new panel classes that inherit from `textual` widgets:

```python
from textual.widgets import Static
from textual.containers import Vertical

class CustomPanel(Vertical):
    def compose(self) -> ComposeResult:
        yield Static("Custom Panel:", classes="panel-title")
        yield Static("Your custom content here")
```

## Error Handling

The GUI includes comprehensive error handling:
- Execution errors are captured and displayed in the log panel
- UI updates are thread-safe and won't crash on exceptions
- Graceful degradation when components fail

## Keyboard Shortcuts

- **Ctrl+C**: Quit the application
- **Tab**: Navigate between focusable elements
- **Enter**: Activate buttons when focused

## Limitations

- **Stop Execution**: Currently requests stop but doesn't forcefully terminate Portia execution
- **Clarifications**: Basic display of clarification requests (no interactive resolution yet)
- **Step-by-step Control**: Execution runs continuously rather than allowing step-by-step manual control

## Future Enhancements

- Interactive clarification resolution
- Step-by-step execution control
- Export logs to file
- Execution history
- Configuration panel
- Themes and customizable styling
- Performance metrics display

## Examples

See `example_gui.py` for complete working examples with different types of queries and configurations. 