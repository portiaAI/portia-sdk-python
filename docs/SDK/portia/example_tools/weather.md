---
sidebar_label: weather
title: portia.example_tools.weather
---

Tool to get the weather from openweathermap.

## WeatherToolSchema Objects

```python
class WeatherToolSchema(BaseModel)
```

Input for WeatherTool.

## WeatherTool Objects

```python
class WeatherTool(Tool[str])
```

Get the weather for a given city.

#### run

```python
def run(city: str) -> str
```

Run the WeatherTool.

