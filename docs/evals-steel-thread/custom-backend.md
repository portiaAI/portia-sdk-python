* [Evals and SteelThread](/evals-steel-thread)
* Custom backends

# Custom backends

SteelThread is designed to allow for metrics to be pushed to other sinks, simply by implementing the correct metrics backend and passing it as config.

```
from abc import ABC, abstractmethod  
from steelthread.streams import StreamMetric, StreamConfig  
from portia import Config  
  
class StreamMetricsBackend(ABC):  
    """Abstract interface for saving metrics."""  
  
    @abstractmethod  
    def save_metrics(self, metrics: list[StreamMetric]) -> None:  
        """Save a list of tagged metrics for a specific evaluation run.  
  
        Args:  
            metrics (list[StreamMetricWithTags]): The metrics to save.  
  
        """  
        raise NotImplementedError  
  
  
class MyMetricsBackend(StreamMetricsBackend):  
    def save_metrics(self, metrics: list[StreamMetric]) -> None:  
        return      
  
config = Config.from_default()  
conf = StreamConfig(stream_name="stream_v1", config=config, metrics_backends=[MyMetricsBackend()])
```

Last updated on **Sep 9, 2025** by **robbie-portia**