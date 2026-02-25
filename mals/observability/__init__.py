"""
MALS Observability — Metrics collection, event recording, and task replay.

This package provides the instrumentation layer for MALS, enabling:
- Real-time metrics collection (token usage, latency, step counts)
- Event recording for full task replay
- Web dashboard for visualization
"""

from mals.observability.metrics import MetricsCollector
from mals.observability.recorder import EventRecorder, EventType

__all__ = ["MetricsCollector", "EventRecorder", "EventType"]
