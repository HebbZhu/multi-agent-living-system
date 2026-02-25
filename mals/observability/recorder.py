"""
EventRecorder — Records every significant event during a MALS task for replay.

The recorder captures a time-ordered stream of events that fully describes
what happened during a task execution. This enables:
- Post-hoc debugging: understand exactly what went wrong and when
- Benchmark analysis: compare different runs side-by-side
- Visualization: feed events into the Dashboard for animated replay

Events are stored in-memory during execution and can be exported to JSON
for persistent storage.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger("mals.observability.recorder")


class EventType(str, Enum):
    """Categories of recordable events."""
    TASK_START = "task_start"
    TASK_END = "task_end"
    STATUS_CHANGE = "status_change"
    CONDUCTOR_THINK = "conductor_think"
    CONDUCTOR_DECIDE = "conductor_decide"
    AGENT_START = "agent_start"
    AGENT_END = "agent_end"
    WORKSPACE_WRITE = "workspace_write"
    WORKSPACE_DELETE = "workspace_delete"
    HYPOTHESIS_PROPOSE = "hypothesis_propose"
    HYPOTHESIS_RESOLVE = "hypothesis_resolve"
    CONSENSUS_START = "consensus_start"
    CONSENSUS_REVIEW = "consensus_review"
    CONSENSUS_END = "consensus_end"
    MEMORY_COMPRESS = "memory_compress"
    LLM_CALL = "llm_call"
    ERROR = "error"


@dataclass
class Event:
    """A single recorded event."""
    type: EventType
    timestamp: float = field(default_factory=time.time)
    data: dict[str, Any] = field(default_factory=dict)
    step: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type.value,
            "timestamp": self.timestamp,
            "step": self.step,
            "data": self.data,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Event":
        return cls(
            type=EventType(d["type"]),
            timestamp=d["timestamp"],
            step=d.get("step", 0),
            data=d.get("data", {}),
        )


class EventRecorder:
    """
    Records a time-ordered stream of events during MALS task execution.

    Usage:
        recorder = EventRecorder()
        recorder.record(EventType.TASK_START, data={"objective": "..."})
        ...
        recorder.export_json("task_replay.json")
    """

    def __init__(self) -> None:
        self._events: list[Event] = []
        self._current_step: int = 0
        self._task_id: str = ""
        self._objective: str = ""

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(self, event_type: EventType, data: dict[str, Any] | None = None) -> Event:
        """
        Record a new event.

        Args:
            event_type: The type of event.
            data: Optional data payload for the event.

        Returns:
            The recorded Event object.
        """
        event = Event(
            type=event_type,
            step=self._current_step,
            data=data or {},
        )
        self._events.append(event)
        logger.debug("Event recorded: %s (step %d)", event_type.value, self._current_step)
        return event

    def set_step(self, step: int) -> None:
        """Update the current step counter."""
        self._current_step = step

    def set_task_info(self, task_id: str, objective: str) -> None:
        """Set task metadata for the recording."""
        self._task_id = task_id
        self._objective = objective

    # ------------------------------------------------------------------
    # Convenience recording methods
    # ------------------------------------------------------------------

    def record_task_start(self, task_id: str, objective: str) -> None:
        self.set_task_info(task_id, objective)
        self.record(EventType.TASK_START, {
            "task_id": task_id,
            "objective": objective,
        })

    def record_task_end(self, status: str, summary: dict[str, Any] | None = None) -> None:
        self.record(EventType.TASK_END, {
            "status": status,
            **(summary or {}),
        })

    def record_status_change(self, from_status: str, to_status: str, reason: str) -> None:
        self.record(EventType.STATUS_CHANGE, {
            "from": from_status,
            "to": to_status,
            "reason": reason,
        })

    def record_conductor_think(self, dashboard: str) -> None:
        self.record(EventType.CONDUCTOR_THINK, {
            "dashboard": dashboard,
        })

    def record_conductor_decide(self, action: str, agent_name: str | None = None, reasoning: str = "") -> None:
        self.record(EventType.CONDUCTOR_DECIDE, {
            "action": action,
            "agent_name": agent_name,
            "reasoning": reasoning,
        })

    def record_agent_start(self, agent_name: str, context_fields: list[str] | None = None) -> None:
        self.record(EventType.AGENT_START, {
            "agent_name": agent_name,
            "context_fields": context_fields or [],
        })

    def record_agent_end(
        self,
        agent_name: str,
        status: str,
        latency: float,
        input_tokens: int = 0,
        output_tokens: int = 0,
        error: str | None = None,
    ) -> None:
        self.record(EventType.AGENT_END, {
            "agent_name": agent_name,
            "status": status,
            "latency_s": round(latency, 3),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "error": error,
        })

    def record_workspace_write(self, field: str, value_preview: str) -> None:
        self.record(EventType.WORKSPACE_WRITE, {
            "field": field,
            "preview": value_preview[:200],
        })

    def record_consensus_start(self, target_field: str) -> None:
        self.record(EventType.CONSENSUS_START, {
            "target_field": target_field,
        })

    def record_consensus_review(self, reviewer: str, verdict: str, critique: str) -> None:
        self.record(EventType.CONSENSUS_REVIEW, {
            "reviewer": reviewer,
            "verdict": verdict,
            "critique": critique[:300],
        })

    def record_consensus_end(self, target_field: str, outcome: str, iterations: int) -> None:
        self.record(EventType.CONSENSUS_END, {
            "target_field": target_field,
            "outcome": outcome,
            "iterations": iterations,
        })

    def record_memory_compress(self, field: str, summary: str) -> None:
        self.record(EventType.MEMORY_COMPRESS, {
            "field": field,
            "summary": summary[:200],
        })

    def record_llm_call(self, caller: str, model: str, input_tokens: int, output_tokens: int, latency: float) -> None:
        self.record(EventType.LLM_CALL, {
            "caller": caller,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency_s": round(latency, 3),
        })

    def record_error(self, source: str, error: str) -> None:
        self.record(EventType.ERROR, {
            "source": source,
            "error": error,
        })

    # ------------------------------------------------------------------
    # Export / Import
    # ------------------------------------------------------------------

    @property
    def events(self) -> list[Event]:
        """Return all recorded events."""
        return list(self._events)

    @property
    def event_count(self) -> int:
        return len(self._events)

    def to_dict(self) -> dict[str, Any]:
        """Export the full recording as a dictionary."""
        return {
            "task_id": self._task_id,
            "objective": self._objective,
            "event_count": len(self._events),
            "events": [e.to_dict() for e in self._events],
        }

    def export_json(self, path: str | Path) -> None:
        """Export the recording to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info("Recording exported to %s (%d events)", path, len(self._events))

    @classmethod
    def load_json(cls, path: str | Path) -> "EventRecorder":
        """Load a recording from a JSON file."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        recorder = cls()
        recorder._task_id = data.get("task_id", "")
        recorder._objective = data.get("objective", "")
        recorder._events = [Event.from_dict(e) for e in data.get("events", [])]
        if recorder._events:
            recorder._current_step = max(e.step for e in recorder._events)
        logger.info("Recording loaded from %s (%d events)", path, len(recorder._events))
        return recorder

    # ------------------------------------------------------------------
    # Replay helpers
    # ------------------------------------------------------------------

    def events_by_type(self, event_type: EventType) -> list[Event]:
        """Filter events by type."""
        return [e for e in self._events if e.type == event_type]

    def events_in_step(self, step: int) -> list[Event]:
        """Get all events that occurred during a specific step."""
        return [e for e in self._events if e.step == step]

    def timeline(self) -> list[dict[str, Any]]:
        """
        Generate a simplified timeline for visualization.

        Returns a list of dicts with: step, timestamp, type, summary.
        """
        timeline_items = []
        for event in self._events:
            summary = _event_summary(event)
            timeline_items.append({
                "step": event.step,
                "timestamp": event.timestamp,
                "type": event.type.value,
                "summary": summary,
            })
        return timeline_items


def _event_summary(event: Event) -> str:
    """Generate a one-line human-readable summary of an event."""
    d = event.data
    match event.type:
        case EventType.TASK_START:
            return f"Task started: {d.get('objective', '')[:60]}"
        case EventType.TASK_END:
            return f"Task ended: {d.get('status', 'unknown')}"
        case EventType.STATUS_CHANGE:
            return f"Status: {d.get('from', '?')} → {d.get('to', '?')}"
        case EventType.CONDUCTOR_DECIDE:
            agent = d.get('agent_name', '')
            return f"Conductor → {d.get('action', '?')}" + (f" ({agent})" if agent else "")
        case EventType.AGENT_START:
            return f"Agent started: {d.get('agent_name', '?')}"
        case EventType.AGENT_END:
            return f"Agent finished: {d.get('agent_name', '?')} ({d.get('status', '?')}, {d.get('latency_s', 0):.1f}s)"
        case EventType.WORKSPACE_WRITE:
            return f"Workspace write: {d.get('field', '?')}"
        case EventType.CONSENSUS_REVIEW:
            return f"Review by {d.get('reviewer', '?')}: {d.get('verdict', '?')}"
        case EventType.CONSENSUS_END:
            return f"Consensus {d.get('outcome', '?')} for {d.get('target_field', '?')}"
        case EventType.MEMORY_COMPRESS:
            return f"Memory compressed: {d.get('field', '?')}"
        case EventType.ERROR:
            return f"Error in {d.get('source', '?')}: {d.get('error', '')[:60]}"
        case _:
            return event.type.value
