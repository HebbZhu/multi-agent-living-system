"""Tests for the MALS observability layer: MetricsCollector, EventRecorder, Dashboard."""

import json
import tempfile
from pathlib import Path

import pytest

from mals.observability.metrics import MetricsCollector
from mals.observability.recorder import EventRecorder, EventType, Event


# ============================================================================
# MetricsCollector Tests
# ============================================================================

class TestMetricsCollector:
    """Tests for MetricsCollector."""

    def test_init(self):
        mc = MetricsCollector()
        assert mc._task_start > 0
        assert mc._agent_metrics == {}
        assert mc._conductor.total_steps == 0

    def test_record_conductor_step(self):
        mc = MetricsCollector()
        mc.record_conductor_step(
            action="invoke_agent",
            agent_name="planner",
            latency=0.5,
            input_tokens=100,
            output_tokens=50,
        )
        assert mc._conductor.total_steps == 1
        assert mc._conductor.decision_counts["invoke_agent"] == 1
        assert mc._conductor.routing_counts["planner"] == 1
        assert mc._conductor.total_input_tokens == 100
        assert mc._conductor.total_output_tokens == 50

    def test_record_agent_invocation_success(self):
        mc = MetricsCollector()
        mc.record_agent_invocation(
            agent_name="code_generator",
            latency=2.5,
            input_tokens=500,
            output_tokens=300,
            success=True,
        )
        assert "code_generator" in mc._agent_metrics
        m = mc._agent_metrics["code_generator"]
        assert m.invocation_count == 1
        assert m.success_count == 1
        assert m.latencies == [2.5]

    def test_record_agent_invocation_failure(self):
        mc = MetricsCollector()
        mc.record_agent_invocation(
            agent_name="code_generator",
            latency=1.0,
            success=False,
            error="LLM timeout",
        )
        m = mc._agent_metrics["code_generator"]
        assert m.error_count == 1
        assert m.success_count == 0

    def test_record_consensus_cycle(self):
        mc = MetricsCollector()
        mc.record_consensus_cycle(iterations=2, outcome="approved_after_revision")
        assert mc._consensus.total_cycles == 1
        assert mc._consensus.total_iterations == 2
        assert mc._consensus.approved_after_revision == 1

    def test_record_memory_compression(self):
        mc = MetricsCollector()
        mc.record_memory_compression()
        mc.record_memory_compression()
        assert mc._memory_compressions == 2

    def test_record_status_transition(self):
        mc = MetricsCollector()
        mc.record_status_transition("PLANNING", "EXECUTING", "First agent invoked")
        assert len(mc._status_transitions) == 1
        assert mc._status_transitions[0]["from"] == "PLANNING"
        assert mc._status_transitions[0]["to"] == "EXECUTING"

    def test_to_dict_structure(self):
        mc = MetricsCollector()
        mc.record_conductor_step("invoke_agent", "planner", 0.5, 100, 50)
        mc.record_agent_invocation("planner", 1.0, 200, 100, True)
        mc.record_consensus_cycle(1, "approved_first_try")
        mc.mark_task_complete()

        data = mc.to_dict()
        assert "task_summary" in data
        assert "agents" in data
        assert "conductor" in data
        assert "consensus" in data
        assert "status_history" in data

        ts = data["task_summary"]
        assert ts["total_steps"] == 1
        assert ts["total_agent_invocations"] == 1
        assert ts["total_tokens"] > 0

    def test_agent_stats(self):
        mc = MetricsCollector()
        mc.record_agent_invocation("planner", 1.0, 200, 100, True)
        mc.record_agent_invocation("planner", 2.0, 300, 150, True)
        mc.record_agent_invocation("planner", 1.5, 250, 120, False, error="timeout")
        mc.mark_task_complete()

        data = mc.to_dict()
        planner = data["agents"]["planner"]
        assert planner["invocation_count"] == 3
        assert planner["success_count"] == 2
        assert abs(planner["success_rate"] - 2 / 3) < 0.01
        assert planner["total_input_tokens"] == 750
        assert planner["total_output_tokens"] == 370
        assert abs(planner["avg_latency_s"] - 1.5) < 0.01

    def test_consensus_stats(self):
        mc = MetricsCollector()
        mc.record_consensus_cycle(1, "approved_first_try")
        mc.record_consensus_cycle(3, "approved_after_revision")
        mc.record_consensus_cycle(1, "approved_first_try")
        mc.mark_task_complete()

        data = mc.to_dict()
        cons = data["consensus"]
        assert cons["total_cycles"] == 3
        assert abs(cons["first_try_approval_rate"] - 2 / 3) < 0.01

    def test_summary_text(self):
        mc = MetricsCollector()
        mc.record_conductor_step("invoke_agent", "planner", 0.5, 100, 50)
        mc.record_agent_invocation("planner", 1.0, 200, 100, True)
        mc.mark_task_complete()

        text = mc.summary_text()
        assert "MALS Task Metrics" in text
        assert "planner" in text


# ============================================================================
# EventRecorder Tests
# ============================================================================

class TestEventRecorder:
    """Tests for EventRecorder."""

    def test_init(self):
        rec = EventRecorder()
        assert rec.event_count == 0
        assert rec.events == []

    def test_record_basic(self):
        rec = EventRecorder()
        event = rec.record(EventType.TASK_START, {"objective": "test"})
        assert event.type == EventType.TASK_START
        assert event.data["objective"] == "test"
        assert rec.event_count == 1

    def test_set_step(self):
        rec = EventRecorder()
        rec.set_step(5)
        event = rec.record(EventType.AGENT_START, {"agent_name": "planner"})
        assert event.step == 5

    def test_record_task_start(self):
        rec = EventRecorder()
        rec.record_task_start("task-123", "Build something")
        assert rec.event_count == 1
        assert rec._task_id == "task-123"
        assert rec._objective == "Build something"

    def test_record_task_end(self):
        rec = EventRecorder()
        rec.record_task_end("COMPLETED", {"steps": 10})
        assert rec.event_count == 1
        event = rec.events[0]
        assert event.data["status"] == "COMPLETED"
        assert event.data["steps"] == 10

    def test_record_agent_lifecycle(self):
        rec = EventRecorder()
        rec.record_agent_start("code_generator", ["code", "requirements"])
        rec.record_agent_end("code_generator", "completed", 2.5, 500, 300)
        assert rec.event_count == 2

    def test_record_consensus(self):
        rec = EventRecorder()
        rec.record_consensus_start("code")
        rec.record_consensus_review("critic", "REJECTED", "Missing error handling")
        rec.record_consensus_end("code", "approved_after_revision", 2)
        assert rec.event_count == 3

    def test_record_error(self):
        rec = EventRecorder()
        rec.record_error("planner", "LLM timeout")
        assert rec.events[0].data["error"] == "LLM timeout"

    def test_export_and_load_json(self):
        rec = EventRecorder()
        rec.record_task_start("task-456", "Test export")
        rec.set_step(1)
        rec.record_agent_start("planner", ["objective"])
        rec.record_agent_end("planner", "completed", 1.0, 100, 50)
        rec.record_task_end("COMPLETED")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_recording.json"
            rec.export_json(path)

            # Verify file exists and is valid JSON
            assert path.exists()
            with open(path) as f:
                data = json.load(f)
            assert data["task_id"] == "task-456"
            assert data["event_count"] == 4
            assert len(data["events"]) == 4

            # Load back
            loaded = EventRecorder.load_json(path)
            assert loaded.event_count == 4
            assert loaded._task_id == "task-456"

    def test_events_by_type(self):
        rec = EventRecorder()
        rec.record_agent_start("a", [])
        rec.record_agent_start("b", [])
        rec.record_agent_end("a", "completed", 1.0)
        rec.record_error("x", "err")

        agent_starts = rec.events_by_type(EventType.AGENT_START)
        assert len(agent_starts) == 2

        errors = rec.events_by_type(EventType.ERROR)
        assert len(errors) == 1

    def test_events_in_step(self):
        rec = EventRecorder()
        rec.set_step(1)
        rec.record_agent_start("a", [])
        rec.record_agent_end("a", "completed", 1.0)
        rec.set_step(2)
        rec.record_agent_start("b", [])

        step1 = rec.events_in_step(1)
        assert len(step1) == 2

        step2 = rec.events_in_step(2)
        assert len(step2) == 1

    def test_timeline(self):
        rec = EventRecorder()
        rec.record_task_start("t1", "Test")
        rec.set_step(1)
        rec.record_agent_start("planner", [])
        rec.record_agent_end("planner", "completed", 1.5, 100, 50)
        rec.record_task_end("COMPLETED")

        tl = rec.timeline()
        assert len(tl) == 4
        assert tl[0]["type"] == "task_start"
        assert "summary" in tl[0]

    def test_to_dict(self):
        rec = EventRecorder()
        rec.record_task_start("t1", "Test objective")
        rec.record_task_end("COMPLETED")

        data = rec.to_dict()
        assert data["task_id"] == "t1"
        assert data["objective"] == "Test objective"
        assert data["event_count"] == 2
        assert len(data["events"]) == 2


class TestEvent:
    """Tests for Event dataclass."""

    def test_to_dict(self):
        event = Event(type=EventType.TASK_START, step=0, data={"objective": "test"})
        d = event.to_dict()
        assert d["type"] == "task_start"
        assert d["step"] == 0
        assert d["data"]["objective"] == "test"

    def test_from_dict(self):
        d = {"type": "agent_start", "timestamp": 1234567890.0, "step": 3, "data": {"agent_name": "planner"}}
        event = Event.from_dict(d)
        assert event.type == EventType.AGENT_START
        assert event.step == 3
        assert event.data["agent_name"] == "planner"


# ============================================================================
# Dashboard Tests
# ============================================================================

class TestDashboard:
    """Tests for the Dashboard app factory."""

    def test_create_dashboard_app(self):
        from mals.observability.dashboard import create_dashboard_app
        app = create_dashboard_app(
            metrics_data={"task_summary": {"total_steps": 5}},
            recording_data={"task_id": "t1", "events": []},
        )
        assert app is not None
        # Verify routes exist
        routes = [r.path for r in app.routes]
        assert "/" in routes
        assert "/api/metrics" in routes
        assert "/api/recording" in routes
        assert "/api/timeline" in routes

    def test_create_dashboard_from_files(self):
        from mals.observability.dashboard import create_dashboard_app

        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = Path(tmpdir) / "metrics.json"
            recording_path = Path(tmpdir) / "recording.json"

            with open(metrics_path, "w") as f:
                json.dump({"task_summary": {}}, f)
            with open(recording_path, "w") as f:
                json.dump({"task_id": "t1", "events": [
                    {"type": "task_start", "timestamp": 0, "step": 0, "data": {}}
                ]}, f)

            app = create_dashboard_app(
                metrics_file=str(metrics_path),
                recording_file=str(recording_path),
            )
            assert app is not None
