"""Tests for core data models."""

import time

from mals.core.models import (
    AgentInvocationRecord,
    BlackboardState,
    ConsensusState,
    ConsensusStatus,
    GlobalStatus,
    Hypothesis,
    HypothesisStatus,
    MemoryTier,
    ReviewRecord,
    StatusChange,
)


class TestBlackboardState:
    def test_default_creation(self) -> None:
        state = BlackboardState()
        assert state.task_id  # auto-generated
        assert state.global_status == GlobalStatus.PLANNING
        assert state.workspace == {}
        assert state.hypothesis_thread == []
        assert state.consensus is None

    def test_with_objective(self) -> None:
        state = BlackboardState(objective="Build a REST API")
        assert state.objective == "Build a REST API"

    def test_touch_updates_timestamp(self) -> None:
        state = BlackboardState()
        old_ts = state.updated_at
        time.sleep(0.01)
        state.touch()
        assert state.updated_at > old_ts

    def test_serialization_roundtrip(self) -> None:
        state = BlackboardState(
            objective="Test task",
            workspace={"code": "print('hello')"},
            active_constraints=["no deps"],
        )
        json_str = state.model_dump_json()
        restored = BlackboardState.model_validate_json(json_str)
        assert restored.objective == "Test task"
        assert restored.workspace["code"] == "print('hello')"
        assert restored.active_constraints == ["no deps"]


class TestHypothesis:
    def test_default_status(self) -> None:
        h = Hypothesis(content="Maybe X is the cause", author_agent="debugger")
        assert h.status == HypothesisStatus.PROPOSED
        assert h.id  # auto-generated
        assert h.evidence == []


class TestConsensusState:
    def test_default(self) -> None:
        cs = ConsensusState(target_field="code")
        assert cs.status == ConsensusStatus.PENDING_REVIEW
        assert cs.current_iteration == 0
        assert cs.max_iterations == 3


class TestMemoryTier:
    def test_default(self) -> None:
        m = MemoryTier()
        assert m.hot == []
        assert m.warm == {}
        assert m.cold_ref == ""
