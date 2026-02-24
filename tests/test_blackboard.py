"""Tests for the Dynamic Blackboard."""

import pytest

from mals.core.blackboard import Blackboard, InMemoryBackend
from mals.core.models import ConsensusStatus, GlobalStatus, HypothesisStatus


class TestBlackboard:
    def setup_method(self) -> None:
        self.board = Blackboard(backend=InMemoryBackend())
        self.board.initialize("Test objective")

    def test_initialize(self) -> None:
        assert self.board.state.objective == "Test objective"
        assert self.board.state.global_status == GlobalStatus.PLANNING

    def test_set_status(self) -> None:
        self.board.set_status(GlobalStatus.EXECUTING, reason="Starting work")
        assert self.board.state.global_status == GlobalStatus.EXECUTING
        assert len(self.board.state.status_history) == 1
        assert self.board.state.status_history[0].to_status == "EXECUTING"

    def test_workspace_read_write(self) -> None:
        self.board.write_workspace("code", "print('hello')")
        assert self.board.read_workspace("code") == "print('hello')"
        assert "code" in self.board.state.memory.hot

    def test_workspace_delete(self) -> None:
        self.board.write_workspace("temp", "data")
        self.board.delete_workspace("temp")
        assert self.board.read_workspace("temp") is None

    def test_workspace_read_nonexistent(self) -> None:
        assert self.board.read_workspace("nonexistent") is None

    def test_hypothesis_lifecycle(self) -> None:
        h = self.board.propose_hypothesis("X might be the cause", "debugger")
        assert h.status == HypothesisStatus.PROPOSED

        self.board.resolve_hypothesis(h.id, HypothesisStatus.VALIDATED, "Confirmed by test")
        found = [x for x in self.board.state.hypothesis_thread if x.id == h.id][0]
        assert found.status == HypothesisStatus.VALIDATED
        assert "Confirmed by test" in found.evidence

    def test_hypothesis_not_found(self) -> None:
        with pytest.raises(ValueError, match="not found"):
            self.board.resolve_hypothesis("nonexistent", HypothesisStatus.REJECTED)

    def test_consensus_lifecycle(self) -> None:
        self.board.write_workspace("code", "def foo(): pass")
        cs = self.board.start_consensus("code")
        assert cs.target_field == "code"

        # Reject first
        self.board.submit_review("critic", ConsensusStatus.REJECTED, "Missing docstring")
        assert self.board.state.consensus.status == ConsensusStatus.REJECTED
        assert self.board.state.consensus.current_iteration == 1

        # Approve second
        self.board.submit_review("critic", ConsensusStatus.APPROVED, "Looks good now")
        assert self.board.state.consensus.status == ConsensusStatus.APPROVED

    def test_consensus_max_iterations(self) -> None:
        self.board.write_workspace("code", "bad code")
        self.board.start_consensus("code", max_iterations=2)

        self.board.submit_review("critic", ConsensusStatus.REJECTED, "Bad")
        self.board.submit_review("critic", ConsensusStatus.REJECTED, "Still bad")
        # Should be force-approved after max iterations
        assert self.board.state.consensus.status == ConsensusStatus.APPROVED

    def test_no_consensus_active(self) -> None:
        with pytest.raises(RuntimeError, match="No active consensus"):
            self.board.submit_review("critic", ConsensusStatus.APPROVED, "ok")

    def test_invocation_logging(self) -> None:
        record = self.board.log_invocation_start("planner")
        assert record.agent_name == "planner"
        assert record.status == "running"

        self.board.log_invocation_end(record, status="completed", input_tokens=100, output_tokens=50)
        assert record.status == "completed"
        assert record.input_tokens == 100
        assert record.finished_at is not None

    def test_persistence_roundtrip(self) -> None:
        backend = InMemoryBackend()
        board1 = Blackboard(backend=backend)
        state = board1.initialize("Persist test")
        board1.write_workspace("data", "important")

        # Resume from same backend
        board2 = Blackboard(backend=backend)
        board2.resume(state.task_id)
        assert board2.state.objective == "Persist test"
        assert board2.read_workspace("data") == "important"

    def test_resume_not_found(self) -> None:
        board = Blackboard(backend=InMemoryBackend())
        with pytest.raises(ValueError, match="not found"):
            board.resume("nonexistent_id")

    def test_not_initialized_error(self) -> None:
        board = Blackboard()
        with pytest.raises(RuntimeError, match="not initialized"):
            _ = board.state
