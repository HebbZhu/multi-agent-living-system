"""Tests for the Memory Manager."""

from mals.core.models import BlackboardState, GlobalStatus
from mals.memory.manager import MemoryManager


class TestMemoryManager:
    def setup_method(self) -> None:
        self.mm = MemoryManager(llm_client=None)  # No LLM for unit tests
        self.state = BlackboardState(
            objective="Test memory management",
            global_status=GlobalStatus.EXECUTING,
        )

    def test_generate_dashboard_basic(self) -> None:
        self.state.workspace["code"] = "print('hello')"
        dashboard = self.mm.generate_dashboard(self.state)
        assert "Test memory management" in dashboard
        assert "EXECUTING" in dashboard
        assert "code" in dashboard

    def test_generate_dashboard_with_warm(self) -> None:
        self.state.workspace["old_code"] = "..."
        self.state.memory.warm["old_code"] = "Previously generated code that passed all tests"
        dashboard = self.mm.generate_dashboard(self.state)
        assert "completed" in dashboard
        assert "Previously generated" in dashboard

    def test_generate_dashboard_truncates_long_values(self) -> None:
        self.state.workspace["big"] = "x" * 500
        dashboard = self.mm.generate_dashboard(self.state)
        assert "500 chars" in dashboard

    def test_slice_context_basic(self) -> None:
        self.state.workspace["code"] = "print('hello')"
        self.state.workspace["tests"] = "def test_foo(): pass"
        self.state.workspace["irrelevant"] = "should not appear"

        ctx = self.mm.slice_context(self.state, relevant_fields=["code"])
        assert "code" in ctx["workspace"]
        assert "irrelevant" not in ctx["workspace"]
        assert ctx["objective"] == "Test memory management"

    def test_slice_context_includes_warm_summary(self) -> None:
        self.state.memory.warm["old_code"] = "Summary of old code"
        ctx = self.mm.slice_context(self.state, relevant_fields=["old_code"])
        assert "old_code_summary" in ctx["workspace"]

    def test_slice_context_with_hypotheses(self) -> None:
        from mals.core.models import Hypothesis
        self.state.hypothesis_thread.append(
            Hypothesis(content="Maybe X", author_agent="debugger")
        )
        ctx = self.mm.slice_context(self.state, relevant_fields=[], include_hypotheses=True)
        assert "hypotheses" in ctx
        assert len(ctx["hypotheses"]) == 1

    def test_mark_hot(self) -> None:
        self.mm.mark_hot(self.state, "new_field")
        assert "new_field" in self.state.memory.hot

    def test_mark_hot_idempotent(self) -> None:
        self.mm.mark_hot(self.state, "field")
        self.mm.mark_hot(self.state, "field")
        assert self.state.memory.hot.count("field") == 1

    def test_get_hot_fields(self) -> None:
        self.state.memory.hot = ["a", "b", "c"]
        assert self.mm.get_hot_fields(self.state) == ["a", "b", "c"]
