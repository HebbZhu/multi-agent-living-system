"""Tests for the Agent Registry."""

import pytest

from mals.agents.registry import AgentRegistry, AgentSpec, specialist
from mals.core.blackboard import Blackboard


class TestAgentRegistry:
    def setup_method(self) -> None:
        self.registry = AgentRegistry()

    def test_register_and_get(self) -> None:
        agent = AgentSpec(name="test_agent", description="A test agent")
        self.registry.register(agent)
        assert self.registry.get("test_agent") is agent

    def test_register_duplicate(self) -> None:
        agent = AgentSpec(name="dup", description="First")
        self.registry.register(agent)
        with pytest.raises(ValueError, match="already registered"):
            self.registry.register(AgentSpec(name="dup", description="Second"))

    def test_get_nonexistent(self) -> None:
        assert self.registry.get("nonexistent") is None

    def test_list_agents(self) -> None:
        self.registry.register(AgentSpec(name="a", description="Agent A"))
        self.registry.register(AgentSpec(name="b", description="Agent B"))
        agents = self.registry.list_agents()
        assert len(agents) == 2

    def test_describe_all(self) -> None:
        self.registry.register(AgentSpec(
            name="coder",
            description="Writes code",
            input_fields=["plan"],
            output_fields=["code"],
        ))
        desc = self.registry.describe_all()
        assert "coder" in desc
        assert "Writes code" in desc
        assert "plan" in desc
        assert "code" in desc

    def test_describe_all_empty(self) -> None:
        desc = self.registry.describe_all()
        assert "No agents" in desc

    def test_len_and_contains(self) -> None:
        self.registry.register(AgentSpec(name="x", description="X"))
        assert len(self.registry) == 1
        assert "x" in self.registry
        assert "y" not in self.registry


class TestSpecialistDecorator:
    def test_decorator_creates_agent_spec(self) -> None:
        @specialist(
            name="my_agent",
            description="Does things",
            input_fields=["input"],
            output_fields=["output"],
        )
        async def my_agent(context: dict, board: Blackboard) -> dict:
            return {"status": "ok"}

        assert isinstance(my_agent, AgentSpec)
        assert my_agent.name == "my_agent"
        assert my_agent.description == "Does things"
        assert my_agent.execute is not None
