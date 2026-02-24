"""
Agent Registry — Registration, discovery, and scheduling of specialist agents.

Provides a decorator-based API for defining specialist agents, and a registry
for the Conductor to discover and invoke them.

Usage:
    from mals.agents.registry import specialist, AgentRegistry

    @specialist(
        name="code_generator",
        description="Generates code based on requirements.",
        output_fields=["code"],
    )
    async def code_generator(context: dict, board: Blackboard) -> dict:
        # ... agent logic ...
        return {"output": "generated code"}

    registry = AgentRegistry()
    registry.register(code_generator)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine

from mals.core.blackboard import Blackboard

logger = logging.getLogger("mals.agents")

# Type alias for agent execute functions
AgentExecuteFn = Callable[[dict[str, Any], Blackboard], Coroutine[Any, Any, dict[str, Any]]]


@dataclass
class AgentSpec:
    """Specification for a specialist agent."""
    name: str
    description: str
    input_fields: list[str] = field(default_factory=list)
    output_fields: list[str] = field(default_factory=list)
    execute: AgentExecuteFn | None = None

    async def __call__(self, context: dict[str, Any], board: Blackboard) -> dict[str, Any]:
        """Invoke the agent's execute function."""
        if self.execute is None:
            raise RuntimeError(f"Agent '{self.name}' has no execute function.")
        return await self.execute(context, board)


def specialist(
    name: str,
    description: str,
    input_fields: list[str] | None = None,
    output_fields: list[str] | None = None,
) -> Callable[[AgentExecuteFn], AgentSpec]:
    """
    Decorator to define a specialist agent.

    Args:
        name: Unique name for the agent (used by the Conductor to reference it).
        description: Human-readable description of what the agent does.
            This is shown to the Conductor LLM to help it make routing decisions.
        input_fields: Workspace fields this agent typically needs to read.
        output_fields: Workspace fields this agent typically writes to.

    Returns:
        A decorator that wraps a function into an AgentSpec.

    Example:
        @specialist(
            name="code_generator",
            description="Generates Python code based on requirements and specifications.",
            input_fields=["requirements"],
            output_fields=["code"],
        )
        async def code_generator(context: dict, board: Blackboard) -> dict:
            # Your agent logic here
            board.write_workspace("code", generated_code)
            return {"status": "ok"}
    """
    def decorator(fn: AgentExecuteFn) -> AgentSpec:
        spec = AgentSpec(
            name=name,
            description=description,
            input_fields=input_fields or [],
            output_fields=output_fields or [],
            execute=fn,
        )
        return spec
    return decorator


class AgentRegistry:
    """
    Registry for specialist agents.

    The Conductor uses this registry to discover available agents and their
    capabilities, and to invoke them during the collaboration loop.
    """

    def __init__(self) -> None:
        self._agents: dict[str, AgentSpec] = {}

    def register(self, agent: AgentSpec) -> None:
        """
        Register a specialist agent.

        Args:
            agent: The AgentSpec to register.

        Raises:
            ValueError: If an agent with the same name is already registered.
        """
        if agent.name in self._agents:
            raise ValueError(f"Agent '{agent.name}' is already registered.")
        self._agents[agent.name] = agent
        logger.info("Registered agent: %s — %s", agent.name, agent.description)

    def get(self, name: str) -> AgentSpec | None:
        """Look up an agent by name. Returns None if not found."""
        return self._agents.get(name)

    def list_agents(self) -> list[AgentSpec]:
        """Return all registered agents."""
        return list(self._agents.values())

    def describe_all(self) -> str:
        """
        Generate a human-readable description of all registered agents.

        This is injected into the Conductor's system prompt so it knows
        which agents are available and what they do.
        """
        if not self._agents:
            return "(No agents registered)"

        lines: list[str] = []
        for agent in self._agents.values():
            parts = [f"- **{agent.name}**: {agent.description}"]
            if agent.input_fields:
                parts.append(f"  Reads: {', '.join(agent.input_fields)}")
            if agent.output_fields:
                parts.append(f"  Writes: {', '.join(agent.output_fields)}")
            lines.extend(parts)
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self._agents)

    def __contains__(self, name: str) -> bool:
        return name in self._agents
