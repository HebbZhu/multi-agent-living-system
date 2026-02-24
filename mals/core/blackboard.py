"""
Dynamic Blackboard — The Single Source of Truth for MALS.

The Blackboard is the central shared knowledge store that all agents read from
and write to. No agent communicates directly with another agent; all interaction
happens through the blackboard.

Supports two storage backends:
- InMemoryBackend: Zero-dependency, ideal for quick experiments and testing.
- RedisBackend: Persistent, production-ready, supports task recovery after interruption.
"""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Any

from mals.core.models import (
    AgentInvocationRecord,
    BlackboardState,
    ConsensusState,
    ConsensusStatus,
    GlobalStatus,
    Hypothesis,
    HypothesisStatus,
    ReviewRecord,
    StatusChange,
)

logger = logging.getLogger("mals.blackboard")


# ---------------------------------------------------------------------------
# Storage Backends
# ---------------------------------------------------------------------------

class BlackboardBackend(ABC):
    """Abstract interface for blackboard persistence."""

    @abstractmethod
    def save(self, state: BlackboardState) -> None:
        """Persist the full blackboard state."""

    @abstractmethod
    def load(self, task_id: str) -> BlackboardState | None:
        """Load a blackboard state by task ID. Returns None if not found."""

    @abstractmethod
    def exists(self, task_id: str) -> bool:
        """Check if a task exists in storage."""


class InMemoryBackend(BlackboardBackend):
    """In-memory storage backend. Data is lost when the process exits."""

    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    def save(self, state: BlackboardState) -> None:
        self._store[state.task_id] = state.model_dump_json()

    def load(self, task_id: str) -> BlackboardState | None:
        raw = self._store.get(task_id)
        if raw is None:
            return None
        return BlackboardState.model_validate_json(raw)

    def exists(self, task_id: str) -> bool:
        return task_id in self._store


class RedisBackend(BlackboardBackend):
    """Redis-backed persistent storage. Supports task recovery after interruption."""

    KEY_PREFIX = "mals:blackboard:"

    def __init__(self, redis_url: str = "redis://localhost:6379/0") -> None:
        try:
            import redis
        except ImportError as e:
            raise ImportError(
                "Redis backend requires the 'redis' package. "
                "Install it with: pip install redis"
            ) from e
        self._client = redis.Redis.from_url(redis_url, decode_responses=True)

    def save(self, state: BlackboardState) -> None:
        key = f"{self.KEY_PREFIX}{state.task_id}"
        self._client.set(key, state.model_dump_json())

    def load(self, task_id: str) -> BlackboardState | None:
        key = f"{self.KEY_PREFIX}{task_id}"
        raw = self._client.get(key)
        if raw is None:
            return None
        return BlackboardState.model_validate_json(raw)

    def exists(self, task_id: str) -> bool:
        key = f"{self.KEY_PREFIX}{task_id}"
        return bool(self._client.exists(key))


# ---------------------------------------------------------------------------
# Blackboard — the main API
# ---------------------------------------------------------------------------

class Blackboard:
    """
    The Dynamic Blackboard — central shared state for multi-agent collaboration.

    All reads and writes go through this class. It provides a structured API
    that prevents agents from corrupting the shared state, while maintaining
    full observability of every mutation.

    Usage:
        board = Blackboard()
        board.initialize("Build a REST API with authentication")
        board.write_workspace("code", "def hello(): ...")
        dashboard = board.dashboard()
    """

    def __init__(self, backend: BlackboardBackend | None = None) -> None:
        self._backend = backend or InMemoryBackend()
        self._state: BlackboardState | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self, objective: str, constraints: list[str] | None = None) -> BlackboardState:
        """
        Initialize a new blackboard for a task.

        Args:
            objective: The high-level goal of the task.
            constraints: Optional list of constraints the agents must respect.

        Returns:
            The newly created BlackboardState.
        """
        self._state = BlackboardState(
            objective=objective,
            active_constraints=constraints or [],
            global_status=GlobalStatus.PLANNING,
        )
        self._persist()
        logger.info("Blackboard initialized: task_id=%s, objective='%s'", self._state.task_id, objective)
        return self._state

    def resume(self, task_id: str) -> BlackboardState:
        """
        Resume a previously persisted task.

        Args:
            task_id: The ID of the task to resume.

        Returns:
            The loaded BlackboardState.

        Raises:
            ValueError: If the task is not found in storage.
        """
        state = self._backend.load(task_id)
        if state is None:
            raise ValueError(f"Task '{task_id}' not found in storage backend.")
        self._state = state
        logger.info("Blackboard resumed: task_id=%s, status=%s", task_id, state.global_status.value)
        return self._state

    @property
    def state(self) -> BlackboardState:
        """Access the current blackboard state. Raises if not initialized."""
        if self._state is None:
            raise RuntimeError("Blackboard not initialized. Call initialize() or resume() first.")
        return self._state

    # ------------------------------------------------------------------
    # Status Management
    # ------------------------------------------------------------------

    def set_status(self, new_status: GlobalStatus, reason: str = "") -> None:
        """
        Transition the global task status.

        Args:
            new_status: The target status.
            reason: Human-readable reason for the transition.
        """
        old_status = self.state.global_status
        self.state.status_history.append(
            StatusChange(
                from_status=old_status.value,
                to_status=new_status.value,
                reason=reason,
            )
        )
        self.state.global_status = new_status
        self._persist()
        logger.info("Status: %s -> %s (reason: %s)", old_status.value, new_status.value, reason)

    # ------------------------------------------------------------------
    # Workspace Read / Write
    # ------------------------------------------------------------------

    def read_workspace(self, field: str) -> Any:
        """Read a field from the workspace. Returns None if not found."""
        return self.state.workspace.get(field)

    def write_workspace(self, field: str, value: Any) -> None:
        """
        Write a value to a workspace field.

        This is the primary way agents produce output. Every write is logged
        and the field is automatically marked as "hot" in memory.
        """
        self.state.workspace[field] = value
        if field not in self.state.memory.hot:
            self.state.memory.hot.append(field)
        self._persist()
        logger.debug("Workspace write: field='%s', size=%s", field, _size_hint(value))

    def delete_workspace(self, field: str) -> None:
        """Remove a field from the workspace."""
        self.state.workspace.pop(field, None)
        if field in self.state.memory.hot:
            self.state.memory.hot.remove(field)
        self._persist()

    # ------------------------------------------------------------------
    # Hypothesis Thread
    # ------------------------------------------------------------------

    def propose_hypothesis(self, content: str, author_agent: str) -> Hypothesis:
        """Add a new hypothesis to the thread."""
        h = Hypothesis(content=content, author_agent=author_agent)
        self.state.hypothesis_thread.append(h)
        self._persist()
        logger.info("Hypothesis proposed by %s: %s", author_agent, content[:60])
        return h

    def resolve_hypothesis(self, hypothesis_id: str, status: HypothesisStatus, evidence: str = "") -> None:
        """Mark a hypothesis as validated or rejected."""
        for h in self.state.hypothesis_thread:
            if h.id == hypothesis_id:
                h.status = status
                if evidence:
                    h.evidence.append(evidence)
                self._persist()
                logger.info("Hypothesis %s -> %s", hypothesis_id, status.value)
                return
        raise ValueError(f"Hypothesis '{hypothesis_id}' not found.")

    # ------------------------------------------------------------------
    # Consensus Cycle
    # ------------------------------------------------------------------

    def start_consensus(self, target_field: str, max_iterations: int = 3) -> ConsensusState:
        """
        Initiate a consensus review cycle for a workspace artifact.

        Args:
            target_field: The workspace field to review.
            max_iterations: Maximum number of write-critique-revise cycles.

        Returns:
            The new ConsensusState.
        """
        self.state.consensus = ConsensusState(
            target_field=target_field,
            max_iterations=max_iterations,
        )
        self._persist()
        logger.info("Consensus started for field '%s' (max %d iterations)", target_field, max_iterations)
        return self.state.consensus

    def submit_review(self, reviewer_agent: str, verdict: ConsensusStatus, critique: str) -> None:
        """
        Submit a review for the current consensus cycle.

        Args:
            reviewer_agent: Name of the reviewing agent.
            verdict: APPROVED or REJECTED.
            critique: The review text.
        """
        if self.state.consensus is None:
            raise RuntimeError("No active consensus cycle.")

        record = ReviewRecord(
            reviewer_agent=reviewer_agent,
            verdict=verdict,
            critique=critique,
        )
        self.state.consensus.review_history.append(record)
        self.state.consensus.current_iteration += 1

        if verdict == ConsensusStatus.APPROVED:
            self.state.consensus.status = ConsensusStatus.APPROVED
            logger.info("Consensus APPROVED by %s for '%s'", reviewer_agent, self.state.consensus.target_field)
        elif self.state.consensus.current_iteration >= self.state.consensus.max_iterations:
            # Max iterations reached — force approve with warning
            self.state.consensus.status = ConsensusStatus.APPROVED
            logger.warning(
                "Consensus force-approved for '%s' after %d iterations",
                self.state.consensus.target_field,
                self.state.consensus.max_iterations,
            )
        else:
            self.state.consensus.status = ConsensusStatus.REJECTED
            logger.info(
                "Consensus REJECTED by %s (iteration %d/%d)",
                reviewer_agent,
                self.state.consensus.current_iteration,
                self.state.consensus.max_iterations,
            )

        self._persist()

    def clear_consensus(self) -> None:
        """Clear the current consensus cycle after it completes."""
        self.state.consensus = None
        self._persist()

    # ------------------------------------------------------------------
    # Invocation Logging
    # ------------------------------------------------------------------

    def log_invocation_start(self, agent_name: str) -> AgentInvocationRecord:
        """Record the start of an agent invocation."""
        record = AgentInvocationRecord(agent_name=agent_name)
        self.state.invocation_log.append(record)
        self._persist()
        return record

    def log_invocation_end(
        self,
        record: AgentInvocationRecord,
        status: str = "completed",
        input_tokens: int = 0,
        output_tokens: int = 0,
        error: str | None = None,
    ) -> None:
        """Record the completion of an agent invocation."""
        record.finished_at = time.time()
        record.status = status
        record.input_tokens = input_tokens
        record.output_tokens = output_tokens
        record.error = error
        self._persist()

    # ------------------------------------------------------------------
    # Conductor Notes
    # ------------------------------------------------------------------

    def set_conductor_notes(self, notes: str) -> None:
        """Update the conductor's scratch-pad notes."""
        self.state.conductor_notes = notes
        self._persist()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _persist(self) -> None:
        """Save the current state to the backend."""
        if self._state is not None:
            self._state.touch()
            self._backend.save(self._state)


def _size_hint(value: Any) -> str:
    """Return a human-readable size hint for a value."""
    if isinstance(value, str):
        return f"{len(value)} chars"
    if isinstance(value, dict):
        return f"{len(value)} keys"
    if isinstance(value, list):
        return f"{len(value)} items"
    return type(value).__name__
