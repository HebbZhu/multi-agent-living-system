"""
Core data models for MALS.

Defines the structured state of the Dynamic Blackboard, including task metadata,
workspace, hypothesis threads, consensus state, and tiered memory management.
All models use Pydantic v2 for strict validation and serialization.
"""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class GlobalStatus(str, Enum):
    """Lifecycle states of a MALS task."""
    PLANNING = "PLANNING"
    EXECUTING = "EXECUTING"
    REFINING = "REFINING"
    VERIFYING = "VERIFYING"
    WAITING_USER = "WAITING_USER"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class HypothesisStatus(str, Enum):
    """Status of a hypothesis in the hypothesis thread."""
    PROPOSED = "proposed"
    VALIDATED = "validated"
    REJECTED = "rejected"


class ConsensusStatus(str, Enum):
    """Status of a consensus review cycle."""
    PENDING_REVIEW = "pending_review"
    IN_REVIEW = "in_review"
    APPROVED = "approved"
    REJECTED = "rejected"


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

class StatusChange(BaseModel):
    """Records a single status transition on the blackboard."""
    from_status: str
    to_status: str
    reason: str
    timestamp: float = Field(default_factory=time.time)


class Hypothesis(BaseModel):
    """A hypothesis proposed by an agent during task execution."""
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    content: str
    author_agent: str
    status: HypothesisStatus = HypothesisStatus.PROPOSED
    evidence: list[str] = Field(default_factory=list)
    timestamp: float = Field(default_factory=time.time)


class ReviewRecord(BaseModel):
    """A single review entry within a consensus cycle."""
    reviewer_agent: str
    verdict: ConsensusStatus
    critique: str
    timestamp: float = Field(default_factory=time.time)


class ConsensusState(BaseModel):
    """Tracks the consensus review cycle for a workspace artifact."""
    target_field: str
    status: ConsensusStatus = ConsensusStatus.PENDING_REVIEW
    review_history: list[ReviewRecord] = Field(default_factory=list)
    max_iterations: int = 3
    current_iteration: int = 0


class MemoryTier(BaseModel):
    """Three-tier memory management model (hot / warm / cold)."""
    hot: list[str] = Field(
        default_factory=list,
        description="Pointers to currently active and relevant blackboard fields.",
    )
    warm: dict[str, str] = Field(
        default_factory=dict,
        description="Summaries of completed work. Key = field name, Value = summary text.",
    )
    cold_ref: str = Field(
        default="",
        description="Reference URI to external cold storage (file path, S3 URI, etc.).",
    )


class AgentInvocationRecord(BaseModel):
    """Records a single agent invocation for observability."""
    agent_name: str
    started_at: float = Field(default_factory=time.time)
    finished_at: float | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    status: str = "running"
    error: str | None = None


# ---------------------------------------------------------------------------
# BlackboardState — the central data structure
# ---------------------------------------------------------------------------

class BlackboardState(BaseModel):
    """
    The complete state of the Dynamic Blackboard.

    This is the Single Source of Truth for the entire MALS system.
    All agents read from and write to this structure via the Blackboard API.
    No agent communicates directly with another agent.
    """

    # -- Metadata --
    task_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    objective: str = ""
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)

    # -- Global control --
    global_status: GlobalStatus = GlobalStatus.PLANNING
    status_history: list[StatusChange] = Field(default_factory=list)
    active_constraints: list[str] = Field(default_factory=list)

    # -- Workspace: where agent outputs live --
    workspace: dict[str, Any] = Field(default_factory=dict)

    # -- Hypothesis thread --
    hypothesis_thread: list[Hypothesis] = Field(default_factory=list)

    # -- Consensus & quality control --
    consensus: ConsensusState | None = None

    # -- Memory management --
    memory: MemoryTier = Field(default_factory=MemoryTier)

    # -- Observability --
    invocation_log: list[AgentInvocationRecord] = Field(default_factory=list)

    # -- Conductor scratch-pad (lightweight notes for the conductor) --
    conductor_notes: str = ""

    def touch(self) -> None:
        """Update the `updated_at` timestamp."""
        self.updated_at = time.time()
