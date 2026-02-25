"""
MetricsCollector — Centralized metrics collection for MALS.

Collects and aggregates performance metrics across the entire system:
- Per-agent token usage (input/output)
- Per-agent latency (min/max/avg/p95)
- Conductor decision counts and routing patterns
- Consensus loop statistics (approval rate, avg iterations)
- Memory tier transitions
- Overall task-level summaries

All metrics are stored in-memory and can be exported as JSON or dict
for dashboard rendering or benchmark reporting.
"""

from __future__ import annotations

import logging
import statistics
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("mals.observability.metrics")


@dataclass
class AgentMetrics:
    """Aggregated metrics for a single specialist agent."""
    name: str
    invocation_count: int = 0
    success_count: int = 0
    error_count: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    latencies: list[float] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    @property
    def avg_latency(self) -> float:
        return statistics.mean(self.latencies) if self.latencies else 0.0

    @property
    def p95_latency(self) -> float:
        if len(self.latencies) < 2:
            return self.latencies[0] if self.latencies else 0.0
        sorted_lats = sorted(self.latencies)
        idx = int(len(sorted_lats) * 0.95)
        return sorted_lats[min(idx, len(sorted_lats) - 1)]

    @property
    def min_latency(self) -> float:
        return min(self.latencies) if self.latencies else 0.0

    @property
    def max_latency(self) -> float:
        return max(self.latencies) if self.latencies else 0.0

    @property
    def success_rate(self) -> float:
        if self.invocation_count == 0:
            return 0.0
        return self.success_count / self.invocation_count

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "invocation_count": self.invocation_count,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "avg_latency_s": round(self.avg_latency, 3),
            "p95_latency_s": round(self.p95_latency, 3),
            "min_latency_s": round(self.min_latency, 3),
            "max_latency_s": round(self.max_latency, 3),
            "success_rate": round(self.success_rate, 4),
        }


@dataclass
class ConsensusMetrics:
    """Aggregated metrics for the consensus loop."""
    total_cycles: int = 0
    approved_first_try: int = 0
    approved_after_revision: int = 0
    force_approved: int = 0
    total_iterations: int = 0
    iterations_per_cycle: list[int] = field(default_factory=list)

    @property
    def avg_iterations(self) -> float:
        return statistics.mean(self.iterations_per_cycle) if self.iterations_per_cycle else 0.0

    @property
    def first_try_approval_rate(self) -> float:
        if self.total_cycles == 0:
            return 0.0
        return self.approved_first_try / self.total_cycles

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_cycles": self.total_cycles,
            "approved_first_try": self.approved_first_try,
            "approved_after_revision": self.approved_after_revision,
            "force_approved": self.force_approved,
            "total_iterations": self.total_iterations,
            "avg_iterations_per_cycle": round(self.avg_iterations, 2),
            "first_try_approval_rate": round(self.first_try_approval_rate, 4),
        }


@dataclass
class ConductorMetrics:
    """Aggregated metrics for the Conductor."""
    total_steps: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    decision_counts: dict[str, int] = field(default_factory=dict)
    routing_counts: dict[str, int] = field(default_factory=dict)
    latencies: list[float] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    @property
    def avg_latency(self) -> float:
        return statistics.mean(self.latencies) if self.latencies else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_steps": self.total_steps,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "avg_latency_s": round(self.avg_latency, 3),
            "decision_counts": dict(self.decision_counts),
            "routing_counts": dict(self.routing_counts),
        }


class MetricsCollector:
    """
    Centralized metrics collector for the MALS system.

    Provides methods to record events and aggregate them into
    structured metrics for dashboard rendering and benchmark reporting.

    Usage:
        collector = MetricsCollector()
        collector.record_agent_invocation("code_generator", latency=2.3, ...)
        report = collector.to_dict()
    """

    def __init__(self) -> None:
        self._task_start: float = time.time()
        self._task_end: float | None = None
        self._agent_metrics: dict[str, AgentMetrics] = {}
        self._conductor = ConductorMetrics()
        self._consensus = ConsensusMetrics()
        self._memory_compressions: int = 0
        self._status_transitions: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Agent Metrics
    # ------------------------------------------------------------------

    def record_agent_invocation(
        self,
        agent_name: str,
        latency: float,
        input_tokens: int = 0,
        output_tokens: int = 0,
        success: bool = True,
        error: str | None = None,
    ) -> None:
        """Record a single agent invocation."""
        if agent_name not in self._agent_metrics:
            self._agent_metrics[agent_name] = AgentMetrics(name=agent_name)

        m = self._agent_metrics[agent_name]
        m.invocation_count += 1
        m.latencies.append(latency)
        m.total_input_tokens += input_tokens
        m.total_output_tokens += output_tokens

        if success:
            m.success_count += 1
        else:
            m.error_count += 1

        logger.debug(
            "Agent metric: %s latency=%.2fs tokens=%d success=%s",
            agent_name, latency, input_tokens + output_tokens, success,
        )

    # ------------------------------------------------------------------
    # Conductor Metrics
    # ------------------------------------------------------------------

    def record_conductor_step(
        self,
        action: str,
        agent_name: str | None = None,
        latency: float = 0.0,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        """Record a single Conductor decision step."""
        self._conductor.total_steps += 1
        self._conductor.total_input_tokens += input_tokens
        self._conductor.total_output_tokens += output_tokens
        self._conductor.latencies.append(latency)

        # Track decision type distribution
        self._conductor.decision_counts[action] = (
            self._conductor.decision_counts.get(action, 0) + 1
        )

        # Track routing distribution
        if agent_name:
            self._conductor.routing_counts[agent_name] = (
                self._conductor.routing_counts.get(agent_name, 0) + 1
            )

    # ------------------------------------------------------------------
    # Consensus Metrics
    # ------------------------------------------------------------------

    def record_consensus_cycle(
        self,
        iterations: int,
        outcome: str,  # "approved_first_try", "approved_after_revision", "force_approved"
    ) -> None:
        """Record a completed consensus cycle."""
        self._consensus.total_cycles += 1
        self._consensus.total_iterations += iterations
        self._consensus.iterations_per_cycle.append(iterations)

        if outcome == "approved_first_try":
            self._consensus.approved_first_try += 1
        elif outcome == "approved_after_revision":
            self._consensus.approved_after_revision += 1
        elif outcome == "force_approved":
            self._consensus.force_approved += 1

    # ------------------------------------------------------------------
    # Memory Metrics
    # ------------------------------------------------------------------

    def record_memory_compression(self) -> None:
        """Record a memory compression event (hot → warm)."""
        self._memory_compressions += 1

    # ------------------------------------------------------------------
    # Status Transitions
    # ------------------------------------------------------------------

    def record_status_transition(
        self, from_status: str, to_status: str, reason: str
    ) -> None:
        """Record a global status transition."""
        self._status_transitions.append({
            "from": from_status,
            "to": to_status,
            "reason": reason,
            "timestamp": time.time(),
        })

    # ------------------------------------------------------------------
    # Task Lifecycle
    # ------------------------------------------------------------------

    def mark_task_complete(self) -> None:
        """Mark the task as complete and record the end time."""
        self._task_end = time.time()

    @property
    def elapsed_time(self) -> float:
        """Total elapsed time in seconds."""
        end = self._task_end or time.time()
        return end - self._task_start

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """
        Export all collected metrics as a structured dictionary.

        Suitable for JSON serialization, dashboard rendering, or benchmark reporting.
        """
        # Aggregate token totals
        total_agent_input = sum(m.total_input_tokens for m in self._agent_metrics.values())
        total_agent_output = sum(m.total_output_tokens for m in self._agent_metrics.values())
        total_agent_invocations = sum(m.invocation_count for m in self._agent_metrics.values())

        return {
            "task_summary": {
                "elapsed_time_s": round(self.elapsed_time, 2),
                "total_steps": self._conductor.total_steps,
                "total_agent_invocations": total_agent_invocations,
                "total_tokens": (
                    self._conductor.total_tokens
                    + total_agent_input
                    + total_agent_output
                ),
                "conductor_tokens": self._conductor.total_tokens,
                "agent_tokens": total_agent_input + total_agent_output,
                "memory_compressions": self._memory_compressions,
                "status_transitions": len(self._status_transitions),
            },
            "conductor": self._conductor.to_dict(),
            "agents": {
                name: m.to_dict()
                for name, m in sorted(self._agent_metrics.items())
            },
            "consensus": self._consensus.to_dict(),
            "status_history": self._status_transitions,
        }

    def summary_text(self) -> str:
        """Generate a human-readable summary of the metrics."""
        d = self.to_dict()
        ts = d["task_summary"]
        lines = [
            "=== MALS Task Metrics ===",
            f"Elapsed Time:        {ts['elapsed_time_s']:.1f}s",
            f"Conductor Steps:     {ts['total_steps']}",
            f"Agent Invocations:   {ts['total_agent_invocations']}",
            f"Total Tokens:        {ts['total_tokens']:,}",
            f"  Conductor:         {ts['conductor_tokens']:,}",
            f"  Agents:            {ts['agent_tokens']:,}",
            f"Memory Compressions: {ts['memory_compressions']}",
            "",
            "--- Per-Agent Breakdown ---",
        ]

        for name, am in d["agents"].items():
            lines.append(
                f"  {name}: {am['invocation_count']} calls, "
                f"{am['total_tokens']:,} tokens, "
                f"avg {am['avg_latency_s']:.2f}s, "
                f"success {am['success_rate']:.0%}"
            )

        cm = d["consensus"]
        if cm["total_cycles"] > 0:
            lines.extend([
                "",
                "--- Consensus ---",
                f"  Cycles:            {cm['total_cycles']}",
                f"  First-try approve: {cm['first_try_approval_rate']:.0%}",
                f"  Avg iterations:    {cm['avg_iterations_per_cycle']:.1f}",
            ])

        return "\n".join(lines)
