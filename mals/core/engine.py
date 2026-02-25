"""
MALSEngine — The top-level orchestrator that wires everything together.

This is the main entry point for using MALS programmatically. It initializes
the blackboard, conductor, memory manager, LLM client, and agent registry,
then runs the conductor loop.

Usage:
    engine = MALSEngine()
    result = await engine.run("Build a REST API with authentication")
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from mals.agents.builtins import create_builtin_agents
from mals.agents.registry import AgentRegistry, AgentSpec
from mals.core.blackboard import Blackboard, InMemoryBackend, RedisBackend
from mals.core.conductor import Conductor
from mals.core.models import GlobalStatus
from mals.llm.client import LLMClient
from mals.memory.manager import MemoryManager
from mals.observability.metrics import MetricsCollector
from mals.observability.recorder import EventRecorder
from mals.utils.config import MALSConfig
from mals.utils.log import setup_logging

logger = logging.getLogger("mals.engine")


class MALSEngine:
    """
    The top-level MALS engine.

    Wires together all components and provides a simple interface for running
    multi-agent tasks. Includes full observability via MetricsCollector and
    EventRecorder.
    """

    def __init__(
        self,
        config: MALSConfig | None = None,
        custom_agents: list[AgentSpec] | None = None,
    ) -> None:
        """
        Initialize the MALS engine.

        Args:
            config: Configuration object. If None, loads from file/env.
            custom_agents: Additional specialist agents to register.
        """
        self._config = config or MALSConfig.load()

        # Setup logging
        setup_logging(self._config.logging.level)

        # Initialize LLM clients
        llm_kwargs: dict[str, Any] = {"max_retries": self._config.llm.max_retries}
        if self._config.llm.api_key:
            llm_kwargs["api_key"] = self._config.llm.api_key
        if self._config.llm.base_url:
            llm_kwargs["base_url"] = self._config.llm.base_url

        self._agent_llm = LLMClient(
            model=self._config.llm.model,
            temperature=self._config.llm.temperature,
            **llm_kwargs,
        )
        self._conductor_llm = LLMClient(
            model=self._config.llm.conductor_model,
            temperature=0.1,  # Conductor should be deterministic
            **llm_kwargs,
        )

        # Initialize blackboard
        if self._config.blackboard.backend == "redis":
            backend = RedisBackend(redis_url=self._config.blackboard.redis_url)
        else:
            backend = InMemoryBackend()
        self._blackboard = Blackboard(backend=backend)

        # Initialize memory manager
        self._memory = MemoryManager(llm_client=self._agent_llm)

        # Initialize agent registry
        self._registry = AgentRegistry()

        # Register built-in agents
        for agent in create_builtin_agents(self._agent_llm):
            self._registry.register(agent)

        # Register custom agents
        if custom_agents:
            for agent in custom_agents:
                self._registry.register(agent)

        # Observability
        self._metrics: MetricsCollector | None = None
        self._recorder: EventRecorder | None = None

        logger.info(
            "MALSEngine initialized: %d agents registered, backend=%s",
            len(self._registry),
            self._config.blackboard.backend,
        )

    async def run(
        self,
        objective: str,
        constraints: list[str] | None = None,
        max_steps: int | None = None,
        record: bool = True,
        export_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        """
        Run a multi-agent task.

        Args:
            objective: The high-level goal of the task.
            constraints: Optional constraints for the agents.
            max_steps: Override the maximum conductor steps.
            record: Whether to enable event recording (default: True).
            export_dir: Directory to export metrics and recording files.
                        If None, files are not auto-exported.

        Returns:
            A dictionary containing the final workspace, status, metrics, and usage stats.
        """
        # Initialize observability
        self._metrics = MetricsCollector()
        self._recorder = EventRecorder() if record else None

        # Reset LLM usage counters
        self._agent_llm.reset_usage()
        self._conductor_llm.reset_usage()

        # Initialize blackboard
        state = self._blackboard.initialize(objective, constraints)
        logger.info("Task started: %s (id=%s)", objective, state.task_id)

        # Create and run conductor
        conductor = Conductor(
            blackboard=self._blackboard,
            llm_client=self._conductor_llm,
            memory_manager=self._memory,
            agent_registry=self._registry,
            max_steps=max_steps or self._config.conductor.max_steps,
            metrics=self._metrics,
            recorder=self._recorder,
        )

        final_status = await conductor.run()

        # Gather results
        result = {
            "task_id": state.task_id,
            "status": final_status.value,
            "objective": objective,
            "workspace": dict(self._blackboard.state.workspace),
            "steps": conductor.step_count,
            "token_usage": {
                "conductor": {
                    "input": self._conductor_llm.total_usage.input_tokens,
                    "output": self._conductor_llm.total_usage.output_tokens,
                },
                "agents": {
                    "input": self._agent_llm.total_usage.input_tokens,
                    "output": self._agent_llm.total_usage.output_tokens,
                },
                "total": (
                    self._conductor_llm.total_usage.total_tokens
                    + self._agent_llm.total_usage.total_tokens
                ),
            },
            "metrics": self._metrics.to_dict() if self._metrics else {},
        }

        # Print metrics summary
        if self._metrics:
            logger.info("\n%s", self._metrics.summary_text())

        # Export files if requested
        if export_dir:
            export_path = Path(export_dir)
            export_path.mkdir(parents=True, exist_ok=True)

            # Export metrics
            metrics_file = export_path / f"{state.task_id}_metrics.json"
            with open(metrics_file, "w", encoding="utf-8") as f:
                json.dump(self._metrics.to_dict(), f, indent=2, ensure_ascii=False)
            logger.info("Metrics exported to %s", metrics_file)

            # Export recording
            if self._recorder:
                recording_file = export_path / f"{state.task_id}_recording.json"
                self._recorder.export_json(recording_file)

            # Export result summary
            result_file = export_path / f"{state.task_id}_result.json"
            with open(result_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False, default=str)
            logger.info("Result exported to %s", result_file)

        logger.info(
            "Task completed: status=%s, steps=%d, total_tokens=%d",
            final_status.value,
            conductor.step_count,
            result["token_usage"]["total"],
        )

        return result

    @property
    def blackboard(self) -> Blackboard:
        """Access the blackboard instance."""
        return self._blackboard

    @property
    def registry(self) -> AgentRegistry:
        """Access the agent registry."""
        return self._registry

    @property
    def metrics(self) -> MetricsCollector | None:
        """Access the metrics collector from the last run."""
        return self._metrics

    @property
    def recorder(self) -> EventRecorder | None:
        """Access the event recorder from the last run."""
        return self._recorder
