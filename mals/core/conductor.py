"""
The Conductor Agent — The central scheduler of the MALS system.

The Conductor implements a continuous Sense-Think-Act loop:
1. SENSE: Read the compact dashboard view of the blackboard.
2. THINK: Use a lightweight LLM to decide which specialist agent to activate next.
3. ACT: Invoke the chosen agent with a minimal context slice, or update the blackboard.

The Conductor never performs domain work itself. It is purely a router and scheduler.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from mals.core.models import ConsensusStatus, GlobalStatus

if TYPE_CHECKING:
    from mals.agents.registry import AgentRegistry
    from mals.core.blackboard import Blackboard
    from mals.llm.client import LLMClient
    from mals.memory.manager import MemoryManager

logger = logging.getLogger("mals.conductor")


# ---------------------------------------------------------------------------
# Conductor Decision — structured output from the Think step
# ---------------------------------------------------------------------------

class ConductorDecision:
    """Represents a routing decision made by the Conductor."""

    def __init__(
        self,
        action: str,
        agent_name: str | None = None,
        relevant_fields: list[str] | None = None,
        include_consensus: bool = False,
        status_update: GlobalStatus | None = None,
        reason: str = "",
    ) -> None:
        self.action = action  # "invoke_agent", "update_status", "wait_user", "complete", "fail"
        self.agent_name = agent_name
        self.relevant_fields = relevant_fields or []
        self.include_consensus = include_consensus
        self.status_update = status_update
        self.reason = reason

    def __repr__(self) -> str:
        return f"ConductorDecision(action={self.action}, agent={self.agent_name}, reason={self.reason!r})"


# ---------------------------------------------------------------------------
# Conductor System Prompt
# ---------------------------------------------------------------------------

CONDUCTOR_SYSTEM_PROMPT = """You are the Conductor of a multi-agent collaboration system called MALS (Multi-Agent Living Space).

Your role is to observe the current state of the shared blackboard and decide the NEXT action to take.
You NEVER perform domain work yourself. You only route tasks to specialist agents.

## Available Agents
{agent_descriptions}

## Decision Rules
1. If the task has just started (status=PLANNING), decide which agent should begin the work.
2. If an agent has produced output and consensus review is needed, invoke the critic agent.
3. If consensus was REJECTED, re-invoke the original agent with the critique.
4. If consensus was APPROVED, decide the next step or mark the task as complete.
5. If all work is done and verified, set status to COMPLETED.
6. If an unrecoverable error occurs, set status to FAILED.

## Response Format
You MUST respond with a valid JSON object (no markdown, no explanation):
{{
  "action": "invoke_agent" | "update_status" | "complete" | "fail",
  "agent_name": "<name of agent to invoke, or null>",
  "relevant_fields": ["<workspace fields the agent needs>"],
  "include_consensus": true | false,
  "reason": "<brief explanation of your decision>"
}}
"""


class Conductor:
    """
    The Conductor Agent — orchestrates the multi-agent collaboration loop.

    The Conductor is intentionally lightweight. It uses a fast, cheap LLM
    (e.g., GPT-4.1-nano) to make routing decisions based on a compact
    dashboard view of the blackboard (~200 tokens).
    """

    def __init__(
        self,
        blackboard: "Blackboard",
        llm_client: "LLMClient",
        memory_manager: "MemoryManager",
        agent_registry: "AgentRegistry",
        max_steps: int = 50,
    ) -> None:
        self._board = blackboard
        self._llm = llm_client
        self._memory = memory_manager
        self._registry = agent_registry
        self._max_steps = max_steps
        self._step_count = 0

    # ------------------------------------------------------------------
    # The Main Loop
    # ------------------------------------------------------------------

    async def run(self) -> GlobalStatus:
        """
        Execute the Conductor loop until the task completes or fails.

        Returns:
            The final GlobalStatus of the task.
        """
        logger.info("Conductor loop started (max_steps=%d)", self._max_steps)

        while self._step_count < self._max_steps:
            self._step_count += 1
            state = self._board.state

            # Check terminal conditions
            if state.global_status in (GlobalStatus.COMPLETED, GlobalStatus.FAILED):
                logger.info("Task reached terminal status: %s", state.global_status.value)
                return state.global_status

            # SENSE: Generate dashboard view
            dashboard = self._memory.generate_dashboard(state)
            logger.info("=== Step %d ===\n%s", self._step_count, dashboard)

            # THINK: Ask LLM for routing decision
            decision = await self._think(dashboard)
            logger.info("Decision: %s", decision)

            # ACT: Execute the decision
            await self._act(decision)

        # Max steps exceeded
        logger.warning("Conductor loop exceeded max_steps (%d). Marking as FAILED.", self._max_steps)
        self._board.set_status(GlobalStatus.FAILED, reason="Max conductor steps exceeded")
        return GlobalStatus.FAILED

    # ------------------------------------------------------------------
    # THINK — LLM-based routing decision
    # ------------------------------------------------------------------

    async def _think(self, dashboard: str) -> ConductorDecision:
        """Use the LLM to decide the next action based on the dashboard view."""
        agent_descriptions = self._registry.describe_all()
        system_prompt = CONDUCTOR_SYSTEM_PROMPT.format(agent_descriptions=agent_descriptions)

        user_prompt = f"Current blackboard state:\n\n{dashboard}\n\nWhat should be the next action?"

        try:
            raw_response = await self._llm.complete(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=300,
                temperature=0.1,
            )

            # Parse the JSON response
            decision_data = _parse_json_response(raw_response)
            return ConductorDecision(
                action=decision_data.get("action", "fail"),
                agent_name=decision_data.get("agent_name"),
                relevant_fields=decision_data.get("relevant_fields", []),
                include_consensus=decision_data.get("include_consensus", False),
                reason=decision_data.get("reason", ""),
            )

        except Exception as e:
            logger.error("Conductor think step failed: %s", e)
            return ConductorDecision(action="fail", reason=f"Conductor error: {e}")

    # ------------------------------------------------------------------
    # ACT — Execute the routing decision
    # ------------------------------------------------------------------

    async def _act(self, decision: ConductorDecision) -> None:
        """Execute the Conductor's routing decision."""

        if decision.action == "invoke_agent":
            await self._invoke_agent(decision)

        elif decision.action == "update_status":
            if decision.status_update:
                self._board.set_status(decision.status_update, reason=decision.reason)

        elif decision.action == "complete":
            self._board.set_status(GlobalStatus.COMPLETED, reason=decision.reason)

        elif decision.action == "fail":
            self._board.set_status(GlobalStatus.FAILED, reason=decision.reason)

        else:
            logger.warning("Unknown action: %s", decision.action)

    async def _invoke_agent(self, decision: ConductorDecision) -> None:
        """Invoke a specialist agent with a context slice from the blackboard."""
        if not decision.agent_name:
            logger.error("invoke_agent decision has no agent_name")
            return

        agent = self._registry.get(decision.agent_name)
        if agent is None:
            logger.error("Agent '%s' not found in registry", decision.agent_name)
            return

        # Prepare context slice
        context = self._memory.slice_context(
            state=self._board.state,
            relevant_fields=decision.relevant_fields,
            include_hypotheses=True,
            include_consensus=decision.include_consensus,
        )

        # Log invocation start
        invocation = self._board.log_invocation_start(decision.agent_name)

        try:
            # Ensure status is EXECUTING
            if self._board.state.global_status == GlobalStatus.PLANNING:
                self._board.set_status(GlobalStatus.EXECUTING, reason=f"First agent invoked: {decision.agent_name}")

            # Call the agent
            result = await agent.execute(context, self._board)

            # Log invocation end
            self._board.log_invocation_end(
                invocation,
                status="completed",
                input_tokens=result.get("input_tokens", 0) if isinstance(result, dict) else 0,
                output_tokens=result.get("output_tokens", 0) if isinstance(result, dict) else 0,
            )

        except Exception as e:
            logger.error("Agent '%s' failed: %s", decision.agent_name, e)
            self._board.log_invocation_end(invocation, status="error", error=str(e))

    @property
    def step_count(self) -> int:
        """Return the current step count."""
        return self._step_count


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_json_response(raw: str) -> dict[str, Any]:
    """Extract a JSON object from an LLM response, handling markdown fences."""
    text = raw.strip()
    # Remove markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last lines (fences)
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    return json.loads(text)
