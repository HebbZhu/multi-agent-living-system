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
import time
from typing import TYPE_CHECKING, Any

from mals.core.models import ConsensusStatus, GlobalStatus

if TYPE_CHECKING:
    from mals.agents.registry import AgentRegistry
    from mals.core.blackboard import Blackboard
    from mals.llm.client import LLMClient
    from mals.memory.manager import MemoryManager
    from mals.observability.metrics import MetricsCollector
    from mals.observability.recorder import EventRecorder

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

CONDUCTOR_SYSTEM_PROMPT = """You are the Conductor of a multi-agent collaboration system called MALS (Multi-Agent Living System).

Your role is to observe the current state of the shared blackboard and decide the NEXT action to take.
You NEVER perform domain work yourself. You only route tasks to specialist agents.

## Available Agents
{agent_descriptions}

## Decision Rules (follow in STRICT priority order — check rule 1 first, then 2, etc.)

1. **COMPLETION CHECK (highest priority):** Look at the workspace fields. If the workspace already contains meaningful content that addresses the objective (code, tests, results, etc.) AND there is NO active consensus review pending, use action="complete". Do NOT wait for every plan step — if the core deliverables exist and look reasonable, COMPLETE.

2. **CONSENSUS PENDING:** If the "Consensus" line shows status=pending_review, invoke the "critic" agent. Set include_consensus=true and include the target field in relevant_fields.

3. **CONSENSUS REJECTED:** If the "Consensus" line shows status=rejected, re-invoke the original producing agent (code_generator or writer) with include_consensus=true so it can see the critique.

4. **NO PLAN YET:** If status=PLANNING and workspace has no "plan" field, invoke the "planner" agent.

5. **NEXT STEP:** If a plan exists and some plan steps have output_fields not yet in workspace, invoke the appropriate agent for the next incomplete step. Use code_generator for code/tests, writer for documentation/reports.

6. **FAILURE:** If an unrecoverable error occurs, use action="fail".

## CRITICAL RULES
- NEVER invoke "critic" unless Consensus line explicitly says status=pending_review. If there is no Consensus line, critic has nothing to review.
- NEVER re-invoke an agent for a field that already has content unless consensus was REJECTED for that specific field.
- If workspace has content for the main deliverables and no consensus is pending, the task is DONE — use action="complete".
- When in doubt, prefer "complete" over invoking more agents. Unnecessary agent calls waste resources.

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
        metrics: "MetricsCollector | None" = None,
        recorder: "EventRecorder | None" = None,
    ) -> None:
        self._board = blackboard
        self._llm = llm_client
        self._memory = memory_manager
        self._registry = agent_registry
        self._max_steps = max_steps
        self._step_count = 0
        self._metrics = metrics
        self._recorder = recorder

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

        if self._recorder:
            self._recorder.record_task_start(
                self._board.state.task_id,
                self._board.state.objective,
            )

        while self._step_count < self._max_steps:
            self._step_count += 1
            state = self._board.state

            if self._recorder:
                self._recorder.set_step(self._step_count)

            # Check terminal conditions
            if state.global_status in (GlobalStatus.COMPLETED, GlobalStatus.FAILED):
                logger.info("Task reached terminal status: %s", state.global_status.value)
                break

            # SENSE: Generate dashboard view
            dashboard = self._memory.generate_dashboard(state)
            logger.info("=== Step %d ===\n%s", self._step_count, dashboard)

            if self._recorder:
                self._recorder.record_conductor_think(dashboard)

            # THINK: Ask LLM for routing decision
            think_start = time.time()
            decision = await self._think(dashboard)
            think_latency = time.time() - think_start
            logger.info("Decision: %s", decision)

            if self._recorder:
                self._recorder.record_conductor_decide(
                    action=decision.action,
                    agent_name=decision.agent_name,
                    reasoning=decision.reason,
                )

            if self._metrics:
                self._metrics.record_conductor_step(
                    action=decision.action,
                    agent_name=decision.agent_name,
                    latency=think_latency,
                    input_tokens=self._llm.total_usage.input_tokens,
                    output_tokens=self._llm.total_usage.output_tokens,
                )

            # ACT: Execute the decision
            await self._act(decision)

        final_status = self._board.state.global_status

        # Handle max steps exceeded
        if self._step_count >= self._max_steps and final_status not in (GlobalStatus.COMPLETED, GlobalStatus.FAILED):
            logger.warning("Conductor loop exceeded max_steps (%d). Marking as FAILED.", self._max_steps)
            self._board.set_status(GlobalStatus.FAILED, reason="Max conductor steps exceeded")
            final_status = GlobalStatus.FAILED

        if self._recorder:
            self._recorder.record_task_end(
                status=final_status.value,
                summary={"steps": self._step_count},
            )

        if self._metrics:
            self._metrics.mark_task_complete()

        return final_status

    # ------------------------------------------------------------------
    # THINK — LLM-based routing decision
    # ------------------------------------------------------------------

    _last_agent: str | None = None
    _repeat_count: int = 0

    async def _think(self, dashboard: str) -> ConductorDecision:
        """Use the LLM to decide the next action based on the dashboard view."""
        agent_descriptions = self._registry.describe_all()
        system_prompt = CONDUCTOR_SYSTEM_PROMPT.format(agent_descriptions=agent_descriptions)

        # Build user prompt with programmatic hints
        hints: list[str] = []
        state = self._board.state

        # Hint: plan already exists
        if "plan" in state.workspace:
            hints.append("IMPORTANT: A plan already exists in workspace. Do NOT invoke planner again.")
            # Find next TODO step
            plan = state.workspace.get("plan")
            if isinstance(plan, dict) and "steps" in plan:
                for step in plan["steps"]:
                    output_field = step.get("output_field", "")
                    if output_field and output_field not in state.workspace:
                        hints.append(f"Next TODO step: '{step.get('title', '')}' -> output_field='{output_field}'. Invoke the appropriate agent (code_generator for code, writer for text).")
                        break
                else:
                    # All steps done
                    if not state.consensus:
                        hints.append("All plan steps are DONE and no consensus is pending. You should use action=\"complete\".")

        # Hint: no consensus active
        if not state.consensus:
            hints.append("No consensus is active. Do NOT invoke critic.")

        # Hint: repeat detection
        if self._repeat_count >= 2:
            hints.append(f"WARNING: You have invoked '{self._last_agent}' {self._repeat_count} times in a row. This is likely a loop. Choose a DIFFERENT action or use action=\"complete\".")

        hint_block = "\n".join(hints)
        user_prompt = f"Current blackboard state:\n\n{dashboard}\n\n{hint_block}\n\nWhat should be the next action?"

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
            if self._recorder:
                self._recorder.record_error("conductor_think", str(e))
            return ConductorDecision(action="fail", reason=f"Conductor error: {e}")

    # ------------------------------------------------------------------
    # ACT — Execute the routing decision
    # ------------------------------------------------------------------

    async def _act(self, decision: ConductorDecision) -> None:
        """Execute the Conductor's routing decision."""

        # Track repeat invocations
        if decision.action == "invoke_agent" and decision.agent_name:
            if decision.agent_name == self._last_agent:
                self._repeat_count += 1
            else:
                self._repeat_count = 1
            self._last_agent = decision.agent_name

            # Hard override: if same agent invoked 3+ times in a row, force complete
            if self._repeat_count >= 3:
                logger.warning(
                    "Agent '%s' invoked %d times in a row — forcing completion.",
                    decision.agent_name, self._repeat_count,
                )
                old_status = self._board.state.global_status.value
                self._board.set_status(
                    GlobalStatus.COMPLETED,
                    reason=f"Auto-completed: agent '{decision.agent_name}' loop detected",
                )
                if self._recorder:
                    self._recorder.record_status_change(old_status, "COMPLETED", "Loop detection override")
                if self._metrics:
                    self._metrics.record_status_transition(old_status, "COMPLETED", "Loop detection override")
                return
        else:
            self._repeat_count = 0
            self._last_agent = None

        if decision.action == "invoke_agent":
            await self._invoke_agent(decision)

        elif decision.action == "update_status":
            if decision.status_update:
                old_status = self._board.state.global_status.value
                self._board.set_status(decision.status_update, reason=decision.reason)
                if self._recorder:
                    self._recorder.record_status_change(
                        old_status, decision.status_update.value, decision.reason,
                    )
                if self._metrics:
                    self._metrics.record_status_transition(
                        old_status, decision.status_update.value, decision.reason,
                    )

        elif decision.action == "complete":
            old_status = self._board.state.global_status.value
            self._board.set_status(GlobalStatus.COMPLETED, reason=decision.reason)
            if self._recorder:
                self._recorder.record_status_change(old_status, "COMPLETED", decision.reason)
            if self._metrics:
                self._metrics.record_status_transition(old_status, "COMPLETED", decision.reason)

        elif decision.action == "fail":
            old_status = self._board.state.global_status.value
            self._board.set_status(GlobalStatus.FAILED, reason=decision.reason)
            if self._recorder:
                self._recorder.record_status_change(old_status, "FAILED", decision.reason)
            if self._metrics:
                self._metrics.record_status_transition(old_status, "FAILED", decision.reason)

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
            if self._recorder:
                self._recorder.record_error("conductor", f"Agent '{decision.agent_name}' not found")
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

        if self._recorder:
            self._recorder.record_agent_start(
                decision.agent_name,
                context_fields=decision.relevant_fields,
            )

        agent_start = time.time()

        try:
            # Ensure status is EXECUTING
            if self._board.state.global_status == GlobalStatus.PLANNING:
                old_status = self._board.state.global_status.value
                self._board.set_status(
                    GlobalStatus.EXECUTING,
                    reason=f"First agent invoked: {decision.agent_name}",
                )
                if self._recorder:
                    self._recorder.record_status_change(
                        old_status, "EXECUTING",
                        f"First agent invoked: {decision.agent_name}",
                    )
                if self._metrics:
                    self._metrics.record_status_transition(
                        old_status, "EXECUTING",
                        f"First agent invoked: {decision.agent_name}",
                    )

            # Call the agent
            result = await agent.execute(context, self._board)

            agent_latency = time.time() - agent_start
            input_tokens = result.get("input_tokens", 0) if isinstance(result, dict) else 0
            output_tokens = result.get("output_tokens", 0) if isinstance(result, dict) else 0

            # Log invocation end
            self._board.log_invocation_end(
                invocation,
                status="completed",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

            if self._recorder:
                self._recorder.record_agent_end(
                    decision.agent_name,
                    status="completed",
                    latency=agent_latency,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )

            if self._metrics:
                self._metrics.record_agent_invocation(
                    decision.agent_name,
                    latency=agent_latency,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    success=True,
                )

            # Check if a consensus cycle just completed
            if self._board.state.consensus and self._board.state.consensus.status == ConsensusStatus.APPROVED:
                iterations = self._board.state.consensus.current_iteration
                if iterations == 1:
                    outcome = "approved_first_try"
                elif iterations < self._board.state.consensus.max_iterations:
                    outcome = "approved_after_revision"
                else:
                    outcome = "force_approved"

                if self._metrics:
                    self._metrics.record_consensus_cycle(iterations, outcome)
                if self._recorder:
                    self._recorder.record_consensus_end(
                        self._board.state.consensus.target_field,
                        outcome, iterations,
                    )

                self._board.clear_consensus()

        except Exception as e:
            agent_latency = time.time() - agent_start
            logger.error("Agent '%s' failed: %s", decision.agent_name, e)
            self._board.log_invocation_end(invocation, status="error", error=str(e))

            if self._recorder:
                self._recorder.record_agent_end(
                    decision.agent_name,
                    status="error",
                    latency=agent_latency,
                    error=str(e),
                )
                self._recorder.record_error(decision.agent_name, str(e))

            if self._metrics:
                self._metrics.record_agent_invocation(
                    decision.agent_name,
                    latency=agent_latency,
                    success=False,
                    error=str(e),
                )

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
