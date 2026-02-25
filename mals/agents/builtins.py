"""
Built-in Specialist Agents for MALS.

These agents provide common capabilities out of the box. Users can use them
directly or as templates for building custom agents.

Each agent follows the same pattern:
1. Receive a context slice from the Conductor (via the blackboard).
2. Use an LLM to perform its specialized task.
3. Write results back to the blackboard.
4. Return token usage for metrics tracking.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from mals.agents.registry import AgentSpec, specialist
from mals.core.blackboard import Blackboard
from mals.core.models import ConsensusStatus
from mals.llm.client import LLMClient

logger = logging.getLogger("mals.agents.builtins")


def create_builtin_agents(llm_client: LLMClient) -> list[AgentSpec]:
    """
    Create and return all built-in specialist agents.

    Args:
        llm_client: The LLM client for agents to use.

    Returns:
        A list of AgentSpec instances ready to be registered.
    """

    # ------------------------------------------------------------------
    # Planner Agent
    # ------------------------------------------------------------------
    @specialist(
        name="planner",
        description="Analyzes the objective and creates a structured execution plan with clear steps. "
                    "Should be invoked first when a new task begins.",
        input_fields=["objective"],
        output_fields=["plan"],
    )
    async def planner(context: dict[str, Any], board: Blackboard) -> dict[str, Any]:
        objective = context.get("objective", "")
        constraints = board.state.active_constraints

        system_prompt = (
            "You are a task planner. Given an objective, create a clear, structured execution plan. "
            "Break the objective into concrete, actionable steps. Each step should produce a distinct workspace field. "
            "Output a JSON object with a 'steps' array, where each step has 'id', 'title', 'description', and 'output_field' (the workspace field it will produce). "
            "Use distinct output_field names for different deliverables (e.g., 'code', 'tests', 'documentation'). "
            "Respond in the same language as the objective."
        )
        user_prompt = f"Objective: {objective}"
        if constraints:
            user_prompt += f"\nConstraints: {', '.join(constraints)}"

        resp = await llm_client.complete_with_usage(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=1000,
        )

        try:
            plan = json.loads(_strip_fences(resp.content))
        except json.JSONDecodeError:
            plan = {"steps": [{"id": 1, "title": "Execute task", "description": resp.content, "output_field": "result"}]}

        board.write_workspace("plan", plan)
        return {"status": "ok", "input_tokens": resp.input_tokens, "output_tokens": resp.output_tokens}

    # ------------------------------------------------------------------
    # Code Generator Agent
    # ------------------------------------------------------------------
    @specialist(
        name="code_generator",
        description="Generates code based on requirements, specifications, or a plan. "
                    "Can write to any workspace field specified in the plan (e.g., 'code', 'tests').",
        input_fields=["plan", "requirements"],
        output_fields=["code", "tests"],
    )
    async def code_generator(context: dict[str, Any], board: Blackboard) -> dict[str, Any]:
        workspace = context.get("workspace", {})
        plan = workspace.get("plan", "")
        consensus = context.get("consensus")

        # Determine which output field to write to based on plan progress
        output_field = _determine_output_field(workspace, plan)

        system_prompt = (
            "You are an expert code generator. Write clean, well-documented, production-quality code "
            "based on the given requirements or plan. Include proper error handling and type hints. "
            "Output ONLY the code, no explanations."
        )

        user_prompt = f"Objective: {context.get('objective', '')}\n"
        if plan:
            user_prompt += f"\nPlan:\n{json.dumps(plan, ensure_ascii=False, indent=2)}"
        if consensus and consensus.get("last_critique"):
            user_prompt += f"\n\nPrevious review feedback (please address these issues):\n{consensus['last_critique']}"

        # Include existing code if writing tests
        if output_field == "tests" and "code" in workspace:
            user_prompt += f"\n\nExisting code to test:\n{workspace['code']}"

        resp = await llm_client.complete_with_usage(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=3000,
        )

        board.write_workspace(output_field, resp.content)
        board.start_consensus(output_field)
        return {"status": "ok", "input_tokens": resp.input_tokens, "output_tokens": resp.output_tokens}

    # ------------------------------------------------------------------
    # Critic Agent (for Consensus Loop)
    # ------------------------------------------------------------------
    @specialist(
        name="critic",
        description="Reviews and critiques workspace artifacts for quality, correctness, and completeness. "
                    "Part of the consensus loop — invoked after another agent produces output.",
        input_fields=["code", "tests", "plan", "result"],
        output_fields=[],
    )
    async def critic(context: dict[str, Any], board: Blackboard) -> dict[str, Any]:
        if board.state.consensus is None:
            return {"status": "no_consensus_active", "input_tokens": 0, "output_tokens": 0}

        target_field = board.state.consensus.target_field
        artifact = context.get("workspace", {}).get(target_field, "")

        system_prompt = (
            "You are a strict but fair code/content reviewer. Review the given artifact for:\n"
            "1. Correctness: Does it fulfill the objective?\n"
            "2. Quality: Is it well-structured, readable, and maintainable?\n"
            "3. Completeness: Are there missing parts or edge cases?\n"
            "4. Bugs: Are there any obvious errors?\n\n"
            "Respond with a JSON object:\n"
            '{"verdict": "APPROVED" or "REJECTED", "critique": "<detailed feedback>"}\n'
            "Only APPROVE if the artifact is genuinely good. Be specific in your critique."
        )

        user_prompt = (
            f"Objective: {context.get('objective', '')}\n\n"
            f"Artifact to review (field: {target_field}):\n\n{artifact}"
        )

        resp = await llm_client.complete_with_usage(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=800,
        )

        try:
            review = json.loads(_strip_fences(resp.content))
            verdict_str = review.get("verdict", "REJECTED").upper()
            verdict = ConsensusStatus.APPROVED if verdict_str == "APPROVED" else ConsensusStatus.REJECTED
            critique = review.get("critique", resp.content)
        except json.JSONDecodeError:
            verdict = ConsensusStatus.REJECTED
            critique = resp.content

        board.submit_review(
            reviewer_agent="critic",
            verdict=verdict,
            critique=critique,
        )

        return {"status": "ok", "verdict": verdict.value, "input_tokens": resp.input_tokens, "output_tokens": resp.output_tokens}

    # ------------------------------------------------------------------
    # Writer Agent
    # ------------------------------------------------------------------
    @specialist(
        name="writer",
        description="Generates written content such as documentation, reports, or summaries. "
                    "Writes output to the specified workspace field (default: 'result').",
        input_fields=["plan", "code", "tests", "requirements"],
        output_fields=["result", "documentation"],
    )
    async def writer(context: dict[str, Any], board: Blackboard) -> dict[str, Any]:
        workspace = context.get("workspace", {})
        consensus = context.get("consensus")

        # Determine output field
        output_field = _determine_output_field(workspace, workspace.get("plan", ""), prefer="result")

        system_prompt = (
            "You are an expert technical writer. Produce clear, well-structured, professional content "
            "based on the given context. Respond in the same language as the objective."
        )

        user_prompt = f"Objective: {context.get('objective', '')}\n"
        for key, value in workspace.items():
            if isinstance(value, str):
                user_prompt += f"\n{key}:\n{value[:2000]}"
            else:
                user_prompt += f"\n{key}:\n{json.dumps(value, ensure_ascii=False)[:2000]}"

        if consensus and consensus.get("last_critique"):
            user_prompt += f"\n\nPrevious review feedback:\n{consensus['last_critique']}"

        resp = await llm_client.complete_with_usage(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=3000,
        )

        board.write_workspace(output_field, resp.content)
        board.start_consensus(output_field)
        return {"status": "ok", "input_tokens": resp.input_tokens, "output_tokens": resp.output_tokens}

    # ------------------------------------------------------------------
    # Summarizer Agent
    # ------------------------------------------------------------------
    @specialist(
        name="summarizer",
        description="Produces a final summary of all completed work. "
                    "Should be invoked near the end of a task to consolidate results.",
        input_fields=["plan", "code", "tests", "result"],
        output_fields=["final_summary"],
    )
    async def summarizer(context: dict[str, Any], board: Blackboard) -> dict[str, Any]:
        workspace = context.get("workspace", {})

        system_prompt = (
            "You are a summarizer. Produce a concise but comprehensive summary of all the work "
            "that has been completed. Include key outcomes, decisions made, and any remaining items. "
            "Respond in the same language as the objective."
        )

        user_prompt = f"Objective: {context.get('objective', '')}\n\nCompleted work:\n"
        for key, value in workspace.items():
            if isinstance(value, str):
                user_prompt += f"\n--- {key} ---\n{value[:1500]}"
            else:
                user_prompt += f"\n--- {key} ---\n{json.dumps(value, ensure_ascii=False)[:1500]}"

        resp = await llm_client.complete_with_usage(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=1000,
        )

        board.write_workspace("final_summary", resp.content)
        return {"status": "ok", "input_tokens": resp.input_tokens, "output_tokens": resp.output_tokens}

    return [planner, code_generator, critic, writer, summarizer]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_fences(text: str) -> str:
    """Remove markdown code fences from LLM output."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    return text


def _determine_output_field(
    workspace: dict[str, Any],
    plan: Any,
    prefer: str = "code",
) -> str:
    """
    Determine which workspace field to write to based on plan progress.

    Looks at the plan steps and finds the first step whose output_field
    doesn't yet have content in the workspace.
    """
    if isinstance(plan, dict) and "steps" in plan:
        for step in plan["steps"]:
            field = step.get("output_field", prefer)
            if field not in workspace or not workspace.get(field):
                return field
    return prefer
