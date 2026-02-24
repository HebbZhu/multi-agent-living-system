"""
Built-in Specialist Agents for MALS.

These agents provide common capabilities out of the box. Users can use them
directly or as templates for building custom agents.

Each agent follows the same pattern:
1. Receive a context slice from the Conductor (via the blackboard).
2. Use an LLM to perform its specialized task.
3. Write results back to the blackboard.
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
            "Break the objective into concrete, actionable steps. "
            "Output a JSON object with a 'steps' array, where each step has 'id', 'title', 'description', and 'output_field' (the workspace field it will produce). "
            "Respond in the same language as the objective."
        )
        user_prompt = f"Objective: {objective}"
        if constraints:
            user_prompt += f"\nConstraints: {', '.join(constraints)}"

        response = await llm_client.complete(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=1000,
        )

        try:
            plan = json.loads(_strip_fences(response))
        except json.JSONDecodeError:
            plan = {"steps": [{"id": 1, "title": "Execute task", "description": response, "output_field": "result"}]}

        board.write_workspace("plan", plan)
        return {"status": "ok"}

    # ------------------------------------------------------------------
    # Code Generator Agent
    # ------------------------------------------------------------------
    @specialist(
        name="code_generator",
        description="Generates code based on requirements, specifications, or a plan. "
                    "Writes the generated code to the 'code' workspace field.",
        input_fields=["plan", "requirements"],
        output_fields=["code"],
    )
    async def code_generator(context: dict[str, Any], board: Blackboard) -> dict[str, Any]:
        workspace = context.get("workspace", {})
        plan = workspace.get("plan", "")
        requirements = workspace.get("requirements", "")
        consensus = context.get("consensus")

        system_prompt = (
            "You are an expert code generator. Write clean, well-documented, production-quality code "
            "based on the given requirements or plan. Include proper error handling and type hints. "
            "Output ONLY the code, no explanations."
        )

        user_prompt = f"Objective: {context.get('objective', '')}\n"
        if plan:
            user_prompt += f"\nPlan:\n{json.dumps(plan, ensure_ascii=False, indent=2)}"
        if requirements:
            user_prompt += f"\nRequirements:\n{requirements}"
        if consensus and consensus.get("last_critique"):
            user_prompt += f"\n\nPrevious review feedback (please address these issues):\n{consensus['last_critique']}"

        response = await llm_client.complete(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=3000,
        )

        board.write_workspace("code", response)
        # Request consensus review
        board.start_consensus("code")
        return {"status": "ok"}

    # ------------------------------------------------------------------
    # Critic Agent (for Consensus Loop)
    # ------------------------------------------------------------------
    @specialist(
        name="critic",
        description="Reviews and critiques workspace artifacts for quality, correctness, and completeness. "
                    "Part of the consensus loop — invoked after another agent produces output.",
        input_fields=["code", "plan", "result"],
        output_fields=[],
    )
    async def critic(context: dict[str, Any], board: Blackboard) -> dict[str, Any]:
        if board.state.consensus is None:
            return {"status": "no_consensus_active"}

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

        response = await llm_client.complete(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=800,
        )

        try:
            review = json.loads(_strip_fences(response))
            verdict_str = review.get("verdict", "REJECTED").upper()
            verdict = ConsensusStatus.APPROVED if verdict_str == "APPROVED" else ConsensusStatus.REJECTED
            critique = review.get("critique", response)
        except json.JSONDecodeError:
            verdict = ConsensusStatus.REJECTED
            critique = response

        board.submit_review(
            reviewer_agent="critic",
            verdict=verdict,
            critique=critique,
        )

        return {"status": "ok", "verdict": verdict.value}

    # ------------------------------------------------------------------
    # Writer Agent
    # ------------------------------------------------------------------
    @specialist(
        name="writer",
        description="Generates written content such as documentation, reports, or summaries. "
                    "Writes output to the 'result' workspace field.",
        input_fields=["plan", "code", "requirements"],
        output_fields=["result"],
    )
    async def writer(context: dict[str, Any], board: Blackboard) -> dict[str, Any]:
        workspace = context.get("workspace", {})
        consensus = context.get("consensus")

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

        response = await llm_client.complete(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=3000,
        )

        board.write_workspace("result", response)
        board.start_consensus("result")
        return {"status": "ok"}

    # ------------------------------------------------------------------
    # Summarizer Agent
    # ------------------------------------------------------------------
    @specialist(
        name="summarizer",
        description="Produces a final summary of all completed work. "
                    "Should be invoked near the end of a task to consolidate results.",
        input_fields=["plan", "code", "result"],
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

        response = await llm_client.complete(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=1000,
        )

        board.write_workspace("final_summary", response)
        return {"status": "ok"}

    return [planner, code_generator, critic, writer, summarizer]


def _strip_fences(text: str) -> str:
    """Remove markdown code fences from LLM output."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    return text
