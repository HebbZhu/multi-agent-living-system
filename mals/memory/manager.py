"""
Memory Manager — Implements the hot/warm/cold tiered memory model.

The memory manager is responsible for:
1. Tracking which blackboard fields are "hot" (actively relevant).
2. Compressing completed work into "warm" summaries.
3. Archiving stale data to "cold" storage.
4. Generating the compact "dashboard view" that the Conductor reads.

This module should be invoked after every agent write-back to keep memory lean.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mals.core.models import BlackboardState
    from mals.llm.client import LLMClient

logger = logging.getLogger("mals.memory")


class MemoryManager:
    """Manages the three-tier memory lifecycle on the blackboard."""

    # Fields that should always remain in hot memory
    PERMANENT_HOT_FIELDS = {"objective", "global_status", "consensus"}

    def __init__(self, llm_client: "LLMClient | None" = None) -> None:
        self._llm = llm_client

    # ------------------------------------------------------------------
    # Dashboard View — the Conductor's "compressed window" into the board
    # ------------------------------------------------------------------

    def generate_dashboard(self, state: "BlackboardState") -> str:
        """
        Generate a compact dashboard view of the blackboard state.

        This is what the Conductor sees — typically ~200-400 tokens,
        regardless of how large the full blackboard is.
        """
        lines: list[str] = []

        # Objective & status
        lines.append(f"Objective: {state.objective}")
        lines.append(f"Status: {state.global_status.value}")

        # Workspace summary (one line per field)
        if state.workspace:
            workspace_lines: list[str] = []
            for key, value in state.workspace.items():
                if key in state.memory.warm:
                    # Use the warm summary instead of full content
                    workspace_lines.append(f"  {key}: [completed] {state.memory.warm[key]}")
                elif isinstance(value, str) and len(value) > 200:
                    workspace_lines.append(f"  {key}: {value[:150]}... ({len(value)} chars)")
                elif isinstance(value, dict):
                    workspace_lines.append(f"  {key}: {json.dumps(value, ensure_ascii=False)[:150]}")
                else:
                    workspace_lines.append(f"  {key}: {value}")
            lines.append("Workspace:")
            lines.extend(workspace_lines)

        # Active consensus
        if state.consensus:
            c = state.consensus
            lines.append(
                f"Consensus: target={c.target_field}, status={c.status.value}, "
                f"iteration={c.current_iteration}/{c.max_iterations}"
            )
            if c.review_history:
                last_review = c.review_history[-1]
                lines.append(f"  Last review by {last_review.reviewer_agent}: {last_review.critique[:100]}")

        # Active hypotheses (only proposed ones)
        proposed = [h for h in state.hypothesis_thread if h.status.value == "proposed"]
        if proposed:
            lines.append(f"Open hypotheses ({len(proposed)}):")
            for h in proposed[-3:]:  # Show at most 3 most recent
                lines.append(f"  [{h.id}] by {h.author_agent}: {h.content[:80]}")

        # Constraints
        if state.active_constraints:
            lines.append(f"Constraints: {', '.join(state.active_constraints)}")

        # Conductor notes
        if state.conductor_notes:
            lines.append(f"Notes: {state.conductor_notes}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Context Slicing — prepare minimal context for a specialist agent
    # ------------------------------------------------------------------

    def slice_context(
        self,
        state: "BlackboardState",
        relevant_fields: list[str],
        include_hypotheses: bool = False,
        include_consensus: bool = False,
    ) -> dict[str, Any]:
        """
        Extract a minimal context slice from the blackboard for a specialist agent.

        Instead of passing the entire blackboard (which could be thousands of tokens),
        this method extracts only the fields that the specialist needs.

        Args:
            state: The full blackboard state.
            relevant_fields: List of workspace field names the agent needs.
            include_hypotheses: Whether to include the hypothesis thread.
            include_consensus: Whether to include the consensus state.

        Returns:
            A dictionary containing only the relevant information.
        """
        context: dict[str, Any] = {
            "objective": state.objective,
            "global_status": state.global_status.value,
            "workspace": {},
        }

        # Extract only requested workspace fields
        for field in relevant_fields:
            if field in state.workspace:
                context["workspace"][field] = state.workspace[field]
            elif field in state.memory.warm:
                context["workspace"][f"{field}_summary"] = state.memory.warm[field]

        if include_hypotheses:
            context["hypotheses"] = [
                {"id": h.id, "content": h.content, "status": h.status.value, "author": h.author_agent}
                for h in state.hypothesis_thread
                if h.status.value == "proposed"
            ]

        if include_consensus and state.consensus:
            c = state.consensus
            context["consensus"] = {
                "target_field": c.target_field,
                "status": c.status.value,
                "iteration": c.current_iteration,
                "last_critique": c.review_history[-1].critique if c.review_history else None,
            }

        return context

    # ------------------------------------------------------------------
    # Memory Compression — move completed work from hot to warm
    # ------------------------------------------------------------------

    async def compress_to_warm(
        self,
        state: "BlackboardState",
        field_name: str,
    ) -> str:
        """
        Compress a workspace field into a warm summary.

        If an LLM client is available, uses it to generate an intelligent summary.
        Otherwise, falls back to a simple truncation strategy.

        Args:
            state: The blackboard state.
            field_name: The workspace field to compress.

        Returns:
            The generated summary string.
        """
        if field_name not in state.workspace:
            return ""

        content = state.workspace[field_name]
        content_str = json.dumps(content, ensure_ascii=False) if not isinstance(content, str) else content

        if self._llm and len(content_str) > 500:
            # Use LLM to generate an intelligent summary
            summary = await self._llm.complete(
                system_prompt="You are a concise summarizer. Summarize the following content in 1-2 sentences, preserving key facts and outcomes. Respond in the same language as the content.",
                user_prompt=f"Summarize this completed work artifact:\n\n{content_str[:3000]}",
                max_tokens=150,
            )
        else:
            # Simple truncation fallback
            summary = content_str[:200] + ("..." if len(content_str) > 200 else "")

        # Update memory tiers
        state.memory.warm[field_name] = summary
        if field_name in state.memory.hot:
            state.memory.hot.remove(field_name)

        logger.info("Compressed field '%s' to warm memory: %s", field_name, summary[:80])
        state.touch()
        return summary

    def mark_hot(self, state: "BlackboardState", field_name: str) -> None:
        """Mark a workspace field as hot (actively relevant)."""
        if field_name not in state.memory.hot:
            state.memory.hot.append(field_name)
            state.touch()

    def get_hot_fields(self, state: "BlackboardState") -> list[str]:
        """Return the list of currently hot workspace fields."""
        return list(state.memory.hot)
