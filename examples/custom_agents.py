"""
Custom Agent Example — How to define and register your own specialist agents.

This example shows how to:
1. Define a custom agent using the @specialist decorator.
2. Register it alongside the built-in agents.
3. Run a task that leverages your custom agent.

Usage:
    export OPENAI_API_KEY="sk-..."
    python examples/custom_agents.py
"""

import asyncio
from typing import Any

from mals.agents.registry import AgentSpec, specialist
from mals.core.blackboard import Blackboard
from mals.core.engine import MALSEngine
from mals.llm.client import LLMClient


# ---------------------------------------------------------------------------
# Step 1: Define your custom agent using the @specialist decorator
# ---------------------------------------------------------------------------

@specialist(
    name="test_generator",
    description="Generates comprehensive unit tests for Python code. "
                "Should be invoked after code has been generated and approved.",
    input_fields=["code"],
    output_fields=["tests"],
)
async def test_generator(context: dict[str, Any], board: Blackboard) -> dict[str, Any]:
    """
    A custom agent that generates unit tests for code.

    Every agent receives:
    - context: A minimal slice of the blackboard (only relevant fields).
    - board: The Blackboard instance for writing results.
    """
    code = context.get("workspace", {}).get("code", "")
    objective = context.get("objective", "")

    # You can use any LLM client here. For simplicity, we create one inline.
    # In production, you'd inject this via dependency injection.
    llm = LLMClient()

    response = await llm.complete(
        system_prompt=(
            "You are a test engineer. Generate comprehensive pytest unit tests for the given code. "
            "Include edge cases, error cases, and happy path tests. "
            "Output ONLY the test code, no explanations."
        ),
        user_prompt=f"Objective: {objective}\n\nCode to test:\n\n{code}",
        max_tokens=2000,
    )

    # Write the result back to the blackboard
    board.write_workspace("tests", response)

    # Optionally request a consensus review for the tests
    board.start_consensus("tests")

    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Step 2: Run the engine with your custom agent
# ---------------------------------------------------------------------------

async def main() -> None:
    # Pass custom agents to the engine
    engine = MALSEngine(custom_agents=[test_generator])

    result = await engine.run(
        objective="Write a Python class for a thread-safe LRU cache with TTL support, "
                  "then generate comprehensive unit tests for it.",
        constraints=[
            "Must be thread-safe using threading locks",
            "Must support TTL (time-to-live) for cache entries",
            "Tests must use pytest",
        ],
    )

    print(f"\nStatus: {result['status']}")
    print(f"Steps: {result['steps']}")

    if "code" in result["workspace"]:
        print("\n--- Generated Code ---")
        print(result["workspace"]["code"])

    if "tests" in result["workspace"]:
        print("\n--- Generated Tests ---")
        print(result["workspace"]["tests"])


if __name__ == "__main__":
    asyncio.run(main())
