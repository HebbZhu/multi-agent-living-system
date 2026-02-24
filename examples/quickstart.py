"""
MALS Quickstart Example — Your first multi-agent task.

This example demonstrates the core MALS workflow:
1. The Conductor reads the objective and invokes the Planner.
2. The Planner creates a structured execution plan.
3. The Conductor routes work to the Code Generator.
4. The Critic reviews the output (consensus loop).
5. If rejected, the Code Generator revises; if approved, the task completes.

Usage:
    export OPENAI_API_KEY="sk-..."
    python examples/quickstart.py
"""

import asyncio

from mals.core.engine import MALSEngine


async def main() -> None:
    # Create the engine with default configuration.
    # It will auto-detect OPENAI_API_KEY from environment variables.
    engine = MALSEngine()

    # Run a multi-agent task
    result = await engine.run(
        objective="Write a Python function that implements binary search on a sorted list. "
                  "Include type hints, docstring, and edge case handling.",
        constraints=[
            "Must be pure Python, no external dependencies",
            "Must handle empty lists and single-element lists",
        ],
    )

    # Print the results
    print(f"\nStatus: {result['status']}")
    print(f"Steps: {result['steps']}")
    print(f"Total tokens: {result['token_usage']['total']}")

    # Print the generated code
    if "code" in result["workspace"]:
        print("\n--- Generated Code ---")
        print(result["workspace"]["code"])

    if "final_summary" in result["workspace"]:
        print("\n--- Summary ---")
        print(result["workspace"]["final_summary"])


if __name__ == "__main__":
    asyncio.run(main())
