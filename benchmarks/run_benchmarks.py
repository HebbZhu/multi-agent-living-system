"""
MALS Benchmark Suite — 3 tasks of increasing complexity.

Task 1 (Simple):   Single-function code generation with tests
Task 2 (Medium):   Multi-file REST API design with review cycle
Task 3 (Complex):  Full project planning → implementation → testing → documentation

Each task is run through the MALS engine with full observability enabled.
Results (metrics, recordings, workspace outputs) are exported to benchmarks/results/.

Usage:
    python benchmarks/run_benchmarks.py              # Run all 3 tasks
    python benchmarks/run_benchmarks.py --task 1     # Run only task 1
    python benchmarks/run_benchmarks.py --task 2     # Run only task 2
    python benchmarks/run_benchmarks.py --task 3     # Run only task 3
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

# Ensure project root is on path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from mals.core.engine import MALSEngine
from mals.utils.config import MALSConfig


# ============================================================================
# Benchmark Task Definitions
# ============================================================================

TASKS = {
    1: {
        "name": "simple_function",
        "complexity": "Simple",
        "objective": (
            "Write a Python function called `merge_sorted_lists` that takes two sorted lists "
            "of integers and returns a single sorted list containing all elements from both lists. "
            "The function should run in O(n+m) time complexity. "
            "Also write 5 unit tests using pytest to verify correctness, including edge cases "
            "like empty lists and lists with duplicate values."
        ),
        "constraints": [
            "Use only Python standard library",
            "Include type hints",
            "Include a docstring with examples",
        ],
        "max_steps": 15,
        "description": (
            "A simple single-function task that tests basic code generation and review. "
            "Expects: 1 planning step, 1 code generation, 1 review cycle, completion."
        ),
    },
    2: {
        "name": "rest_api_design",
        "complexity": "Medium",
        "objective": (
            "Design a REST API for a simple task management system (todo app). "
            "Provide the following deliverables:\n"
            "1. API specification: list all endpoints (method, path, request/response schema)\n"
            "2. Data model: define the Task entity with fields and types\n"
            "3. Implementation: write a complete Flask application with all endpoints\n"
            "4. Error handling: proper HTTP status codes and error response format\n"
            "The API should support: create task, list tasks, get task by ID, update task, delete task, "
            "and mark task as complete."
        ),
        "constraints": [
            "Use Flask framework",
            "Use in-memory storage (dict) for simplicity",
            "Follow RESTful conventions",
            "Include input validation",
            "Return JSON responses with consistent format",
        ],
        "max_steps": 25,
        "description": (
            "A medium-complexity task requiring multi-step planning, code generation across "
            "multiple concerns (routing, models, validation), and quality review. "
            "Tests the conductor's ability to sequence work and the consensus loop's effectiveness."
        ),
    },
    3: {
        "name": "data_pipeline",
        "complexity": "Complex",
        "objective": (
            "Design and implement a complete data processing pipeline that:\n"
            "1. Reads CSV data containing sales records (date, product, quantity, price, region)\n"
            "2. Cleans the data: handle missing values, remove duplicates, validate formats\n"
            "3. Transforms: calculate total revenue per record, add month/quarter columns\n"
            "4. Aggregates: compute summary statistics by product, by region, and by quarter\n"
            "5. Generates a text-based report with key findings and top performers\n"
            "6. Includes comprehensive error handling and logging\n"
            "7. Write unit tests for each pipeline stage\n"
            "8. Write documentation explaining the pipeline architecture and usage"
        ),
        "constraints": [
            "Use Python with pandas for data processing",
            "Modular design: each pipeline stage should be a separate function",
            "Include type hints throughout",
            "Handle edge cases: empty files, malformed rows, negative values",
            "Generate sample CSV data for testing",
            "Follow clean code principles",
        ],
        "max_steps": 40,
        "description": (
            "A complex multi-stage task that requires planning, implementation of multiple "
            "interconnected modules, testing, and documentation. Tests the full MALS pipeline "
            "including multiple agent invocations, consensus cycles, and memory management."
        ),
    },
}


# ============================================================================
# Benchmark Runner
# ============================================================================

async def run_single_benchmark(task_id: int, export_base: Path) -> dict:
    """Run a single benchmark task and return results."""
    task = TASKS[task_id]
    export_dir = export_base / task["name"]
    export_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  BENCHMARK {task_id}: {task['name']} ({task['complexity']})")
    print(f"{'='*70}")
    print(f"  Objective: {task['objective'][:100]}...")
    print(f"  Max Steps: {task['max_steps']}")
    print(f"  Export Dir: {export_dir}")
    print(f"{'='*70}\n")

    # Configure engine
    config = MALSConfig.load()
    config.conductor.max_steps = task["max_steps"]

    engine = MALSEngine(config=config)

    start_time = time.time()

    try:
        result = await engine.run(
            objective=task["objective"],
            constraints=task["constraints"],
            max_steps=task["max_steps"],
            record=True,
            export_dir=str(export_dir),
        )
    except Exception as e:
        result = {
            "status": "ERROR",
            "error": str(e),
            "steps": 0,
            "token_usage": {"total": 0},
        }

    elapsed = time.time() - start_time

    # Enrich result with benchmark metadata
    benchmark_result = {
        "benchmark_id": task_id,
        "task_name": task["name"],
        "complexity": task["complexity"],
        "objective": task["objective"],
        "constraints": task["constraints"],
        "max_steps_allowed": task["max_steps"],
        "actual_steps": result.get("steps", 0),
        "status": result.get("status", "ERROR"),
        "total_elapsed_s": round(elapsed, 2),
        "token_usage": result.get("token_usage", {}),
        "metrics": result.get("metrics", {}),
        "workspace_keys": list(result.get("workspace", {}).keys()),
    }

    # Save benchmark result
    result_file = export_dir / "benchmark_result.json"
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(benchmark_result, f, indent=2, ensure_ascii=False, default=str)

    # Save full workspace output
    workspace_file = export_dir / "workspace_output.json"
    with open(workspace_file, "w", encoding="utf-8") as f:
        json.dump(result.get("workspace", {}), f, indent=2, ensure_ascii=False, default=str)

    # Print summary
    print(f"\n--- Benchmark {task_id} Summary ---")
    print(f"  Status:       {benchmark_result['status']}")
    print(f"  Steps:        {benchmark_result['actual_steps']} / {benchmark_result['max_steps_allowed']}")
    print(f"  Elapsed:      {benchmark_result['total_elapsed_s']:.1f}s")
    total_tokens = result.get("token_usage", {}).get("total", 0)
    print(f"  Total Tokens: {total_tokens:,}")
    print(f"  Output Keys:  {benchmark_result['workspace_keys']}")
    print()

    return benchmark_result


async def run_all_benchmarks(task_ids: list[int] | None = None) -> list[dict]:
    """Run all (or selected) benchmark tasks."""
    export_base = Path(__file__).parent / "results"
    export_base.mkdir(parents=True, exist_ok=True)

    ids_to_run = task_ids or [1, 2, 3]
    results = []

    for task_id in ids_to_run:
        if task_id not in TASKS:
            print(f"Warning: Task {task_id} not found, skipping.")
            continue
        result = await run_single_benchmark(task_id, export_base)
        results.append(result)

    # Save combined results
    combined_file = export_base / "all_benchmarks.json"
    with open(combined_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    # Print comparison table
    print(f"\n{'='*70}")
    print("  BENCHMARK COMPARISON")
    print(f"{'='*70}")
    print(f"  {'Task':<20} {'Status':<12} {'Steps':<10} {'Tokens':<12} {'Time':<10}")
    print(f"  {'-'*64}")
    for r in results:
        total_tokens = r.get("token_usage", {}).get("total", 0)
        print(
            f"  {r['task_name']:<20} "
            f"{r['status']:<12} "
            f"{r['actual_steps']:<10} "
            f"{total_tokens:<12,} "
            f"{r['total_elapsed_s']:<10.1f}s"
        )
    print(f"{'='*70}\n")
    print(f"Results exported to: {export_base}/")

    return results


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="MALS Benchmark Suite")
    parser.add_argument("--task", "-t", type=int, default=None, help="Run a specific task (1, 2, or 3)")
    args = parser.parse_args()

    task_ids = [args.task] if args.task else None
    asyncio.run(run_all_benchmarks(task_ids))


if __name__ == "__main__":
    main()
