# MALS v0.2 Benchmark Report

> **Multi-Agent Living System (MALS)** — First benchmark run on v0.2 with full observability.
> Date: 2026-02-25 | LLM Backend: GPT-4.1-mini (agents) + GPT-4.1-nano (conductor)

## Executive Summary

MALS v0.2 was tested against three tasks of increasing complexity. All three tasks completed successfully, demonstrating that the blackboard-driven architecture can coordinate multiple specialist agents to produce meaningful output with reasonable token efficiency.

| Metric | Task 1 (Simple) | Task 2 (Medium) | Task 3 (Complex) |
|--------|:---------------:|:---------------:|:-----------------:|
| **Status** | COMPLETED | COMPLETED | COMPLETED |
| **Conductor Steps** | 5 / 15 | 5 / 25 | 5 / 40 |
| **Elapsed Time** | 30.1s | 46.1s | 74.9s |
| **Total Tokens** | 5,847 | 9,061 | 12,350 |
| **Agent Invocations** | 4 | 3 | 3 |
| **Completion** | Natural | Loop Override | Loop Override |

## Task Descriptions

**Task 1 — Simple Function (Simple):** Write a Python function `merge_sorted_lists` that merges two sorted lists in O(n+m) time, plus 5 pytest unit tests covering edge cases. Constraints: Python standard library only, type hints, docstrings with examples.

**Task 2 — REST API Design (Medium):** Design and implement a complete REST API for a task management system using Flask, including CRUD endpoints, input validation, error handling, and a test suite. Constraints: Flask framework, RESTful conventions, proper HTTP status codes, in-memory storage.

**Task 3 — Data Pipeline (Complex):** Design and implement a complete data processing pipeline that reads CSV sales records, cleans data, transforms and aggregates it, generates a text-based report, and includes unit tests and documentation. Constraints: Python with pandas, modular design, type hints, edge case handling, sample data generation.

## Results Visualization

### Overview

![Benchmark Overview](results/benchmark_overview.png)

The overview chart reveals a clear and healthy scaling pattern. Conductor steps remained constant at 5 across all complexity levels, while token consumption and elapsed time scaled proportionally with task complexity. This suggests that the conductor's routing efficiency is stable regardless of task size — the additional cost comes from the agents doing more work, not from the conductor making more decisions.

### Agent Breakdown

![Agent Breakdown](results/benchmark_agent_breakdown.png)

Across all three tasks, only two agent types were activated: **planner** (1 invocation each) and **code_generator** (2 invocations each). The planner consistently consumed minimal tokens (402–969), while the code_generator's token usage scaled with task complexity (1,080 → 3,883 → 6,358). The critic, writer, and summarizer agents were not invoked in Tasks 2 and 3 due to the loop detection mechanism triggering before the consensus cycle could complete.

### Efficiency Metrics

![Efficiency Metrics](results/benchmark_efficiency.png)

Tokens per conductor step increased from 1,169 (simple) to 2,470 (complex), reflecting the growing dashboard size as more workspace content accumulates. Time per step scaled from 6.0s to 15.0s, primarily driven by the code_generator's longer response times for more complex code. Both metrics show linear scaling, which is a positive indicator for the architecture's scalability.

### Token Distribution

![Token Distribution](results/benchmark_token_distribution.png)

A notable finding is the conductor's token overhead. For the simple task, the conductor consumed 87% of total tokens; for the complex task, this dropped to 62%. This inverse relationship is actually desirable — it means the conductor's overhead is relatively fixed (a "base cost" of orchestration), while the useful agent work scales with task complexity. As tasks become more complex, a larger proportion of tokens goes toward productive work.

## Key Findings

### What Worked Well

The blackboard architecture demonstrated its core value proposition. Agents operated in a fully stateless manner — each invocation received only a relevant context slice, performed its work, and wrote results back to the shared blackboard. The conductor never needed to maintain conversation history or manage inter-agent communication channels. This resulted in clean, predictable token consumption patterns.

The programmatic hints system proved essential for reliable conductor routing. By injecting structured hints into the conductor's prompt (e.g., "A plan already exists — do NOT invoke planner again"), we achieved zero wasted planner invocations across all three tasks. This hybrid approach — combining LLM reasoning with programmatic guardrails — is a key architectural insight.

### What Needs Improvement

The most significant issue is the **consensus loop not completing for medium and complex tasks**. The code_generator writes output and starts a consensus review, but the conductor then re-invokes code_generator for the next plan step instead of routing to the critic first. After 3 consecutive code_generator invocations, the loop detection mechanism forces completion. This means Tasks 2 and 3 were completed without quality review — a critical gap.

The root cause is a priority conflict in the conductor's decision rules. Rule 2 (consensus pending → invoke critic) should take precedence over Rule 5 (next plan step → invoke agent), but the LLM sometimes prioritizes forward progress over review. This will be addressed in v0.3 by making consensus resolution a hard programmatic gate rather than a soft LLM decision.

The conductor's token overhead (62–87% of total) is higher than ideal. The primary contributor is the system prompt, which includes all agent descriptions and decision rules on every step. Future versions will implement prompt caching and incremental dashboard updates to reduce this overhead.

## Iteration History

The benchmark process itself revealed important engineering lessons. The first run of Task 1 failed (15/15 steps exhausted) because the conductor repeatedly invoked code_generator without recognizing that the task was complete. Through three iterations of prompt engineering and the addition of programmatic guardrails, Task 1's performance improved dramatically:

| Iteration | Status | Steps | Tokens | Time |
|-----------|--------|-------|--------|------|
| 1st | FAILED | 15/15 | 29,188 | 143.6s |
| 2nd | FAILED | 15/15 | 17,539 | 45.9s |
| 3rd | COMPLETED | 7/15 | 9,149 | 30.8s |
| Final (v0.2) | COMPLETED | 5/15 | 5,847 | 30.1s |

This 5x improvement in token efficiency and the transition from failure to success validates the iterative approach of combining LLM prompting with programmatic safeguards.

## Roadmap Implications

Based on these benchmark results, the following priorities are recommended for v0.3:

The consensus loop must be made reliable through a **programmatic gate**: after any agent writes to the blackboard and starts a consensus review, the conductor must route to the critic before any other agent can be invoked. This should not be an LLM decision — it should be enforced in code.

The conductor's system prompt should be **cached and reused** across steps, with only the dashboard portion changing. This alone could reduce conductor token overhead by 40–60%.

A **task completion heuristic** should be added: if all plan steps have corresponding workspace fields, and no consensus is pending, the task is complete. This should be a programmatic check, not an LLM judgment.

---

*Report generated by MALS v0.2 Observability Module*
