# MALS — Multi-Agent Living System

[中文](./README_CN.md) | [English](./README.md)

[![PyPI version](https://badge.fury.io/py/mals.svg)](https://badge.fury.io/py/mals)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/release/python-3110/)

**MALS (Multi-Agent Living System)** 是一个基于**黑板架构 (Blackboard Architecture)** 的多智能体协作框架，它用一个共享的、持久的、自组织的认知空间，取代了脆弱、混乱的智能体直接通信。

## 核心理念：告别部落困境

当前主流的多智能体系统（如 AutoGen, CrewAI）大多依赖于“对话驱动”的协作模式，智能体之间通过点对点消息传递来推进任务。这种模式在任务复杂时会迅速退化为“部落困境”：

- **上下文丢失**：每个智能体都试图在有限的上下文中维护对全局状态的理解，导致信息在传递中失真和遗忘。
- **通信开销巨大**：随着智能体数量增加，通信路径呈指数级增长，大量的 Token 被消耗在同步状态和重复沟通上。
- **错误放大**：一个智能体的微小错误或幻觉，会通过对话链迅速污染整个系统，导致集体跑偏。

**MALS 的核心理念是：智能体不应该直接对话。**

取而代之，所有协作都围绕一个中心化的、结构化的**动态黑板 (Dynamic Blackboard)** 进行。智能体像工匠一样，从黑板上领取任务和原材料，完成自己的部分，然后将产出放回黑板，供其他智能体使用。这种模式从根本上解决了部落困境。

## 架构概览

![MALS Architecture](docs/images/architecture_overview.png)

MALS 由四大核心组件构成：

1.  **动态黑板 (Dynamic Blackboard)**：系统的单一事实来源 (Single Source of Truth)。它是一个结构化的数据存储（内存或 Redis），包含了任务目标、工作空间、假设、共识状态和记忆。所有智能体的输入都来自黑板，输出也写回黑板。

2.  **指挥家 (Conductor)**：系统的“中枢神经”。它是一个轻量级的调度智能体，以极低的 Token 消耗持续执行“感知-思考-行动”循环：
    - **感知 (Sense)**：读取黑板的高度压缩“仪表盘”视图。
    - **思考 (Think)**：决定下一步激活哪个专家智能体。
    - **行动 (Act)**：调用选定的智能体，并为其提供完成任务所需的最小上下文切片。

3.  **专家智能体 (Specialist Agents)**：负责执行具体任务的领域专家（如 `code_generator`, `critic`, `planner`）。每个专家都遵循“单一职责原则”，只做自己最擅长的事。

4.  **记忆管理器 (Memory Manager)**：负责维护黑板的“认知负荷”。它通过三级记忆模型（热-温-冷）自动将完成的工作压缩为摘要，归档陈旧信息，确保指挥家和专家智能体的上下文窗口始终保持精简。

## 关键特性

- **黑板驱动**：彻底取代点对点通信，从根本上降低了系统的通信复杂度和 Token 消耗。
- **共识循环 (Consensus Loop)**：通过“写-批-改”的循环（如 `code_generator` 写代码，`critic` 审查），确保产出质量，有效抑制幻觉。
- **三级记忆管理**：自动化的记忆衰减机制，让系统能处理长期、复杂的任务，而不会被上下文窗口限制。
- **上下文切片器**：指挥家在调用专家时，会动态地从黑板上切分出完成任务所需的最小上下文，极大提升了效率。
- **插件化智能体**：使用简单的 `@specialist` 装饰器即可定义自己的专家智能体，轻松扩展系统能力。
- **开箱即用**：内置一组核心专家智能体，一行命令即可运行一个完整的多智能体协作任务。

## 快速开始

### 1. 安装

```bash
pip install mals
```

### 2. 设置 API Key

```bash
export OPENAI_API_KEY="sk-..."
```

MALS 支持任何与 OpenAI 兼容的 API 接口。你可以通过环境变量 `OPENAI_BASE_URL` 指定自定义端点（如本地模型、Azure 等）。

### 3. 运行你的第一个任务

```bash
mals run "Write a Python function that implements binary search. Include docstrings and tests."
```

系统将启动，指挥家会首先调用 `planner` 制定计划，然后 `code_generator` 写代码，`critic` 审查，最后 `test_generator` 生成测试用例，整个过程全自动完成。

## 使用示例

### 命令行

```bash
# 使用不同的模型
mals run "Create a FastAPI endpoint for user login" --model gpt-4.1-mini

# 增加约束
mals run "Design a database schema for a blog" -k "Use PostgreSQL syntax" -k "Include user, post, and comment tables"

# 保存结果到文件
mals run "Summarize the latest news about AI" -o results.json
```

### Python API

你也可以在 Python 代码中直接使用 MALS：

```python
import asyncio
from mals import MALSEngine

async def main():
    engine = MALSEngine()
    result = await engine.run(
        objective="Write a Python script to fetch and parse RSS feeds.",
        constraints=["Use the 'feedparser' library"]
    )
    print(result["workspace"]["code"])

if __name__ == "__main__":
    asyncio.run(main())
```

## 架构深入

### 指挥家循环

![Conductor Loop](docs/images/conductor_loop.png)

### 共识循环

![Consensus Loop](docs/images/consensus_loop.png)

### 记忆分级管理

![Memory Tiers](docs/images/memory_tiers.png)

## 贡献

我们欢迎任何形式的贡献！请查看 `CONTRIBUTING.md` 了解详情。

## 路线图

- **v0.2**: 可视化仪表盘与任务回放
- **v0.3**: 专家智能体插件市场
- **v0.4**: 跨任务记忆与知识图谱集成
- **v0.5**: 分布式部署与容错

## License

本项目采用 MIT License。
