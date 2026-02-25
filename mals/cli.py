"""
MALS CLI — Command-line interface for running multi-agent tasks.

Usage:
    mals run "Build a REST API with authentication"
    mals run "Write a Python script to analyze CSV data" --max-steps 30 --export ./output
    mals dashboard --recording output/task_recording.json --metrics output/task_metrics.json
"""

from __future__ import annotations

import asyncio
import json
import sys

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


@click.group()
@click.version_option(package_name="mals")
def main() -> None:
    """MALS — Multi-Agent Living System: Blackboard-based multi-agent collaboration."""
    pass


@main.command()
@click.argument("objective")
@click.option("--model", "-m", default=None, help="LLM model for specialist agents (default: gpt-4.1-mini)")
@click.option("--conductor-model", "-c", default=None, help="LLM model for the conductor (default: gpt-4.1-nano)")
@click.option("--max-steps", "-s", default=50, help="Maximum conductor loop steps (default: 50)")
@click.option("--backend", "-b", default="memory", type=click.Choice(["memory", "redis"]), help="Blackboard backend")
@click.option("--redis-url", default="redis://localhost:6379/0", help="Redis URL (only for redis backend)")
@click.option("--constraint", "-k", multiple=True, help="Add a constraint (can be repeated)")
@click.option("--output", "-o", default=None, help="Save results to a JSON file")
@click.option("--export", "-e", default=None, help="Export metrics and recording to a directory")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose (DEBUG) logging")
def run(
    objective: str,
    model: str | None,
    conductor_model: str | None,
    max_steps: int,
    backend: str,
    redis_url: str,
    constraint: tuple[str, ...],
    output: str | None,
    export: str | None,
    verbose: bool,
) -> None:
    """Run a multi-agent task with the given objective."""
    from mals.utils.config import MALSConfig

    # Build config
    config = MALSConfig.load()
    if model:
        config.llm.model = model
    if conductor_model:
        config.llm.conductor_model = conductor_model
    config.conductor.max_steps = max_steps
    config.blackboard.backend = backend
    config.blackboard.redis_url = redis_url
    if verbose:
        config.logging.level = "DEBUG"

    # Display task header
    console.print()
    console.print(Panel(
        f"[bold]{objective}[/bold]",
        title="[cyan]MALS Task[/cyan]",
        subtitle=f"model={config.llm.model} | conductor={config.llm.conductor_model} | max_steps={max_steps}",
        border_style="cyan",
    ))
    console.print()

    # Run the engine
    from mals.core.engine import MALSEngine

    engine = MALSEngine(config=config)
    constraints = list(constraint) if constraint else None

    try:
        result = asyncio.run(engine.run(
            objective,
            constraints=constraints,
            max_steps=max_steps,
            record=True,
            export_dir=export,
        ))
    except KeyboardInterrupt:
        console.print("\n[yellow]Task interrupted by user.[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[red]Task failed: {e}[/red]")
        sys.exit(1)

    # Display results
    _display_results(result)

    # Save to file if requested
    if output:
        with open(output, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)
        console.print(f"\n[green]Results saved to {output}[/green]")

    if export:
        console.print(f"\n[green]Metrics and recording exported to {export}/[/green]")
        console.print(f"[dim]View with: mals dashboard --dir {export}[/dim]")


@main.command()
@click.option("--dir", "-d", default=None, help="Directory containing exported metrics and recording files")
@click.option("--recording", "-r", default=None, help="Path to recording JSON file")
@click.option("--metrics", "-m", default=None, help="Path to metrics JSON file")
@click.option("--port", "-p", default=8765, help="Dashboard server port (default: 8765)")
@click.option("--host", "-h", default="0.0.0.0", help="Dashboard server host (default: 0.0.0.0)")
def dashboard(
    dir: str | None,
    recording: str | None,
    metrics: str | None,
    port: int,
    host: str,
) -> None:
    """Launch the web dashboard to visualize task execution."""
    import glob
    from pathlib import Path

    # Resolve file paths
    recording_file = recording
    metrics_file = metrics

    if dir:
        dir_path = Path(dir)
        if not recording_file:
            candidates = sorted(glob.glob(str(dir_path / "*_recording.json")))
            if candidates:
                recording_file = candidates[-1]  # Most recent
        if not metrics_file:
            candidates = sorted(glob.glob(str(dir_path / "*_metrics.json")))
            if candidates:
                metrics_file = candidates[-1]

    if not recording_file and not metrics_file:
        console.print("[red]Error: No data files found. Provide --dir, --recording, or --metrics.[/red]")
        sys.exit(1)

    console.print(Panel(
        f"Recording: {recording_file or 'N/A'}\nMetrics: {metrics_file or 'N/A'}",
        title="[cyan]MALS Dashboard[/cyan]",
        subtitle=f"http://{host}:{port}",
        border_style="cyan",
    ))

    from mals.observability.dashboard import create_dashboard_app

    app = create_dashboard_app(
        recording_file=recording_file,
        metrics_file=metrics_file,
    )

    try:
        import uvicorn
        uvicorn.run(app, host=host, port=port, log_level="warning")
    except ImportError:
        console.print("[red]Dashboard requires 'uvicorn'. Install with: pip install uvicorn[/red]")
        sys.exit(1)


@main.command()
def agents() -> None:
    """List all available specialist agents."""
    from mals.agents.builtins import create_builtin_agents
    from mals.llm.client import LLMClient

    # Create a dummy client just to get agent specs
    try:
        client = LLMClient()
        builtin = create_builtin_agents(client)
    except Exception:
        # If no API key, still show agent info
        builtin = []

    table = Table(title="Available Specialist Agents", show_lines=True)
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Description", style="white")
    table.add_column("Reads", style="green")
    table.add_column("Writes", style="yellow")

    for agent in builtin:
        table.add_row(
            agent.name,
            agent.description,
            ", ".join(agent.input_fields) or "-",
            ", ".join(agent.output_fields) or "-",
        )

    console.print()
    console.print(table)
    console.print()


@main.command()
def init() -> None:
    """Generate a sample mals.yaml configuration file."""
    sample_config = """# MALS Configuration
# See https://github.com/HebbZhu/multi-agent-living-system for documentation.

llm:
  # Model for specialist agents (any OpenAI-compatible model)
  model: "gpt-4.1-mini"
  # Model for the conductor (lightweight, fast model recommended)
  conductor_model: "gpt-4.1-nano"
  # API key (can also be set via OPENAI_API_KEY env var)
  # api_key: "sk-..."
  # Base URL for custom endpoints (e.g., Ollama, Azure, etc.)
  # base_url: "http://localhost:11434/v1"
  temperature: 0.3

blackboard:
  # Storage backend: "memory" (zero-dependency) or "redis" (persistent)
  backend: "memory"
  # Redis URL (only used when backend is "redis")
  # redis_url: "redis://localhost:6379/0"

conductor:
  # Maximum number of conductor loop iterations
  max_steps: 50
  # Maximum write-critique-revise cycles per artifact
  consensus_max_iterations: 3

logging:
  # Log level: DEBUG, INFO, WARNING, ERROR
  level: "INFO"
"""
    with open("mals.yaml", "w") as f:
        f.write(sample_config)
    console.print("[green]Created mals.yaml with default configuration.[/green]")


def _display_results(result: dict) -> None:
    """Display task results in a formatted table."""
    # Status
    status = result["status"]
    status_style = "green" if status == "COMPLETED" else "red"
    console.print(f"\n[{status_style}]Status: {status}[/{status_style}]")
    console.print(f"Steps: {result.get('steps', 0)}")

    # Token usage
    usage = result.get("token_usage", {})
    usage_table = Table(title="Token Usage", show_lines=True)
    usage_table.add_column("Component", style="cyan")
    usage_table.add_column("Input", justify="right")
    usage_table.add_column("Output", justify="right")

    conductor_usage = usage.get("conductor", {})
    agent_usage = usage.get("agents", {})
    usage_table.add_row(
        "Conductor",
        str(conductor_usage.get("input", 0)),
        str(conductor_usage.get("output", 0)),
    )
    usage_table.add_row(
        "Agents",
        str(agent_usage.get("input", 0)),
        str(agent_usage.get("output", 0)),
    )
    usage_table.add_row(
        "[bold]Total[/bold]",
        str(conductor_usage.get("input", 0) + agent_usage.get("input", 0)),
        str(conductor_usage.get("output", 0) + agent_usage.get("output", 0)),
    )
    console.print(usage_table)

    # Metrics summary
    metrics = result.get("metrics", {})
    if metrics:
        ts = metrics.get("task_summary", {})
        if ts:
            metrics_table = Table(title="Metrics Summary", show_lines=True)
            metrics_table.add_column("Metric", style="cyan")
            metrics_table.add_column("Value", justify="right")
            metrics_table.add_row("Total Steps", str(ts.get("total_steps", 0)))
            metrics_table.add_row("Agent Invocations", str(ts.get("total_agent_invocations", 0)))
            metrics_table.add_row("Elapsed Time", f"{ts.get('elapsed_time_s', 0):.1f}s")
            metrics_table.add_row("Total Tokens", str(ts.get("total_tokens", 0)))
            metrics_table.add_row("Memory Compressions", str(ts.get("memory_compressions", 0)))
            console.print(metrics_table)

    # Workspace output
    workspace = result.get("workspace", {})
    if workspace:
        console.print("\n[bold cyan]Workspace Output:[/bold cyan]")
        for key, value in workspace.items():
            content = str(value)
            if len(content) > 2000:
                content = content[:2000] + f"\n... ({len(content)} chars total)"
            console.print(Panel(content, title=f"[yellow]{key}[/yellow]", border_style="dim"))


if __name__ == "__main__":
    main()
