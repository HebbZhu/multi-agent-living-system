"""
MALS Dashboard — A lightweight web-based visualization for task replay and metrics.

Provides a single-file FastAPI server that serves:
- /              → Dashboard HTML page (embedded, no external dependencies)
- /api/metrics   → Current metrics as JSON
- /api/recording → Event recording as JSON
- /api/timeline  → Simplified timeline for visualization

The dashboard uses vanilla HTML/CSS/JS with Chart.js (loaded from CDN) for
visualization. No build step required.

Usage:
    # From code:
    from mals.observability.dashboard import create_dashboard_app
    app = create_dashboard_app(metrics, recorder)
    uvicorn.run(app, host="0.0.0.0", port=8765)

    # From CLI:
    mals dashboard --recording task_recording.json --metrics task_metrics.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("mals.observability.dashboard")


def create_dashboard_app(
    metrics_data: dict[str, Any] | None = None,
    recording_data: dict[str, Any] | None = None,
    metrics_file: str | Path | None = None,
    recording_file: str | Path | None = None,
) -> Any:
    """
    Create a FastAPI app for the MALS Dashboard.

    Can accept data directly (from a live run) or file paths (for replay).
    """
    try:
        from fastapi import FastAPI
        from fastapi.responses import HTMLResponse, JSONResponse
    except ImportError as e:
        raise ImportError(
            "Dashboard requires 'fastapi' and 'uvicorn'. "
            "Install with: pip install fastapi uvicorn"
        ) from e

    # Load from files if paths are provided
    if metrics_file and not metrics_data:
        with open(metrics_file, encoding="utf-8") as f:
            metrics_data = json.load(f)
    if recording_file and not recording_data:
        with open(recording_file, encoding="utf-8") as f:
            recording_data = json.load(f)

    app = FastAPI(title="MALS Dashboard", version="0.2.0")

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return _dashboard_html()

    @app.get("/api/metrics", response_class=JSONResponse)
    async def get_metrics():
        return metrics_data or {}

    @app.get("/api/recording", response_class=JSONResponse)
    async def get_recording():
        return recording_data or {}

    @app.get("/api/timeline", response_class=JSONResponse)
    async def get_timeline():
        if not recording_data:
            return []
        events = recording_data.get("events", [])
        timeline = []
        for e in events:
            timeline.append({
                "step": e.get("step", 0),
                "timestamp": e.get("timestamp", 0),
                "type": e.get("type", ""),
                "data": e.get("data", {}),
            })
        return timeline

    return app


def _dashboard_html() -> str:
    """Generate the full dashboard HTML page."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MALS Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
:root {
    --bg-primary: #0f1117;
    --bg-secondary: #1a1d29;
    --bg-card: #21253a;
    --text-primary: #e4e6f0;
    --text-secondary: #8b8fa3;
    --accent-blue: #4f8ff7;
    --accent-green: #3dd68c;
    --accent-purple: #a78bfa;
    --accent-orange: #f59e0b;
    --accent-red: #ef4444;
    --border: #2d3148;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: var(--bg-primary);
    color: var(--text-primary);
    min-height: 100vh;
}
.header {
    background: var(--bg-secondary);
    border-bottom: 1px solid var(--border);
    padding: 16px 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.header h1 { font-size: 20px; font-weight: 600; }
.header h1 span { color: var(--accent-blue); }
.header .task-info { color: var(--text-secondary); font-size: 13px; }
.container { max-width: 1400px; margin: 0 auto; padding: 24px; }

/* Summary Cards */
.summary-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 16px;
    margin-bottom: 24px;
}
.card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px;
}
.card .label { font-size: 12px; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.5px; }
.card .value { font-size: 28px; font-weight: 700; margin-top: 4px; }
.card .sub { font-size: 12px; color: var(--text-secondary); margin-top: 2px; }
.card.blue .value { color: var(--accent-blue); }
.card.green .value { color: var(--accent-green); }
.card.purple .value { color: var(--accent-purple); }
.card.orange .value { color: var(--accent-orange); }

/* Chart Section */
.charts-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
    margin-bottom: 24px;
}
.chart-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px;
}
.chart-card h3 { font-size: 14px; margin-bottom: 12px; color: var(--text-secondary); }

/* Agent Table */
.table-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 24px;
}
.table-card h3 { font-size: 14px; margin-bottom: 12px; color: var(--text-secondary); }
table { width: 100%; border-collapse: collapse; }
th, td { padding: 10px 12px; text-align: left; border-bottom: 1px solid var(--border); font-size: 13px; }
th { color: var(--text-secondary); font-weight: 500; font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; }
td { color: var(--text-primary); }
.badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 600;
}
.badge-success { background: rgba(61, 214, 140, 0.15); color: var(--accent-green); }
.badge-error { background: rgba(239, 68, 68, 0.15); color: var(--accent-red); }

/* Timeline */
.timeline-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 24px;
}
.timeline-card h3 { font-size: 14px; margin-bottom: 12px; color: var(--text-secondary); }
.timeline { max-height: 500px; overflow-y: auto; }
.timeline-item {
    display: flex;
    align-items: flex-start;
    padding: 8px 0;
    border-bottom: 1px solid var(--border);
    font-size: 13px;
}
.timeline-item:last-child { border-bottom: none; }
.timeline-step {
    min-width: 40px;
    color: var(--accent-blue);
    font-weight: 600;
    font-size: 12px;
}
.timeline-type {
    min-width: 140px;
    font-size: 11px;
    padding: 2px 6px;
    border-radius: 3px;
    margin-right: 8px;
}
.timeline-type.conductor_decide { background: rgba(79, 143, 247, 0.15); color: var(--accent-blue); }
.timeline-type.agent_start { background: rgba(167, 139, 250, 0.15); color: var(--accent-purple); }
.timeline-type.agent_end { background: rgba(61, 214, 140, 0.15); color: var(--accent-green); }
.timeline-type.consensus_review { background: rgba(245, 158, 11, 0.15); color: var(--accent-orange); }
.timeline-type.status_change { background: rgba(239, 68, 68, 0.15); color: var(--accent-red); }
.timeline-type.error { background: rgba(239, 68, 68, 0.3); color: var(--accent-red); }
.timeline-summary { color: var(--text-primary); flex: 1; }

.loading { text-align: center; padding: 40px; color: var(--text-secondary); }

@media (max-width: 768px) {
    .charts-grid { grid-template-columns: 1fr; }
    .summary-grid { grid-template-columns: repeat(2, 1fr); }
}
</style>
</head>
<body>
<div class="header">
    <h1><span>MALS</span> Dashboard</h1>
    <div class="task-info" id="taskInfo">Loading...</div>
</div>
<div class="container">
    <div id="content"><div class="loading">Loading data...</div></div>
</div>

<script>
async function loadData() {
    const [metricsRes, timelineRes, recordingRes] = await Promise.all([
        fetch('/api/metrics').then(r => r.json()),
        fetch('/api/timeline').then(r => r.json()),
        fetch('/api/recording').then(r => r.json()),
    ]);
    return { metrics: metricsRes, timeline: timelineRes, recording: recordingRes };
}

function formatNumber(n) {
    if (n >= 1000000) return (n / 1000000).toFixed(1) + 'M';
    if (n >= 1000) return (n / 1000).toFixed(1) + 'K';
    return n.toString();
}

function renderDashboard(data) {
    const { metrics, timeline, recording } = data;
    const ts = metrics.task_summary || {};
    const agents = metrics.agents || {};
    const consensus = metrics.consensus || {};
    const conductor = metrics.conductor || {};

    // Task info
    document.getElementById('taskInfo').textContent =
        `Task: ${recording.objective || 'N/A'} | ID: ${recording.task_id || 'N/A'}`;

    let html = '';

    // Summary cards
    html += `<div class="summary-grid">
        <div class="card blue">
            <div class="label">Total Steps</div>
            <div class="value">${ts.total_steps || 0}</div>
            <div class="sub">${ts.total_agent_invocations || 0} agent invocations</div>
        </div>
        <div class="card green">
            <div class="label">Total Tokens</div>
            <div class="value">${formatNumber(ts.total_tokens || 0)}</div>
            <div class="sub">Conductor: ${formatNumber(ts.conductor_tokens || 0)} | Agents: ${formatNumber(ts.agent_tokens || 0)}</div>
        </div>
        <div class="card purple">
            <div class="label">Elapsed Time</div>
            <div class="value">${(ts.elapsed_time_s || 0).toFixed(1)}s</div>
            <div class="sub">${ts.memory_compressions || 0} memory compressions</div>
        </div>
        <div class="card orange">
            <div class="label">Consensus</div>
            <div class="value">${consensus.total_cycles || 0}</div>
            <div class="sub">First-try: ${((consensus.first_try_approval_rate || 0) * 100).toFixed(0)}% | Avg iter: ${(consensus.avg_iterations_per_cycle || 0).toFixed(1)}</div>
        </div>
    </div>`;

    // Charts
    html += `<div class="charts-grid">
        <div class="chart-card"><h3>Token Distribution by Agent</h3><canvas id="tokenChart"></canvas></div>
        <div class="chart-card"><h3>Agent Latency (avg seconds)</h3><canvas id="latencyChart"></canvas></div>
        <div class="chart-card"><h3>Conductor Routing Distribution</h3><canvas id="routingChart"></canvas></div>
        <div class="chart-card"><h3>Conductor Decision Types</h3><canvas id="decisionChart"></canvas></div>
    </div>`;

    // Agent table
    const agentNames = Object.keys(agents);
    html += `<div class="table-card"><h3>Per-Agent Breakdown</h3><table>
        <thead><tr><th>Agent</th><th>Calls</th><th>Tokens</th><th>Avg Latency</th><th>P95 Latency</th><th>Success Rate</th></tr></thead>
        <tbody>`;
    for (const name of agentNames) {
        const a = agents[name];
        const badge = a.success_rate >= 0.9 ? 'badge-success' : 'badge-error';
        html += `<tr>
            <td><strong>${a.name}</strong></td>
            <td>${a.invocation_count}</td>
            <td>${formatNumber(a.total_tokens)}</td>
            <td>${a.avg_latency_s.toFixed(2)}s</td>
            <td>${a.p95_latency_s.toFixed(2)}s</td>
            <td><span class="badge ${badge}">${(a.success_rate * 100).toFixed(0)}%</span></td>
        </tr>`;
    }
    html += `</tbody></table></div>`;

    // Timeline
    html += `<div class="timeline-card"><h3>Event Timeline (${timeline.length} events)</h3><div class="timeline">`;
    for (const item of timeline) {
        const typeClass = item.type.replace(/[^a-z_]/g, '');
        const summary = _eventSummary(item);
        html += `<div class="timeline-item">
            <div class="timeline-step">S${item.step}</div>
            <div class="timeline-type ${typeClass}">${item.type}</div>
            <div class="timeline-summary">${summary}</div>
        </div>`;
    }
    html += `</div></div>`;

    document.getElementById('content').innerHTML = html;

    // Render charts
    _renderTokenChart(agents);
    _renderLatencyChart(agents);
    _renderRoutingChart(conductor.routing_counts || {});
    _renderDecisionChart(conductor.decision_counts || {});
}

function _eventSummary(item) {
    const d = item.data || {};
    switch (item.type) {
        case 'task_start': return `Task started: ${(d.objective || '').substring(0, 60)}`;
        case 'task_end': return `Task ended: ${d.status || '?'}`;
        case 'status_change': return `${d.from || '?'} &rarr; ${d.to || '?'} (${d.reason || ''})`;
        case 'conductor_decide': return `${d.action || '?'}${d.agent_name ? ' → ' + d.agent_name : ''}: ${d.reasoning || ''}`;
        case 'agent_start': return `Started: ${d.agent_name || '?'}`;
        case 'agent_end': return `Finished: ${d.agent_name || '?'} (${d.status}, ${(d.latency_s || 0).toFixed(1)}s, ${(d.input_tokens||0)+(d.output_tokens||0)} tokens)`;
        case 'consensus_review': return `${d.reviewer || '?'}: ${d.verdict || '?'} — ${(d.critique || '').substring(0, 80)}`;
        case 'consensus_end': return `${d.outcome || '?'} for ${d.target_field || '?'} (${d.iterations} iter)`;
        case 'error': return `${d.source || '?'}: ${(d.error || '').substring(0, 80)}`;
        default: return JSON.stringify(d).substring(0, 100);
    }
}

const COLORS = ['#4f8ff7', '#3dd68c', '#a78bfa', '#f59e0b', '#ef4444', '#ec4899', '#06b6d4'];

function _renderTokenChart(agents) {
    const names = Object.keys(agents);
    new Chart(document.getElementById('tokenChart'), {
        type: 'bar',
        data: {
            labels: names,
            datasets: [
                { label: 'Input Tokens', data: names.map(n => agents[n].total_input_tokens), backgroundColor: '#4f8ff7' },
                { label: 'Output Tokens', data: names.map(n => agents[n].total_output_tokens), backgroundColor: '#3dd68c' },
            ]
        },
        options: { responsive: true, scales: { x: { ticks: { color: '#8b8fa3' } }, y: { ticks: { color: '#8b8fa3' } } }, plugins: { legend: { labels: { color: '#e4e6f0' } } } }
    });
}

function _renderLatencyChart(agents) {
    const names = Object.keys(agents);
    new Chart(document.getElementById('latencyChart'), {
        type: 'bar',
        data: {
            labels: names,
            datasets: [{ label: 'Avg Latency (s)', data: names.map(n => agents[n].avg_latency_s), backgroundColor: '#a78bfa' }]
        },
        options: { responsive: true, scales: { x: { ticks: { color: '#8b8fa3' } }, y: { ticks: { color: '#8b8fa3' } } }, plugins: { legend: { labels: { color: '#e4e6f0' } } } }
    });
}

function _renderRoutingChart(routing) {
    const labels = Object.keys(routing);
    new Chart(document.getElementById('routingChart'), {
        type: 'doughnut',
        data: { labels, datasets: [{ data: labels.map(l => routing[l]), backgroundColor: COLORS }] },
        options: { responsive: true, plugins: { legend: { labels: { color: '#e4e6f0' } } } }
    });
}

function _renderDecisionChart(decisions) {
    const labels = Object.keys(decisions);
    new Chart(document.getElementById('decisionChart'), {
        type: 'doughnut',
        data: { labels, datasets: [{ data: labels.map(l => decisions[l]), backgroundColor: COLORS.slice(2) }] },
        options: { responsive: true, plugins: { legend: { labels: { color: '#e4e6f0' } } } }
    });
}

loadData().then(renderDashboard).catch(err => {
    document.getElementById('content').innerHTML = `<div class="loading">Error loading data: ${err.message}</div>`;
});
</script>
</body>
</html>"""
