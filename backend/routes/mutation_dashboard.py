"""
Enhanced Mutation Dashboard Routes
=================================

Provides visual dashboard for monitoring AI strategy mutations with full metadata.
"""

import json
import logging
import os
import sqlite3

from fastapi import APIRouter
from fastapi.responses import HTMLResponse, JSONResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/mutation", tags=["mutation-dashboard"])


@router.get("/dashboard", response_class=HTMLResponse)
async def mutation_dashboard():
    """Enhanced mutation dashboard page with full metadata"""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>ðŸ§¬ Ultimate AI Strategy Evolution Dashboard</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }
            .container {
                max-width: 1600px;
                margin: 0 auto;
                background: white;
                border-radius: 15px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
                overflow: hidden;
            }
            .header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                text-align: center;
            }
            .header h1 {
                margin: 0;
                font-size: 2.5em;
                font-weight: 300;
            }
            .header p {
                margin: 10px 0 0 0;
                opacity: 0.9;
                font-size: 1.1em;
            }
            .stats-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                padding: 30px;
                background: #f8f9fa;
            }
            .stat-card {
                background: white;
                padding: 25px;
                border-radius: 10px;
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
                text-align: center;
                transition: transform 0.3s ease;
            }
            .stat-card:hover {
                transform: translateY(-5px);
            }
            .stat-value {
                font-size: 2.5em;
                font-weight: bold;
                margin: 10px 0;
            }
            .stat-label {
                color: #666;
                font-size: 0.9em;
                text-transform: uppercase;
                letter-spacing: 1px;
            }
            .profit { color: #28a745; }
            .loss { color: #dc3545; }
            .neutral { color: #6c757d; }
            .charts-section {
                padding: 30px;
            }
            .chart-container {
                background: white;
                border-radius: 10px;
                padding: 20px;
                margin-bottom: 30px;
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            }
            .chart-title {
                font-size: 1.5em;
                margin-bottom: 20px;
                color: #333;
                text-align: center;
            }
            .mutations-table {
                padding: 30px;
            }
            .table-container {
                background: white;
                border-radius: 10px;
                overflow: hidden;
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            }
            table {
                width: 100%;
                border-collapse: collapse;
            }
            th, td {
                padding: 15px;
                text-align: left;
                border-bottom: 1px solid #eee;
            }
            th {
                background: #f8f9fa;
                font-weight: 600;
                color: #333;
            }
            tr:hover {
                background: #f8f9fa;
            }
            .status-promoted {
                color: #28a745;
                font-weight: bold;
            }
            .status-rejected {
                color: #dc3545;
                font-weight: bold;
            }
            .status-live {
                color: #007bff;
                font-weight: bold;
                background: #e3f2fd;
            }
            .controls {
                padding: 20px 30px;
                background: #f8f9fa;
                border-top: 1px solid #eee;
                display: flex;
                gap: 15px;
                flex-wrap: wrap;
            }
            .btn {
                padding: 10px 20px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-weight: 500;
                transition: all 0.3s ease;
            }
            .btn-primary {
                background: #007bff;
                color: white;
            }
            .btn-primary:hover {
                background: #0056b3;
            }
            .btn-success {
                background: #28a745;
                color: white;
            }
            .btn-success:hover {
                background: #1e7e34;
            }
            .btn-danger {
                background: #dc3545;
                color: white;
            }
            .btn-danger:hover {
                background: #c82333;
            }
            .refresh-info {
                color: #666;
                font-size: 0.9em;
                margin-left: auto;
            }
            .metadata-section {
                padding: 20px;
                background: #f8f9fa;
                border-top: 1px solid #eee;
            }
            .metadata-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
            }
            .metadata-card {
                background: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }
            .metadata-title {
                font-weight: bold;
                color: #333;
                margin-bottom: 10px;
            }
            .description {
                font-style: italic;
                color: #666;
                margin-top: 10px;
                padding: 10px;
                background: #f8f9fa;
                border-radius: 5px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>ðŸ§¬ Ultimate AI Strategy Evolution Dashboard</h1>
                <p>Real-time monitoring of your self-evolving, version-controlled trading AI</p>
            </div>

            <div class="stats-grid" id="stats-grid">
                <!-- Stats will be loaded here -->
            </div>

            <div class="charts-section">
                <div class="chart-container">
                    <div class="chart-title">ðŸ“ˆ Evolution Profit Over Time</div>
                    <canvas id="profitChart" width="400" height="200"></canvas>
                </div>

                <div class="chart-container">
                    <div class="chart-title">ðŸŽ¯ Promotion Success Rate</div>
                    <canvas id="successChart" width="400" height="200"></canvas>
                </div>
            </div>

            <div class="mutations-table">
                <div class="table-container">
                    <table id="mutations-table">
                        <thead>
                            <tr>
                                <th>Timestamp</th>
                                <th>Strategy Name</th>
                                <th>Type</th>
                                <th>Creator</th>
                                <th>Version</th>
                                <th>Profit (%)</th>
                                <th>Win Rate</th>
                                <th>Trades</th>
                                <th>Status</th>
                            </tr>
                        </thead>
                        <tbody id="mutations-tbody">
                            <!-- Mutations will be loaded here -->
                        </tbody>
                    </table>
                </div>
            </div>

            <div class="metadata-section">
                <h3>ðŸ“‹ Strategy Metadata</h3>
                <div class="metadata-grid" id="metadata-grid">
                    <!-- Metadata will be loaded here -->
                </div>
            </div>

            <div class="controls">
                <button class="btn btn-primary" onclick="refreshData()">ðŸ”„ Refresh Data</button>
                <button class="btn btn-success" onclick="startMutationCycle()">ðŸš€ Start Evolution</button>
                <button class="btn btn-danger" onclick="stopMutationCycle()">â¹ï¸ Stop Evolution</button>
                <button class="btn btn-primary" onclick="viewLeaderboard()">ðŸ† View Leaderboard</button>
                <div class="refresh-info">Auto-refresh every 30 seconds</div>
            </div>
        </div>

        <script>
            let profitChart, successChart;

            // Initialize charts
            function initCharts() {
                const profitCtx = document.getElementById('profitChart').getContext('2d');
                profitChart = new Chart(profitCtx, {
                    type: 'line',
                    data: {
                        labels: [],
                        datasets: [{
                            label: 'Profit (%)',
                            data: [],
                            borderColor: '#28a745',
                            backgroundColor: 'rgba(40, 167, 69, 0.1)',
                            tension: 0.4
                        }]
                    },
                    options: {
                        responsive: true,
                        scales: {
                            y: {
                                beginAtZero: true
                            }
                        }
                    }
                });

                const successCtx = document.getElementById('successChart').getContext('2d');
                successChart = new Chart(successCtx, {
                    type: 'doughnut',
                    data: {
                        labels: ['Promoted', 'Rejected'],
                        datasets: [{
                            data: [0, 0],
                            backgroundColor: ['#28a745', '#dc3545']
                        }]
                    },
                    options: {
                        responsive: true
                    }
                });
            }

            // Load dashboard data
            async function loadDashboardData() {
                try {
                    const response = await fetch('/mutation/api/stats');
                    const data = await response.json();

                    updateStats(data.stats);
                    updateCharts(data.chart_data);
                    updateMutationsTable(data.recent_mutations);
                    updateMetadata(data.metadata);

                } catch (error) {
                    console.error('Error loading dashboard data:', error);
                }
            }

            // Update statistics cards
            function updateStats(stats) {
                const statsGrid = document.getElementById('stats-grid');
                statsGrid.innerHTML = `
                    <div class="stat-card">
                        <div class="stat-label">Total Evolutions</div>
                        <div class="stat-value neutral">${stats.total_mutations}</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Promoted Strategies</div>
                        <div class="stat-value profit">${stats.promoted_mutations}</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Success Rate</div>
                        <div class="stat-value ${stats.promotion_rate > 0.2 ? 'profit' : 'neutral'}">${(stats.promotion_rate * 100).toFixed(1)}%</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Average Profit</div>
                        <div class="stat-value ${stats.average_profit > 0 ? 'profit' : 'loss'}">${stats.average_profit.toFixed(2)}%</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Active Sessions</div>
                        <div class="stat-value neutral">${stats.active_sessions}</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">System Status</div>
                        <div class="stat-value ${stats.is_running ? 'profit' : 'loss'}">${stats.is_running ? 'ðŸŸ¢ Evolving' : 'ðŸ”´ Stopped'}</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Current Version</div>
                        <div class="stat-value neutral">${stats.current_version || 'v1.0.0'}</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Live Strategy</div>
                        <div class="stat-value neutral">${stats.live_strategy || 'None'}</div>
                    </div>
                `;
            }

            // Update charts
            function updateCharts(chartData) {
                // Update profit chart
                profitChart.data.labels = chartData.profit_labels;
                profitChart.data.datasets[0].data = chartData.profit_data;
                profitChart.update();

                // Update success chart
                successChart.data.datasets[0].data = [chartData.promoted_count, chartData.rejected_count];
                successChart.update();
            }

            // Update mutations table
            function updateMutationsTable(mutations) {
                const tbody = document.getElementById('mutations-tbody');
                tbody.innerHTML = mutations.map(mutation => {
                    const statusClass = mutation.promoted ? 'status-promoted' : 'status-rejected';
                    const statusText = mutation.promoted ? 'âœ… Promoted' : 'âŒ Rejected';
                    const isLive = mutation.is_live ? 'status-live' : '';
                    const liveText = mutation.is_live ? ' (LIVE)' : '';

                    return `
                        <tr class="${isLive}">
                            <td>${mutation.timestamp}</td>
                            <td>${mutation.strategy_name}</td>
                            <td>${mutation.strategy_type}</td>
                            <td>${mutation.creator || 'Unknown'}</td>
                            <td>${mutation.version || 'v1.0.0'}</td>
                            <td class="${mutation.profit > 0 ? 'profit' : 'loss'}">${mutation.profit.toFixed(2)}%</td>
                            <td>${(mutation.win_rate * 100).toFixed(1)}%</td>
                            <td>${mutation.num_trades}</td>
                            <td class="${statusClass}">${statusText}${liveText}</td>
                        </tr>
                    `;
                }).join('');
            }

            // Update metadata section
            function updateMetadata(metadata) {
                const metadataGrid = document.getElementById('metadata-grid');
                if (!metadata || metadata.length === 0) {
                    metadataGrid.innerHTML = '<div class="metadata-card"><div class="metadata-title">No metadata available</div></div>';
                    return;
                }

                metadataGrid.innerHTML = metadata.map(item => `
                    <div class="metadata-card">
                        <div class="metadata-title">${item.name}</div>
                        <div><strong>Creator:</strong> ${item.creator}</div>
                        <div><strong>Version:</strong> ${item.version}</div>
                        <div><strong>Parent:</strong> ${item.parent || 'None'}</div>
                        <div><strong>Created:</strong> ${item.created_at}</div>
                        <div class="description">${item.description}</div>
                    </div>
                `).join('');
            }

            // Control functions
            async function refreshData() {
                await loadDashboardData();
            }

            async function startMutationCycle() {
                try {
                    await fetch('/mutation/api/start', { method: 'POST' });
                    alert('Evolution cycle started!');
                    await loadDashboardData();
                } catch (error) {
                    alert('Error starting evolution cycle');
                }
            }

            async function stopMutationCycle() {
                try {
                    await fetch('/mutation/api/stop', { method: 'POST' });
                    alert('Evolution cycle stopped!');
                    await loadDashboardData();
                } catch (error) {
                    alert('Error stopping evolution cycle');
                }
            }

            function viewLeaderboard() {
                window.open('/mutation-leaderboard', '_blank');
            }

            // Initialize dashboard
            document.addEventListener('DOMContentLoaded', function() {
                initCharts();
                loadDashboardData();

                // Auto-refresh every 30 seconds
                setInterval(loadDashboardData, 30000);
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)


@router.get("/api/stats")
async def get_mutation_stats():
    """Get enhanced mutation statistics for the dashboard"""
    try:
        # Get stats from mutation manager
        from backend.ai_mutation.mutation_manager import MutationManager

        manager = MutationManager()
        stats = manager.get_mutation_stats()

        # Get recent mutations with metadata
        recent_mutations = manager.get_recent_mutations(20)

        # Get metadata for promoted strategies
        metadata = get_strategy_metadata()

        # Prepare chart data
        chart_data = {
            "profit_labels": [],
            "profit_data": [],
            "promoted_count": 0,
            "rejected_count": 0,
        }

        for mutation in recent_mutations:
            chart_data["profit_labels"].append(mutation["timestamp"][-8:])  # HH:MM:SS
            chart_data["profit_data"].append(mutation["profit"])

            if mutation["promoted"]:
                chart_data["promoted_count"] += 1
            else:
                chart_data["rejected_count"] += 1

        return JSONResponse(
            {
                "stats": stats,
                "recent_mutations": recent_mutations,
                "chart_data": chart_data,
                "metadata": metadata,
            }
        )

    except Exception as e:
        logger.error(f"Error getting mutation stats: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


def get_strategy_metadata():
    """Get metadata for promoted strategies"""
    try:
        conn = sqlite3.connect("simulation_trades.db")
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT file, simulated_profit, promoted, timestamp
            FROM strategy_mutations
            WHERE promoted = 1
            ORDER BY simulated_profit DESC
            LIMIT 5
        """
        )

        rows = cursor.fetchall()
        conn.close()

        metadata = []
        for row in rows:
            # Try to load strategy file to get metadata
            try:
                strategy_path = os.path.join("strategies", row[0])
                if os.path.exists(strategy_path):
                    with open(strategy_path) as f:
                        strategy_data = json.load(f)

                    meta = strategy_data.get("metadata", {})
                    metadata.append(
                        {
                            "name": row[0],
                            "creator": meta.get("created_by", "Unknown"),
                            "version": meta.get("ai_version", "v1.0.0"),
                            "parent": meta.get("parent", ""),
                            "created_at": meta.get("created_at", row[3]),
                            "description": meta.get("description", "No description available"),
                        }
                    )
            except Exception as e:
                logger.error(f"Error loading strategy metadata: {e}")
                metadata.append(
                    {
                        "name": row[0],
                        "creator": "Unknown",
                        "version": "v1.0.0",
                        "parent": "",
                        "created_at": row[3],
                        "description": "Metadata not available",
                    }
                )

        return metadata

    except Exception as e:
        logger.error(f"Error getting strategy metadata: {e}")
        return []


@router.post("/api/start")
async def start_mutation_cycle():
    """Start the mutation cycle"""
    try:
        # This would integrate with your mutation manager
        # For now, return success
        return JSONResponse({"status": "success", "message": "Evolution cycle started"})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@router.post("/api/stop")
async def stop_mutation_cycle():
    """Stop the mutation cycle"""
    try:
        # This would integrate with your mutation manager
        # For now, return success
        return JSONResponse({"status": "success", "message": "Evolution cycle stopped"})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


