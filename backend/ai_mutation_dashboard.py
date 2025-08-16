"""
AI Mutation Engine Dashboard
Advanced dashboard for monitoring and controlling AI strategy evolution
"""

import asyncio
import json
import os
import redis
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import uvicorn
from dotenv import load_dotenv

# Load environment variables
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))


class MutationDashboard:
    def __init__(self):
        """Initialize Mutation Dashboard"""
        self.redis_client = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            db=int(os.getenv("REDIS_DB", 0)),
            decode_responses=True,
        )
        self.app = FastAPI(title="AI Mutation Dashboard", version="1.0.0")
        self.active_connections: List[WebSocket] = []

        # Setup CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Setup routes
        self.setup_routes()

    def setup_routes(self):
        """Setup API routes"""

        @self.app.get("/")
        async def get_dashboard():
            """Get dashboard HTML"""
            return HTMLResponse(self.get_dashboard_html())

        @self.app.get("/api/status")
        async def get_system_status():
            """Get overall system status"""
            return await self.get_system_status()

        @self.app.get("/api/strategies")
        async def get_strategies():
            """Get all AI strategies"""
            return await self.get_all_strategies()

        @self.app.get("/api/strategies/{strategy_id}")
        async def get_strategy(strategy_id: str):
            """Get specific strategy details"""
            return await self.get_strategy_details(strategy_id)

        @self.app.get("/api/genetic-population")
        async def get_genetic_population():
            """Get genetic algorithm population"""
            return await self.get_genetic_population()

        @self.app.get("/api/evolution-history")
        async def get_evolution_history():
            """Get evolution history"""
            return await self.get_evolution_history()

        @self.app.get("/api/model-versions")
        async def get_model_versions():
            """Get model versions"""
            return await self.get_model_versions()

        @self.app.get("/api/retrain-queue")
        async def get_retrain_queue():
            """Get retrain queue"""
            return await self.get_retrain_queue()

        @self.app.post("/api/strategies/{strategy_id}/retrain")
        async def trigger_retrain(strategy_id: str):
            """Trigger retraining for a strategy"""
            return await self.trigger_strategy_retrain(strategy_id)

        @self.app.post("/api/strategies/{strategy_id}/deploy")
        async def deploy_strategy(strategy_id: str):
            """Deploy a strategy"""
            return await self.deploy_strategy(strategy_id)

        @self.app.post("/api/genetic/evolve")
        async def trigger_evolution():
            """Trigger genetic evolution"""
            return await self.trigger_genetic_evolution()

        @self.app.post("/api/genetic/reset")
        async def reset_population():
            """Reset genetic population"""
            return await self.reset_genetic_population()

        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time updates"""
            await websocket.accept()
            self.active_connections.append(websocket)

            try:
                while True:
                    # Send real-time updates
                    data = await self.get_realtime_data()
                    await websocket.send_text(json.dumps(data))
                    await asyncio.sleep(5)  # Update every 5 seconds
            except WebSocketDisconnect:
                self.active_connections.remove(websocket)

    async def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        try:
            # Get service statuses
            services = {
                "ai_strategy_generator": self.redis_client.exists("ai_strategy_generator_status"),
                "genetic_algorithm": self.redis_client.exists("genetic_algorithm_status"),
                "model_versioning": self.redis_client.exists("model_versioning_status"),
                "auto_retrain": self.redis_client.exists("auto_retrain_status"),
            }

            # Get counts
            strategy_count = self.redis_client.llen("ai_strategies")
            population_size = self.redis_client.llen("genetic_population")
            retrain_queue_size = self.redis_client.llen("retrain_queue")

            # Get recent activity
            recent_activity = await self.get_recent_activity()

            return {
                "status": ("operational" if any(services.values()) else "offline"),
                "services": services,
                "counts": {
                    "strategies": strategy_count,
                    "population_size": population_size,
                    "retrain_queue": retrain_queue_size,
                },
                "recent_activity": recent_activity,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    async def get_all_strategies(self) -> List[Dict[str, Any]]:
        """Get all AI strategies"""
        try:
            strategy_ids = self.redis_client.lrange("ai_strategies", 0, -1)
            strategies = []

            for strategy_id in strategy_ids:
                strategy_data = self.redis_client.get(f"ai_strategy:{strategy_id}")
                if strategy_data:
                    strategy = json.loads(strategy_data)
                    strategies.append(strategy)

            return strategies

        except Exception:
            return []

    async def get_strategy_details(self, strategy_id: str) -> Dict[str, Any]:
        """Get specific strategy details"""
        try:
            strategy_data = self.redis_client.get(f"ai_strategy:{strategy_id}")
            if not strategy_data:
                raise HTTPException(status_code=404, detail="Strategy not found")

            strategy = json.loads(strategy_data)

            # Get performance history
            performance_history = await self.get_strategy_performance_history(strategy_id)

            # Get lineage
            lineage = await self.get_strategy_lineage(strategy_id)

            return {
                "strategy": strategy,
                "performance_history": performance_history,
                "lineage": lineage,
            }

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    async def get_genetic_population(self) -> Dict[str, Any]:
        """Get genetic algorithm population"""
        try:
            population_data = self.redis_client.get("genetic_population")
            if not population_data:
                return {"population": [], "generation": 0, "best_fitness": 0.0}

            population = json.loads(population_data)

            # Get evolution history
            evolution_data = self.redis_client.get("evolution_history")
            evolution_history = json.loads(evolution_data) if evolution_data else []

            return {
                "population": population,
                "generation": len(evolution_history),
                "best_fitness": (
                    evolution_history[-1]["best_fitness"] if evolution_history else 0.0
                ),
                "evolution_history": evolution_history,
            }

        except Exception as e:
            return {
                "population": [],
                "generation": 0,
                "best_fitness": 0.0,
                "error": str(e),
            }

    async def get_evolution_history(self) -> List[Dict[str, Any]]:
        """Get evolution history"""
        try:
            evolution_data = self.redis_client.get("evolution_history")
            return json.loads(evolution_data) if evolution_data else []

        except Exception:
            return []

    async def get_model_versions(self) -> List[Dict[str, Any]]:
        """Get model versions"""
        try:
            # This would typically query the model versioning database
            # For now, return simulated data
            versions = []

            for i in range(5):
                version = {
                    "version_id": f"v1.{i}.0",
                    "model_name": f"AI_Strategy_{i}",
                    "model_type": "lstm" if i % 2 == 0 else "transformer",
                    "created_at": ((datetime.now() - timedelta(days=i)).isoformat()),
                    "performance": {
                        "accuracy": np.random.uniform(0.6, 0.8),
                        "total_return": np.random.uniform(0.05, 0.15),
                        "sharpe_ratio": np.random.uniform(0.8, 1.5),
                    },
                    "status": "active",
                }
                versions.append(version)

            return versions

        except Exception:
            return []

    async def get_retrain_queue(self) -> List[Dict[str, Any]]:
        """Get retrain queue"""
        try:
            queue_items = self.redis_client.lrange("retrain_queue", 0, -1)
            queue = []

            for item in queue_items:
                queue.append(json.loads(item))

            return queue

        except Exception:
            return []

    async def trigger_strategy_retrain(self, strategy_id: str) -> Dict[str, Any]:
        """Trigger retraining for a strategy"""
        try:
            # Add to retrain queue
            retrain_request = {
                "model_id": strategy_id,
                "reason": "manual_trigger",
                "timestamp": datetime.now().isoformat(),
            }

            self.redis_client.lpush("retrain_queue", json.dumps(retrain_request))

            return {
                "status": "success",
                "message": f"Retraining triggered for strategy {strategy_id}",
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    async def deploy_strategy(self, strategy_id: str) -> Dict[str, Any]:
        """Deploy a strategy"""
        try:
            # Add to deployment queue
            deployment_request = {
                "strategy_id": strategy_id,
                "environment": "production",
                "timestamp": datetime.now().isoformat(),
            }

            self.redis_client.lpush("deployment_queue", json.dumps(deployment_request))

            return {
                "status": "success",
                "message": f"Strategy {strategy_id} queued for deployment",
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    async def trigger_genetic_evolution(self) -> Dict[str, Any]:
        """Trigger genetic evolution"""
        try:
            # Add evolution request
            evolution_request = {
                "action": "evolve",
                "timestamp": datetime.now().isoformat(),
            }

            self.redis_client.lpush("genetic_requests", json.dumps(evolution_request))

            return {
                "status": "success",
                "message": "Genetic evolution triggered",
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    async def reset_genetic_population(self) -> Dict[str, Any]:
        """Reset genetic population"""
        try:
            # Add reset request
            reset_request = {
                "action": "reset",
                "timestamp": datetime.now().isoformat(),
            }

            self.redis_client.lpush("genetic_requests", json.dumps(reset_request))

            return {
                "status": "success",
                "message": "Genetic population reset triggered",
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    async def get_recent_activity(self) -> List[Dict[str, Any]]:
        """Get recent system activity"""
        try:
            # This would typically query logs or activity database
            # For now, return simulated data
            activities = []

            activity_types = [
                "strategy_generated",
                "model_retrained",
                "evolution_completed",
                "strategy_deployed",
            ]

            for i in range(10):
                activity = {
                    "type": np.random.choice(activity_types),
                    "description": f"Activity {i}",
                    "timestamp": ((datetime.now() - timedelta(minutes=i * 5)).isoformat()),
                    "status": "completed",
                }
                activities.append(activity)

            return activities

        except Exception:
            return []

    async def get_strategy_performance_history(self, strategy_id: str) -> List[Dict[str, Any]]:
        """Get strategy performance history"""
        try:
            # This would typically query performance database
            # For now, return simulated data
            history = []

            for i in range(30):
                performance = {
                    "date": (datetime.now() - timedelta(days=i)).isoformat(),
                    "accuracy": np.random.uniform(0.6, 0.8),
                    "total_return": np.random.uniform(-0.05, 0.1),
                    "sharpe_ratio": np.random.uniform(0.5, 1.5),
                    "win_rate": np.random.uniform(0.4, 0.7),
                }
                history.append(performance)

            return history

        except Exception:
            return []

    async def get_strategy_lineage(self, strategy_id: str) -> List[Dict[str, Any]]:
        """Get strategy lineage"""
        try:
            # This would typically query lineage database
            # For now, return simulated data
            lineage = [
                {
                    "version_id": strategy_id,
                    "parent_id": None,
                    "relationship": "original",
                    "created_at": datetime.now().isoformat(),
                }
            ]

            return lineage

        except Exception:
            return []

    async def get_realtime_data(self) -> Dict[str, Any]:
        """Get real-time data for WebSocket updates"""
        try:
            return {
                "type": "realtime_update",
                "system_status": await self.get_system_status(),
                "active_strategies": len(await self.get_all_strategies()),
                "genetic_population": await self.get_genetic_population(),
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            return {
                "type": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def get_dashboard_html(self) -> str:
        """Get dashboard HTML"""
        return """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>AI Mutation Dashboard</title>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <style>
                body {
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                }
                .container {
                    max-width: 1400px;
                    margin: 0 auto;
                }
                .header {
                    text-align: center;
                    margin-bottom: 30px;
                }
                .header h1 {
                    font-size: 2.5em;
                    margin: 0;
                    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
                }
                .grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                    gap: 20px;
                    margin-bottom: 30px;
                }
                .card {
                    background: rgba(255,255,255,0.1);
                    backdrop-filter: blur(10px);
                    border-radius: 15px;
                    padding: 20px;
                    box-shadow: 0 8px 32px rgba(0,0,0,0.1);
                    border: 1px solid rgba(255,255,255,0.2);
                }
                .card h3 {
                    margin-top: 0;
                    color: #fff;
                    border-bottom: 2px solid rgba(255,255,255,0.3);
                    padding-bottom: 10px;
                }
                .status-indicator {
                    display: inline-block;
                    width: 12px;
                    height: 12px;
                    border-radius: 50%;
                    margin-right: 8px;
                }
                .status-online { background-color: #4CAF50; }
                .status-offline { background-color: #f44336; }
                .status-warning { background-color: #ff9800; }
                .metric {
                    display: flex;
                    justify-content: space-between;
                    margin: 10px 0;
                    padding: 8px;
                    background: rgba(255,255,255,0.1);
                    border-radius: 8px;
                }
                .metric-value {
                    font-weight: bold;
                    color: #4CAF50;
                }
                .chart-container {
                    height: 300px;
                    margin-top: 20px;
                }
                .button {
                    background: linear-gradient(45deg, #4CAF50, #45a049);
                    color: white;
                    border: none;
                    padding: 10px 20px;
                    border-radius: 8px;
                    cursor: pointer;
                    margin: 5px;
                    transition: transform 0.2s;
                }
                .button:hover {
                    transform: translateY(-2px);
                }
                .button.danger {
                    background: linear-gradient(45deg, #f44336, #da190b);
                }
                .activity-feed {
                    max-height: 300px;
                    overflow-y: auto;
                }
                .activity-item {
                    padding: 8px;
                    margin: 5px 0;
                    background: rgba(255,255,255,0.1);
                    border-radius: 8px;
                    border-left: 4px solid #4CAF50;
                }
                .timestamp {
                    font-size: 0.8em;
                    color: rgba(255,255,255,0.7);
                }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>ðŸ§¬ AI Mutation Dashboard</h1>
                    <p>Advanced AI Strategy Evolution & Management</p>
                </div>

                <div class="grid">
                    <div class="card">
                        <h3>ðŸ”„ System Status</h3>
                        <div id="system-status">Loading...</div>
                    </div>

                    <div class="card">
                        <h3>ðŸ“Š Active Strategies</h3>
                        <div id="strategies-count">Loading...</div>
                        <div class="chart-container">
                            <canvas id="strategies-chart"></canvas>
                        </div>
                    </div>

                    <div class="card">
                        <h3>ðŸ§¬ Genetic Population</h3>
                        <div id="genetic-status">Loading...</div>
                        <div class="chart-container">
                            <canvas id="evolution-chart"></canvas>
                        </div>
                    </div>

                    <div class="card">
                        <h3>ðŸ“¦ Model Versions</h3>
                        <div id="model-versions">Loading...</div>
                    </div>

                    <div class="card">
                        <h3>ðŸ”„ Retrain Queue</h3>
                        <div id="retrain-queue">Loading...</div>
                        <button class="button" onclick="triggerEvolution()">Trigger Evolution</button>
                        <button class="button danger" onclick="resetPopulation()">Reset Population</button>
                    </div>

                    <div class="card">
                        <h3>ðŸ“ˆ Recent Activity</h3>
                        <div id="activity-feed" class="activity-feed">Loading...</div>
                    </div>
                </div>
            </div>

            <script>
                let ws;
                let strategiesChart, evolutionChart;

                // Initialize WebSocket
                function initWebSocket() {
                    ws = new WebSocket(`ws://${window.location.host}/ws`);
                    ws.onmessage = function(event) {
                        const data = JSON.parse(event.data);
                        updateDashboard(data);
                    };
                    ws.onclose = function() {
                        setTimeout(initWebSocket, 1000);
                    };
                }

                // Update dashboard with real-time data
                function updateDashboard(data) {
                    if (data.system_status) {
                        updateSystemStatus(data.system_status);
                    }
                    if (data.genetic_population) {
                        updateGeneticStatus(data.genetic_population);
                    }
                }

                // Update system status
                function updateSystemStatus(status) {
                    const statusDiv = document.getElementById('system-status');
                    let html = '';

                    for (const [service, isOnline] of Object.entries(status.services)) {
                        const statusClass = isOnline ? 'status-online' : 'status-offline';
                        const statusText = isOnline ? 'Online' : 'Offline';
                        html += `<div class="metric">
                            <span><span class="status-indicator ${statusClass}"></span>${service}</span>
                            <span class="metric-value">${statusText}</span>
                        </div>`;
                    }

                    html += `<div class="metric">
                        <span>Active Strategies</span>
                        <span class="metric-value">${status.counts.strategies}</span>
                    </div>`;

                    html += `<div class="metric">
                        <span>Population Size</span>
                        <span class="metric-value">${status.counts.population_size}</span>
                    </div>`;

                    statusDiv.innerHTML = html;
                }

                // Update genetic status
                function updateGeneticStatus(data) {
                    const statusDiv = document.getElementById('genetic-status');
                    statusDiv.innerHTML = `
                        <div class="metric">
                            <span>Generation</span>
                            <span class="metric-value">${data.generation}</span>
                        </div>
                        <div class="metric">
                            <span>Best Fitness</span>
                            <span class="metric-value">${data.best_fitness.toFixed(4)}</span>
                        </div>
                        <div class="metric">
                            <span>Population Size</span>
                            <span class="metric-value">${data.population.length}</span>
                        </div>
                    `;

                    updateEvolutionChart(data.evolution_history);
                }

                // Initialize charts
                function initCharts() {
                    const strategiesCtx = document.getElementById('strategies-chart').getContext('2d');
                    strategiesChart = new Chart(strategiesCtx, {
                        type: 'line',
                        data: {
                            labels: [],
                            datasets: [{
                                label: 'Active Strategies',
                                data: [],
                                borderColor: '#4CAF50',
                                backgroundColor: 'rgba(76, 175, 80, 0.1)',
                                tension: 0.4
                            }]
                        },
                        options: {
                            responsive: true,
                            maintainAspectRatio: false,
                            scales: {
                                y: {
                                    beginAtZero: true,
                                    grid: {
                                        color: 'rgba(255,255,255,0.1)'
                                    },
                                    ticks: { color: 'white' }
                                },
                                x: {
                                    grid: {
                                        color: 'rgba(255,255,255,0.1)'
                                    },
                                    ticks: { color: 'white' }
                                }
                            },
                            plugins: {
                                legend: {
                                    labels: { color: 'white' }
                                }
                            }
                        }
                    });

                    const evolutionCtx = document.getElementById('evolution-chart').getContext('2d');
                    evolutionChart = new Chart(evolutionCtx, {
                        type: 'line',
                        data: {
                            labels: [],
                            datasets: [{
                                label: 'Best Fitness',
                                data: [],
                                borderColor: '#FF6B6B',
                                backgroundColor: 'rgba(255, 107, 107, 0.1)',
                                tension: 0.4
                            }]
                        },
                        options: {
                            responsive: true,
                            maintainAspectRatio: false,
                            scales: {
                                y: {
                                    beginAtZero: true,
                                    grid: {
                                        color: 'rgba(255,255,255,0.1)'
                                    },
                                    ticks: { color: 'white' }
                                },
                                x: {
                                    grid: {
                                        color: 'rgba(255,255,255,0.1)'
                                    },
                                    ticks: { color: 'white' }
                                }
                            },
                            plugins: {
                                legend: {
                                    labels: { color: 'white' }
                                }
                            }
                        }
                    });
                }

                // Update evolution chart
                function updateEvolutionChart(history) {
                    if (!evolutionChart || !history) return;

                    const labels = history.map((_, i) => `Gen ${i + 1}`);
                    const data = history.map(h => h.best_fitness);

                    evolutionChart.data.labels = labels;
                    evolutionChart.data.datasets[0].data = data;
                    evolutionChart.update();
                }

                // API functions
                async function triggerEvolution() {
                    try {
                        const response = await fetch('/api/genetic/evolve', { method: 'POST' });
                        const result = await response.json();
                        alert(result.message);
                    } catch (error) {
                        alert('Error triggering evolution: ' + error);
                    }
                }

                async function resetPopulation() {
                    if (confirm('Are you sure you want to reset the genetic population?')) {
                        try {
                            const response = await fetch('/api/genetic/reset', { method: 'POST' });
                            const result = await response.json();
                            alert(result.message);
                        } catch (error) {
                            alert('Error resetting population: ' + error);
                        }
                    }
                }

                // Load initial data
                async function loadInitialData() {
                    try {
                        const [statusRes, strategiesRes, geneticRes, versionsRes, queueRes] = await Promise.all([
                            fetch('/api/status'),
                            fetch('/api/strategies'),
                            fetch('/api/genetic-population'),
                            fetch('/api/model-versions'),
                            fetch('/api/retrain-queue')
                        ]);

                        const status = await statusRes.json();
                        const strategies = await strategiesRes.json();
                        const genetic = await geneticRes.json();
                        const versions = await versionsRes.json();
                        const queue = await queueRes.json();

                        updateSystemStatus(status);
                        updateGeneticStatus(genetic);

                        // Update other sections
                        document.getElementById('strategies-count').innerHTML = `
                            <div class="metric">
                                <span>Total Strategies</span>
                                <span class="metric-value">${strategies.length}</span>
                            </div>
                        `;

                        document.getElementById('model-versions').innerHTML = `
                            <div class="metric">
                                <span>Total Versions</span>
                                <span class="metric-value">${versions.length}</span>
                            </div>
                        `;

                        document.getElementById('retrain-queue').innerHTML = `
                            <div class="metric">
                                <span>Queue Size</span>
                                <span class="metric-value">${queue.length}</span>
                            </div>
                        `;

                    } catch (error) {
                        console.error('Error loading initial data:', error);
                    }
                }

                // Initialize
                document.addEventListener('DOMContentLoaded', function() {
                    initCharts();
                    initWebSocket();
                    loadInitialData();
                });
            </script>
        </body>
        </html>
        """

    async def start(self, host: str = "0.0.0.0", port: int = 8080):
        """Start the dashboard server"""
        print(f"ðŸš€ Starting AI Mutation Dashboard on {host}:{port}")
        config = uvicorn.Config(self.app, host=host, port=port, log_level="info")
        server = uvicorn.Server(config)
        await server.serve()


async def main():
    """Main function"""
    dashboard = MutationDashboard()

    try:
        await dashboard.start()
    except KeyboardInterrupt:
        print("ðŸ›‘ Received interrupt signal")
    except Exception as e:
        print(f"âŒ Error in main: {e}")


if __name__ == "__main__":
    asyncio.run(main())


