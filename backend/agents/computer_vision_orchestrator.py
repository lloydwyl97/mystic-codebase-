"""
Computer Vision Orchestrator
Coordinates all computer vision agents and provides unified API
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.base_agent import BaseAgent


class ComputerVisionOrchestrator(BaseAgent):
    """Computer Vision Orchestrator - Coordinates all CV agents"""

    def __init__(self, agent_id: str = "computer_vision_orchestrator_001"):
        super().__init__(agent_id, "computer_vision_orchestrator")

        # Orchestrator-specific state
        self.state.update(
            {
                "cv_agents": {},
                "analysis_results": {},
                "coordination_history": {},
                "last_coordination": None,
                "coordination_count": 0,
            }
        )

        # CV agent configuration
        self.cv_config = {
            "agents": {
                "chart_pattern_agent": {
                    "type": "chart_pattern",
                    "priority": 1,
                    "enabled": True,
                },
                "technical_indicator_agent": {
                    "type": "technical_indicator",
                    "priority": 2,
                    "enabled": True,
                },
                "market_visualization_agent": {
                    "type": "market_visualization",
                    "priority": 3,
                    "enabled": True,
                },
            },
            "coordination_settings": {
                "analysis_interval": 300,  # 5 minutes
                "result_retention": 3600,  # 1 hour
                "max_concurrent_analyses": 5,
            },
        }

        # Register orchestrator-specific handlers
        self.register_handler("coordinate_analysis", self.handle_coordinate_analysis)
        self.register_handler("get_cv_results", self.handle_get_cv_results)
        self.register_handler("update_cv_config", self.handle_update_cv_config)
        self.register_handler("cv_agent_update", self.handle_cv_agent_update)

        print(f"🎯 Computer Vision Orchestrator {agent_id} initialized")

    async def initialize(self):
        """Initialize computer vision orchestrator resources"""
        try:
            # Load CV configuration
            await self.load_cv_config()

            # Initialize CV agent tracking
            await self.initialize_cv_agent_tracking()

            # Start coordination monitoring
            await self.start_coordination_monitoring()

            print(f"✅ Computer Vision Orchestrator {self.agent_id} initialized successfully")

        except Exception as e:
            print(f"❌ Error initializing Computer Vision Orchestrator: {e}")
            self.update_health_status("error")

    async def process_loop(self):
        """Main coordination processing loop"""
        while self.running:
            try:
                # Coordinate CV agent activities
                await self.coordinate_cv_agents()

                # Aggregate analysis results
                await self.aggregate_analysis_results()

                # Update coordination metrics
                await self.update_coordination_metrics()

                # Clean up old results
                await self.cleanup_old_results()

                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                print(f"❌ Error in coordination processing loop: {e}")
                await asyncio.sleep(120)

    async def load_cv_config(self):
        """Load CV configuration from Redis"""
        try:
            # Load CV configuration
            config_data = self.redis_client.get("computer_vision_config")
            if config_data:
                self.cv_config = json.loads(config_data)

            print(f"📋 CV configuration loaded: {len(self.cv_config['agents'])} agents")

        except Exception as e:
            print(f"❌ Error loading CV configuration: {e}")

    async def initialize_cv_agent_tracking(self):
        """Initialize CV agent tracking"""
        try:
            # Initialize agent tracking for each configured agent
            for agent_id, agent_config in self.cv_config["agents"].items():
                if agent_config["enabled"]:
                    self.state["cv_agents"][agent_id] = {
                        "config": agent_config,
                        "status": "unknown",
                        "last_update": None,
                        "analysis_count": 0,
                        "last_analysis": None,
                    }

            print(f"📊 CV agent tracking initialized: {len(self.state['cv_agents'])} agents")

        except Exception as e:
            print(f"❌ Error initializing CV agent tracking: {e}")

    async def start_coordination_monitoring(self):
        """Start coordination monitoring"""
        try:
            # Subscribe to CV agent updates
            pubsub = self.redis_client.pubsub()
            pubsub.subscribe("cv_agent_updates")

            # Start agent update listener
            asyncio.create_task(self.listen_cv_agent_updates(pubsub))

            print("📡 Coordination monitoring started")

        except Exception as e:
            print(f"❌ Error starting coordination monitoring: {e}")

    async def listen_cv_agent_updates(self, pubsub):
        """Listen for CV agent updates"""
        try:
            for message in pubsub.listen():
                if not self.running:
                    break

                if message["type"] == "message":
                    agent_update = json.loads(message["data"])
                    await self.process_cv_agent_update(agent_update)

        except Exception as e:
            print(f"❌ Error in CV agent update listener: {e}")
        finally:
            pubsub.close()

    async def process_cv_agent_update(self, agent_update: Dict[str, Any]):
        """Process CV agent update"""
        try:
            agent_id = agent_update.get("agent_id")
            update_type = agent_update.get("type")

            if agent_id in self.state["cv_agents"]:
                # Update agent status
                self.state["cv_agents"][agent_id]["status"] = agent_update.get("status", "unknown")
                self.state["cv_agents"][agent_id]["last_update"] = datetime.now().isoformat()

                # Handle specific update types
                if update_type == "analysis_complete":
                    await self.handle_agent_analysis_complete(agent_id, agent_update)
                elif update_type == "pattern_detected":
                    await self.handle_pattern_detection(agent_id, agent_update)
                elif update_type == "indicator_signal":
                    await self.handle_indicator_signal(agent_id, agent_update)
                elif update_type == "chart_generated":
                    await self.handle_chart_generation(agent_id, agent_update)

        except Exception as e:
            print(f"❌ Error processing CV agent update: {e}")

    async def handle_agent_analysis_complete(self, agent_id: str, update: Dict[str, Any]):
        """Handle agent analysis completion"""
        try:
            # Update agent analysis count
            self.state["cv_agents"][agent_id]["analysis_count"] += 1
            self.state["cv_agents"][agent_id]["last_analysis"] = datetime.now().isoformat()

            # Store analysis result
            symbol = update.get("symbol")
            if symbol:
                if symbol not in self.state["analysis_results"]:
                    self.state["analysis_results"][symbol] = {}

                self.state["analysis_results"][symbol][agent_id] = {
                    "result": update.get("result"),
                    "timestamp": datetime.now().isoformat(),
                }

            print(f"✅ Analysis complete from {agent_id} for {symbol}")

        except Exception as e:
            print(f"❌ Error handling agent analysis complete: {e}")

    async def handle_pattern_detection(self, agent_id: str, update: Dict[str, Any]):
        """Handle pattern detection from chart pattern agent"""
        try:
            symbol = update.get("symbol")
            patterns = update.get("patterns", [])

            if symbol and patterns:
                # Store pattern detection result
                if symbol not in self.state["analysis_results"]:
                    self.state["analysis_results"][symbol] = {}

                self.state["analysis_results"][symbol]["pattern_detection"] = {
                    "patterns": patterns,
                    "agent_id": agent_id,
                    "timestamp": datetime.now().isoformat(),
                }

                # Broadcast pattern detection to other agents
                await self.broadcast_pattern_detection(symbol, patterns)

        except Exception as e:
            print(f"❌ Error handling pattern detection: {e}")

    async def handle_indicator_signal(self, agent_id: str, update: Dict[str, Any]):
        """Handle indicator signal from technical indicator agent"""
        try:
            symbol = update.get("symbol")
            signal = update.get("signal")

            if symbol and signal:
                # Store indicator signal result
                if symbol not in self.state["analysis_results"]:
                    self.state["analysis_results"][symbol] = {}

                self.state["analysis_results"][symbol]["indicator_signal"] = {
                    "signal": signal,
                    "agent_id": agent_id,
                    "timestamp": datetime.now().isoformat(),
                }

                # Broadcast indicator signal to other agents
                await self.broadcast_indicator_signal(symbol, signal)

        except Exception as e:
            print(f"❌ Error handling indicator signal: {e}")

    async def handle_chart_generation(self, agent_id: str, update: Dict[str, Any]):
        """Handle chart generation from market visualization agent"""
        try:
            symbol = update.get("symbol")
            charts = update.get("charts", {})

            if symbol and charts:
                # Store chart generation result
                if symbol not in self.state["analysis_results"]:
                    self.state["analysis_results"][symbol] = {}

                self.state["analysis_results"][symbol]["chart_generation"] = {
                    "charts": charts,
                    "agent_id": agent_id,
                    "timestamp": datetime.now().isoformat(),
                }

                # Broadcast chart generation to other agents
                await self.broadcast_chart_generation(symbol, charts)

        except Exception as e:
            print(f"❌ Error handling chart generation: {e}")

    async def coordinate_cv_agents(self):
        """Coordinate CV agent activities"""
        try:
            print(f"🎯 Coordinating {len(self.state['cv_agents'])} CV agents...")

            # Check agent health and status
            for agent_id, agent_data in self.state["cv_agents"].items():
                if agent_data["config"]["enabled"]:
                    await self.check_agent_health(agent_id)

            # Trigger coordinated analysis if needed
            await self.trigger_coordinated_analysis()

            # Update coordination count
            self.state["coordination_count"] += 1
            self.state["last_coordination"] = datetime.now().isoformat()

            print("✅ CV agent coordination complete")

        except Exception as e:
            print(f"❌ Error coordinating CV agents: {e}")

    async def check_agent_health(self, agent_id: str):
        """Check health of a specific CV agent"""
        try:
            # Get agent metrics from Redis
            metrics_key = f"agent_metrics:{agent_id}"
            metrics_data = self.redis_client.get(metrics_key)

            if metrics_data:
                metrics = json.loads(metrics_data)

                # Update agent status
                self.state["cv_agents"][agent_id]["status"] = "healthy"
                self.state["cv_agents"][agent_id]["last_update"] = metrics.get("timestamp")

                # Check if agent is responsive
                last_update = metrics.get("timestamp")
                if last_update:
                    last_update_time = datetime.fromisoformat(last_update)
                    if datetime.now() - last_update_time > timedelta(minutes=5):
                        self.state["cv_agents"][agent_id]["status"] = "stale"
            else:
                self.state["cv_agents"][agent_id]["status"] = "unknown"

        except Exception as e:
            print(f"❌ Error checking health for {agent_id}: {e}")
            self.state["cv_agents"][agent_id]["status"] = "error"

    async def trigger_coordinated_analysis(self):
        """Trigger coordinated analysis across all CV agents"""
        try:
            # Get symbols that need analysis
            symbols = await self.get_symbols_for_analysis()

            for symbol in symbols[
                : self.cv_config["coordination_settings"]["max_concurrent_analyses"]
            ]:
                # Trigger analysis for each agent
                for agent_id, agent_data in self.state["cv_agents"].items():
                    if agent_data["config"]["enabled"] and agent_data["status"] == "healthy":
                        await self.trigger_agent_analysis(agent_id, symbol)

        except Exception as e:
            print(f"❌ Error triggering coordinated analysis: {e}")

    async def get_symbols_for_analysis(self) -> List[str]:
        """Get symbols that need analysis"""
        try:
            # Get trading symbols from Redis
            symbols_data = self.redis_client.get("trading_symbols")
            if symbols_data:
                return json.loads(symbols_data)

            # Default symbols
            return ["BTC", "ETH", "ADA", "DOT", "LINK", "UNI", "AAVE"]

        except Exception as e:
            print(f"❌ Error getting symbols for analysis: {e}")
            return []

    async def trigger_agent_analysis(self, agent_id: str, symbol: str):
        """Trigger analysis for a specific agent and symbol"""
        try:
            # Send analysis request to agent
            analysis_request = {
                "type": "trigger_analysis",
                "symbol": symbol,
                "from_agent": self.agent_id,
                "timestamp": datetime.now().isoformat(),
            }

            await self.send_message(agent_id, analysis_request)

        except Exception as e:
            print(f"❌ Error triggering analysis for {agent_id}: {e}")

    async def aggregate_analysis_results(self):
        """Aggregate analysis results from all CV agents"""
        try:
            aggregated_results = {}

            for symbol, results in self.state["analysis_results"].items():
                aggregated_result = {
                    "symbol": symbol,
                    "timestamp": datetime.now().isoformat(),
                    "pattern_analysis": results.get("pattern_detection"),
                    "indicator_analysis": results.get("indicator_signal"),
                    "visual_analysis": results.get("chart_generation"),
                    "composite_signal": await self.generate_composite_signal(symbol, results),
                }

                aggregated_results[symbol] = aggregated_result

            # Store aggregated results
            self.state["coordination_history"]["aggregated_results"] = aggregated_results

            # Broadcast aggregated results
            await self.broadcast_aggregated_results(aggregated_results)

        except Exception as e:
            print(f"❌ Error aggregating analysis results: {e}")

    async def generate_composite_signal(
        self, symbol: str, results: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Generate composite signal from all analysis results"""
        try:
            composite_signal = {
                "symbol": symbol,
                "signal_type": "neutral",
                "confidence": 0.0,
                "factors": [],
                "timestamp": datetime.now().isoformat(),
            }

            # Analyze pattern detection
            pattern_analysis = results.get("pattern_detection")
            if pattern_analysis and pattern_analysis.get("patterns"):
                patterns = pattern_analysis["patterns"]
                if patterns:
                    # Find highest confidence pattern
                    best_pattern = max(patterns, key=lambda x: x.get("confidence", 0))
                    composite_signal["factors"].append(
                        {
                            "type": "pattern",
                            "pattern": best_pattern.get("pattern"),
                            "confidence": best_pattern.get("confidence", 0),
                            "direction": (
                                best_pattern.get("template", {}).get("direction", "neutral")
                            ),
                        }
                    )

            # Analyze indicator signals
            indicator_analysis = results.get("indicator_signal")
            if indicator_analysis and indicator_analysis.get("signal"):
                signal = indicator_analysis["signal"]
                composite_signal["factors"].append(
                    {
                        "type": "indicator",
                        "signal_type": signal.get("signal_type"),
                        "confidence": signal.get("confidence", 0),
                        "analysis": signal.get("analysis", {}),
                    }
                )

            # Determine composite signal
            if composite_signal["factors"]:
                # Calculate weighted confidence
                total_confidence = 0
                buy_signals = 0
                sell_signals = 0

                for factor in composite_signal["factors"]:
                    confidence = factor.get("confidence", 0)
                    total_confidence += confidence

                    if factor.get("type") == "pattern":
                        direction = factor.get("direction")
                        if direction == "bullish":
                            buy_signals += confidence
                        elif direction == "bearish":
                            sell_signals += confidence
                    elif factor.get("type") == "indicator":
                        signal_type = factor.get("signal_type")
                        if signal_type == "buy":
                            buy_signals += confidence
                        elif signal_type == "sell":
                            sell_signals += confidence

                # Determine overall signal
                if buy_signals > sell_signals:
                    composite_signal["signal_type"] = "buy"
                    composite_signal["confidence"] = buy_signals / total_confidence
                elif sell_signals > buy_signals:
                    composite_signal["signal_type"] = "sell"
                    composite_signal["confidence"] = sell_signals / total_confidence
                else:
                    composite_signal["signal_type"] = "neutral"
                    composite_signal["confidence"] = 0.5

            return composite_signal

        except Exception as e:
            print(f"❌ Error generating composite signal for {symbol}: {e}")
            return None

    async def broadcast_pattern_detection(self, symbol: str, patterns: List[Dict[str, Any]]):
        """Broadcast pattern detection to other agents"""
        try:
            pattern_update = {
                "type": "pattern_detection_broadcast",
                "symbol": symbol,
                "patterns": patterns,
                "from_orchestrator": self.agent_id,
                "timestamp": datetime.now().isoformat(),
            }

            # Broadcast to all agents
            await self.broadcast_message(pattern_update)

        except Exception as e:
            print(f"❌ Error broadcasting pattern detection: {e}")

    async def broadcast_indicator_signal(self, symbol: str, signal: Dict[str, Any]):
        """Broadcast indicator signal to other agents"""
        try:
            signal_update = {
                "type": "indicator_signal_broadcast",
                "symbol": symbol,
                "signal": signal,
                "from_orchestrator": self.agent_id,
                "timestamp": datetime.now().isoformat(),
            }

            # Broadcast to all agents
            await self.broadcast_message(signal_update)

        except Exception as e:
            print(f"❌ Error broadcasting indicator signal: {e}")

    async def broadcast_chart_generation(self, symbol: str, charts: Dict[str, Any]):
        """Broadcast chart generation to other agents"""
        try:
            chart_update = {
                "type": "chart_generation_broadcast",
                "symbol": symbol,
                "charts": charts,
                "from_orchestrator": self.agent_id,
                "timestamp": datetime.now().isoformat(),
            }

            # Broadcast to all agents
            await self.broadcast_message(chart_update)

        except Exception as e:
            print(f"❌ Error broadcasting chart generation: {e}")

    async def broadcast_aggregated_results(self, aggregated_results: Dict[str, Any]):
        """Broadcast aggregated results to other agents"""
        try:
            aggregated_update = {
                "type": "aggregated_results_update",
                "results": aggregated_results,
                "from_orchestrator": self.agent_id,
                "timestamp": datetime.now().isoformat(),
            }

            # Broadcast to all agents
            await self.broadcast_message(aggregated_update)

            # Send to specific agents
            await self.send_message("strategy_agent", aggregated_update)
            await self.send_message("execution_agent", aggregated_update)

        except Exception as e:
            print(f"❌ Error broadcasting aggregated results: {e}")

    async def handle_coordinate_analysis(self, message: Dict[str, Any]):
        """Handle manual coordination request"""
        try:
            symbol = message.get("symbol")

            print(f"🎯 Manual coordination requested for {symbol}")

            if symbol:
                # Trigger analysis for all agents
                for agent_id, agent_data in self.state["cv_agents"].items():
                    if agent_data["config"]["enabled"]:
                        await self.trigger_agent_analysis(agent_id, symbol)

                # Wait for results and aggregate
                await asyncio.sleep(10)  # Wait for agents to process
                await self.aggregate_analysis_results()

            # Send response
            response = {
                "type": "coordination_complete",
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
            }

            sender = message.get("from_agent")
            if sender:
                await self.send_message(sender, response)

        except Exception as e:
            print(f"❌ Error handling coordination request: {e}")
            await self.broadcast_error(f"Coordination error: {e}")

    async def handle_get_cv_results(self, message: Dict[str, Any]):
        """Handle CV results request"""
        try:
            symbol = message.get("symbol")

            print(f"📊 CV results requested for {symbol}")

            # Get CV results
            if symbol and symbol in self.state["analysis_results"]:
                results = self.state["analysis_results"][symbol]

                response = {
                    "type": "cv_results_response",
                    "symbol": symbol,
                    "results": results,
                    "timestamp": datetime.now().isoformat(),
                }
            else:
                response = {
                    "type": "cv_results_response",
                    "symbol": symbol,
                    "results": None,
                    "timestamp": datetime.now().isoformat(),
                }

            sender = message.get("from_agent")
            if sender:
                await self.send_message(sender, response)

        except Exception as e:
            print(f"❌ Error handling CV results request: {e}")
            await self.broadcast_error(f"CV results error: {e}")

    async def handle_update_cv_config(self, message: Dict[str, Any]):
        """Handle CV configuration update request"""
        try:
            new_config = message.get("config", {})

            print("⚙️ CV configuration update requested")

            # Update configuration
            if "agents" in new_config:
                self.cv_config["agents"].update(new_config["agents"])

            if "coordination_settings" in new_config:
                self.cv_config["coordination_settings"].update(new_config["coordination_settings"])

            # Save to Redis
            self.redis_client.set("computer_vision_config", json.dumps(self.cv_config))

            # Update agent tracking
            await self.initialize_cv_agent_tracking()

            # Send response
            response = {
                "type": "cv_config_update_complete",
                "config": self.cv_config,
                "timestamp": datetime.now().isoformat(),
            }

            sender = message.get("from_agent")
            if sender:
                await self.send_message(sender, response)

        except Exception as e:
            print(f"❌ Error handling CV config update: {e}")
            await self.broadcast_error(f"CV config update error: {e}")

    async def handle_cv_agent_update(self, message: Dict[str, Any]):
        """Handle CV agent update"""
        try:
            agent_id = message.get("agent_id")
            update_data = message.get("update_data", {})

            if agent_id in self.state["cv_agents"]:
                # Update agent data
                self.state["cv_agents"][agent_id].update(update_data)
                self.state["cv_agents"][agent_id]["last_update"] = datetime.now().isoformat()

        except Exception as e:
            print(f"❌ Error handling CV agent update: {e}")

    async def update_coordination_metrics(self):
        """Update coordination metrics"""
        try:
            metrics = {
                "agent_id": self.agent_id,
                "cv_agents_count": len(self.cv_config["agents"]),
                "active_agents": len(
                    [a for a in self.state["cv_agents"].values() if a["status"] == "healthy"]
                ),
                "analysis_results_count": len(self.state["analysis_results"]),
                "coordination_count": self.state["coordination_count"],
                "last_coordination": self.state["last_coordination"],
                "timestamp": datetime.now().isoformat(),
            }

            # Store metrics in Redis
            self.redis_client.set(f"agent_metrics:{self.agent_id}", json.dumps(metrics), ex=300)

        except Exception as e:
            print(f"❌ Error updating coordination metrics: {e}")

    async def cleanup_old_results(self):
        """Clean up old analysis results"""
        try:
            current_time = datetime.now()
            retention_time = timedelta(
                seconds=self.cv_config["coordination_settings"]["result_retention"]
            )

            # Clean up old results
            for symbol in list(self.state["analysis_results"].keys()):
                results = self.state["analysis_results"][symbol]

                # Check if results are old
                for result_type, result_data in list(results.items()):
                    if "timestamp" in result_data:
                        result_time = datetime.fromisoformat(result_data["timestamp"])
                        if current_time - result_time > retention_time:
                            del results[result_type]

                # Remove symbol if no results left
                if not results:
                    del self.state["analysis_results"][symbol]

        except Exception as e:
            print(f"❌ Error cleaning up old results: {e}")
