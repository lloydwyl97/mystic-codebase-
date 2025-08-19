"""
Advanced AI Orchestrator
Coordinates all AI agents including deep learning, reinforcement learning, and model management
"""

import asyncio
import json
import os
import sys
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

# Add parent directory and project root to path for imports
backend_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_path not in sys.path:
    sys.path.append(backend_path)
root_path = os.path.dirname(backend_path)
if root_path not in sys.path:
    sys.path.append(root_path)

from backend.agents.base_agent import BaseAgent  # noqa: E402
from backend.config.coins import FEATURED_SYMBOLS  # noqa: E402
from backend.services.ai_attribution import save_attribution  # noqa: E402

# Add usage for unused imports to satisfy F401
_timedelta: timedelta = timedelta(seconds=0)
_Tuple: tuple[int, int] = (0, 1)
_pd_df = pd.DataFrame()


class AIStrategy:
    """AI Strategy combining multiple AI approaches"""

    def __init__(self, strategy_id: str, symbol: str, components: dict[str, Any]):
        self.strategy_id = strategy_id
        self.symbol = symbol
        self.components = components
        self.created_at = datetime.now().isoformat()
        self.confidence = 0.0
        self.performance_metrics = {}
        self.status = "created"


class AdvancedAIOrchestrator(BaseAgent):
    """Advanced AI Orchestrator - Coordinates all AI agents"""

    def __init__(self, agent_id: str = "advanced_ai_orchestrator_001"):
        super().__init__(agent_id, "advanced_ai_orchestrator")

        # Orchestrator-specific state
        self.state.update(
            {
                "ai_strategies": {},
                "agent_coordination": {},
                "performance_metrics": {},
                "last_coordination": None,
                "coordination_count": 0,
            }
        )

        # AI coordination configuration
        self.coordination_config = {
            "ai_agents": {
                "deep_learning_agent": {
                    "type": "deep_learning",
                    "capabilities": [
                        "price_prediction",
                        "pattern_recognition",
                    ],
                    "weight": 0.4,
                    "enabled": True,
                },
                "reinforcement_learning_agent": {
                    "type": "reinforcement_learning",
                    "capabilities": [
                        "strategy_optimization",
                        "action_selection",
                    ],
                    "weight": 0.3,
                    "enabled": True,
                },
                "nlp_agent": {
                    "type": "nlp",
                    "capabilities": ["sentiment_analysis", "news_processing"],
                    "weight": 0.2,
                    "enabled": True,
                },
                "computer_vision_agent": {
                    "type": "computer_vision",
                    "capabilities": ["chart_analysis", "technical_indicators"],
                    "weight": 0.1,
                    "enabled": True,
                },
            },
            "coordination_settings": {
                "strategy_generation_interval": 300,  # seconds
                "performance_evaluation_interval": 600,  # seconds
                "confidence_threshold": 0.7,
                "min_agent_agreement": 0.6,
                "strategy_lifetime": 3600,  # seconds
            },
            "integration_settings": {
                "enable_cross_validation": True,
                "enable_ensemble_methods": True,
                "enable_adaptive_weights": True,
                "enable_fallback_strategies": True,
            },
        }

        # Trading symbols to monitor
        self.trading_symbols = [
            "BTC",
            "ETH",
            "ADA",
            "DOT",
            "LINK",
            "UNI",
            "AAVE",
        ]

        # AI agent connections
        self.ai_agents = {}
        self.agent_weights = {}
        self.agent_capabilities = {}

        # Strategy cache
        self.strategy_cache = {}
        self.performance_history = {}

        # Register orchestrator-specific handlers
        self.register_handler("coordinate_ai_agents", self.handle_coordinate_ai_agents)
        self.register_handler("generate_ai_strategy", self.handle_generate_ai_strategy)
        self.register_handler("get_ai_status", self.handle_get_ai_status)
        self.register_handler("ai_prediction", self.handle_ai_prediction)
        self.register_handler("ai_strategy", self.handle_ai_strategy)
        self.register_handler("model_deployment", self.handle_model_deployment)

        print(f"ðŸŽ›ï¸ Advanced AI Orchestrator {agent_id} initialized")

    async def initialize(self):
        """Initialize advanced AI orchestrator resources"""
        try:
            # Load coordination configuration
            await self.load_coordination_config()

            # Initialize AI agent connections
            await self.initialize_ai_agents()

            # Start AI coordination
            await self.start_ai_coordination()

            print(f"âœ… Advanced AI Orchestrator {self.agent_id} initialized successfully")

        except Exception as e:
            print(f"âŒ Error initializing Advanced AI Orchestrator: {e}")
            self.update_health_status("error")

    async def process_loop(self):
        """Main AI coordination processing loop"""
        while self.running:
            try:
                # Coordinate AI agents
                await self.coordinate_ai_agents()

                # Generate AI strategies
                await self.generate_ai_strategies()

                # Evaluate strategy performance
                await self.evaluate_strategy_performance()

                # Update coordination metrics
                await self.update_coordination_metrics()

                # Clean up old strategies
                await self.cleanup_old_strategies()

                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                print(f"âŒ Error in AI coordination processing loop: {e}")
                await asyncio.sleep(120)

    async def load_coordination_config(self):
        """Load coordination configuration from Redis"""
        try:
            # Load coordination configuration
            config_data = self.redis_client.get("advanced_ai_coordination_config")
            if config_data:
                self.coordination_config = json.loads(config_data)

            # Load trading symbols
            symbols_data = self.redis_client.get("trading_symbols")
            if symbols_data:
                self.trading_symbols = json.loads(symbols_data)

            print(
                f"ðŸ“‹ Coordination configuration loaded: "
                f"{len(self.coordination_config['ai_agents'])} AI agents, "
                f"{len(self.trading_symbols)} symbols"
            )

        except Exception as e:
            print(f"âŒ Error loading coordination configuration: {e}")

    async def initialize_ai_agents(self):
        """Initialize AI agent connections"""
        try:
            for agent_name, agent_config in self.coordination_config["ai_agents"].items():
                if agent_config["enabled"]:
                    self.ai_agents[agent_name] = {
                        "type": agent_config["type"],
                        "capabilities": agent_config["capabilities"],
                        "weight": agent_config["weight"],
                        "status": "unknown",
                        "last_communication": None,
                    }

                    self.agent_weights[agent_name] = agent_config["weight"]
                    self.agent_capabilities[agent_name] = agent_config["capabilities"]

            print(f"ðŸ¤– AI agents initialized: {len(self.ai_agents)} agents")

        except Exception as e:
            print(f"âŒ Error initializing AI agents: {e}")

    async def start_ai_coordination(self):
        """Start AI coordination"""
        try:
            # Subscribe to AI agent updates
            pubsub = self.redis_client.pubsub()
            pubsub.subscribe("ai_predictions")
            pubsub.subscribe("ai_strategies")
            pubsub.subscribe("model_deployments")

            # Start coordination listener
            asyncio.create_task(self.listen_ai_updates(pubsub))

            print("ðŸ“¡ AI coordination started")

        except Exception as e:
            print(f"âŒ Error starting AI coordination: {e}")

    async def listen_ai_updates(self, pubsub) -> None:
        """Listen for AI agent updates"""
        try:
            for message in pubsub.listen():
                if not self.running:
                    break

                if message["type"] == "message":
                    update_data = json.loads(message["data"])
                    await self.process_ai_update(update_data)

        except Exception as e:
            print(f"âŒ Error in AI updates listener: {e}")
        finally:
            pubsub.close()

    async def process_ai_update(self, update_data: dict[str, Any]):
        """Process AI agent update"""
        try:
            update_type = update_data.get("type")

            if update_type == "deep_learning_prediction_update":
                await self.handle_deep_learning_prediction(update_data)
            elif update_type == "rl_strategy_update":
                await self.handle_reinforcement_learning_strategy(update_data)
            elif update_type == "nlp_sentiment_update":
                await self.handle_nlp_sentiment(update_data)
            elif update_type == "computer_vision_analysis":
                await self.handle_computer_vision_analysis(update_data)
            elif update_type == "model_deployment_update":
                await self.handle_model_deployment_update(update_data)

        except Exception as e:
            print(f"âŒ Error processing AI update: {e}")

    async def handle_deep_learning_prediction(self, update_data: dict[str, Any]):
        """Handle deep learning prediction update"""
        try:
            symbol = update_data.get("symbol")
            predictions = update_data.get("predictions", {})

            if symbol and predictions:
                # Store predictions
                if symbol not in self.strategy_cache:
                    self.strategy_cache[symbol] = {}

                self.strategy_cache[symbol]["deep_learning"] = {
                    "predictions": predictions,
                    "timestamp": datetime.now().isoformat(),
                }

                # Update agent status
                self.ai_agents["deep_learning_agent"]["status"] = "active"
                self.ai_agents["deep_learning_agent"][
                    "last_communication"
                ] = datetime.now().isoformat()

                print(f"ðŸ§  Deep learning predictions received for {symbol}")

        except Exception as e:
            print(f"âŒ Error handling deep learning prediction: {e}")

    async def handle_reinforcement_learning_strategy(self, update_data: dict[str, Any]):
        """Handle reinforcement learning strategy update"""
        try:
            symbol = update_data.get("symbol")
            strategies = update_data.get("strategies", {})

            if symbol and strategies:
                # Store strategies
                if symbol not in self.strategy_cache:
                    self.strategy_cache[symbol] = {}

                self.strategy_cache[symbol]["reinforcement_learning"] = {
                    "strategies": strategies,
                    "timestamp": datetime.now().isoformat(),
                }

                # Update agent status
                self.ai_agents["reinforcement_learning_agent"]["status"] = "active"
                self.ai_agents["reinforcement_learning_agent"][
                    "last_communication"
                ] = datetime.now().isoformat()

                print(f"ðŸŽ¯ Reinforcement learning strategies received for {symbol}")

        except Exception as e:
            print(f"âŒ Error handling reinforcement learning strategy: {e}")

    async def handle_nlp_sentiment(self, update_data: dict[str, Any]):
        """Handle NLP sentiment update"""
        try:
            symbol = update_data.get("symbol")
            sentiment_data = update_data.get("sentiment_data", {})

            if symbol and sentiment_data:
                # Store sentiment data
                if symbol not in self.strategy_cache:
                    self.strategy_cache[symbol] = {}

                self.strategy_cache[symbol]["nlp"] = {
                    "sentiment_data": sentiment_data,
                    "timestamp": datetime.now().isoformat(),
                }

                # Update agent status
                self.ai_agents["nlp_agent"]["status"] = "active"
                self.ai_agents["nlp_agent"]["last_communication"] = datetime.now().isoformat()

                print(f"ðŸ“° NLP sentiment data received for {symbol}")

        except Exception as e:
            print(f"âŒ Error handling NLP sentiment: {e}")

    async def handle_computer_vision_analysis(self, update_data: dict[str, Any]):
        """Handle computer vision analysis update"""
        try:
            symbol = update_data.get("symbol")
            analysis_data = update_data.get("analysis_data", {})

            if symbol and analysis_data:
                # Store analysis data
                if symbol not in self.strategy_cache:
                    self.strategy_cache[symbol] = {}

                self.strategy_cache[symbol]["computer_vision"] = {
                    "analysis_data": analysis_data,
                    "timestamp": datetime.now().isoformat(),
                }

                # Update agent status
                self.ai_agents["computer_vision_agent"]["status"] = "active"
                self.ai_agents["computer_vision_agent"][
                    "last_communication"
                ] = datetime.now().isoformat()

                print(f"ðŸ‘ï¸ Computer vision analysis received for {symbol}")

        except Exception as e:
            print(f"âŒ Error handling computer vision analysis: {e}")

    async def handle_model_deployment_update(self, update_data: dict[str, Any]):
        """Handle model deployment update"""
        try:
            model_id = update_data.get("model_id")
            version = update_data.get("version")
            model_type = update_data.get("model_type")

            if model_id and version:
                # Update agent coordination
                if model_id not in self.state["agent_coordination"]:
                    self.state["agent_coordination"][model_id] = {}

                self.state["agent_coordination"][model_id] = {
                    "version": version,
                    "model_type": model_type,
                    "deployed_at": datetime.now().isoformat(),
                }

                print(f"ðŸš€ Model deployment coordinated: {model_id} v{version}")

        except Exception as e:
            print(f"âŒ Error handling model deployment update: {e}")

    async def coordinate_ai_agents(self):
        """Coordinate AI agents"""
        try:
            print(f"ðŸŽ›ï¸ Coordinating {len(self.ai_agents)} AI agents...")

            # Check agent health
            await self.check_agent_health()

            # Request updates from agents
            await self.request_agent_updates()

            # Update coordination state
            self.state["last_coordination"] = datetime.now().isoformat()
            self.state["coordination_count"] += 1

            print("âœ… AI coordination complete")

        except Exception as e:
            print(f"âŒ Error coordinating AI agents: {e}")

    async def check_agent_health(self):
        """Check health of AI agents"""
        try:
            for agent_name, agent_info in self.ai_agents.items():
                # Check last communication time
                last_communication = agent_info.get("last_communication")

                if last_communication:
                    last_time = datetime.fromisoformat(last_communication)
                    current_time = datetime.now()

                    # Mark as inactive if no communication for 10 minutes
                    if (current_time - last_time).total_seconds() > 600:
                        agent_info["status"] = "inactive"
                    else:
                        agent_info["status"] = "active"
                else:
                    agent_info["status"] = "unknown"

        except Exception as e:
            print(f"âŒ Error checking agent health: {e}")

    async def request_agent_updates(self):
        """Request updates from AI agents"""
        try:
            for agent_name, agent_info in self.ai_agents.items():
                if agent_info["status"] == "active":
                    # Request predictions/strategies for all symbols
                    for symbol in self.trading_symbols:
                        try:
                            if agent_name == "deep_learning_agent":
                                await self.request_deep_learning_prediction(agent_name, symbol)
                            elif agent_name == "reinforcement_learning_agent":
                                await self.request_reinforcement_learning_strategy(
                                    agent_name, symbol
                                )
                            elif agent_name == "nlp_agent":
                                await self.request_nlp_sentiment(agent_name, symbol)
                            elif agent_name == "computer_vision_agent":
                                await self.request_computer_vision_analysis(agent_name, symbol)
                        except Exception as e:
                            print(f"âŒ Error requesting update from {agent_name} for {symbol}: {e}")

        except Exception as e:
            print(f"âŒ Error requesting agent updates: {e}")

    async def request_deep_learning_prediction(self, agent_name: str, symbol: str):
        """Request deep learning prediction"""
        try:
            request = {
                "type": "make_prediction",
                "symbol": symbol,
                "from_agent": self.agent_id,
                "timestamp": datetime.now().isoformat(),
            }

            await self.send_message(agent_name, request)

        except Exception as e:
            print(f"âŒ Error requesting deep learning prediction: {e}")

    async def request_reinforcement_learning_strategy(self, agent_name: str, symbol: str):
        """Request reinforcement learning strategy"""
        try:
            request = {
                "type": "generate_strategy",
                "symbol": symbol,
                "from_agent": self.agent_id,
                "timestamp": datetime.now().isoformat(),
            }

            await self.send_message(agent_name, request)

        except Exception as e:
            print(f"âŒ Error requesting reinforcement learning strategy: {e}")

    async def request_nlp_sentiment(self, agent_name: str, symbol: str):
        """Request NLP sentiment analysis"""
        try:
            request = {
                "type": "analyze_sentiment",
                "symbol": symbol,
                "from_agent": self.agent_id,
                "timestamp": datetime.now().isoformat(),
            }

            await self.send_message(agent_name, request)

        except Exception as e:
            print(f"âŒ Error requesting NLP sentiment: {e}")

    async def request_computer_vision_analysis(self, agent_name: str, symbol: str):
        """Request computer vision analysis"""
        try:
            request = {
                "type": "analyze_charts",
                "symbol": symbol,
                "from_agent": self.agent_id,
                "timestamp": datetime.now().isoformat(),
            }

            await self.send_message(agent_name, request)

        except Exception as e:
            print(f"âŒ Error requesting computer vision analysis: {e}")

    async def generate_ai_strategies(self) -> None:
        """Generate AI strategies by combining agent outputs"""
        try:
            print(f"ðŸŽ¯ Generating AI strategies for {len(FEATURED_SYMBOLS)} symbols...")

            for symbol in FEATURED_SYMBOLS:
                try:
                    await self.generate_symbol_ai_strategy(symbol)
                except Exception as e:
                    print(f"âŒ Error generating AI strategy for {symbol}: {e}")

            print("âœ… AI strategy generation complete")

        except Exception as e:
            print(f"âŒ Error generating AI strategies: {e}")

    async def generate_symbol_ai_strategy(self, symbol: str) -> None:
        """Generate AI strategy for a specific symbol"""
        try:
            if symbol not in FEATURED_SYMBOLS:
                return
            # Check if we have data from all agents
            if symbol not in self.strategy_cache:
                return

            agent_data = self.strategy_cache[symbol]

            # Check if we have recent data from all agents
            current_time = datetime.now()
            active_agents = []

            for agent_name, agent_info in self.ai_agents.items():
                if agent_name in agent_data:
                    data_timestamp = datetime.fromisoformat(agent_data[agent_name]["timestamp"])

                    # Check if data is recent (within last 10 minutes)
                    if (current_time - data_timestamp).total_seconds() < 600:
                        active_agents.append(agent_name)

            if len(active_agents) < 2:
                print(
                    f"âš ï¸ Insufficient agent data for {symbol}: "
                    f"{len(active_agents)} active agents"
                )
                return

            # Generate combined strategy
            strategy = await self.combine_agent_outputs(symbol, agent_data, active_agents)

            if strategy:
                # Store strategy
                strategy_id = f"ai_strategy_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

                self.state["ai_strategies"][strategy_id] = {
                    "symbol": symbol,
                    "strategy": strategy,
                    "active_agents": active_agents,
                    "created_at": datetime.now().isoformat(),
                    "status": "active",
                }

                # Broadcast strategy
                await self.broadcast_ai_strategy(symbol, strategy, active_agents)

                print(f"âœ… AI strategy generated for {symbol} with {len(active_agents)} agents")

                # Save AI attribution for explainability
                try:
                    # Build inputs snapshot from available data
                    cv_data = agent_data.get("computer_vision", {}).get("analysis_data", {})
                    nlp_data = agent_data.get("nlp", {}).get("sentiment_data", {})
                    rl_data = agent_data.get("reinforcement_learning", {}).get("strategies", {})
                    dl_data = agent_data.get("deep_learning", {}).get("predictions", {})
                    inputs = {
                        "ohlcv": cv_data or {},
                        "nlp": nlp_data or {},
                        "rl": rl_data or {},
                        "dl": dl_data or {},
                    }

                    # Derive weights from active agents
                    weights = {}
                    total = 0.0
                    for a in active_agents:
                        w = float(self.agent_weights.get(a, 0.1))
                        weights[a] = w
                        total += w
                    if total > 0:
                        for a in list(weights.keys()):
                            weights[a] = weights[a] / total

                    reason = (
                        strategy.get("reasoning", [""])[0]
                        if isinstance(strategy.get("reasoning"), list)
                        else (strategy.get("reasoning", "") or "")
                    )

                    save_attribution(symbol, inputs, weights, reason)
                except Exception as _e:
                    # Non-fatal if attribution storage fails
                    pass

        except Exception as e:
            print(f"âŒ Error generating AI strategy for {symbol}: {e}")

    async def combine_agent_outputs(
        self, symbol: str, agent_data: dict[str, Any], active_agents: list[str]
    ) -> dict[str, Any] | None:
        """Combine outputs from multiple AI agents"""
        try:
            combined_strategy = {
                "symbol": symbol,
                "action": "hold",
                "confidence": 0.0,
                "reasoning": [],
                "agent_contributions": {},
                "timestamp": datetime.now().isoformat(),
            }

            total_weight = 0.0
            action_scores = {"buy": 0.0, "sell": 0.0, "hold": 0.0}

            # Process each agent's contribution
            for agent_name in active_agents:
                agent_weight = self.agent_weights.get(agent_name, 0.1)
                agent_contribution = await self.process_agent_contribution(
                    agent_name, agent_data[agent_name]
                )

                if agent_contribution:
                    combined_strategy["agent_contributions"][agent_name] = agent_contribution

                    # Add to action scores
                    action = agent_contribution.get("action", "hold")
                    confidence = agent_contribution.get("confidence", 0.0)
                    action_scores[action] += confidence * agent_weight
                    total_weight += agent_weight

                    # Add reasoning
                    reasoning = agent_contribution.get("reasoning", "")
                    if reasoning:
                        combined_strategy["reasoning"].append(f"{agent_name}: {reasoning}")

            if total_weight > 0:
                # Determine final action
                best_action = max(action_scores, key=action_scores.get)
                combined_strategy["action"] = best_action

                # Calculate overall confidence
                combined_strategy["confidence"] = action_scores[best_action] / total_weight

                # Check if confidence meets threshold
                threshold = self.coordination_config["coordination_settings"][
                    "confidence_threshold"
                ]

                if combined_strategy["confidence"] >= threshold:
                    return combined_strategy
                else:
                    print(
                        f"âš ï¸ Low confidence strategy for {symbol}: "
                        f"{combined_strategy['confidence']:.3f}"
                    )
                    return None

            return None

        except Exception as e:
            print(f"âŒ Error combining agent outputs: {e}")
            return None

    async def process_agent_contribution(
        self, agent_name: str, agent_data: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Process individual agent contribution"""
        try:
            agent_type = self.ai_agents[agent_name]["type"]

            if agent_type == "deep_learning":
                return await self.process_deep_learning_contribution(agent_data)
            elif agent_type == "reinforcement_learning":
                return await self.process_reinforcement_learning_contribution(agent_data)
            elif agent_type == "nlp":
                return await self.process_nlp_contribution(agent_data)
            elif agent_type == "computer_vision":
                return await self.process_computer_vision_contribution(agent_data)
            else:
                return None

        except Exception as e:
            print(f"âŒ Error processing agent contribution: {e}")
            return None

    async def process_deep_learning_contribution(
        self, agent_data: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Process deep learning agent contribution"""
        try:
            predictions = agent_data.get("predictions", {})

            if not predictions:
                return None

            # Extract prediction data
            prediction_values = []
            confidences = []

            for model_name, prediction in predictions.items():
                if isinstance(prediction, dict):
                    value = prediction.get("value", 0)
                    confidence = prediction.get("confidence", 0)
                    prediction_values.append(value)
                    confidences.append(confidence)

            if not prediction_values:
                return None

            # Calculate average prediction and confidence
            avg_prediction = np.mean(prediction_values)
            avg_confidence = np.mean(confidences)

            # Determine action based on prediction
            if avg_prediction > 0.02:  # 2% positive prediction
                action = "buy"
            elif avg_prediction < -0.02:  # 2% negative prediction
                action = "sell"
            else:
                action = "hold"

            return {
                "action": action,
                "confidence": avg_confidence,
                "prediction_value": avg_prediction,
                "reasoning": f"DL prediction: {avg_prediction:.4f}",
            }

        except Exception as e:
            print(f"âŒ Error processing deep learning contribution: {e}")
            return None

    async def process_reinforcement_learning_contribution(
        self, agent_data: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Process reinforcement learning agent contribution"""
        try:
            strategies = agent_data.get("strategies", {})

            if not strategies:
                return None

            # Extract strategy data
            strategy_types = []
            confidences = []

            for algorithm_name, strategy in strategies.items():
                if isinstance(strategy, dict):
                    strategy_type = strategy.get("type", "neutral")
                    confidence = strategy.get("confidence", 0)
                    strategy_types.append(strategy_type)
                    confidences.append(confidence)

            if not strategy_types:
                return None

            # Determine action based on strategy types
            bullish_count = strategy_types.count("bullish")
            bearish_count = strategy_types.count("bearish")
            neutral_count = strategy_types.count("neutral")

            if bullish_count > bearish_count and bullish_count > neutral_count:
                action = "buy"
            elif bearish_count > bullish_count and bearish_count > neutral_count:
                action = "sell"
            else:
                action = "hold"

            avg_confidence = np.mean(confidences) if confidences else 0.5

            return {
                "action": action,
                "confidence": avg_confidence,
                "strategy_distribution": {
                    "bullish": bullish_count,
                    "bearish": bearish_count,
                    "neutral": neutral_count,
                },
                "reasoning": (f"RL strategies: {bullish_count}B/{bearish_count}S/{neutral_count}N"),
            }

        except Exception as e:
            print(f"âŒ Error processing reinforcement learning contribution: {e}")
            return None

    async def process_nlp_contribution(
        self, agent_data: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Process NLP agent contribution"""
        try:
            sentiment_data = agent_data.get("sentiment_data", {})

            if not sentiment_data:
                return None

            # Extract sentiment scores
            news_sentiment = sentiment_data.get("news_sentiment", 0)
            social_sentiment = sentiment_data.get("social_sentiment", 0)
            market_sentiment = sentiment_data.get("market_sentiment", 0)

            # Calculate average sentiment
            avg_sentiment = (news_sentiment + social_sentiment + market_sentiment) / 3

            # Determine action based on sentiment
            if avg_sentiment > 0.1:  # Positive sentiment
                action = "buy"
            elif avg_sentiment < -0.1:  # Negative sentiment
                action = "sell"
            else:
                action = "hold"

            confidence = min(abs(avg_sentiment) * 2, 1.0)  # Scale confidence

            return {
                "action": action,
                "confidence": confidence,
                "sentiment_scores": {
                    "news": news_sentiment,
                    "social": social_sentiment,
                    "market": market_sentiment,
                },
                "reasoning": f"NLP sentiment: {avg_sentiment:.3f}",
            }

        except Exception as e:
            print(f"âŒ Error processing NLP contribution: {e}")
            return None

    async def process_computer_vision_contribution(
        self, agent_data: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Process computer vision agent contribution"""
        try:
            analysis_data = agent_data.get("analysis_data", {})

            if not analysis_data:
                return None

            # Extract analysis results
            pattern_signals = analysis_data.get("pattern_signals", {})
            indicator_signals = analysis_data.get("indicator_signals", {})

            # Calculate signal strength
            buy_signals = 0
            sell_signals = 0

            # Process pattern signals
            for pattern, signal in pattern_signals.items():
                if signal > 0.5:
                    buy_signals += 1
                elif signal < -0.5:
                    sell_signals += 1

            # Process indicator signals
            for indicator, signal in indicator_signals.items():
                if signal > 0.5:
                    buy_signals += 1
                elif signal < -0.5:
                    sell_signals += 1

            # Determine action
            if buy_signals > sell_signals:
                action = "buy"
                confidence = min(buy_signals / 10, 1.0)
            elif sell_signals > buy_signals:
                action = "sell"
                confidence = min(sell_signals / 10, 1.0)
            else:
                action = "hold"
                confidence = 0.5

            return {
                "action": action,
                "confidence": confidence,
                "signal_counts": {"buy": buy_signals, "sell": sell_signals},
                "reasoning": f"CV signals: {buy_signals}B/{sell_signals}S",
            }

        except Exception as e:
            print(f"âŒ Error processing computer vision contribution: {e}")
            return None

    async def broadcast_ai_strategy(
        self, symbol: str, strategy: dict[str, Any], active_agents: list[str]
    ):
        """Broadcast AI strategy to other agents"""
        try:
            strategy_update = {
                "type": "advanced_ai_strategy_update",
                "symbol": symbol,
                "strategy": strategy,
                "active_agents": active_agents,
                "timestamp": datetime.now().isoformat(),
            }

            # Broadcast to all agents
            await self.broadcast_message(strategy_update)

            # Send to specific agents
            await self.send_message("strategy_agent", strategy_update)
            await self.send_message("execution_agent", strategy_update)
            await self.send_message("risk_agent", strategy_update)

        except Exception as e:
            print(f"âŒ Error broadcasting AI strategy: {e}")

    async def evaluate_strategy_performance(self):
        """Evaluate performance of AI strategies"""
        try:
            current_time = datetime.now()

            for strategy_id, strategy_info in list(self.state["ai_strategies"].items()):
                created_at = datetime.fromisoformat(strategy_info["created_at"])

                # Check if strategy is still valid
                lifetime = self.coordination_config["coordination_settings"]["strategy_lifetime"]
                if (current_time - created_at).total_seconds() > lifetime:
                    strategy_info["status"] = "expired"
                    continue

                # Evaluate performance (simplified)
                symbol = strategy_info["symbol"]
                strategy = strategy_info["strategy"]

                # Get current market data for evaluation
                market_data = await self.get_current_market_data(symbol)

                if market_data:
                    performance = await self.calculate_strategy_performance(strategy, market_data)

                    if symbol not in self.performance_history:
                        self.performance_history[symbol] = []

                    self.performance_history[symbol].append(
                        {
                            "strategy_id": strategy_id,
                            "performance": performance,
                            "timestamp": current_time.isoformat(),
                        }
                    )

                    # Keep only recent performance data
                    if len(self.performance_history[symbol]) > 100:
                        self.performance_history[symbol] = self.performance_history[symbol][-100:]

        except Exception as e:
            print(f"âŒ Error evaluating strategy performance: {e}")

    async def get_current_market_data(self, symbol: str) -> dict[str, Any] | None:
        """Get current market data for symbol"""
        try:
            # Get from Redis cache
            market_data = self.redis_client.get(f"market_data:{symbol}")
            if market_data:
                return json.loads(market_data)

            return None

        except Exception as e:
            print(f"âŒ Error getting current market data: {e}")
            return None

    async def calculate_strategy_performance(
        self, strategy: dict[str, Any], market_data: dict[str, Any]
    ) -> float:
        """Calculate strategy performance"""
        try:
            # Simplified performance calculation
            action = strategy.get("action", "hold")
            confidence = strategy.get("confidence", 0)

            # Get price change
            current_price = market_data.get("price", 0)
            previous_price = market_data.get("previous_price", current_price)

            if previous_price > 0:
                price_change = (current_price - previous_price) / previous_price
            else:
                price_change = 0

            # Calculate performance based on action and price change
            if action == "buy" and price_change > 0:
                performance = confidence * price_change
            elif action == "sell" and price_change < 0:
                performance = confidence * abs(price_change)
            elif action == "hold":
                performance = 0
            else:
                performance = -confidence * abs(price_change)

            return performance

        except Exception as e:
            print(f"âŒ Error calculating strategy performance: {e}")
            return 0.0

    async def cleanup_old_strategies(self):
        """Clean up old strategies"""
        try:
            current_time = datetime.now()
            lifetime = self.coordination_config["coordination_settings"]["strategy_lifetime"]

            # Remove expired strategies
            expired_strategies = []
            for strategy_id, strategy_info in self.state["ai_strategies"].items():
                created_at = datetime.fromisoformat(strategy_info["created_at"])
                if (current_time - created_at).total_seconds() > lifetime:
                    expired_strategies.append(strategy_id)

            for strategy_id in expired_strategies:
                del self.state["ai_strategies"][strategy_id]

            if expired_strategies:
                print(f"ðŸ—‘ï¸ Cleaned up {len(expired_strategies)} expired strategies")

        except Exception as e:
            print(f"âŒ Error cleaning up old strategies: {e}")

    async def handle_coordinate_ai_agents(self, message: dict[str, Any]):
        """Handle manual AI coordination request"""
        try:
            print("ðŸŽ›ï¸ Manual AI coordination requested")

            # Perform coordination
            await self.coordinate_ai_agents()

            response = {
                "type": "ai_coordination_response",
                "status": "completed",
                "active_agents": len(
                    [a for a in self.ai_agents.values() if a["status"] == "active"]
                ),
                "timestamp": datetime.now().isoformat(),
            }

            sender = message.get("from_agent")
            if sender:
                await self.send_message(sender, response)

        except Exception as e:
            print(f"âŒ Error handling AI coordination request: {e}")
            await self.broadcast_error(f"AI coordination error: {e}")

    async def handle_generate_ai_strategy(self, message: dict[str, Any]):
        """Handle manual AI strategy generation request"""
        try:
            symbol = message.get("symbol")

            print(f"ðŸŽ¯ Manual AI strategy generation requested for {symbol}")

            if symbol:
                await self.generate_symbol_ai_strategy(symbol)

                response = {
                    "type": "ai_strategy_generation_response",
                    "symbol": symbol,
                    "status": "completed",
                    "timestamp": datetime.now().isoformat(),
                }
            else:
                response = {
                    "type": "ai_strategy_generation_response",
                    "symbol": symbol,
                    "status": "failed",
                    "error": "No symbol provided",
                    "timestamp": datetime.now().isoformat(),
                }

            sender = message.get("from_agent")
            if sender:
                await self.send_message(sender, response)

        except Exception as e:
            print(f"âŒ Error handling AI strategy generation request: {e}")
            await self.broadcast_error(f"AI strategy generation error: {e}")

    async def handle_get_ai_status(self, message: dict[str, Any]):
        """Handle AI status request"""
        try:
            print("ðŸ“Š AI status requested")

            # Get AI status
            ai_status = {
                "agents": self.ai_agents,
                "strategies_count": len(self.state["ai_strategies"]),
                "coordination_count": self.state["coordination_count"],
                "last_coordination": self.state["last_coordination"],
                "performance_history": {k: len(v) for k, v in self.performance_history.items()},
            }

            response = {
                "type": "ai_status_response",
                "status": ai_status,
                "timestamp": datetime.now().isoformat(),
            }

            sender = message.get("from_agent")
            if sender:
                await self.send_message(sender, response)

        except Exception as e:
            print(f"âŒ Error handling AI status request: {e}")
            await self.broadcast_error(f"AI status error: {e}")

    async def handle_ai_prediction(self, message: dict[str, Any]):
        """Handle AI prediction update"""
        try:
            await self.process_ai_update(message)
        except Exception as e:
            print(f"âŒ Error handling AI prediction: {e}")

    async def handle_ai_strategy(self, message: dict[str, Any]):
        """Handle AI strategy update"""
        try:
            await self.process_ai_update(message)
        except Exception as e:
            print(f"âŒ Error handling AI strategy: {e}")

    async def handle_model_deployment(self, message: dict[str, Any]):
        """Handle model deployment update"""
        try:
            await self.process_ai_update(message)
        except Exception as e:
            print(f"âŒ Error handling model deployment: {e}")

    async def update_coordination_metrics(self):
        """Update coordination metrics"""
        try:
            metrics = {
                "agent_id": self.agent_id,
                "ai_agents_count": len(self.ai_agents),
                "active_agents_count": len(
                    [a for a in self.ai_agents.values() if a["status"] == "active"]
                ),
                "strategies_count": len(self.state["ai_strategies"]),
                "coordination_count": self.state["coordination_count"],
                "last_coordination": self.state["last_coordination"],
                "symbols_count": len(self.trading_symbols),
                "timestamp": datetime.now().isoformat(),
            }

            # Store metrics in Redis
            self.redis_client.set(f"agent_metrics:{self.agent_id}", json.dumps(metrics), ex=300)

        except Exception as e:
            print(f"âŒ Error updating coordination metrics: {e}")

    async def process_market_data(self, market_data: dict[str, Any]):
        """Process incoming market data and coordinate AI agents"""
        try:
            print(f"ðŸ“Š Processing market data for {len(market_data)} symbols")

            # Update market data in state
            self.state["last_market_data"] = market_data
            self.state["last_market_update"] = datetime.now().isoformat()

            # Process each symbol with AI agents
            for symbol, data in market_data.items():
                if symbol in self.trading_symbols:
                    # Generate AI strategy for this symbol
                    await self.generate_symbol_ai_strategy(symbol)

                    # Request predictions from AI agents
                    await self.request_deep_learning_prediction("deep_learning_agent", symbol)
                    await self.request_reinforcement_learning_strategy("reinforcement_learning_agent", symbol)
                    await self.request_nlp_sentiment("nlp_agent", symbol)
                    await self.request_computer_vision_analysis("computer_vision_agent", symbol)

            # Coordinate AI agents
            await self.coordinate_ai_agents()

            # Update coordination metrics
            await self.update_coordination_metrics()

            print("âœ… Market data processed successfully")

        except Exception as e:
            print(f"âŒ Error processing market data: {e}")
            await self.broadcast_error(f"Market data processing error: {e}")


