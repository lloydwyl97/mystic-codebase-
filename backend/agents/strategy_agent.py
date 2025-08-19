"""
Strategy Agent
Handles AI strategy generation, analysis, and optimization
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from typing import Any

import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.agents.base_agent import BaseAgent

# import backend.ai as ai components with fallback
try:
    from ai_strategy_generator import AIStrategyGenerator
except ImportError:
    try:
        from ..ai_strategy_generator import AIStrategyGenerator
    except ImportError:
        AIStrategyGenerator = None

try:
    from ai_auto_retrain import AutoRetrainService
except ImportError:
    try:
        from ..ai_auto_retrain import AutoRetrainService
    except ImportError:
        AutoRetrainService = None

logger = logging.getLogger(__name__)


class StrategyAgent(BaseAgent):
    """AI Strategy Agent - Handles strategy generation and analysis"""

    def __init__(self, agent_id: str = "strategy_agent_001"):
        super().__init__(agent_id, "strategy")

        # Strategy-specific state
        self.state.update(
            {
                "active_strategies": [],
                "strategy_performance": {},
                "generation_count": 0,
                "last_generation": None,
                "optimization_running": False,
            }
        )

        # Initialize strategy components
        self.strategy_generator = None
        self.auto_retrain = None

        # Register strategy-specific handlers
        self.register_handler("generate_strategy", self.handle_generate_strategy)
        self.register_handler("analyze_strategy", self.handle_analyze_strategy)
        self.register_handler("optimize_strategy", self.handle_optimize_strategy)
        self.register_handler("market_data", self.handle_market_data)
        self.register_handler("performance_update", self.handle_performance_update)

        print(f"ðŸ§  Strategy Agent {agent_id} initialized")

    async def initialize(self):
        """Initialize strategy agent resources"""
        try:
            # Initialize AI strategy generator
            self.strategy_generator = AIStrategyGenerator()

            # Initialize auto-retrain system
            self.auto_retrain = AutoRetrainService()

            # Load existing strategies
            await self.load_active_strategies()

            # Subscribe to market data
            await self.subscribe_to_market_data()

            print(f"âœ… Strategy Agent {self.agent_id} initialized successfully")

        except Exception as e:
            print(f"âŒ Error initializing Strategy Agent: {e}")
            self.update_health_status("error")

    async def process_loop(self):
        """Main strategy processing loop"""
        while self.running:
            try:
                # Monitor strategy performance
                await self.monitor_strategy_performance()

                # Check for strategy optimization opportunities
                await self.check_optimization_opportunities()

                # Update strategy metrics
                await self.update_strategy_metrics()

                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                print(f"âŒ Error in strategy processing loop: {e}")
                await asyncio.sleep(120)

    async def load_active_strategies(self):
        """Load active strategies from Redis"""
        try:
            from utils.redis_helpers import to_str_list
            active_strategies = to_str_list(self.redis_client.lrange("ai_strategies", 0, -1))

            for strategy_id in active_strategies:
                strategy_data = self.redis_client.get(f"ai_strategy:{strategy_id}")
                if strategy_data:
                    strategy = json.loads(strategy_data)
                    self.state["active_strategies"].append(strategy)

            print(f"ðŸ“Š Loaded {len(self.state['active_strategies'])} active strategies")

        except Exception as e:
            print(f"âŒ Error loading active strategies: {e}")

    async def subscribe_to_market_data(self):
        """Subscribe to market data updates"""
        try:
            # Subscribe to market data channel
            pubsub = self.redis_client.pubsub()
            pubsub.subscribe("market_data")

            # Start market data listener
            asyncio.create_task(self.listen_market_data(pubsub))

            print("ðŸ“¡ Strategy Agent subscribed to market data")

        except Exception as e:
            print(f"âŒ Error subscribing to market data: {e}")

    async def listen_market_data(self, pubsub):
        """Listen for market data updates"""
        try:
            for message in pubsub.listen():
                if not self.running:
                    break

                if message["type"] == "message":
                    market_data = json.loads(message["data"])
                    await self.process_market_data(market_data)

        except Exception as e:
            print(f"âŒ Error in market data listener: {e}")
        finally:
            pubsub.close()

    async def process_market_data(self, market_data: dict[str, Any]):
        """Process incoming market data"""
        try:
            symbol = market_data.get("symbol")
            price = market_data.get("price")
            timestamp = market_data.get("timestamp")

            # Update strategy performance with new market data
            await self.update_strategy_performance(symbol, price, timestamp)

            # Check for strategy signals
            await self.check_strategy_signals(market_data)

        except Exception as e:
            print(f"âŒ Error processing market data: {e}")

    async def handle_generate_strategy(self, message: dict[str, Any]):
        """Handle strategy generation request"""
        try:
            symbol = message.get("symbol", "BTCUSDT")
            strategy_type = message.get("strategy_type", "lstm")
            parameters = message.get("parameters", {})

            print(f"ðŸŽ¯ Generating strategy for {symbol} ({strategy_type})")

            # Generate new strategy
            strategy = await self.generate_strategy(symbol, strategy_type, parameters)

            # Send response
            response = {
                "type": "strategy_generated",
                "strategy_id": strategy.get("id"),
                "symbol": symbol,
                "strategy_type": strategy_type,
                "performance": strategy.get("performance", {}),
                "timestamp": datetime.now().isoformat(),
            }

            sender = message.get("from_agent")
            if sender:
                await self.send_message(sender, response)

            # Broadcast strategy generation
            await self.broadcast_message({"type": "strategy_generated", "strategy": strategy})

        except Exception as e:
            print(f"âŒ Error generating strategy: {e}")
            await self.broadcast_error(f"Strategy generation error: {e}")

    async def handle_analyze_strategy(self, message: dict[str, Any]):
        """Handle strategy analysis request"""
        try:
            strategy_id = message.get("strategy_id")

            print(f"ðŸ“Š Analyzing strategy {strategy_id}")

            # Analyze strategy
            analysis = await self.analyze_strategy(strategy_id)

            # Send response
            response = {
                "type": "strategy_analysis",
                "strategy_id": strategy_id,
                "analysis": analysis,
                "timestamp": datetime.now().isoformat(),
            }

            sender = message.get("from_agent")
            if sender:
                await self.send_message(sender, response)

        except Exception as e:
            print(f"âŒ Error analyzing strategy: {e}")
            await self.broadcast_error(f"Strategy analysis error: {e}")

    async def handle_optimize_strategy(self, message: dict[str, Any]):
        """Handle strategy optimization request"""
        try:
            strategy_id = message.get("strategy_id")
            optimization_type = message.get("optimization_type", "genetic")

            print(f"âš¡ Optimizing strategy {strategy_id} ({optimization_type})")

            # Start optimization
            await self.optimize_strategy(strategy_id, optimization_type)

            # Send response
            response = {
                "type": "optimization_started",
                "strategy_id": strategy_id,
                "optimization_type": optimization_type,
                "timestamp": datetime.now().isoformat(),
            }

            sender = message.get("from_agent")
            if sender:
                await self.send_message(sender, response)

        except Exception as e:
            print(f"âŒ Error optimizing strategy: {e}")
            await self.broadcast_error(f"Strategy optimization error: {e}")

    async def handle_performance_update(self, message: dict[str, Any]):
        """Handle performance update"""
        try:
            strategy_id = message.get("strategy_id")
            performance = message.get("performance", {})

            # Update strategy performance
            self.state["strategy_performance"][strategy_id] = performance

            # Check if retrain is needed
            await self.check_retrain_needed(strategy_id, performance)

        except Exception as e:
            print(f"âŒ Error handling performance update: {e}")

    async def handle_market_data(self, message: dict[str, Any]):
        """Handle market data message"""
        try:
            market_data = message.get("market_data", {})
            print(f"ðŸ“Š Strategy Agent received market data for {len(market_data)} symbols")
            
            # Process market data
            await self.process_market_data(market_data)
            
            # Check strategy signals
            await self.check_strategy_signals(market_data)
            
        except Exception as e:
            print(f"âŒ Error handling market data: {e}")
            await self.broadcast_error(f"Market data handling error: {e}")

    async def generate_strategy(
        self, symbol: str, strategy_type: str, parameters: dict[str, Any]
    ) -> dict[str, Any]:
        """Generate a new AI strategy"""
        try:
            # Use strategy generator to create new strategy
            strategy = await self.strategy_generator.generate_strategy(
                symbol=symbol,
                strategy_type=strategy_type,
                parameters=parameters,
            )

            # Add to active strategies
            self.state["active_strategies"].append(strategy)
            self.state["generation_count"] += 1
            self.state["last_generation"] = datetime.now().isoformat()

            # Store in Redis
            self.redis_client.set(f"ai_strategy:{strategy['id']}", json.dumps(strategy), ex=86400)
            self.redis_client.lpush("ai_strategies", strategy["id"])

            print(f"âœ… Generated strategy {strategy['id']} for {symbol}")

            return strategy

        except Exception as e:
            print(f"âŒ Error generating strategy: {e}")
            raise

    async def analyze_strategy(self, strategy_id: str) -> dict[str, Any]:
        """Analyze strategy performance"""
        try:
            # Get strategy data
            strategy_data = self.redis_client.get(f"ai_strategy:{strategy_id}")
            if not strategy_data:
                from backend.utils.exceptions import ErrorCode, StrategyException
                raise StrategyException(
                    message=f"Strategy {strategy_id} not found",
                    error_code=ErrorCode.VALIDATION_ERROR,
                    details={"strategy_id": strategy_id}
                )

            strategy = json.loads(strategy_data)

            # Perform analysis
            analysis = {
                "strategy_id": strategy_id,
                "performance_metrics": strategy.get("performance", {}),
                "risk_metrics": await self.calculate_risk_metrics(strategy),
                "optimization_potential": (await self.assess_optimization_potential(strategy)),
                "market_conditions": await self.analyze_market_conditions(strategy),
                "recommendations": await self.generate_recommendations(strategy),
            }

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing strategy: {e}")
            raise

    async def optimize_strategy(self, strategy_id: str, optimization_type: str):
        """Optimize strategy using genetic algorithm"""
        try:
            self.state["optimization_running"] = True

            # Get strategy data
            strategy_data = self.redis_client.get(f"ai_strategy:{strategy_id}")
            if not strategy_data:
                from backend.utils.exceptions import ErrorCode, StrategyException
                raise StrategyException(
                    message=f"Strategy {strategy_id} not found",
                    error_code=ErrorCode.VALIDATION_ERROR,
                    details={"strategy_id": strategy_id}
                )

            strategy = json.loads(strategy_data)

            # Run optimization
            optimized_strategy = await self.strategy_generator.optimize_strategy(
                strategy=strategy, optimization_type=optimization_type
            )

            # Update strategy
            self.redis_client.set(
                f"ai_strategy:{strategy_id}",
                json.dumps(optimized_strategy),
                ex=86400,
            )

            # Broadcast optimization complete
            await self.broadcast_message(
                {
                    "type": "optimization_complete",
                    "strategy_id": strategy_id,
                    "improvements": optimized_strategy.get("improvements", {}),
                }
            )

            self.state["optimization_running"] = False

        except Exception as e:
            logger.error(f"Error optimizing strategy: {e}")
            self.state["optimization_running"] = False
            raise

    async def monitor_strategy_performance(self):
        """Monitor performance of all active strategies"""
        try:
            for strategy in self.state["active_strategies"]:
                strategy_id = strategy.get("id")

                # Get current performance
                performance = self.state["strategy_performance"].get(strategy_id, {})

                # Check for performance degradation
                if await self.is_performance_degrading(performance):
                    print(f"âš ï¸ Performance degrading for strategy {strategy_id}")

                    # Trigger retrain
                    await self.trigger_retrain(strategy_id)

        except Exception as e:
            print(f"âŒ Error monitoring strategy performance: {e}")

    async def check_optimization_opportunities(self):
        """Check for optimization opportunities"""
        try:
            for strategy in self.state["active_strategies"]:
                strategy_id = strategy.get("id")

                # Check if optimization is beneficial
                if await self.should_optimize(strategy):
                    print(f"ðŸŽ¯ Optimization opportunity detected for strategy {strategy_id}")

                    # Send optimization request to self
                    await self.send_message(
                        self.agent_id,
                        {
                            "type": "optimize_strategy",
                            "strategy_id": strategy_id,
                            "optimization_type": "genetic",
                        },
                    )

        except Exception as e:
            print(f"âŒ Error checking optimization opportunities: {e}")

    async def update_strategy_metrics(self):
        """Update strategy metrics"""
        try:
            metrics = {
                "agent_id": self.agent_id,
                "active_strategies_count": len(self.state["active_strategies"]),
                "generation_count": self.state["generation_count"],
                "optimization_running": self.state["optimization_running"],
                "timestamp": datetime.now().isoformat(),
            }

            # Store metrics in Redis
            self.redis_client.set(f"agent_metrics:{self.agent_id}", json.dumps(metrics), ex=300)

        except Exception as e:
            print(f"âŒ Error updating strategy metrics: {e}")

    async def calculate_risk_metrics(self, strategy: dict[str, Any]) -> dict[str, Any]:
        """Calculate risk metrics for strategy"""
        try:
            performance = strategy.get("performance", {})

            # Calculate various risk metrics
            risk_metrics = {
                "max_drawdown": performance.get("max_drawdown", 0),
                "volatility": performance.get("volatility", 0),
                "var_95": performance.get("var_95", 0),
                "sharpe_ratio": performance.get("sharpe_ratio", 0),
                "risk_score": self.calculate_risk_score(performance),
            }

            return risk_metrics

        except Exception as e:
            print(f"âŒ Error calculating risk metrics: {e}")
            return {}

    def calculate_risk_score(self, performance: dict[str, Any]) -> float:
        """Calculate overall risk score"""
        try:
            # Simple risk score calculation
            sharpe = performance.get("sharpe_ratio", 0)
            max_dd = performance.get("max_drawdown", 0)
            volatility = performance.get("volatility", 0)

            # Risk score (0-100, lower is better)
            risk_score = max(0, 100 - (sharpe * 20) + (max_dd * 50) + (volatility * 30))

            return min(100, risk_score)

        except Exception as e:
            print(f"âŒ Error calculating risk score: {e}")
            return 50.0

    async def assess_optimization_potential(self, strategy: dict[str, Any]) -> dict[str, Any]:
        """Assess optimization potential"""
        try:
            performance = strategy.get("performance", {})

            # Assess different optimization opportunities
            optimization_potential = {
                "parameter_tuning": self.assess_parameter_tuning(performance),
                "feature_engineering": self.assess_feature_engineering(performance),
                "model_architecture": self.assess_model_architecture(performance),
                "overall_potential": 0,
            }

            # Calculate overall potential
            optimization_potential["overall_potential"] = (
                optimization_potential["parameter_tuning"]
                + optimization_potential["feature_engineering"]
                + optimization_potential["model_architecture"]
            ) / 3

            return optimization_potential

        except Exception as e:
            print(f"âŒ Error assessing optimization potential: {e}")
            return {}

    def assess_parameter_tuning(self, performance: dict[str, Any]) -> float:
        """Assess parameter tuning potential (0-100)"""
        accuracy = performance.get("accuracy", 0)
        if accuracy < 0.7:
            return 80  # High potential for improvement
        elif accuracy < 0.8:
            return 50  # Medium potential
        else:
            return 20  # Low potential

    def assess_feature_engineering(self, performance: dict[str, Any]) -> float:
        """Assess feature engineering potential (0-100)"""
        # Simple assessment based on performance
        return 60  # Medium potential

    def assess_model_architecture(self, performance: dict[str, Any]) -> float:
        """Assess model architecture potential (0-100)"""
        # Simple assessment based on performance
        return 40  # Low-medium potential

    async def analyze_market_conditions(self, strategy: dict[str, Any]) -> dict[str, Any]:
        """Analyze current market conditions for strategy"""
        try:
            symbol = strategy.get("symbol", "BTCUSDT")

            # Get recent market data
            market_data = self.redis_client.get(f"market_data:{symbol}")
            if market_data:
                market = json.loads(market_data)

                # Analyze market conditions
                market_conditions = {
                    "volatility": self.calculate_market_volatility(market),
                    "trend": self.analyze_market_trend(market),
                    "volume": self.analyze_market_volume(market),
                    "suitability": self.assess_market_suitability(strategy, market),
                }

                return market_conditions

            return {}

        except Exception as e:
            print(f"âŒ Error analyzing market conditions: {e}")
            return {}

    def calculate_market_volatility(self, market_data: dict[str, Any]) -> float:
        """Calculate market volatility"""
        try:
            # Simple volatility calculation
            return 0.15  # Placeholder
        except Exception as e:
            print(f"âŒ Error calculating market volatility: {e}")
            return 0.0

    def analyze_market_trend(self, market_data: dict[str, Any]) -> str:
        """Analyze market trend"""
        try:
            # Simple trend analysis
            return "bullish"  # Placeholder
        except Exception as e:
            print(f"âŒ Error analyzing market trend: {e}")
            return "neutral"

    def analyze_market_volume(self, market_data: dict[str, Any]) -> str:
        """Analyze market volume"""
        try:
            # Simple volume analysis
            return "normal"  # Placeholder
        except Exception as e:
            print(f"âŒ Error analyzing market volume: {e}")
            return "normal"

    def assess_market_suitability(
        self, strategy: dict[str, Any], market_data: dict[str, Any]
    ) -> float:
        """Assess how suitable current market conditions are for strategy"""
        try:
            # Simple suitability assessment
            return 0.75  # Placeholder
        except Exception as e:
            print(f"âŒ Error assessing market suitability: {e}")
            return 0.5

    async def generate_recommendations(self, strategy: dict[str, Any]) -> list[str]:
        """Generate recommendations for strategy improvement"""
        try:
            recommendations = []
            performance = strategy.get("performance", {})

            # Generate recommendations based on performance
            if performance.get("accuracy", 0) < 0.7:
                recommendations.append("Consider retraining with more data")

            if performance.get("sharpe_ratio", 0) < 1.0:
                recommendations.append("Optimize risk management parameters")

            if performance.get("max_drawdown", 0) > 0.2:
                recommendations.append("Implement stricter stop-loss mechanisms")

            if not recommendations:
                recommendations.append("Strategy performing well - monitor for changes")

            return recommendations

        except Exception as e:
            print(f"âŒ Error generating recommendations: {e}")
            return ["Unable to generate recommendations"]

    async def is_performance_degrading(self, performance: dict[str, Any]) -> bool:
        """Check if strategy performance is degrading"""
        try:
            accuracy = performance.get("accuracy", 0)
            sharpe = performance.get("sharpe_ratio", 0)

            # Simple degradation check
            return accuracy < 0.6 or sharpe < 0.5

        except Exception as e:
            print(f"âŒ Error checking performance degradation: {e}")
            return False

    async def should_optimize(self, strategy: dict[str, Any]) -> bool:
        """Check if strategy should be optimized"""
        try:
            performance = strategy.get("performance", {})
            accuracy = performance.get("accuracy", 0)

            # Simple optimization trigger
            return accuracy < 0.75 and not self.state["optimization_running"]

        except Exception as e:
            print(f"âŒ Error checking optimization need: {e}")
            return False

    async def trigger_retrain(self, strategy_id: str):
        """Trigger retrain for strategy"""
        try:
            print(f"ðŸ”„ Triggering retrain for strategy {strategy_id}")

            # Send retrain request to auto-retrain service
            await self.send_message(
                "auto_retrain_agent",
                {
                    "type": "retrain_request",
                    "strategy_id": strategy_id,
                    "reason": "performance_degradation",
                    "priority": "high",
                },
            )

        except Exception as e:
            print(f"âŒ Error triggering retrain: {e}")

    async def check_strategy_signals(self, market_data: dict[str, Any]):
        """Check for trading signals from strategies"""
        try:
            symbol = market_data.get("symbol")

            # Check each strategy for signals
            for strategy in self.state["active_strategies"]:
                if strategy.get("symbol") == symbol:
                    signal = await self.generate_signal(strategy, market_data)

                    if signal:
                        # Send signal to execution agent
                        await self.send_message(
                            "execution_agent",
                            {
                                "type": "trading_signal",
                                "strategy_id": strategy.get("id"),
                                "signal": signal,
                                "market_data": market_data,
                            },
                        )

        except Exception as e:
            print(f"âŒ Error checking strategy signals: {e}")

    async def generate_signal(
        self, strategy: dict[str, Any], market_data: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Generate trading signal from strategy"""
        try:
            # Simple signal generation (placeholder)
            # In real implementation, this would use the actual strategy model

            price = market_data.get("price", 0)
            if price > 0:
                # Generate random signal for demonstration
                signal_type = np.random.choice(["buy", "sell", "hold"], p=[0.3, 0.3, 0.4])

                if signal_type != "hold":
                    return {
                        "type": signal_type,
                        "confidence": np.random.uniform(0.6, 0.9),
                        "price": price,
                        "timestamp": datetime.now().isoformat(),
                    }

            return None

        except Exception as e:
            print(f"âŒ Error generating signal: {e}")
            return None


