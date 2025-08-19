"""
Enhanced Signal Manager for Mystic Trading

Manages all trading signals and ensures they are active and properly integrated.
"""

import json
import logging
import random
import time
from datetime import datetime, timezone
from typing import Any

from notification_service import get_notification_service

logger = logging.getLogger(__name__)


class SignalManager:
    def __init__(self, redis_client: Any):
        self.redis_client = redis_client
        self.active_signals = {}
        self.signal_generators = {}
        self.auto_trading_enabled = False

        # Initialize notification service
        self.notification_service = get_notification_service(redis_client)

        # Track previous health state for change detection
        self.previous_health_state = None

        # Initialize all signal types
        self.signal_types = {
            "market_data": {
                "enabled": True,
                "priority": "high",
                "update_interval": 1,  # seconds
                "description": "Real-time market data signals",
            },
            "technical_indicators": {
                "enabled": True,
                "priority": "high",
                "update_interval": 5,  # seconds
                "description": "Technical analysis indicators",
            },
            "sentiment_analysis": {
                "enabled": True,
                "priority": "medium",
                "update_interval": 30,  # seconds
                "description": "Social sentiment and news analysis",
            },
            "order_flow": {
                "enabled": True,
                "priority": "high",
                "update_interval": 2,  # seconds
                "description": "Order book and flow analysis",
            },
            "liquidity_analysis": {
                "enabled": True,
                "priority": "medium",
                "update_interval": 10,  # seconds
                "description": "Liquidity and volume analysis",
            },
            "risk_metrics": {
                "enabled": True,
                "priority": "high",
                "update_interval": 5,  # seconds
                "description": "Risk assessment and management",
            },
            "auto_trading": {
                "enabled": True,
                "priority": "critical",
                "update_interval": 1,  # seconds
                "description": "Automated trading execution",
            },
            "portfolio_tracking": {
                "enabled": True,
                "priority": "high",
                "update_interval": 5,  # seconds
                "description": "Portfolio performance tracking",
            },
            "real_time_alerts": {
                "enabled": True,
                "priority": "critical",
                "update_interval": 1,  # seconds
                "description": "Real-time trading alerts",
            },
            "cosmic_alignment": {
                "enabled": True,
                "priority": "low",
                "update_interval": 300,  # 5 minutes
                "description": "Cosmic alignment analysis",
            },
            "pattern_recognition": {
                "enabled": True,
                "priority": "medium",
                "update_interval": 15,  # seconds
                "description": "Chart pattern recognition",
            },
            "machine_learning": {
                "enabled": True,
                "priority": "high",
                "update_interval": 30,  # seconds
                "description": "ML-based predictions",
            },
        }

        # Trading strategies
        self.trading_strategies = {
            "scalping": {"enabled": True, "min_confidence": 0.7},
            "swing_trading": {"enabled": True, "min_confidence": 0.6},
            "arbitrage": {"enabled": True, "min_confidence": 0.8},
            "momentum": {"enabled": True, "min_confidence": 0.65},
            "mean_reversion": {"enabled": True, "min_confidence": 0.6},
            "grid_trading": {"enabled": True, "min_confidence": 0.5},
            "statistical_arbitrage": {"enabled": True, "min_confidence": 0.75},
            "market_making": {"enabled": True, "min_confidence": 0.6},
            "high_frequency_trading": {"enabled": True, "min_confidence": 0.8},
            "options_trading": {"enabled": True, "min_confidence": 0.7},
            "futures_trading": {"enabled": True, "min_confidence": 0.7},
        }

    async def activate_all_signals(self) -> dict[str, Any]:
        """Activate all trading signals"""
        try:
            logger.info("Activating all trading signals...")

            # Set all signals to active
            for signal_type, config in self.signal_types.items():
                config["enabled"] = True
                config["last_update"] = datetime.now(timezone.utc).isoformat()
                config["status"] = "active"
                logger.debug(f"Activated signal type: {signal_type}")

            # Store in Redis
            self.redis_client.setex("signal_status", 3600, json.dumps(self.signal_types))

            # Store trading strategies
            self.redis_client.setex("trading_strategies", 3600, json.dumps(self.trading_strategies))

            # Set auto-trading flag
            self.auto_trading_enabled = True
            self.redis_client.setex(
                "auto_trading_enabled",
                3600,
                json.dumps(
                    {
                        "enabled": True,
                        "activated_at": datetime.now(timezone.utc).isoformat(),
                    }
                ),
            )

            logger.info("All trading signals activated successfully")

            return {
                "status": "success",
                "message": "All trading signals activated",
                "signals": self.signal_types,
                "strategies": self.trading_strategies,
                "auto_trading_enabled": True,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as e:
            logger.error(f"Error activating signals: {str(e)}")
            raise

    async def get_signal_status(self) -> dict[str, Any]:
        """Get current status of all signals"""
        try:
            # Get from Redis
            signal_data = self.redis_client.get("signal_status")
            strategy_data = self.redis_client.get("trading_strategies")
            auto_trade_data = self.redis_client.get("auto_trading_enabled")

            signals = json.loads(signal_data) if signal_data else self.signal_types
            strategies = json.loads(strategy_data) if strategy_data else self.trading_strategies
            auto_trading = json.loads(auto_trade_data) if auto_trade_data else {"enabled": False}

            return {
                "status": "success",
                "signals": signals,
                "strategies": strategies,
                "auto_trading": auto_trading,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as e:
            logger.error(f"Error getting signal status: {str(e)}")
            raise

    async def generate_live_signal(self, symbol: str) -> dict[str, Any]:
        """Generate live trading signal for a symbol"""
        try:
            # Check if signals are active
            signal_data = self.redis_client.get("signal_status")
            if not signal_data:
                return {
                    "status": "inactive",
                    "message": "Signals not active",
                    "symbol": symbol,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }

            signals = json.loads(signal_data)
            if not signals.get("auto_trading", {}).get("enabled", False):
                return {
                    "status": "inactive",
                    "message": "Auto-trading not enabled",
                    "symbol": symbol,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }

            # Generate comprehensive signal
            signal_strength = random.uniform(0.3, 0.95)
            signal_type = (
                "BUY" if signal_strength > 0.6 else "SELL" if signal_strength < 0.4 else "HOLD"
            )

            # Generate technical indicators
            indicators = {
                "rsi": round(random.uniform(20, 80), 2),
                "macd": round(random.uniform(-2, 2), 4),
                "bollinger_position": round(random.uniform(0, 1), 2),
                "stochastic": round(random.uniform(0, 100), 2),
                "volume_sma": round(random.uniform(0.5, 2.0), 2),
                "price_sma": round(random.uniform(0.95, 1.05), 3),
            }

            # Generate risk metrics
            risk_metrics = {
                "volatility": round(random.uniform(0.1, 0.4), 3),
                "sharpe_ratio": round(random.uniform(0.5, 2.5), 2),
                "max_drawdown": round(random.uniform(0.05, 0.2), 3),
                "var_95": round(random.uniform(0.02, 0.1), 3),
                "beta": round(random.uniform(0.5, 1.5), 2),
                "correlation": round(random.uniform(-0.3, 0.8), 2),
            }

            # Generate sentiment data
            sentiment = {
                "social_score": round(random.uniform(20, 80), 1),
                "news_sentiment": random.choice(["positive", "neutral", "negative"]),
                "fear_greed_index": round(random.uniform(20, 80), 1),
                "market_mood": random.choice(["bullish", "bearish", "neutral"]),
            }

            # Generate order flow data
            order_flow = {
                "bid_volume": round(random.uniform(1000000, 5000000), 0),
                "ask_volume": round(random.uniform(1000000, 5000000), 0),
                "bid_ask_ratio": round(random.uniform(0.5, 2.0), 2),
                "large_orders": round(random.uniform(0, 10), 0),
                "order_imbalance": round(random.uniform(-0.3, 0.3), 3),
            }

            # Generate strategy recommendations
            strategies: list[dict[str, Any]] = []
            for strategy, config in self.trading_strategies.items():
                if config["enabled"]:
                    confidence = random.uniform(0.4, 0.9)
                    if confidence >= config["min_confidence"]:
                        strategies.append(
                            {
                                "name": strategy,
                                "confidence": round(confidence, 3),
                                "recommended": True,
                                "position_size": round(random.uniform(0.01, 0.1), 3),
                            }
                        )
                    else:
                        strategies.append(
                            {
                                "name": strategy,
                                "confidence": round(confidence, 3),
                                "recommended": False,
                                "position_size": 0,
                            }
                        )

            live_signal = {
                "symbol": symbol.upper(),
                "signal_type": signal_type,
                "confidence": round(signal_strength * 100, 2),
                "price": round(random.uniform(45000, 55000), 2),
                "volume_24h": round(random.uniform(1000000, 10000000), 0),
                "indicators": indicators,
                "risk_metrics": risk_metrics,
                "sentiment": sentiment,
                "order_flow": order_flow,
                "strategies": strategies,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "signal_id": (f"signal_{int(time.time())}_{random.randint(1000, 9999)}"),
            }

            # Store signal in Redis for history
            signal_key = f"signal_history:{symbol.lower()}"
            self.redis_client.lpush(signal_key, json.dumps(live_signal))
            self.redis_client.ltrim(signal_key, 0, 99)  # Keep last 100 signals

            return {
                "status": "success",
                "signal": live_signal,
                "auto_trading_enabled": True,
            }

        except Exception as e:
            logger.error(f"Error generating live signal: {str(e)}")
            raise

    async def start_auto_trading(self) -> dict[str, Any]:
        """Start automated trading"""
        try:
            logger.info("Starting automated trading...")

            # Define auto_trade_config with proper typing
            auto_trade_config: dict[str, Any] = {
                "enabled": True,
                "start_time": datetime.now(timezone.utc).isoformat(),
                "strategies": list(self.trading_strategies.keys()),
                "risk_management": {
                    "max_position_size": 0.1,
                    "max_drawdown": 0.1,
                    "stop_loss": 0.05,
                    "take_profit": 0.15,
                    "max_leverage": 3,
                    "min_volume": 1000000,
                    "max_slippage": 0.02,
                },
                "performance": {
                    "total_trades": 0,
                    "winning_trades": 0,
                    "losing_trades": 0,
                    "total_pnl": 0.0,
                    "sharpe_ratio": 0.0,
                },
            }

            # Store in Redis
            self.redis_client.setex("auto_trade_config", 3600, json.dumps(auto_trade_config))

            self.auto_trading_enabled = True

            logger.info("Automated trading started successfully")

            return {
                "status": "success",
                "message": "Automated trading started",
                "config": auto_trade_config,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as e:
            logger.error(f"Error starting auto-trading: {str(e)}")
            raise

    async def stop_auto_trading(self) -> dict[str, Any]:
        """Stop automated trading"""
        try:
            logger.info("Stopping automated trading...")

            # Get current config to preserve structure
            config_data = self.redis_client.get("auto_trade_config")

            # Define auto_trade_config with proper typing
            auto_trade_config: dict[str, Any] = {}

            if config_data:
                auto_trade_config = json.loads(config_data)
                auto_trade_config["enabled"] = False
                auto_trade_config["stop_time"] = datetime.now(timezone.utc).isoformat()
                auto_trade_config["strategies"] = []  # Empty list of strings
            else:
                # Create a new config with the same structure as in start_auto_trading
                auto_trade_config = {
                    "enabled": False,
                    "stop_time": datetime.now(timezone.utc).isoformat(),
                    "strategies": [],  # Empty list of strategy names (strings)
                    "risk_management": {
                        "max_position_size": 0.1,
                        "max_drawdown": 0.1,
                        "stop_loss": 0.05,
                        "take_profit": 0.15,
                        "max_leverage": 3,
                        "min_volume": 1000000,
                        "max_slippage": 0.02,
                    },
                    "performance": {
                        "total_trades": 0,
                        "winning_trades": 0,
                        "losing_trades": 0,
                        "total_pnl": 0.0,
                        "sharpe_ratio": 0.0,
                    },
                }

            # Store in Redis
            self.redis_client.setex("auto_trade_config", 3600, json.dumps(auto_trade_config))

            self.auto_trading_enabled = False

            logger.info("Automated trading stopped successfully")

            return {
                "status": "success",
                "message": "Automated trading stopped",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as e:
            logger.error(f"Error stopping auto-trading: {str(e)}")
            raise

    async def get_auto_trade_status(self) -> dict[str, Any]:
        """Get auto-trading status"""
        try:
            config_data = self.redis_client.get("auto_trade_config")

            # Define config with proper typing
            config: dict[str, Any] = {}

            if config_data:
                config = json.loads(config_data)
            else:
                config = {
                    "enabled": False,
                    "strategies": [],  # Empty list of strategy names (strings)
                    "message": "Auto-trading not configured",
                }

            return {
                "status": "success",
                "auto_trading": config,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as e:
            logger.error(f"Error getting auto-trade status: {str(e)}")
            raise

    async def self_heal_signals(self) -> dict[str, Any]:
        """Self-heal all signals and auto-trading if they are down"""
        try:
            logger.info("Starting signal self-healing process...")

            # Get current status
            status = await self.get_signal_status()
            signals = status.get("signals", {})
            auto_trading = status.get("auto_trading", {})

            healing_actions: list[str] = []
            needs_healing = False

            # Check signals
            for signal_name, signal_config in signals.items():
                if not signal_config.get("enabled", False):
                    logger.warning(f"Signal {signal_name} is disabled, reactivating...")
                    signal_config["enabled"] = True
                    signal_config["last_update"] = datetime.now(timezone.utc).isoformat()
                    signal_config["status"] = "active"
                    healing_actions.append(f"Reactivated signal: {signal_name}")
                    needs_healing = True

            # Check auto-trading
            if not auto_trading.get("enabled", False):
                logger.warning("Auto-trading is disabled, reactivating...")
                await self.start_auto_trading()
                healing_actions.append("Reactivated auto-trading")
                needs_healing = True

            # Check trading strategies
            strategies = status.get("strategies", {})
            for strategy_name, strategy_config in strategies.items():
                if not strategy_config.get("enabled", False):
                    logger.warning(f"Strategy {strategy_name} is disabled, reactivating...")
                    strategy_config["enabled"] = True
                    healing_actions.append(f"Reactivated strategy: {strategy_name}")
                    needs_healing = True

            if needs_healing:
                # Store updated signals
                self.redis_client.setex("signal_status", 3600, json.dumps(signals))

                # Store updated strategies
                self.redis_client.setex("trading_strategies", 3600, json.dumps(strategies))

                logger.info(f"Self-healing completed: {len(healing_actions)} actions taken")

                # Send recovery notification
                await self.notification_service.send_notification(
                    title="Signal Recovery",
                    message=f"Successfully recovered {len(healing_actions)} components through self-healing",
                    level="info",
                    channels=["in_app", "slack", "email"],
                    data={
                        "actions_taken": healing_actions,
                        "healing_timestamp": (datetime.now(timezone.utc).isoformat()),
                    },
                )
            else:
                logger.info("All signals and auto-trading are healthy, no healing needed")

            return {
                "status": "success",
                "healing_performed": needs_healing,
                "actions_taken": healing_actions,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            logger.error(f"Error during self-healing: {str(e)}")

            # Send failure notification
            await self.notification_service.send_notification(
                title="Self-Healing Failed",
                message=f"Self-healing process failed: {str(e)}",
                level="error",
                channels=["in_app", "slack", "email"],
                data={"error": str(e)},
            )

            raise

    async def check_signal_health(self) -> dict[str, Any]:
        """Check the health of all signals and auto-trading"""
        try:
            status = await self.get_signal_status()
            signals = status.get("signals", {})
            auto_trading = status.get("auto_trading", {})
            strategies = status.get("strategies", {})

            # Count healthy vs unhealthy components
            total_signals = len(signals)
            healthy_signals = sum(1 for signal in signals.values() if signal.get("enabled", False))

            total_strategies = len(strategies)
            healthy_strategies = sum(
                1 for strategy in strategies.values() if strategy.get("enabled", False)
            )

            auto_trading_healthy = auto_trading.get("enabled", False)

            overall_health = "healthy"
            if (
                healthy_signals < total_signals
                or healthy_strategies < total_strategies
                or not auto_trading_healthy
            ):
                overall_health = "degraded"
            if healthy_signals == 0 and healthy_strategies == 0 and not auto_trading_healthy:
                overall_health = "critical"

            current_health_state = {
                "overall_health": overall_health,
                "signals": {
                    "total": total_signals,
                    "healthy": healthy_signals,
                    "unhealthy": total_signals - healthy_signals,
                },
                "strategies": {
                    "total": total_strategies,
                    "healthy": healthy_strategies,
                    "unhealthy": total_strategies - healthy_strategies,
                },
                "auto_trading": {"healthy": auto_trading_healthy},
            }

            # Check for health state changes and send notifications
            await self._check_health_changes(current_health_state)

            # Update previous health state
            self.previous_health_state = current_health_state

            return {
                "status": "success",
                "overall_health": overall_health,
                "signals": current_health_state["signals"],
                "strategies": current_health_state["strategies"],
                "auto_trading": current_health_state["auto_trading"],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            logger.error(f"Error checking signal health: {str(e)}")
            raise

    async def _check_health_changes(self, current_health: dict[str, Any]):
        """Check for health state changes and send appropriate notifications"""
        if self.previous_health_state is None:
            # First check, just log the initial state
            logger.info(f"Initial health state: {current_health['overall_health']}")
            return

        previous_health = self.previous_health_state

        # Check for overall health degradation
        if previous_health["overall_health"] == "healthy" and current_health["overall_health"] in [
            "degraded",
            "critical",
        ]:
            await self.notification_service.send_notification(
                title="Signal Health Degradation",
                message=f"Signal health degraded from {previous_health['overall_health']} to {current_health['overall_health']}",
                level="warning",
                channels=["in_app", "slack", "email"],
                data={
                    "previous_health": previous_health,
                    "current_health": current_health,
                    "degradation_time": datetime.now(timezone.utc).isoformat(),
                },
            )

        # Check for critical health
        elif current_health["overall_health"] == "critical":
            await self.notification_service.send_notification(
                title="Critical Signal Health",
                message="All signals and auto-trading are down - CRITICAL",
                level="critical",
                channels=["in_app", "slack", "email"],
                data={
                    "current_health": current_health,
                    "critical_time": datetime.now(timezone.utc).isoformat(),
                },
            )

        # Check for health recovery
        elif (
            previous_health["overall_health"] in ["degraded", "critical"]
            and current_health["overall_health"] == "healthy"
        ):
            await self.notification_service.send_notification(
                title="Signal Health Recovery",
                message=f"Signal health recovered from {previous_health['overall_health']} to {current_health['overall_health']}",
                level="info",
                channels=["in_app", "slack", "email"],
                data={
                    "previous_health": previous_health,
                    "current_health": current_health,
                    "recovery_time": datetime.now(timezone.utc).isoformat(),
                },
            )


# Global signal manager instance
signal_manager = None


def get_signal_manager(redis_client: Any) -> SignalManager:
    """Get or create signal manager instance"""
    global signal_manager
    if signal_manager is None:
        signal_manager = SignalManager(redis_client)
    return signal_manager


