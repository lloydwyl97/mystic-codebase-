"""
Enhanced Signal Manager Manager for Mystic Trading

Manages all trading signals and ensures they are active and properly integrated.
"""

import json
import logging
import random
import time
from datetime import timezone, datetime
from typing import Any, Dict, List

from notification_service import get_notification_service

logger = logging.getLogger(__name__)


class SignalManagerManager:
    """Manager for the Signal Manager system"""

    def __init__(self, redis_client: Any):
        self.redis_client = redis_client
        self.signal_manager = SignalManager(redis_client)
        self._health_check_interval = 30  # seconds
        self._last_health_check = 0

    async def activate_all_signals(self) -> Dict[str, Any]:
        """Activate all trading signals"""
        return await self.signal_manager.activate_all_signals()

    async def get_signal_status(self) -> Dict[str, Any]:
        """Get current status of all signals"""
        return await self.signal_manager.get_signal_status()

    async def generate_live_signal(self, symbol: str) -> Dict[str, Any]:
        """Generate live trading signal for a symbol"""
        return await self.signal_manager.generate_live_signal(symbol)

    async def start_auto_trading(self) -> Dict[str, Any]:
        """Start auto-trading"""
        return await self.signal_manager.start_auto_trading()

    async def stop_auto_trading(self) -> Dict[str, Any]:
        """Stop auto-trading"""
        return await self.signal_manager.stop_auto_trading()

    async def get_auto_trade_status(self) -> Dict[str, Any]:
        """Get auto-trading status"""
        return await self.signal_manager.get_auto_trade_status()

    async def self_heal_signals(self) -> Dict[str, Any]:
        """Self-heal signals"""
        return await self.signal_manager.self_heal_signals()

    async def check_signal_health(self) -> Dict[str, Any]:
        """Check signal health"""
        return await self.signal_manager.check_signal_health()

    async def get_signal_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for all signals"""
        try:
            signal_status = await self.get_signal_status()
            signals = signal_status.get("signals", {})

            metrics: Dict[str, Any] = {
                "total_signals": len(signals),
                "active_signals": 0,
                "inactive_signals": 0,
                "high_priority_signals": 0,
                "medium_priority_signals": 0,
                "low_priority_signals": 0,
                "critical_priority_signals": 0,
                "average_update_interval": 0,
                "signal_types": {},
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            total_interval = 0
            interval_count = 0

            for signal_type, config in signals.items():
                if config.get("enabled", False):
                    metrics["active_signals"] += 1
                else:
                    metrics["inactive_signals"] += 1

                priority = config.get("priority", "medium")
                if priority == "high":
                    metrics["high_priority_signals"] += 1
                elif priority == "medium":
                    metrics["medium_priority_signals"] += 1
                elif priority == "low":
                    metrics["low_priority_signals"] += 1
                elif priority == "critical":
                    metrics["critical_priority_signals"] += 1

                update_interval = config.get("update_interval", 0)
                if update_interval > 0:
                    total_interval += update_interval
                    interval_count += 1

                metrics["signal_types"][signal_type] = {
                    "enabled": config.get("enabled", False),
                    "priority": priority,
                    "update_interval": update_interval,
                    "last_update": config.get("last_update"),
                    "status": config.get("status", "unknown"),
                }

            if interval_count > 0:
                metrics["average_update_interval"] = total_interval / interval_count

            return metrics
        except Exception as e:
            logger.error(f"Error getting signal performance metrics: {str(e)}")
            raise

    async def get_trading_strategy_metrics(self) -> Dict[str, Any]:
        """Get metrics for trading strategies"""
        try:
            signal_status = await self.get_signal_status()
            strategies = signal_status.get("strategies", {})

            metrics: Dict[str, Any] = {
                "total_strategies": len(strategies),
                "enabled_strategies": 0,
                "disabled_strategies": 0,
                "average_confidence": 0.0,
                "strategy_details": {},
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            total_confidence = 0.0
            confidence_count = 0

            for strategy_name, config in strategies.items():
                if config.get("enabled", False):
                    metrics["enabled_strategies"] += 1
                else:
                    metrics["disabled_strategies"] += 1

                confidence = config.get("min_confidence", 0.0)
                if confidence > 0:
                    total_confidence += confidence
                    confidence_count += 1

                metrics["strategy_details"][strategy_name] = {
                    "enabled": config.get("enabled", False),
                    "min_confidence": confidence,
                    "description": self._get_strategy_description(strategy_name),
                }

            if confidence_count > 0:
                metrics["average_confidence"] = total_confidence / confidence_count

            return metrics
        except Exception as e:
            logger.error(f"Error getting trading strategy metrics: {str(e)}")
            raise

    def _get_strategy_description(self, strategy_name: str) -> str:
        """Get description for a trading strategy"""
        descriptions = {
            "scalping": "High-frequency trading with small profits",
            "swing_trading": "Medium-term position holding",
            "arbitrage": "Exploiting price differences across exchanges",
            "momentum": "Following price momentum trends",
            "mean_reversion": "Trading price reversals to the mean",
            "grid_trading": "Automated grid-based trading",
            "statistical_arbitrage": "Statistical arbitrage opportunities",
            "market_making": "Providing liquidity to the market",
            "high_frequency_trading": "Ultra-fast algorithmic trading",
            "options_trading": "Options and derivatives trading",
            "futures_trading": "Futures contract trading",
        }
        return descriptions.get(strategy_name, "Unknown strategy")

    async def get_system_health_summary(self) -> Dict[str, Any]:
        """Get comprehensive system health summary"""
        try:
            health_data = await self.check_signal_health()
            performance_metrics = await self.get_signal_performance_metrics()
            strategy_metrics = await self.get_trading_strategy_metrics()
            auto_trade_status = await self.get_auto_trade_status()

            summary = {
                "overall_health": health_data.get("overall_health", "unknown"),
                "signal_health": health_data.get("signal_health", {}),
                "performance_metrics": performance_metrics,
                "strategy_metrics": strategy_metrics,
                "auto_trading": auto_trade_status,
                "recommendations": self._generate_health_recommendations(
                    health_data, performance_metrics, strategy_metrics
                ),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            return summary
        except Exception as e:
            logger.error(f"Error getting system health summary: {str(e)}")
            raise

    def _generate_health_recommendations(
        self,
        health_data: Dict[str, Any],
        performance_metrics: Dict[str, Any],
        strategy_metrics: Dict[str, Any],
    ) -> List[str]:
        """Generate health recommendations based on current state"""
        recommendations: List[str] = []

        # Check signal health
        overall_health = health_data.get("overall_health", "unknown")
        if overall_health != "healthy":
            recommendations.append(
                f"Signal system health is {overall_health}. Consider running self-healing."
            )

        # Check inactive signals
        inactive_signals = performance_metrics.get("inactive_signals", 0)
        if inactive_signals > 0:
            recommendations.append(
                f"{inactive_signals} signals are inactive. Review and reactivate if needed."
            )

        # Check auto-trading
        auto_trading = health_data.get("auto_trading", {})
        if not auto_trading.get("enabled", False):
            recommendations.append("Auto-trading is disabled. Enable for automated trading.")

        # Check strategy confidence
        avg_confidence = strategy_metrics.get("average_confidence", 0.0)
        if avg_confidence < 0.6:
            recommendations.append(
                "Average strategy confidence is low. Review strategy parameters."
            )

        # Check update intervals
        avg_interval = performance_metrics.get("average_update_interval", 0)
        if avg_interval > 60:
            recommendations.append(
                "Average signal update interval is high. Consider optimizing for faster updates."
            )

        if not recommendations:
            recommendations.append("System is healthy. No immediate action required.")

        return recommendations

    async def optimize_signal_performance(self) -> Dict[str, Any]:
        """Optimize signal performance based on current metrics"""
        try:
            performance_metrics = await self.get_signal_performance_metrics()
            strategy_metrics = await self.get_trading_strategy_metrics()

            optimizations: Dict[str, Any] = {
                "signal_optimizations": [],
                "strategy_optimizations": [],
                "performance_improvements": [],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            # Signal optimizations
            inactive_signals = performance_metrics.get("inactive_signals", 0)
            if inactive_signals > 0:
                optimizations["signal_optimizations"].append(
                    f"Reactivate {inactive_signals} inactive signals"
                )

            avg_interval = performance_metrics.get("average_update_interval", 0)
            if avg_interval > 30:
                optimizations["signal_optimizations"].append(
                    "Reduce signal update intervals for faster response"
                )

            # Strategy optimizations
            avg_confidence = strategy_metrics.get("average_confidence", 0.0)
            if avg_confidence < 0.7:
                optimizations["strategy_optimizations"].append(
                    "Increase minimum confidence thresholds for better signal quality"
                )

            disabled_strategies = strategy_metrics.get("disabled_strategies", 0)
            if disabled_strategies > 0:
                optimizations["strategy_optimizations"].append(
                    f"Review and potentially enable {disabled_strategies} disabled strategies"
                )

            # Performance improvements
            critical_signals = performance_metrics.get("critical_priority_signals", 0)
            if critical_signals == 0:
                optimizations["performance_improvements"].append(
                    "Add critical priority signals for essential trading functions"
                )

            return optimizations
        except Exception as e:
            logger.error(f"Error optimizing signal performance: {str(e)}")
            raise


class SignalManager:
    def __init__(self, redis_client: Any):
        self.redis_client = redis_client
        self.active_signals = {}
        self.signal_generators = {}
        self.auto_trading_enabled = False
        self._health_check_interval = 30  # seconds
        self._last_health_check = 0

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

    async def activate_all_signals(self) -> Dict[str, Any]:
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

    async def get_signal_status(self) -> Dict[str, Any]:
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

    async def generate_live_signal(self, symbol: str) -> Dict[str, Any]:
        """Generate live trading signal for a symbol"""
        try:
            # Check if signals are active
            signal_data = self.redis_client.get("signal_status")
            if not signal_data:
                return {
                    "status": "error",
                    "message": "Signals not active",
                    "symbol": symbol,
                }

            # Generate mock signal (replace with actual signal generation)
            signal = {
                "symbol": symbol,
                "signal": random.choice(["buy", "sell", "hold"]),
                "confidence": random.uniform(0.5, 0.95),
                "strength": random.uniform(0.3, 0.9),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source": "live_signal_generator",
            }

            return {
                "status": "success",
                "signal": signal,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as e:
            logger.error(f"Error generating live signal: {str(e)}")
            raise

    async def start_auto_trading(self) -> Dict[str, Any]:
        """Start auto-trading"""
        try:
            logger.info("Starting auto-trading...")

            # Check if signals are active
            signal_data = self.redis_client.get("signal_status")
            if not signal_data:
                return {
                    "status": "error",
                    "message": "Cannot start auto-trading: signals not active",
                }

            # Set auto-trading flag
            self.auto_trading_enabled = True
            self.redis_client.setex(
                "auto_trading_enabled",
                3600,
                json.dumps(
                    {
                        "enabled": True,
                        "started_at": datetime.now(timezone.utc).isoformat(),
                        "status": "running",
                    }
                ),
            )

            # Send notification
            await self.notification_service.send_notification(
                "Auto-trading started",
                "Automated trading has been activated",
                "info",
            )

            logger.info("Auto-trading started successfully")

            return {
                "status": "success",
                "message": "Auto-trading started",
                "auto_trading_enabled": True,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as e:
            logger.error(f"Error starting auto-trading: {str(e)}")
            raise

    async def stop_auto_trading(self) -> Dict[str, Any]:
        """Stop auto-trading"""
        try:
            logger.info("Stopping auto-trading...")

            # Set auto-trading flag
            self.auto_trading_enabled = False
            self.redis_client.setex(
                "auto_trading_enabled",
                3600,
                json.dumps(
                    {
                        "enabled": False,
                        "stopped_at": datetime.now(timezone.utc).isoformat(),
                        "status": "stopped",
                    }
                ),
            )

            # Send notification
            await self.notification_service.send_notification(
                "Auto-trading stopped",
                "Automated trading has been deactivated",
                "warning",
            )

            logger.info("Auto-trading stopped successfully")

            return {
                "status": "success",
                "message": "Auto-trading stopped",
                "auto_trading_enabled": False,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as e:
            logger.error(f"Error stopping auto-trading: {str(e)}")
            raise

    async def get_auto_trade_status(self) -> Dict[str, Any]:
        """Get auto-trading status"""
        try:
            auto_trade_data = self.redis_client.get("auto_trading_enabled")
            auto_trading = json.loads(auto_trade_data) if auto_trade_data else {"enabled": False}

            return {
                "status": "success",
                "auto_trading": auto_trading,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as e:
            logger.error(f"Error getting auto-trade status: {str(e)}")
            raise

    async def self_heal_signals(self) -> Dict[str, Any]:
        """Self-heal signals"""
        try:
            logger.info("Starting signal self-healing...")

            healed_signals: List[str] = []
            failed_signals: List[Dict[str, str]] = []

            # Check each signal type
            for signal_type, config in self.signal_types.items():
                try:
                    # Check if signal is healthy
                    if not config.get("enabled", False):
                        # Reactivate disabled signal
                        config["enabled"] = True
                        config["status"] = "healed"
                        config["last_update"] = datetime.now(timezone.utc).isoformat()
                        healed_signals.append(signal_type)
                        logger.info(f"Healed signal: {signal_type}")

                except Exception as e:
                    failed_signals.append({"signal": signal_type, "error": str(e)})
                    logger.error(f"Failed to heal signal {signal_type}: {str(e)}")

            # Update Redis
            self.redis_client.setex("signal_status", 3600, json.dumps(self.signal_types))

            # Send notification
            if healed_signals:
                await self.notification_service.send_notification(
                    "Signals healed",
                    f"Successfully healed {len(healed_signals)} signals",
                    "success",
                )

            logger.info(
                f"Self-healing completed. Healed: {len(healed_signals)}, Failed: {len(failed_signals)}"
            )

            return {
                "status": "success",
                "healed_signals": healed_signals,
                "failed_signals": failed_signals,
                "total_healed": len(healed_signals),
                "total_failed": len(failed_signals),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as e:
            logger.error(f"Error during self-healing: {str(e)}")
            raise

    async def check_signal_health(self) -> Dict[str, Any]:
        """Check signal health"""
        try:
            current_time = time.time()

            # Rate limit health checks
            if current_time - self._last_health_check < self._health_check_interval:
                return {
                    "status": "success",
                    "message": "Health check rate limited",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }

            self._last_health_check = current_time

            # Get current signal status
            signal_data = self.redis_client.get("signal_status")
            signals = json.loads(signal_data) if signal_data else self.signal_types

            # Check health of each signal
            signal_health = {}
            healthy_signals = 0
            unhealthy_signals = 0

            for signal_type, config in signals.items():
                is_healthy = config.get("enabled", False) and config.get("status") == "active"

                signal_health[signal_type] = {
                    "healthy": is_healthy,
                    "enabled": config.get("enabled", False),
                    "status": config.get("status", "unknown"),
                    "last_update": config.get("last_update"),
                    "priority": config.get("priority", "medium"),
                }

                if is_healthy:
                    healthy_signals += 1
                else:
                    unhealthy_signals += 1

            # Determine overall health
            total_signals = len(signals)
            health_percentage = (healthy_signals / total_signals) * 100 if total_signals > 0 else 0

            if health_percentage >= 90:
                overall_health = "healthy"
            elif health_percentage >= 70:
                overall_health = "warning"
            else:
                overall_health = "critical"

            # Check for health changes
            current_health = {
                "overall_health": overall_health,
                "health_percentage": health_percentage,
                "healthy_signals": healthy_signals,
                "unhealthy_signals": unhealthy_signals,
            }

            await self._check_health_changes(current_health)

            return {
                "status": "success",
                "overall_health": overall_health,
                "health_percentage": health_percentage,
                "healthy_signals": healthy_signals,
                "unhealthy_signals": unhealthy_signals,
                "signal_health": signal_health,
                "auto_trading": await self.get_auto_trade_status(),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as e:
            logger.error(f"Error checking signal health: {str(e)}")
            raise

    async def _check_health_changes(self, current_health: Dict[str, Any]):
        """Check for health state changes and notify"""
        if self.previous_health_state is None:
            self.previous_health_state = current_health
            return

        # Check for significant changes
        prev_health = self.previous_health_state.get("overall_health")
        curr_health = current_health.get("overall_health")

        if prev_health != curr_health:
            # Health state changed
            message = f"Signal health changed from {prev_health} to {curr_health}"
            notification_type = "warning" if curr_health in ["warning", "critical"] else "info"

            await self.notification_service.send_notification(
                "Signal Health Change", message, notification_type
            )

        self.previous_health_state = current_health


def get_signal_manager_manager(redis_client: Any) -> SignalManagerManager:
    """Get Signal Manager Manager instance"""
    return SignalManagerManager(redis_client)
