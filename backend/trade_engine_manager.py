#!/usr/bin/env python3
"""
Unified Trade Decision Engine Manager
Combines all three tiers of signals and makes trading decisions every 3-10 seconds
"""

import asyncio
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from .services.websocket_manager import websocket_manager

logger = logging.getLogger(__name__)


class SignalStrength(Enum):
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"


class TradeAction(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    CLOSE = "CLOSE"


@dataclass
class CoinState:
    symbol: str
    last_price: float
    last_volume: float
    rsi: float
    macd: dict[str, float]
    mystic_alignment_score: float
    is_active_buy_signal: bool
    is_active_sell_signal: bool
    signal_strength: SignalStrength
    confidence: float
    last_update: str
    api_source: str


@dataclass
class TradeDecision:
    symbol: str
    action: TradeAction
    confidence: float
    price: float
    reason: str
    signal_strength: SignalStrength
    tier1_signals: dict[str, Any]
    tier2_signals: dict[str, Any]
    tier3_signals: dict[str, Any]
    timestamp: str


class TradeEngineManager:
    """Manager for the Trade Engine system"""

    def __init__(self, redis_client: Any):
        self.redis_client = redis_client
        self.trade_engine = TradeEngine(redis_client)
        self._performance_metrics: dict[str, int | float | str] = {
            "total_decisions": 0,
            "buy_decisions": 0,
            "sell_decisions": 0,
            "hold_decisions": 0,
            "average_confidence": 0.0,
            "average_signal_strength": 0.0,
            "last_update": datetime.now(timezone.utc).isoformat(),
        }

    async def start_engine(self) -> dict[str, Any]:
        """Start the trade engine"""
        try:
            if self.trade_engine.is_running:
                return {
                    "status": "warning",
                    "message": "Trade engine is already running",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }

            # Start the engine
            self.trade_engine.is_running = True

            # Start the engine loop in background
            asyncio.create_task(self.trade_engine.run())

            logger.info("Trade engine started successfully")

            return {
                "status": "success",
                "message": "Trade engine started",
                "engine_running": True,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as e:
            logger.error(f"Error starting trade engine: {str(e)}")
            raise

    async def stop_engine(self) -> dict[str, Any]:
        """Stop the trade engine"""
        try:
            if not self.trade_engine.is_running:
                return {
                    "status": "warning",
                    "message": "Trade engine is not running",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }

            # Stop the engine
            self.trade_engine.is_running = False

            logger.info("Trade engine stopped successfully")

            return {
                "status": "success",
                "message": "Trade engine stopped",
                "engine_running": False,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as e:
            logger.error(f"Error stopping trade engine: {str(e)}")
            raise

    async def get_engine_status(self) -> dict[str, Any]:
        """Get trade engine status"""
        try:
            engine_status = self.trade_engine.get_status()

            return {
                "status": "success",
                "engine_status": engine_status,
                "performance_metrics": self._performance_metrics,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as e:
            logger.error(f"Error getting engine status: {str(e)}")
            raise

    async def get_trade_decisions(self) -> dict[str, Any]:
        """Get current trade decisions"""
        try:
            decisions = await self.trade_engine.get_trade_decisions()

            return {
                "status": "success",
                "decisions": decisions,
                "total_decisions": len(decisions),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as e:
            logger.error(f"Error getting trade decisions: {str(e)}")
            raise

    async def get_coin_states(self) -> dict[str, Any]:
        """Get all coin states"""
        try:
            coin_states = await self.trade_engine.get_all_coin_states()

            # Convert to serializable format
            serializable_states = {}
            for symbol, state in coin_states.items():
                serializable_states[symbol] = {
                    "symbol": state.symbol,
                    "last_price": state.last_price,
                    "last_volume": state.last_volume,
                    "rsi": state.rsi,
                    "macd": state.macd,
                    "mystic_alignment_score": state.mystic_alignment_score,
                    "is_active_buy_signal": state.is_active_buy_signal,
                    "is_active_sell_signal": state.is_active_sell_signal,
                    "signal_strength": state.signal_strength.value,
                    "confidence": state.confidence,
                    "last_update": state.last_update,
                    "api_source": state.api_source,
                }

            return {
                "status": "success",
                "coin_states": serializable_states,
                "total_coins": len(coin_states),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as e:
            logger.error(f"Error getting coin states: {str(e)}")
            raise

    async def get_signal_analysis(self, symbol: str) -> dict[str, Any]:
        """Get detailed signal analysis for a specific coin"""
        try:
            # Get all tier signals
            tier1_signals = await self.trade_engine.get_tier1_signals()
            tier2_signals = await self.trade_engine.get_tier2_signals()
            tier3_signals = await self.trade_engine.get_tier3_signals()

            # Get coin state
            coin_state = await self.trade_engine.get_coin_state(symbol)

            if not coin_state:
                return {
                    "status": "error",
                    "message": f"No data available for {symbol}",
                    "symbol": symbol,
                }

            # Access nested signal structures
            tier1_data = tier1_signals.get("prices", {}).get(symbol, {})
            tier2_data = tier2_signals.get("indicators", {}).get(symbol, {})
            tier3_data = tier3_signals.get("cosmic_signals", {})

            # Calculate signal strength and confidence
            signal_strength = self.trade_engine.calculate_signal_strength(
                tier1_data, tier2_data, tier3_data
            )

            confidence = self.trade_engine.calculate_confidence(tier1_data, tier2_data, tier3_data)

            # Determine trade action
            action, _ = self.trade_engine.determine_trade_action(
                symbol, tier1_data, tier2_data, tier3_data
            )

            analysis = {
                "symbol": symbol,
                "signal_strength": signal_strength.value,
                "confidence": confidence,
                "recommended_action": action.value,
                "reason": _ if _ else "No reason provided",
                "tier1_signals": tier1_data,
                "tier2_signals": tier2_data,
                "tier3_signals": tier3_data,
                "coin_state": {
                    "last_price": coin_state.last_price,
                    "last_volume": coin_state.last_volume,
                    "rsi": coin_state.rsi,
                    "macd": coin_state.macd,
                    "mystic_alignment_score": (coin_state.mystic_alignment_score),
                    "is_active_buy_signal": coin_state.is_active_buy_signal,
                    "is_active_sell_signal": coin_state.is_active_sell_signal,
                    "last_update": coin_state.last_update,
                    "api_source": coin_state.api_source,
                },
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            return {"status": "success", "analysis": analysis}
        except Exception as e:
            logger.error(f"Error getting signal analysis for {symbol}: {str(e)}")
            raise

    async def update_performance_metrics(self, decisions: list[TradeDecision]):
        """Update performance metrics based on new decisions"""
        try:
            self._performance_metrics["total_decisions"] = int(
                self._performance_metrics["total_decisions"]
            ) + len(decisions)

            total_confidence = 0.0
            total_strength = 0.0
            strength_count = 0

            for decision in decisions:
                if decision.action == TradeAction.BUY:
                    self._performance_metrics["buy_decisions"] = (
                        int(self._performance_metrics["buy_decisions"]) + 1
                    )
                elif decision.action == TradeAction.SELL:
                    self._performance_metrics["sell_decisions"] = (
                        int(self._performance_metrics["sell_decisions"]) + 1
                    )
                else:
                    self._performance_metrics["hold_decisions"] = (
                        int(self._performance_metrics["hold_decisions"]) + 1
                    )

                total_confidence += decision.confidence

                # Convert signal strength to numeric value
                strength_value = {
                    SignalStrength.WEAK: 1,
                    SignalStrength.MODERATE: 2,
                    SignalStrength.STRONG: 3,
                    SignalStrength.VERY_STRONG: 4,
                }.get(decision.signal_strength, 1)

                total_strength += strength_value
                strength_count += 1

            # Update averages
            if int(self._performance_metrics["total_decisions"]) > 0:
                self._performance_metrics["average_confidence"] = total_confidence / int(
                    self._performance_metrics["total_decisions"]
                )

            if strength_count > 0:
                self._performance_metrics["average_signal_strength"] = (
                    total_strength / strength_count
                )

            self._performance_metrics["last_update"] = datetime.now(timezone.utc).isoformat()

            # Broadcast performance metrics update
            await websocket_manager.broadcast_json(
                {
                    "type": "engine_performance",
                    "data": {
                        "total_decisions": self._performance_metrics["total_decisions"],
                        "buy_decisions": self._performance_metrics["buy_decisions"],
                        "sell_decisions": self._performance_metrics["sell_decisions"],
                        "hold_decisions": self._performance_metrics["hold_decisions"],
                        "average_confidence": self._performance_metrics["average_confidence"],
                        "average_signal_strength": self._performance_metrics[
                            "average_signal_strength"
                        ],
                        "last_update": self._performance_metrics["last_update"],
                    },
                }
            )

        except Exception as e:
            logger.error(f"Error updating performance metrics: {str(e)}")

    async def get_performance_report(self) -> dict[str, Any]:
        """Get comprehensive performance report"""
        try:
            engine_status = self.trade_engine.get_status()
            coin_states = await self.trade_engine.get_all_coin_states()

            # Calculate additional metrics
            active_coins = len(
                [
                    state
                    for state in coin_states.values()
                    if state.is_active_buy_signal or state.is_active_sell_signal
                ]
            )

            total_coins = len(coin_states)
            activation_rate = (active_coins / total_coins * 100) if total_coins > 0 else 0

            report = {
                "engine_status": engine_status,
                "performance_metrics": self._performance_metrics,
                "coin_metrics": {
                    "total_coins": total_coins,
                    "active_coins": active_coins,
                    "activation_rate": activation_rate,
                    "coins_with_buy_signals": len(
                        [s for s in coin_states.values() if s.is_active_buy_signal]
                    ),
                    "coins_with_sell_signals": len(
                        [s for s in coin_states.values() if s.is_active_sell_signal]
                    ),
                },
                "signal_distribution": {
                    "weak_signals": len(
                        [
                            s
                            for s in coin_states.values()
                            if s.signal_strength == SignalStrength.WEAK
                        ]
                    ),
                    "moderate_signals": len(
                        [
                            s
                            for s in coin_states.values()
                            if s.signal_strength == SignalStrength.MODERATE
                        ]
                    ),
                    "strong_signals": len(
                        [
                            s
                            for s in coin_states.values()
                            if s.signal_strength == SignalStrength.STRONG
                        ]
                    ),
                    "very_strong_signals": len(
                        [
                            s
                            for s in coin_states.values()
                            if s.signal_strength == SignalStrength.VERY_STRONG
                        ]
                    ),
                },
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            return {"status": "success", "report": report}
        except Exception as e:
            logger.error(f"Error getting performance report: {str(e)}")
            raise

    async def optimize_engine_config(self) -> dict[str, Any]:
        """Optimize engine configuration based on performance"""
        try:
            performance = self._performance_metrics

            optimizations: dict[str, Any] = {
                "config_changes": [],
                "recommendations": [],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            # Analyze decision distribution
            total_decisions = int(performance["total_decisions"])
            if total_decisions > 0:
                buy_rate = int(performance["buy_decisions"]) / total_decisions
                sell_rate = int(performance["sell_decisions"]) / total_decisions
                hold_rate = int(performance["hold_decisions"]) / total_decisions

                # Check for imbalance
                if buy_rate > 0.7:
                    optimizations["recommendations"].append(
                        "High buy rate detected. Consider increasing confidence thresholds."
                    )
                elif sell_rate > 0.7:
                    optimizations["recommendations"].append(
                        "High sell rate detected. Consider adjusting risk parameters."
                    )
                elif hold_rate > 0.8:
                    optimizations["recommendations"].append(
                        "High hold rate detected. Consider reducing confidence thresholds."
                    )

            # Check average confidence
            avg_confidence = float(performance["average_confidence"])
            if avg_confidence < 0.6:
                optimizations["recommendations"].append(
                    "Low average confidence. Consider improving signal quality."
                )
            elif avg_confidence > 0.9:
                optimizations["recommendations"].append(
                    "Very high confidence. Consider if thresholds are too conservative."
                )

            # Check signal strength
            avg_strength = float(performance["average_signal_strength"])
            if avg_strength < 2.0:
                optimizations["recommendations"].append(
                    "Low average signal strength. Consider signal enhancement."
                )

            if not optimizations["recommendations"]:
                optimizations["recommendations"].append("Engine configuration appears optimal.")

            return {"status": "success", "optimizations": optimizations}
        except Exception as e:
            logger.error(f"Error optimizing engine config: {str(e)}")
            raise


class TradeEngine:
    def __init__(self, redis_client: Any):
        self.redis_client = redis_client
        self.is_running = False

        # Engine Configuration
        self.config = {
            "decision_interval": 5,  # 3-10 seconds
            "cache_ttl": 60,  # 1 minute
            "min_confidence": 0.6,
            "max_confidence": 0.95,
            "price_deviation_threshold": 0.02,  # 2%
            "volume_spike_threshold": 0.2,  # 20%
            "momentum_flip_threshold": 0.05,  # 5%
        }

        # Coin state tracking
        self.coin_states: dict[str, CoinState] = {}

        # Trading thresholds
        self.thresholds = {
            "rsi_oversold": 30,
            "rsi_overbought": 70,
            "macd_bullish": 0.001,
            "macd_bearish": -0.001,
            "cosmic_alignment_min": 60,
            "volatility_max": 80,
        }

        logger.info("Trade Engine initialized")

    async def get_tier1_signals(self) -> dict[str, Any]:
        """Get Tier 1 signals from cache"""
        try:
            tier1_data = self.redis_client.get("tier1_signals")
            if tier1_data:
                return json.loads(tier1_data)
            return {}
        except Exception as e:
            logger.error(f"Error getting Tier 1 signals: {e}")
            return {}

    async def get_tier2_signals(self) -> dict[str, Any]:
        """Get Tier 2 signals from cache"""
        try:
            tier2_data = self.redis_client.get("tier2_indicators")
            if tier2_data:
                return json.loads(tier2_data)
            return {}
        except Exception as e:
            logger.error(f"Error getting Tier 2 signals: {e}")
            return {}

    async def get_tier3_signals(self) -> dict[str, Any]:
        """Get Tier 3 signals from cache"""
        try:
            tier3_data = self.redis_client.get("cosmic_signals")
            if tier3_data:
                return json.loads(tier3_data)
            return {}
        except Exception as e:
            logger.error(f"Error getting Tier 3 signals: {e}")
            return {}

    def calculate_signal_strength(
        self,
        tier1: dict[str, Any],
        tier2: dict[str, Any],
        tier3: dict[str, Any],
    ) -> SignalStrength:
        """Calculate overall signal strength from all tiers"""
        try:
            strength_score = 0
            factors = 0

            # Tier 1 factors (price momentum)
            if "change_1m" in tier1:
                momentum = abs(tier1["change_1m"])
                if momentum > 5:
                    strength_score += 3
                elif momentum > 2:
                    strength_score += 2
                elif momentum > 1:
                    strength_score += 1
                factors += 1

            # Tier 2 factors (technical indicators)
            if "rsi" in tier2:
                rsi = tier2["rsi"]
                if rsi < 20 or rsi > 80:
                    strength_score += 3
                elif rsi < 30 or rsi > 70:
                    strength_score += 2
                factors += 1

            if "macd" in tier2 and "histogram" in tier2["macd"]:
                macd_hist = abs(tier2["macd"]["histogram"])
                if macd_hist > 0.01:
                    strength_score += 2
                elif macd_hist > 0.005:
                    strength_score += 1
                factors += 1

            # Tier 3 factors (cosmic alignment)
            if "cosmic_timing_score" in tier3:
                cosmic_score = tier3["cosmic_timing_score"]
                if cosmic_score > 80:
                    strength_score += 2
                elif cosmic_score > 60:
                    strength_score += 1
                factors += 1

            # Calculate average strength
            if factors > 0:
                avg_strength = strength_score / factors

                if avg_strength >= 2.5:
                    return SignalStrength.VERY_STRONG
                elif avg_strength >= 2.0:
                    return SignalStrength.STRONG
                elif avg_strength >= 1.5:
                    return SignalStrength.MODERATE
                else:
                    return SignalStrength.WEAK

            return SignalStrength.WEAK

        except Exception as e:
            logger.error(f"Error calculating signal strength: {e}")
            return SignalStrength.WEAK

    def calculate_confidence(
        self,
        tier1: dict[str, Any],
        tier2: dict[str, Any],
        tier3: dict[str, Any],
    ) -> float:
        """Calculate trading confidence from all tiers"""
        try:
            confidence_factors: list[float] = []

            # Price stability (Tier 1)
            if "price" in tier1:
                confidence_factors.append(0.8)  # Base confidence for price data

            # Technical indicator agreement (Tier 2)
            if "rsi" in tier2 and "macd" in tier2:
                rsi = tier2["rsi"]
                macd_hist = tier2["macd"].get("histogram", 0)

                # Check if RSI and MACD agree
                rsi_bullish = rsi < 70
                macd_bullish = macd_hist > 0

                if rsi_bullish == macd_bullish:
                    confidence_factors.append(0.9)  # High confidence for agreement
                else:
                    confidence_factors.append(0.6)  # Lower confidence for disagreement

            # Cosmic alignment (Tier 3)
            if "cosmic_timing_score" in tier3:
                cosmic_score = tier3["cosmic_timing_score"]
                if cosmic_score > 70:
                    confidence_factors.append(0.85)
                elif cosmic_score > 50:
                    confidence_factors.append(0.7)
                else:
                    confidence_factors.append(0.5)

            # Calculate average confidence
            if confidence_factors:
                avg_confidence = sum(confidence_factors) / len(confidence_factors)
                return min(
                    self.config["max_confidence"],
                    max(self.config["min_confidence"], avg_confidence),
                )

            return self.config["min_confidence"]

        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return self.config["min_confidence"]

    def determine_trade_action(
        self,
        symbol: str,
        tier1: dict[str, Any],
        tier2: dict[str, Any],
        tier3: dict[str, Any],
    ) -> tuple[TradeAction, str]:
        """Determine trading action based on all signal tiers"""
        try:
            reasons: list[str] = []
            buy_signals = 0
            sell_signals = 0

            # Tier 1: Price momentum analysis
            if "change_1m" in tier1:
                momentum = tier1["change_1m"]
                if momentum > self.config["momentum_flip_threshold"] * 100:
                    buy_signals += 1
                    reasons.append(f"Strong upward momentum ({momentum:.2f}%)")
                elif momentum < -self.config["momentum_flip_threshold"] * 100:
                    sell_signals += 1
                    reasons.append(f"Strong downward momentum ({momentum:.2f}%)")

            # Tier 2: Technical analysis
            if "rsi" in tier2:
                rsi = tier2["rsi"]
                if rsi < self.thresholds["rsi_oversold"]:
                    buy_signals += 1
                    reasons.append(f"RSI oversold ({rsi:.1f})")
                elif rsi > self.thresholds["rsi_overbought"]:
                    sell_signals += 1
                    reasons.append(f"RSI overbought ({rsi:.1f})")

            if "macd" in tier2 and "histogram" in tier2["macd"]:
                macd_hist = tier2["macd"]["histogram"]
                if macd_hist > self.thresholds["macd_bullish"]:
                    buy_signals += 1
                    reasons.append("MACD bullish")
                elif macd_hist < self.thresholds["macd_bearish"]:
                    sell_signals += 1
                    reasons.append("MACD bearish")

            # Tier 3: Cosmic alignment
            if "cosmic_timing_score" in tier3:
                cosmic_score = tier3["cosmic_timing_score"]
                if cosmic_score > self.thresholds["cosmic_alignment_min"]:
                    buy_signals += 1
                    reasons.append(f"Strong cosmic alignment ({cosmic_score:.1f})")
                elif cosmic_score < 30:
                    sell_signals += 1
                    reasons.append(f"Weak cosmic alignment ({cosmic_score:.1f})")

            # Determine action based on signal balance
            if buy_signals > sell_signals and buy_signals >= 2:
                return TradeAction.BUY, " | ".join(reasons)
            elif sell_signals > buy_signals and sell_signals >= 2:
                return TradeAction.SELL, " | ".join(reasons)
            else:
                return TradeAction.HOLD, "Insufficient signal strength"

        except Exception as e:
            logger.error(f"Error determining trade action for {symbol}: {e}")
            return TradeAction.HOLD, f"Error in analysis: {str(e)}"

    async def update_coin_state(
        self,
        symbol: str,
        tier1: dict[str, Any],
        tier2: dict[str, Any],
        tier3: dict[str, Any],
    ):
        """Update coin state with latest signals"""
        try:
            # Extract data from tiers
            price = tier1.get("price", 0.0)
            volume = tier1.get("volume_1m", 0.0)
            rsi = tier2.get("rsi", 50.0)
            macd = tier2.get(
                "macd",
                {"macd_line": 0.0, "signal_line": 0.0, "histogram": 0.0},
            )
            cosmic_score = tier3.get("cosmic_timing_score", 50.0)
            api_source = tier1.get("api_source", "unknown")

            # Calculate signal strength and confidence
            signal_strength = self.calculate_signal_strength(tier1, tier2, tier3)
            confidence = self.calculate_confidence(tier1, tier2, tier3)

            # Determine trade action
            action, _ = self.determine_trade_action(symbol, tier1, tier2, tier3)

            # Update coin state
            self.coin_states[symbol] = CoinState(
                symbol=symbol,
                last_price=price,
                last_volume=volume,
                rsi=rsi,
                macd=macd,
                mystic_alignment_score=cosmic_score,
                is_active_buy_signal=(action == TradeAction.BUY),
                is_active_sell_signal=(action == TradeAction.SELL),
                signal_strength=signal_strength,
                confidence=confidence,
                last_update=datetime.now(timezone.utc).isoformat(),
                api_source=api_source,
            )

        except Exception as e:
            logger.error(f"Error updating coin state for {symbol}: {e}")

    async def generate_trade_decisions(self) -> list[TradeDecision]:
        """Generate trade decisions for all coins"""
        try:
            decisions: list[TradeDecision] = []

            # Get all signals
            tier1_signals = await self.get_tier1_signals()
            tier2_signals = await self.get_tier2_signals()
            tier3_signals = await self.get_tier3_signals()

            # Process each coin
            for symbol in tier1_signals.get("prices", {}).keys():
                try:
                    # Access nested signal structures
                    tier1 = tier1_signals.get("prices", {}).get(symbol, {})
                    tier2 = tier2_signals.get("indicators", {}).get(symbol, {})
                    tier3 = tier3_signals.get("cosmic_signals", {})

                    # Update coin state
                    await self.update_coin_state(symbol, tier1, tier2, tier3)

                    # Generate decision
                    action, reason = self.determine_trade_action(symbol, tier1, tier2, tier3)
                    confidence = self.calculate_confidence(tier1, tier2, tier3)
                    signal_strength = self.calculate_signal_strength(tier1, tier2, tier3)

                    decision = TradeDecision(
                        symbol=symbol,
                        action=action,
                        confidence=confidence,
                        price=tier1.get("price", 0),
                        reason=reason,
                        signal_strength=signal_strength,
                        tier1_signals=tier1,
                        tier2_signals=tier2,
                        tier3_signals=tier3,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                    )

                    decisions.append(decision)

                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}")
                    continue

            # Cache decisions
            await self._cache_trade_decisions(decisions)

            return decisions

        except Exception as e:
            logger.error(f"Error generating trade decisions: {e}")
            return []

    async def _cache_trade_decisions(self, decisions: list[TradeDecision]):
        """Cache trade decisions in Redis"""
        try:
            decisions_data = [asdict(decision) for decision in decisions]
            self.redis_client.setex(
                "trade_decisions",
                self.config["cache_ttl"],
                json.dumps(decisions_data),
            )
        except Exception as e:
            logger.error(f"Error caching trade decisions: {e}")

    async def _cache_coin_states(self):
        """Cache coin states in Redis"""
        try:
            states_data = {}
            for symbol, state in self.coin_states.items():
                states_data[symbol] = asdict(state)
                states_data[symbol]["signal_strength"] = state.signal_strength.value

            self.redis_client.setex(
                "coin_states",
                self.config["cache_ttl"],
                json.dumps(states_data),
            )
        except Exception as e:
            logger.error(f"Error caching coin states: {e}")

    async def run(self):
        """Main engine loop"""
        logger.info("Trade engine started")

        while self.is_running:
            try:
                # Generate trade decisions
                decisions = await self.generate_trade_decisions()

                # Cache coin states
                await self._cache_coin_states()

                # Log decision count
                if decisions:
                    logger.info(f"Generated {len(decisions)} trade decisions")

                # Wait for next cycle
                await asyncio.sleep(self.config["decision_interval"])

            except Exception as e:
                logger.error(f"Error in trade engine loop: {e}")
                await asyncio.sleep(self.config["decision_interval"])

        logger.info("Trade engine stopped")

    def get_status(self) -> dict[str, Any]:
        """Get engine status"""
        return {
            "is_running": self.is_running,
            "config": self.config,
            "thresholds": self.thresholds,
            "coin_count": len(self.coin_states),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    async def get_coin_state(self, symbol: str) -> CoinState | None:
        """Get state for a specific coin"""
        return self.coin_states.get(symbol)

    async def get_all_coin_states(self) -> dict[str, CoinState]:
        """Get all coin states"""
        return self.coin_states.copy()

    async def get_trade_decisions(self) -> list[dict[str, Any]]:
        """Get cached trade decisions"""
        try:
            decisions_data = self.redis_client.get("trade_decisions")
            if decisions_data:
                return json.loads(decisions_data)
            return []
        except Exception as e:
            logger.error(f"Error getting trade decisions: {e}")
            return []


def get_trade_engine_manager(redis_client: Any) -> TradeEngineManager:
    """Get Trade Engine Manager instance"""
    return TradeEngineManager(redis_client)


