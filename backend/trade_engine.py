#!/usr/bin/env python3
"""
Unified Trade Decision Engine
Combines all three tiers of signals and makes trading decisions every 3-10 seconds
"""

import asyncio
import json
import logging
from dataclasses import asdict, dataclass
from datetime import timezone, datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

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
    macd: Dict[str, float]
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
    tier1_signals: Dict[str, Any]
    tier2_signals: Dict[str, Any]
    tier3_signals: Dict[str, Any]
    timestamp: str


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
        self.coin_states: Dict[str, CoinState] = {}

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

    async def get_tier1_signals(self) -> Dict[str, Any]:
        """Get Tier 1 signals from cache"""
        try:
            tier1_data = self.redis_client.get("tier1_signals")
            if tier1_data:
                return json.loads(tier1_data)
            return {}
        except Exception as e:
            logger.error(f"Error getting Tier 1 signals: {e}")
            return {}

    async def get_tier2_signals(self) -> Dict[str, Any]:
        """Get Tier 2 signals from cache"""
        try:
            tier2_data = self.redis_client.get("tier2_indicators")
            if tier2_data:
                return json.loads(tier2_data)
            return {}
        except Exception as e:
            logger.error(f"Error getting Tier 2 signals: {e}")
            return {}

    async def get_tier3_signals(self) -> Dict[str, Any]:
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
        tier1: Dict[str, Any],
        tier2: Dict[str, Any],
        tier3: Dict[str, Any],
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
        tier1: Dict[str, Any],
        tier2: Dict[str, Any],
        tier3: Dict[str, Any],
    ) -> float:
        """Calculate trading confidence from all tiers"""
        try:
            confidence_factors: List[float] = []

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
        tier1: Dict[str, Any],
        tier2: Dict[str, Any],
        tier3: Dict[str, Any],
    ) -> Tuple[TradeAction, str]:
        """Determine trading action based on all signal tiers"""
        try:
            reasons: List[str] = []
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
        tier1: Dict[str, Any],
        tier2: Dict[str, Any],
        tier3: Dict[str, Any],
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

    async def generate_trade_decisions(self) -> List[TradeDecision]:
        """Generate trade decisions for all coins"""
        try:
            # Get all signal tiers
            tier1_signals = await self.get_tier1_signals()
            tier2_signals = await self.get_tier2_signals()
            tier3_signals = await self.get_tier3_signals()

            decisions: List[TradeDecision] = []

            # Process each coin
            for symbol in tier1_signals.get("prices", {}):
                try:
                    tier1 = tier1_signals["prices"].get(symbol, {})
                    tier2 = tier2_signals.get("indicators", {}).get(symbol, {})
                    tier3 = tier3_signals.get("cosmic_signals", {})

                    # Update coin state
                    await self.update_coin_state(symbol, tier1, tier2, tier3)

                    # Generate trade decision
                    action, reason = self.determine_trade_action(symbol, tier1, tier2, tier3)
                    confidence = self.calculate_confidence(tier1, tier2, tier3)
                    price = tier1.get("price", 0.0)

                    decision = TradeDecision(
                        symbol=symbol,
                        action=action,
                        confidence=confidence,
                        price=price,
                        reason=reason,
                        tier1_signals=tier1,
                        tier2_signals=tier2,
                        tier3_signals=tier3,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                    )

                    decisions.append(decision)

                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}")
                    continue

            return decisions

        except Exception as e:
            logger.error(f"Error generating trade decisions: {e}")
            return []

    async def _cache_trade_decisions(self, decisions: List[TradeDecision]):
        """Cache trade decisions"""
        try:
            decisions_data = [asdict(d) for d in decisions]
            self.redis_client.setex(
                "trade_decisions",
                self.config["cache_ttl"],
                json.dumps(decisions_data),
            )
        except Exception as e:
            logger.error(f"Error caching trade decisions: {e}")

    async def _cache_coin_states(self):
        """Cache coin states"""
        try:
            states_data = {symbol: asdict(state) for symbol, state in self.coin_states.items()}
            self.redis_client.setex(
                "coin_states",
                self.config["cache_ttl"],
                json.dumps(states_data),
            )
        except Exception as e:
            logger.error(f"Error caching coin states: {e}")

    async def run(self):
        """Main trade engine loop"""
        logger.info("Starting Unified Trade Engine...")
        self.is_running = True

        try:
            while self.is_running:
                try:
                    # Generate trade decisions
                    decisions = await self.generate_trade_decisions()

                    # Cache decisions and states
                    await self._cache_trade_decisions(decisions)
                    await self._cache_coin_states()

                    # Log significant decisions
                    for decision in decisions:
                        if decision.confidence > 0.8 and decision.action != TradeAction.HOLD:
                            logger.info(
                                f"Strong signal: {decision.symbol} {decision.action.value} "
                                f"(confidence: {decision.confidence:.2f}) - {decision.reason}"
                            )

                    logger.debug(f"Generated {len(decisions)} trade decisions")

                    # Wait for next cycle
                    await asyncio.sleep(self.config["decision_interval"])

                except Exception as e:
                    logger.error(f"Error in trade engine loop: {e}")
                    await asyncio.sleep(10)  # Wait 10 seconds on error

        except Exception as e:
            logger.error(f"Fatal error in trade engine: {e}")
        finally:
            self.is_running = False

    def get_status(self) -> Dict[str, Any]:
        """Get trade engine status"""
        return {
            "status": "running" if self.is_running else "stopped",
            "config": self.config,
            "coin_states_count": len(self.coin_states),
            "thresholds": self.thresholds,
        }

    async def get_coin_state(self, symbol: str) -> Optional[CoinState]:
        """Get current state for a specific coin"""
        return self.coin_states.get(symbol)

    async def get_all_coin_states(self) -> Dict[str, CoinState]:
        """Get all coin states"""
        return self.coin_states.copy()

    async def get_trade_decisions(self) -> List[Dict[str, Any]]:
        """Get cached trade decisions"""
        try:
            decisions_data = self.redis_client.get("trade_decisions")
            if decisions_data:
                return json.loads(decisions_data)
            return []
        except Exception as e:
            logger.error(f"Error getting trade decisions: {e}")
            return []
