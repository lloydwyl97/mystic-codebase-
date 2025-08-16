"""
Global Overlord for Mystic AI Trading Platform
Orchestrates AI decisions from multiple agents using weighted voting and confidence scoring.
"""

import logging
import random
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
import sys
import os

# Add backend to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from backend.modules.ai.persistent_cache import PersistentCache
from backend.modules.ai.signal_engine import SignalEngine
from backend.modules.ai.self_replication_engine import SelfReplicationEngine

logger = logging.getLogger(__name__)


class CosmicPatternRecognizer:
    """Mock cosmic pattern recognizer for optional cosmic overlays"""

    def __init__(self):
        self.cache = PersistentCache()
        self.cosmic_weights = {
            "moon_phase": 0.1,
            "solar_activity": 0.05,
            "cosmic_rays": 0.03,
            "planetary_alignment": 0.02
        }

    def get_cosmic_factors(self, symbol: str) -> Dict[str, float]:
        """Get cosmic factors that might influence trading"""
        try:
            # Mock cosmic factors (in real implementation, would fetch from cosmic APIs)
            factors = {
                "moon_phase": random.uniform(0.8, 1.2),
                "solar_activity": random.uniform(0.9, 1.1),
                "cosmic_rays": random.uniform(0.95, 1.05),
                "planetary_alignment": random.uniform(0.85, 1.15)
            }

            return factors

        except Exception as e:
            logger.error(f"Failed to get cosmic factors: {e}")
            return {"moon_phase": 1.0, "solar_activity": 1.0, "cosmic_rays": 1.0, "planetary_alignment": 1.0}

    def adjust_confidence(self, base_confidence: float, symbol: str) -> float:
        """Adjust confidence based on cosmic factors"""
        try:
            cosmic_factors = self.get_cosmic_factors(symbol)

            # Calculate cosmic adjustment
            adjustment = 1.0
            for factor, weight in self.cosmic_weights.items():
                adjustment *= (cosmic_factors[factor] * weight + (1 - weight))

            # Apply adjustment to confidence
            adjusted_confidence = base_confidence * adjustment

            # Ensure confidence stays within bounds
            return max(0.0, min(1.0, adjusted_confidence))

        except Exception as e:
            logger.error(f"Failed to adjust confidence: {e}")
            return base_confidence


class GlobalOverlord:
    def __init__(self):
        """Initialize global overlord with all AI components"""
        self.cache = PersistentCache()
        self.signal_engine = SignalEngine()
        self.self_replication_engine = SelfReplicationEngine()

        # Decision weights for different signal sources
        self.signal_weights = {
            "signal_engine": 0.4,
            "self_replication_agents": 0.5,
            "cosmic_patterns": 0.1
        }

        # Confidence thresholds
        self.min_confidence = 0.6
        self.high_confidence = 0.8

        # Voting parameters
        self.min_agents_for_decision = 3
        self.consensus_threshold = 0.7

        # Initialize cosmic pattern recognizer if available
        try:
            self.cosmic_recognizer = CosmicPatternRecognizer()
            self.cosmic_available = True
        except Exception as e:
            logger.warning(f"Cosmic pattern recognizer not available: {e}")
            self.cosmic_available = False

        logger.info("âœ… GlobalOverlord initialized")

    def _collect_signal_engine_signals(self, exchange: str, symbol: str) -> List[Dict[str, Any]]:
        """Collect signals from the signal engine"""
        try:
            signals = []

            # Get recent signals from signal engine
            signal_engine_signals = self.signal_engine.get_recent_signals(symbol, limit=5)

            for signal in signal_engine_signals:
                # Apply cosmic adjustment if available
                if self.cosmic_available:
                    adjusted_confidence = self.cosmic_recognizer.adjust_confidence(
                        signal.get("confidence", 0.0), symbol
                    )
                else:
                    adjusted_confidence = signal.get("confidence", 0.0)

                signals.append({
                    "source": "signal_engine",
                    "signal": signal.get("signal_type"),
                    "confidence": adjusted_confidence,
                    "weight": self.signal_weights["signal_engine"],
                    "metadata": signal.get("metadata", {})
                })

            return signals

        except Exception as e:
            logger.error(f"Failed to collect signal engine signals: {e}")
            return []

    def _collect_agent_signals(self, exchange: str, symbol: str) -> List[Dict[str, Any]]:
        """Collect signals from self-replication agents"""
        try:
            signals = []

            # Get recent agent data
            agent_data = self.self_replication_engine.get_recent_agents(limit=10)

            for agent in agent_data:
                # Generate signal from agent data
                agent_signal = self._generate_agent_signal(agent, symbol)
                if agent_signal:
                    signals.append(agent_signal)

            return signals

        except Exception as e:
            logger.error(f"Failed to collect agent signals: {e}")
            return []

    def _generate_agent_signal(self, agent_data: Dict[str, Any], symbol: str) -> Optional[Dict[str, Any]]:
        """Generate trading signal from agent data"""
        try:
            # Mock price data for technical analysis
            prices = [100 + i * random.uniform(-2, 2) for i in range(50)]

            # Calculate technical indicators
            rsi_values = self._calculate_rsi(prices)
            ema_20 = self._calculate_ema(prices, 20)
            ema_50 = self._calculate_ema(prices, 50)

            if not rsi_values or not ema_20 or not ema_50:
                return None

            # Generate signal based on technical analysis
            current_rsi = rsi_values[-1] if rsi_values else 50
            current_ema_20 = ema_20[-1] if ema_20 else prices[-1]
            current_ema_50 = ema_50[-1] if ema_50 else prices[-1]

            signal = "HOLD"
            confidence = 0.5

            # RSI-based signals
            if current_rsi < 30:
                signal = "BUY"
                confidence = 0.7
            elif current_rsi > 70:
                signal = "SELL"
                confidence = 0.7

            # EMA crossover signals
            if current_ema_20 > current_ema_50:
                if signal == "BUY":
                    confidence += 0.1
                elif signal == "HOLD":
                    signal = "BUY"
                    confidence = 0.6
            elif current_ema_20 < current_ema_50:
                if signal == "SELL":
                    confidence += 0.1
                elif signal == "HOLD":
                    signal = "SELL"
                    confidence = 0.6

            # Apply agent-specific adjustments
            agent_confidence = agent_data.get("performance_score", 0.5)
            confidence = (confidence + agent_confidence) / 2

            return {
                "source": f"agent_{agent_data.get('agent_id', 'unknown')}",
                "signal": signal,
                "confidence": min(confidence, 1.0),
                "weight": self.signal_weights["self_replication_agents"],
                "metadata": {
                    "agent_id": agent_data.get("agent_id"),
                    "performance_score": agent_confidence,
                    "rsi": current_rsi,
                    "ema_20": current_ema_20,
                    "ema_50": current_ema_50
                }
            }

        except Exception as e:
            logger.error(f"Failed to generate agent signal: {e}")
            return None

    def _calculate_rsi(self, prices: List[float], period: int = 14) -> List[float]:
        """Calculate RSI for price series"""
        try:
            if len(prices) < period + 1:
                return []

            rsi_values = []
            gains = []
            losses = []

            # Calculate price changes
            for i in range(1, len(prices)):
                change = prices[i] - prices[i-1]
                gains.append(max(change, 0))
                losses.append(max(-change, 0))

            # Calculate initial average gain and loss
            avg_gain = sum(gains[:period]) / period
            avg_loss = sum(losses[:period]) / period

            # Calculate RSI for each period
            for i in range(period, len(prices)):
                avg_gain = sum(gains[i-period:i]) / period
                avg_loss = sum(losses[i-period:i]) / period

                if avg_loss == 0:
                    rsi = 100
                else:
                    rs = avg_gain / avg_loss
                    rsi = 100 - (100 / (1 + rs))

                rsi_values.append(rsi)

            return rsi_values

        except Exception as e:
            logger.error(f"Failed to calculate RSI: {e}")
            return []

    def _calculate_ema(self, prices: List[float], period: int) -> List[float]:
        """Calculate EMA for price series"""
        try:
            if len(prices) < period:
                return []

            ema_values = []
            multiplier = 2 / (period + 1)

            # First EMA is SMA
            sma = sum(prices[:period]) / period
            ema_values.append(sma)

            # Calculate subsequent EMAs
            for i in range(period, len(prices)):
                ema = (prices[i] * multiplier) + (ema_values[-1] * (1 - multiplier))
                ema_values.append(ema)

            return ema_values

        except Exception as e:
            logger.error(f"Failed to calculate EMA: {e}")
            return []

    def _apply_weighted_voting(self, signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Apply weighted voting to determine final decision"""
        try:
            if not signals:
                return {
                    "decision": "HOLD",
                    "confidence": 0.0,
                    "source_agents": [],
                    "reason": "No signals available"
                }

            # Count votes by signal type
            buy_votes = 0.0
            sell_votes = 0.0
            hold_votes = 0.0
            total_weight = 0.0

            source_agents = []

            for signal in signals:
                weight = signal.get("weight", 1.0)
                total_weight += weight

                if signal.get("signal") == "BUY":
                    buy_votes += weight
                elif signal.get("signal") == "SELL":
                    sell_votes += weight
                else:
                    hold_votes += weight

                source_agents.append({
                    "source": signal.get("source"),
                    "signal": signal.get("signal"),
                    "confidence": signal.get("confidence"),
                    "weight": weight,
                    "metadata": signal.get("metadata", {})
                })

            # Calculate percentages
            if total_weight > 0:
                buy_percentage = (buy_votes / total_weight) * 100
                sell_percentage = (sell_votes / total_weight) * 100
                hold_percentage = (hold_votes / total_weight) * 100
            else:
                buy_percentage = sell_percentage = hold_percentage = 0.0

            # Determine final decision
            decision = "HOLD"
            confidence = 0.0

            if buy_percentage > sell_percentage and buy_percentage > hold_percentage:
                decision = "BUY"
                confidence = buy_percentage
            elif sell_percentage > buy_percentage and sell_percentage > hold_percentage:
                decision = "SELL"
                confidence = sell_percentage
            else:
                decision = "HOLD"
                confidence = hold_percentage

            # Check if we have enough agents for a decision
            if len(signals) < self.min_agents_for_decision:
                decision = "HOLD"
                confidence = 0.0

            # Check consensus threshold
            if confidence < (self.consensus_threshold * 100):
                decision = "HOLD"
                confidence = 0.0

            return {
                "decision": decision,
                "confidence": confidence,
                "source_agents": source_agents,
                "vote_breakdown": {
                    "buy_percentage": buy_percentage,
                    "sell_percentage": sell_percentage,
                    "hold_percentage": hold_percentage,
                    "total_agents": len(signals)
                }
            }

        except Exception as e:
            logger.error(f"Failed to apply weighted voting: {e}")
            return {
                "decision": "HOLD",
                "confidence": 0.0,
                "source_agents": [],
                "error": str(e)
            }

    def decide_trade(self, exchange: str, symbol: str) -> Dict[str, Any]:
        """Make final trading decision using weighted voting"""
        try:
            logger.info(f"ðŸ¤– GlobalOverlord making decision for {symbol}")

            # Collect signals from all sources
            signal_engine_signals = self._collect_signal_engine_signals(exchange, symbol)
            agent_signals = self._collect_agent_signals(exchange, symbol)

            # Combine all signals
            all_signals = signal_engine_signals + agent_signals

            # Apply weighted voting
            decision_result = self._apply_weighted_voting(all_signals)

            # Create overlord decision
            overlord_decision = {
                "exchange": exchange,
                "symbol": symbol,
                "decision": decision_result["decision"],
                "confidence": decision_result["confidence"] / 100.0,
                "source_agents": decision_result["source_agents"],
                "vote_breakdown": decision_result.get("vote_breakdown", {}),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "cosmic_available": self.cosmic_available
            }

            # Store decision in cache
            decision_id = f"overlord_{symbol}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
            self.cache.store_signal(
                signal_id=decision_id,
                symbol=symbol,
                signal_type="OVERLORD_DECISION",
                confidence=decision_result["confidence"] / 100.0,
                strategy="weighted_voting",
                metadata=overlord_decision
            )

            logger.info(f"âœ… Overlord decision: {decision_result['decision']} with {decision_result['confidence']:.1f}% confidence")

            return overlord_decision

        except Exception as e:
            logger.error(f"Failed to decide trade: {e}")
            return {
                "exchange": exchange,
                "symbol": symbol,
                "decision": "HOLD",
                "confidence": 0.0,
                "source_agents": [],
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

    def get_decision_history(self, symbol: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent overlord decisions for a symbol"""
        try:
            # Get recent signals from cache
            signals = self.cache.get_signals_by_type("OVERLORD_DECISION", limit=limit)

            # Filter by symbol
            symbol_decisions = [
                signal for signal in signals
                if signal.get("symbol") == symbol
            ]

            return symbol_decisions

        except Exception as e:
            logger.error(f"Failed to get decision history: {e}")
            return []

    def get_consensus_metrics(self) -> Dict[str, Any]:
        """Get consensus metrics from recent decisions"""
        try:
            # Get recent overlord decisions
            recent_decisions = self.cache.get_signals_by_type("OVERLORD_DECISION", limit=100)
            
            if not recent_decisions:
                return {
                    "total_decisions": 0,
                    "consensus_rate": 0.0,
                    "average_confidence": 0.0,
                    "decision_distribution": {},
                    "recent_decisions": []
                }
            
            # Calculate metrics
            total_decisions = len(recent_decisions)
            decisions_with_consensus = [
                d for d in recent_decisions 
                if d.get("metadata", {}).get("confidence", 0) >= self.consensus_threshold
            ]
            consensus_rate = (len(decisions_with_consensus) / total_decisions * 100) if total_decisions > 0 else 0
            
            # Calculate average confidence
            confidences = [
                d.get("metadata", {}).get("confidence", 0) 
                for d in recent_decisions
            ]
            average_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            # Decision distribution
            decision_distribution = {}
            for decision in recent_decisions:
                decision_type = decision.get("metadata", {}).get("decision", "UNKNOWN")
                decision_distribution[decision_type] = decision_distribution.get(decision_type, 0) + 1
            
            return {
                "total_decisions": total_decisions,
                "consensus_rate": consensus_rate,
                "average_confidence": average_confidence,
                "decision_distribution": decision_distribution,
                "recent_decisions": recent_decisions[:10]
            }
            
        except Exception as e:
            logger.error(f"Failed to get consensus metrics: {e}")
            return {
                "total_decisions": 0,
                "consensus_rate": 0.0,
                "average_confidence": 0.0,
                "decision_distribution": {},
                "recent_decisions": []
            }

    def get_overlord_status(self) -> Dict[str, Any]:
        """Get current overlord status and configuration"""
        try:
            return {
                "service": "GlobalOverlord",
                "status": "active",
                "signal_weights": self.signal_weights,
                "confidence_thresholds": {
                    "min_confidence": self.min_confidence,
                    "high_confidence": self.high_confidence
                },
                "voting_parameters": {
                    "min_agents_for_decision": self.min_agents_for_decision,
                    "consensus_threshold": self.consensus_threshold
                },
                "cosmic_available": self.cosmic_available,
                "components": {
                    "signal_engine": "active",
                    "self_replication_engine": "active",
                    "persistent_cache": "active"
                }
            }

        except Exception as e:
            logger.error(f"Failed to get overlord status: {e}")
            return {"success": False, "error": str(e)}


# Global overlord instance
global_overlord = GlobalOverlord()


def get_global_overlord() -> GlobalOverlord:
    """Get the global overlord instance"""
    return global_overlord


if __name__ == "__main__":
    # Test the global overlord
    overlord = GlobalOverlord()
    print(f"âœ… GlobalOverlord initialized: {overlord}")

    # Test decision making
    decision = overlord.decide_trade('coinbase', 'BTC-USD')
    print(f"Overlord decision: {decision}")

    # Test status
    status = overlord.get_overlord_status()
    print(f"Overlord status: {status['status']}")
    print(f"Cosmic available: {status['cosmic_available']}")


