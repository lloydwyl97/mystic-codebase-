"""
Capital Allocation Engine for Mystic AI Trading Platform
Calculates optimal USD allocation per symbol based on risk, confidence, and diversification.
"""

import logging
import numpy as np
from typing import Dict, List, Any
from datetime import datetime, timezone
import sys
import os

# Add backend to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from modules.ai.persistent_cache import PersistentCache

logger = logging.getLogger(__name__)


class CapitalAllocationEngine:
    def __init__(self):
        """Initialize capital allocation engine with risk parameters"""
        self.cache = PersistentCache()

        # Risk management parameters
        self.max_allocation_per_symbol = 0.4  # 40% max per symbol
        self.min_allocation_per_symbol = 0.05  # 5% min per symbol
        self.max_portfolio_risk = 0.25  # 25% max portfolio risk
        self.diversification_target = 0.8  # 80% diversification target

        # Confidence weighting parameters
        self.confidence_weight = 0.4  # 40% weight for confidence
        self.volatility_weight = 0.3  # 30% weight for volatility
        self.diversification_weight = 0.3  # 30% weight for diversification

        # Volatility thresholds
        self.low_volatility_threshold = 0.02  # 2% daily volatility
        self.high_volatility_threshold = 0.05  # 5% daily volatility

        # Top symbols for allocation
        self.top_symbols = [
            'BTC-USD', 'ETH-USD', 'ADA-USD', 'SOL-USD', 'DOT-USD',
            'LINK-USD', 'MATIC-USD', 'AVAX-USD', 'UNI-USD', 'ATOM-USD'
        ]

        # Exchange diversification
        self.exchanges = ['coinbase', 'binanceus', 'kraken']

        logger.info("✅ CapitalAllocationEngine initialized")

    def _get_signal_confidence(self, symbol: str) -> float:
        """Get confidence score from SignalEngine"""
        try:
            # Get recent signals from cache
            signals = self.cache.get_signals_by_type("SIGNAL_ENGINE", limit=5)

            # Filter by symbol
            symbol_signals = [
                signal for signal in signals
                if signal.get("symbol") == symbol
            ]

            if symbol_signals:
                # Use average confidence from recent signals
                confidences = [signal.get("confidence", 0.0) for signal in symbol_signals]
                return np.mean(confidences)

            return 0.5  # Default confidence

        except Exception as e:
            logger.error(f"Failed to get signal confidence for {symbol}: {e}")
            return 0.5

    def _get_overlord_decision(self, symbol: str) -> Dict[str, Any]:
        """Get final decision from GlobalOverlord"""
        try:
            # Get recent overlord decisions from cache
            signals = self.cache.get_signals_by_type("OVERLORD_DECISION", limit=5)

            # Filter by symbol
            symbol_decisions = [
                signal for signal in signals
                if signal.get("symbol") == symbol
            ]

            if symbol_decisions:
                latest_decision = symbol_decisions[0]
                return {
                    "decision": latest_decision.get("metadata", {}).get("decision", "HOLD"),
                    "confidence": latest_decision.get("confidence", 0.0),
                    "timestamp": latest_decision.get("timestamp")
                }

            return {
                "decision": "HOLD",
                "confidence": 0.0,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            logger.error(f"Failed to get overlord decision for {symbol}: {e}")
            return {
                "decision": "HOLD",
                "confidence": 0.0,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

    def _get_time_optimization(self, symbol: str) -> Dict[str, Any]:
        """Get time optimization from TimeAwareTradeOptimizer"""
        try:
            # Get recent time optimizations from cache
            signals = self.cache.get_signals_by_type("TIME_OPTIMIZATION", limit=5)

            # Filter by symbol
            symbol_optimizations = [
                signal for signal in signals
                if signal.get("symbol") == symbol
            ]

            if symbol_optimizations:
                latest_optimization = symbol_optimizations[0]
                return {
                    "confidence": latest_optimization.get("confidence", 0.0),
                    "best_entry": latest_optimization.get("metadata", {}).get("best_entry", "14:00 UTC"),
                    "best_exit": latest_optimization.get("metadata", {}).get("best_exit", "21:00 UTC"),
                    "timestamp": latest_optimization.get("timestamp")
                }

            return {
                "confidence": 0.5,
                "best_entry": "14:00 UTC",
                "best_exit": "21:00 UTC",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            logger.error(f"Failed to get time optimization for {symbol}: {e}")
            return {
                "confidence": 0.5,
                "best_entry": "14:00 UTC",
                "best_exit": "21:00 UTC",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

    def _calculate_volatility(self, symbol: str) -> float:
        """Calculate volatility for a symbol"""
        try:
            # Get recent price history
            price_history = self.cache.get_price_history('aggregated', symbol, limit=30)

            if not price_history or len(price_history) < 10:
                return 0.03  # Default volatility

            # Calculate daily returns
            prices = [float(data['price']) for data in price_history]
            returns = []

            for i in range(1, len(prices)):
                daily_return = (prices[i] - prices[i-1]) / prices[i-1]
                returns.append(daily_return)

            # Calculate volatility
            volatility = np.std(returns) * np.sqrt(365)  # Annualized

            return max(0.01, min(0.10, volatility))  # Clamp between 1% and 10%

        except Exception as e:
            logger.error(f"Failed to calculate volatility for {symbol}: {e}")
            return 0.03  # Default volatility

    def _calculate_risk_score(self, symbol: str) -> float:
        """Calculate risk score for a symbol"""
        try:
            # Get volatility
            volatility = self._calculate_volatility(symbol)

            # Get overlord decision
            overlord_decision = self._get_overlord_decision(symbol)

            # Calculate risk score based on volatility and decision
            base_risk = volatility / 0.05  # Normalize to 5% volatility

            # Adjust risk based on overlord decision
            decision_risk = {
                "BUY": 0.8,  # Lower risk for buy signals
                "SELL": 0.9,  # Higher risk for sell signals
                "HOLD": 1.0   # Normal risk for hold
            }

            decision_multiplier = decision_risk.get(overlord_decision["decision"], 1.0)

            risk_score = base_risk * decision_multiplier

            return max(0.1, min(1.0, risk_score))  # Clamp between 0.1 and 1.0

        except Exception as e:
            logger.error(f"Failed to calculate risk score for {symbol}: {e}")
            return 0.5  # Default risk score

    def _calculate_allocation_score(self, symbol: str) -> float:
        """Calculate allocation score for a symbol"""
        try:
            # Get confidence from signal engine
            signal_confidence = self._get_signal_confidence(symbol)

            # Get overlord confidence
            overlord_decision = self._get_overlord_decision(symbol)
            overlord_confidence = overlord_decision.get("confidence", 0.0)

            # Get time optimization confidence
            time_optimization = self._get_time_optimization(symbol)
            time_confidence = time_optimization.get("confidence", 0.0)

            # Calculate weighted confidence
            weighted_confidence = (
                signal_confidence * 0.4 +
                overlord_confidence * 0.4 +
                time_confidence * 0.2
            )

            # Get risk score
            risk_score = self._calculate_risk_score(symbol)

            # Calculate allocation score (higher confidence, lower risk = higher allocation)
            allocation_score = weighted_confidence * (1 - risk_score * 0.5)

            return max(0.0, min(1.0, allocation_score))

        except Exception as e:
            logger.error(f"Failed to calculate allocation score for {symbol}: {e}")
            return 0.5  # Default allocation score

    def _apply_diversification_constraints(self, allocations: Dict[str, float],
                                        portfolio_balance: float) -> Dict[str, float]:
        """Apply diversification constraints to allocations"""
        try:
            # Sort symbols by allocation score
            sorted_symbols = sorted(
                allocations.items(),
                key=lambda x: x[1],
                reverse=True
            )

            # Calculate total allocation
            sum(allocations.values())

            # Apply maximum allocation per symbol
            adjusted_allocations = {}
            remaining_balance = portfolio_balance

            for symbol, allocation in sorted_symbols:
                # Calculate maximum allowed allocation
                max_allocation = portfolio_balance * self.max_allocation_per_symbol
                min_allocation = portfolio_balance * self.min_allocation_per_symbol

                # Apply constraints
                adjusted_allocation = max(min_allocation, min(max_allocation, allocation))

                # Ensure we don't exceed portfolio balance
                if adjusted_allocation > remaining_balance:
                    adjusted_allocation = remaining_balance

                if adjusted_allocation > 0:
                    adjusted_allocations[symbol] = adjusted_allocation
                    remaining_balance -= adjusted_allocation

                if remaining_balance <= 0:
                    break

            return adjusted_allocations

        except Exception as e:
            logger.error(f"Failed to apply diversification constraints: {e}")
            return allocations

    def get_allocation_plan(self, portfolio_balance: float) -> Dict[str, Any]:
        """Get optimal allocation plan for portfolio"""
        try:
            logger.info(f"💰 Calculating allocation plan for ${portfolio_balance:,.2f} portfolio")

            # Calculate allocation scores for all symbols
            allocation_scores = {}
            for symbol in self.top_symbols:
                score = self._calculate_allocation_score(symbol)
                allocation_scores[symbol] = score

            # Convert scores to dollar amounts
            total_score = sum(allocation_scores.values())
            if total_score > 0:
                # Distribute based on scores
                allocations = {}
                for symbol, score in allocation_scores.items():
                    allocation = (score / total_score) * portfolio_balance
                    allocations[symbol] = allocation
            else:
                # Equal distribution if no scores
                equal_allocation = portfolio_balance / len(self.top_symbols)
                allocations = {symbol: equal_allocation for symbol in self.top_symbols}

            # Apply diversification constraints
            final_allocations = self._apply_diversification_constraints(allocations, portfolio_balance)

            # Calculate allocation statistics
            total_allocated = sum(final_allocations.values())
            allocation_percentage = (total_allocated / portfolio_balance) * 100 if portfolio_balance > 0 else 0

            # Create allocation plan
            allocation_plan = {
                "portfolio_balance": portfolio_balance,
                "allocations": final_allocations,
                "total_allocated": total_allocated,
                "allocation_percentage": allocation_percentage,
                "unallocated_balance": portfolio_balance - total_allocated,
                "diversification_score": len(final_allocations) / len(self.top_symbols),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

            # Store allocation plan in cache
            plan_id = f"allocation_plan_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
            self.cache.store_signal(
                signal_id=plan_id,
                symbol="PORTFOLIO",
                signal_type="CAPITAL_ALLOCATION",
                confidence=allocation_percentage / 100.0,
                strategy="risk_based_allocation",
                metadata=allocation_plan
            )

            logger.info(f"✅ Allocation plan complete: ${total_allocated:,.2f} allocated across {len(final_allocations)} symbols")

            return allocation_plan

        except Exception as e:
            logger.error(f"Failed to get allocation plan: {e}")
            return {
                "portfolio_balance": portfolio_balance,
                "allocations": {},
                "total_allocated": 0.0,
                "allocation_percentage": 0.0,
                "unallocated_balance": portfolio_balance,
                "diversification_score": 0.0,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

    def allocate_for_symbol(self, symbol: str, balance: float) -> Dict[str, Any]:
        """Get allocation recommendation for a specific symbol"""
        try:
            logger.info(f"💰 Calculating allocation for {symbol} with ${balance:,.2f}")

            # Calculate allocation score
            allocation_score = self._calculate_allocation_score(symbol)

            # Get risk assessment
            risk_score = self._calculate_risk_score(symbol)

            # Get overlord decision
            overlord_decision = self._get_overlord_decision(symbol)

            # Get time optimization
            time_optimization = self._get_time_optimization(symbol)

            # Calculate recommended allocation
            max_allocation = balance * self.max_allocation_per_symbol
            recommended_allocation = balance * allocation_score

            # Apply constraints
            final_allocation = max(
                balance * self.min_allocation_per_symbol,
                min(max_allocation, recommended_allocation)
            )

            # Create allocation recommendation
            recommendation = {
                "symbol": symbol,
                "balance": balance,
                "recommended_allocation": final_allocation,
                "allocation_percentage": (final_allocation / balance) * 100 if balance > 0 else 0,
                "allocation_score": allocation_score,
                "risk_score": risk_score,
                "overlord_decision": overlord_decision,
                "time_optimization": time_optimization,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

            # Store recommendation in cache
            recommendation_id = f"allocation_{symbol}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
            self.cache.store_signal(
                signal_id=recommendation_id,
                symbol=symbol,
                signal_type="SYMBOL_ALLOCATION",
                confidence=allocation_score,
                strategy="risk_based_allocation",
                metadata=recommendation
            )

            logger.info(f"✅ Allocation recommendation: ${final_allocation:,.2f} ({recommendation['allocation_percentage']:.1f}%)")

            return recommendation

        except Exception as e:
            logger.error(f"Failed to allocate for {symbol}: {e}")
            return {
                "symbol": symbol,
                "balance": balance,
                "recommended_allocation": 0.0,
                "allocation_percentage": 0.0,
                "allocation_score": 0.0,
                "risk_score": 0.5,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

    def get_allocation_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get capital allocation history"""
        try:
            # Get recent allocation signals from cache
            signals = self.cache.get_signals_by_type("CAPITAL_ALLOCATION", limit=limit)

            return signals

        except Exception as e:
            logger.error(f"Failed to get allocation history: {e}")
            return []

    def get_engine_status(self) -> Dict[str, Any]:
        """Get current allocation engine status"""
        try:
            return {
                "service": "CapitalAllocationEngine",
                "status": "active",
                "risk_parameters": {
                    "max_allocation_per_symbol": self.max_allocation_per_symbol,
                    "min_allocation_per_symbol": self.min_allocation_per_symbol,
                    "max_portfolio_risk": self.max_portfolio_risk,
                    "diversification_target": self.diversification_target
                },
                "weighting_parameters": {
                    "confidence_weight": self.confidence_weight,
                    "volatility_weight": self.volatility_weight,
                    "diversification_weight": self.diversification_weight
                },
                "volatility_thresholds": {
                    "low_volatility": self.low_volatility_threshold,
                    "high_volatility": self.high_volatility_threshold
                },
                "monitored_symbols": self.top_symbols,
                "exchanges": self.exchanges
            }

        except Exception as e:
            logger.error(f"Failed to get engine status: {e}")
            return {"success": False, "error": str(e)}


# Global capital allocation engine instance
capital_allocation_engine = CapitalAllocationEngine()


def get_capital_allocation_engine() -> CapitalAllocationEngine:
    """Get the global capital allocation engine instance"""
    return capital_allocation_engine


if __name__ == "__main__":
    # Test the capital allocation engine
    engine = CapitalAllocationEngine()
    print(f"✅ CapitalAllocationEngine initialized: {engine}")

    # Test allocation plan
    plan = engine.get_allocation_plan(1000)
    print(f"Allocation plan: {plan}")

    # Test symbol allocation
    recommendation = engine.allocate_for_symbol('BTC-USD', 1000)
    print(f"Symbol allocation: {recommendation}")

    # Test status
    status = engine.get_engine_status()
    print(f"Engine status: {status['status']}")
