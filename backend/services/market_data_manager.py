"""
Market Data Manager

Handles business logic for market data operations including signal calculation,
API health management, and data processing.
"""

import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class MarketDataManager:
    """Manages market data business logic and operations."""

    def __init__(self):
        self.api_health: Dict[str, Dict[str, Any]] = {
            "binance": {"healthy": True, "last_check": time.time()},
            "coinbase": {"healthy": True, "last_check": time.time()},
        }
        self.current_api = "binance"

    def calculate_signal_strength(self, data: Dict[str, Any]) -> str:
        """Calculate signal strength from market data."""
        change_24h = data.get("change_24h", 0)

        if change_24h > 10:
            return "STRONG_BUY"
        elif change_24h > 5:
            return "BUY"
        elif change_24h > 0:
            return "WEAK_BUY"
        elif change_24h > -5:
            return "HOLD"
        elif change_24h > -10:
            return "SELL"
        else:
            return "STRONG_SELL"

    def calculate_confidence(self, data: Dict[str, Any]) -> float:
        """Calculate confidence level from market data."""
        confidence_factors: List[float] = []

        # Price momentum
        change_24h = abs(data.get("change_24h", 0))
        if change_24h > 10:
            confidence_factors.append(0.9)
        elif change_24h > 5:
            confidence_factors.append(0.8)
        elif change_24h > 2:
            confidence_factors.append(0.7)
        else:
            confidence_factors.append(0.6)

        # Volume
        volume_24h = data.get("volume_24h", 0)
        if volume_24h > 1000000:
            confidence_factors.append(0.85)
        elif volume_24h > 100000:
            confidence_factors.append(0.75)
        else:
            confidence_factors.append(0.65)

        # Calculate average confidence
        if confidence_factors:
            avg_confidence = sum(confidence_factors) / len(confidence_factors)
            return min(0.95, max(0.6, avg_confidence))

        return 0.6

    def determine_trade_action(self, data: Dict[str, Any]) -> Tuple[str, str]:
        """Determine trade action from market data."""
        reasons: List[str] = []
        buy_signals = 0
        sell_signals = 0

        # Price momentum
        change_24h = data.get("change_24h", 0)
        if change_24h > 2:
            buy_signals += 1
            reasons.append(f"Strong upward momentum ({change_24h:.2f}%)")
        elif change_24h < -2:
            sell_signals += 1
            reasons.append(f"Strong downward momentum ({change_24h:.2f}%)")

        # Volume analysis
        volume_24h = data.get("volume_24h", 0)
        if volume_24h > 1000000:
            buy_signals += 1
            reasons.append("High volume activity")

        # Determine action
        if buy_signals > sell_signals and buy_signals >= 1:
            return "BUY", " | ".join(reasons)
        elif sell_signals > buy_signals and sell_signals >= 1:
            return "SELL", " | ".join(reasons)
        else:
            return "HOLD", "Insufficient signal strength"

    def update_api_health(self, api_name: str, success: bool, error: Optional[str] = None):
        """Update API health status."""
        self.api_health[api_name] = {
            "healthy": success,
            "last_check": time.time(),
            "last_error": error if not success else None,
        }

    def get_healthy_api(self) -> str:
        """Get the healthiest available API."""
        healthy_apis = [
            api for api, health in self.api_health.items() if health.get("healthy", True)
        ]

        if not healthy_apis:
            # If no healthy APIs, return the current one
            return self.current_api

        # Prefer the current API if it's healthy
        if self.current_api in healthy_apis:
            return self.current_api

        # Otherwise, rotate to the first healthy API
        self.current_api = healthy_apis[0]
        return self.current_api

    def process_market_data(
        self, cached_data: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """Process market data into signals format."""
        signals: Dict[str, Dict[str, Any]] = {}

        for coin, data in cached_data.items():
            signals[coin] = {
                "price": data.get("price", 0.0),
                "change_24h": data.get("change_24h", 0.0),
                "signal_strength": self.calculate_signal_strength(data),
                "confidence": self.calculate_confidence(data),
                "action": self.determine_trade_action(data)[0],
                "reason": self.determine_trade_action(data)[1],
                "api_source": data.get("api_source", "unknown"),
                "timestamp": data.get("timestamp", ""),
            }

        return signals

    def format_markets_data(
        self, cached_data: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """Format cached data into markets overview format."""
        markets_data: Dict[str, Dict[str, Any]] = {}
        for symbol, data in cached_data.items():
            markets_data[symbol] = {
                "symbol": symbol,
                "price": data.get("price", 0.0),
                "change_24h": data.get("change_24h", 0.0),
                "volume_24h": data.get("volume_24h", 0.0),
                "high_24h": data.get("high_24h", 0.0),
                "low_24h": data.get("low_24h", 0.0),
                "api_source": data.get("api_source", "unknown"),
                "timestamp": data.get(
                    "timestamp",
                    datetime.now(timezone.timezone.utc).isoformat(),
                ),
            }

        return markets_data


# Global instance
market_data_manager = MarketDataManager()


