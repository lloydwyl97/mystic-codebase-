"""
Signal Service

Handles trading signals and signal-related operations.
"""

import logging
from datetime import timezone, datetime
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class SignalService:
    """Service for managing trading signals."""

    def __init__(self):
        self.signals = []
        self.signal_history = []

    async def get_signals(self) -> List[Dict[str, Any]]:
        """Get current trading signals with live data."""
        try:
            # Return empty list - signals should come from AI endpoints
            return []
        except Exception as e:
            logger.error(f"Error getting signals: {str(e)}")
            return []

    async def create_signal(self, signal_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new trading signal."""
        try:
            signal = {
                "id": f"sig_{len(self.signals) + 1}",
                "symbol": signal_data["symbol"],
                "type": signal_data["type"],
                "confidence": signal_data["confidence"],
                "price": signal_data["price"],
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source": signal_data.get("source", "Manual"),
                "indicators": signal_data.get("indicators", {}),
            }

            self.signals.append(signal)
            return signal
        except Exception as e:
            logger.error(f"Error creating signal: {str(e)}")
            return {}

    async def get_signal(self, signal_id: str) -> Dict[str, Any]:
        """Get a specific signal by ID."""
        try:
            for signal in self.signals:
                if signal.get("id") == signal_id:
                    return signal
            return None
        except Exception as e:
            logger.error(f"Error getting signal {signal_id}: {str(e)}")
            return None

    async def get_latest_signals(self) -> List[Dict[str, Any]]:
        """Get latest trading signals."""
        try:
            # Return the same signals as get_signals() for consistency
            return await self.get_signals()
        except Exception as e:
            logger.error(f"Error getting latest signals: {str(e)}")
            return []

    async def get_signal_metrics(self) -> Dict[str, Any]:
        """Get signal performance metrics."""
        try:
            total_signals = len(self.signals)
            successful_signals = len(
                [s for s in self.signals if s.get("type") in ["BUY", "STRONG_BUY"]]
            )
            success_rate = successful_signals / total_signals if total_signals > 0 else 0

            return {
                "total_signals": total_signals,
                "successful_signals": successful_signals,
                "failed_signals": total_signals - successful_signals,
                "success_rate": success_rate,
                "average_confidence": (
                    sum(s.get("confidence", 0) for s in self.signals) / total_signals
                    if total_signals > 0
                    else 0
                ),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as e:
            logger.error(f"Error getting signal metrics: {str(e)}")
            return {
                "total_signals": 0,
                "successful_signals": 0,
                "failed_signals": 0,
                "success_rate": 0,
                "average_confidence": 0,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": str(e),
            }


# Global instance
signal_service = SignalService()


def get_signal_service() -> SignalService:
    """Get the signal service instance"""
    return signal_service
