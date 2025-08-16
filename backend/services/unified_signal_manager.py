"""
Unified Signal Manager Service
Handles trading signals from multiple sources
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class UnifiedSignalManager:
    def __init__(self):
        self.signals: List[Dict[str, Any]] = []
        self.signal_sources = ["technical", "sentiment", "ai", "social"]
        self.is_running = False

    async def add_signal(self, signal_data: Dict[str, Any]) -> Dict[str, Any]:
        """Add a new trading signal"""
        try:
            signal_id = f"signal_{len(self.signals) + 1}"
            signal = {
                "id": signal_id,
                "symbol": signal_data.get("symbol"),
                "type": signal_data.get("type", "unknown"),
                "strength": signal_data.get("strength", 0),
                "direction": signal_data.get("direction", "neutral"),
                "source": signal_data.get("source", "unknown"),
                "timestamp": datetime.now(timezone.timezone.utc).isoformat(),
                "data": signal_data.get("data", {}),
            }

            self.signals.append(signal)
            logger.info(f"Signal added: {signal_id}")
            return {
                "status": "success",
                "signal_id": signal_id,
                "signal": signal,
            }

        except Exception as e:
            logger.error(f"Error adding signal: {e}")
            return {"status": "error", "message": str(e)}

    async def get_signals(
        self, symbol: Optional[str] = None, source: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get signals with optional filtering"""
        filtered_signals = self.signals

        if symbol:
            filtered_signals = [s for s in filtered_signals if s.get("symbol") == symbol]

        if source:
            filtered_signals = [s for s in filtered_signals if s.get("source") == source]

        return filtered_signals

    async def get_latest_signals(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get latest signals"""
        return sorted(self.signals, key=lambda x: x.get("timestamp", ""), reverse=True)[:limit]

    async def clear_old_signals(self, hours: int = 24) -> int:
        """Clear signals older than specified hours"""
        try:
            cutoff_time = datetime.now(timezone.timezone.utc).timestamp() - (hours * 3600)
            original_count = len(self.signals)

            self.signals = [
                signal
                for signal in self.signals
                if datetime.fromisoformat(signal.get("timestamp", "")).timestamp() > cutoff_time
            ]

            cleared_count = original_count - len(self.signals)
            logger.info(f"Cleared {cleared_count} old signals")
            return cleared_count

        except Exception as e:
            logger.error(f"Error clearing old signals: {e}")
            return 0


# Global instance
unified_signal_manager = UnifiedSignalManager()


