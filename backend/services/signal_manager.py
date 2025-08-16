"""
Signal Manager Service
Handles trading signal processing and management
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class SignalManager:
    def __init__(self):
        self.signals: List[Dict[str, Any]] = []
        self.signal_processors: Dict[str, Any] = {}
        self.is_running = False

    async def process_signal(self, signal_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a trading signal"""
        try:
            signal_id = f"signal_{len(self.signals) + 1}"
            signal = {
                "id": signal_id,
                "symbol": signal_data.get("symbol"),
                "type": signal_data.get("type", "unknown"),
                "action": signal_data.get("action", "hold"),
                "confidence": signal_data.get("confidence", 0.0),
                "strength": signal_data.get("strength", 0.0),
                "timestamp": datetime.now(timezone.timezone.utc).isoformat(),
                "source": signal_data.get("source", "unknown"),
                "data": signal_data.get("data", {}),
                "processed": False,
            }

            # Process the signal
            processed_signal = await self._apply_signal_logic(signal)
            processed_signal["processed"] = True

            self.signals.append(processed_signal)
            logger.info(f"Signal processed: {signal_id}")
            return {
                "status": "success",
                "signal_id": signal_id,
                "signal": processed_signal,
            }

        except Exception as e:
            logger.error(f"Error processing signal: {e}")
            return {"status": "error", "message": str(e)}

    async def _apply_signal_logic(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """Apply signal processing logic"""
        try:
            # Simple signal strength calculation
            confidence = signal.get("confidence", 0.0)
            strength = signal.get("strength", 0.0)

            # Calculate composite score
            composite_score = (confidence + strength) / 2.0

            # Determine action based on score
            if composite_score > 0.7:
                signal["action"] = "strong_buy"
            elif composite_score > 0.5:
                signal["action"] = "buy"
            elif composite_score < -0.7:
                signal["action"] = "strong_sell"
            elif composite_score < -0.5:
                signal["action"] = "sell"
            else:
                signal["action"] = "hold"

            signal["composite_score"] = composite_score
            return signal

        except Exception as e:
            logger.error(f"Error applying signal logic: {e}")
            return signal

    async def get_signals(
        self, symbol: Optional[str] = None, action: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get signals with optional filtering"""
        filtered_signals = self.signals

        if symbol:
            filtered_signals = [s for s in filtered_signals if s.get("symbol") == symbol]

        if action:
            filtered_signals = [s for s in filtered_signals if s.get("action") == action]

        return filtered_signals

    async def get_latest_signals(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get latest signals"""
        return sorted(self.signals, key=lambda x: x.get("timestamp", ""), reverse=True)[:limit]

    async def get_signal_stats(self) -> Dict[str, Any]:
        """Get signal statistics"""
        try:
            total_signals = len(self.signals)
            processed_signals = len([s for s in self.signals if s.get("processed", False)])

            action_counts = {}
            for signal in self.signals:
                action = signal.get("action", "unknown")
                action_counts[action] = action_counts.get(action, 0) + 1

            return {
                "total_signals": total_signals,
                "processed_signals": processed_signals,
                "action_distribution": action_counts,
                "processing_rate": (processed_signals / total_signals if total_signals > 0 else 0),
            }
        except Exception as e:
            logger.error(f"Error getting signal stats: {e}")
            return {}


# Global instance
signal_manager = SignalManager()


