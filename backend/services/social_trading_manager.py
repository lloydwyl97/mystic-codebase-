"""
Social Trading Manager Service
Handles social trading features and community signals
"""

import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


class SocialTradingManager:
    def __init__(self):
        self.active_traders = {}
        self.community_signals = []
        self.performance_leaderboard = []
        logger.info("âœ… SocialTradingManager initialized")

    async def get_social_status(self) -> dict[str, Any]:
        """Get status of social trading features"""
        return {
            "active_traders": len(self.active_traders),
            "community_signals": len(self.community_signals),
            "status": "operational",
            "timestamp": datetime.now(timezone.timezone.utc).isoformat(),
        }

    async def add_trader(self, trader_id: str, name: str, strategy: str) -> dict[str, Any]:
        """Add a new trader to the social trading network"""
        try:
            self.active_traders[trader_id] = {
                "name": name,
                "strategy": strategy,
                "performance": {
                    "win_rate": 0.0,
                    "total_trades": 0,
                    "pnl": 0.0,
                },
                "followers": 0,
                "joined_at": datetime.now(timezone.timezone.utc).isoformat(),
                "status": "active",
            }
            logger.info(f"âœ… Added trader: {name}")
            return {"success": True, "trader_id": trader_id}
        except Exception as e:
            logger.error(f"âŒ Error adding trader: {e}")
            return {"success": False, "error": str(e)}

    async def get_leaderboard(self) -> list[dict[str, Any]]:
        """Get performance leaderboard"""
        try:
            leaderboard = []
            for trader_id, trader in self.active_traders.items():
                leaderboard.append(
                    {
                        "trader_id": trader_id,
                        "name": trader["name"],
                        "win_rate": trader["performance"]["win_rate"],
                        "total_trades": trader["performance"]["total_trades"],
                        "pnl": trader["performance"]["pnl"],
                        "followers": trader["followers"],
                    }
                )

            # Sort by PnL
            leaderboard.sort(key=lambda x: x["pnl"], reverse=True)
            return leaderboard[:10]  # Top 10
        except Exception as e:
            logger.error(f"âŒ Error getting leaderboard: {e}")
            return []

    async def add_community_signal(
        self, trader_id: str, pair: str, signal_type: str, confidence: float
    ) -> dict[str, Any]:
        """Add a community trading signal"""
        try:
            signal = {
                "trader_id": trader_id,
                "trader_name": (self.active_traders.get(trader_id, {}).get("name", "Unknown")),
                "pair": pair,
                "signal_type": signal_type,  # buy, sell, hold
                "confidence": confidence,
                "timestamp": datetime.now(timezone.timezone.utc).isoformat(),
                "votes": 0,
            }
            self.community_signals.append(signal)
            logger.info(f"âœ… Added community signal from {signal['trader_name']}")
            return {"success": True, "signal_id": len(self.community_signals)}
        except Exception as e:
            logger.error(f"âŒ Error adding community signal: {e}")
            return {"success": False, "error": str(e)}

    async def get_recent_signals(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get recent community signals"""
        try:
            return self.community_signals[-limit:] if self.community_signals else []
        except Exception as e:
            logger.error(f"âŒ Error getting recent signals: {e}")
            return []

    async def update_trader_performance(
        self, trader_id: str, trade_result: dict[str, Any]
    ) -> dict[str, Any]:
        """Update trader performance after a trade"""
        try:
            if trader_id in self.active_traders:
                trader = self.active_traders[trader_id]
                trader["performance"]["total_trades"] += 1

                if trade_result.get("success"):
                    trader["performance"]["pnl"] += trade_result.get("pnl", 0)
                    # Update win rate
                    wins = int(
                        trader["performance"]["win_rate"] * trader["performance"]["total_trades"]
                    )
                    if trade_result.get("pnl", 0) > 0:
                        wins += 1
                    trader["performance"]["win_rate"] = wins / trader["performance"]["total_trades"]

                logger.info(f"âœ… Updated performance for trader {trader_id}")
                return {"success": True}
            return {"success": False, "error": "Trader not found"}
        except Exception as e:
            logger.error(f"âŒ Error updating trader performance: {e}")
            return {"success": False, "error": str(e)}


# Global instance
social_trading_manager = SocialTradingManager()


