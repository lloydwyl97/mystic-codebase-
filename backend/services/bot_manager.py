"""
Bot Management Service

Handles bot control, monitoring, and auto-buy functionality.
"""

import logging
from datetime import timezone, datetime
from typing import Any, Dict

from config import AUTO_BUY_CONFIG, BOT_MONITORING

logger = logging.getLogger(__name__)


class BotManager:
    """Manages trading bot operations and monitoring."""

    def __init__(self):
        self.auto_buy_config = AUTO_BUY_CONFIG.copy()
        self.bot_monitoring = BOT_MONITORING.copy()

    def get_bot_status(self) -> Dict[str, Any]:
        """Get current bot status"""
        return {
            "bot_status": self.bot_monitoring,
            "auto_buy_config": self.auto_buy_config,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def start_bot(self) -> Dict[str, Any]:
        """Start the trading bot"""
        self.bot_monitoring["is_running"] = True
        self.auto_buy_config["bot_status"] = "running"
        self.bot_monitoring["logs"].append(
            f"[{datetime.now(timezone.utc).isoformat()}] Bot started"
        )

        return {
            "success": True,
            "message": "Bot started successfully",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def stop_bot(self) -> Dict[str, Any]:
        """Stop the trading bot"""
        self.bot_monitoring["is_running"] = False
        self.auto_buy_config["bot_status"] = "stopped"
        self.bot_monitoring["logs"].append(
            f"[{datetime.now(timezone.utc).isoformat()}] Bot stopped"
        )

        return {
            "success": True,
            "message": "Bot stopped successfully",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def configure_auto_buy(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Configure auto-buy settings"""
        try:
            required_fields = [
                "enabled",
                "max_investment",
                "stop_loss",
                "take_profit",
            ]
            for field in required_fields:
                if field not in config:
                    raise ValueError(f"Missing required field: {field}")

            self.auto_buy_config.update(config)

            return {
                "success": True,
                "message": "Auto-buy configuration updated",
                "config": self.auto_buy_config,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as e:
            logger.error(f"Error configuring auto-buy: {e}")
            raise

    def get_auto_buy_config(self) -> Dict[str, Any]:
        """Get current auto-buy configuration"""
        return self.auto_buy_config

    def get_bot_logs(self, limit: int = 100) -> Dict[str, Any]:
        """Get bot logs"""
        try:
            logs = self.bot_monitoring["logs"]
            if limit > 0:
                logs = logs[-limit:]

            return {
                "logs": logs,
                "total_logs": len(self.bot_monitoring["logs"]),
                "returned_logs": len(logs),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as e:
            logger.error(f"Error getting bot logs: {e}")
            raise

    def execute_trade(
        self,
        symbol: str,
        action: str,
        amount: float = 1000,
        strategy: str = "default",
    ) -> Dict[str, Any]:
        """Execute a trade"""
        try:
            # Simulate trade execution
            trade_result = {
                "symbol": symbol,
                "action": action,
                "amount": amount,
                "strategy": strategy,
                "status": "executed",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            # Update bot monitoring
            self.bot_monitoring["total_trades"] += 1
            self.bot_monitoring["last_trade"] = trade_result

            return trade_result
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            raise


# Global instance
bot_manager = BotManager()


