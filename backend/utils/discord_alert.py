"""
Discord Alert Module
===================

Sends Discord notifications when strategies are promoted or important events occur.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any

import requests

logger = logging.getLogger(__name__)

# Default webhook URL - replace with your actual Discord webhook
DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/your_webhook_url_here"


class DiscordNotifier:
    """Handles Discord notifications for the AI mutation system"""

    def __init__(self, webhook_url: str | None = None):
        self.webhook_url = webhook_url or DISCORD_WEBHOOK_URL
        self.enabled = webhook_url != "your_webhook_url_here"

    def send_strategy_promoted_alert(
        self, strategy_name: str, backtest_results: dict[str, Any]
    ) -> bool:
        """
        Send enhanced alert when a strategy is promoted with rich formatting

        Args:
            strategy_name: Name of the promoted strategy
            backtest_results: Results from backtest

        Returns:
            True if alert was sent successfully, False otherwise
        """
        if not self.enabled:
            logger.warning("Discord notifications disabled - no webhook URL configured")
            return False

        win_rate = backtest_results.get("win_rate", 0)
        total_profit = backtest_results.get("total_profit", 0)
        num_trades = backtest_results.get("num_trades", 0)
        sharpe_ratio = backtest_results.get("sharpe_ratio", 0)
        max_drawdown = backtest_results.get("max_drawdown", 0)
        strategy_type = backtest_results.get("strategy_type", "unknown")
        symbol = backtest_results.get("symbol", "BTCUSDT")

        # Determine color based on performance
        if total_profit > 10 and win_rate > 0.6:
            color = 0x00FF00  # Green for excellent performance
        elif total_profit > 5 and win_rate > 0.55:
            color = 0x00FF88  # Light green for good performance
        elif total_profit > 0:
            color = 0xFFFF00  # Yellow for positive performance
        else:
            color = 0xFF8800  # Orange for minimal performance

        # Create performance emoji
        if total_profit > 10:
            performance_emoji = "ðŸš€"
        elif total_profit > 5:
            performance_emoji = "ðŸ“ˆ"
        elif total_profit > 0:
            performance_emoji = "ðŸ“Š"
        else:
            performance_emoji = "âš ï¸"

        embed = {
            "title": f"{performance_emoji} New AI Strategy Promoted!",
            "description": (
                f"**{strategy_name}** has been automatically promoted to live trading by the AI mutation system."
            ),
            "color": color,
            "fields": [
                {
                    "name": "ðŸ’° Total Profit",
                    "value": f"`${total_profit:.2f}`",
                    "inline": True,
                },
                {
                    "name": "ðŸ“Š Win Rate",
                    "value": f"`{win_rate:.1%}`",
                    "inline": True,
                },
                {
                    "name": "ðŸ“ˆ Total Trades",
                    "value": f"`{num_trades}`",
                    "inline": True,
                },
                {
                    "name": "âš¡ Sharpe Ratio",
                    "value": f"`{sharpe_ratio:.2f}`",
                    "inline": True,
                },
                {
                    "name": "ðŸ“‰ Max Drawdown",
                    "value": f"`{max_drawdown:.1%}`",
                    "inline": True,
                },
                {
                    "name": "ðŸŽ¯ Strategy Type",
                    "value": f"`{strategy_type}`",
                    "inline": True,
                },
                {
                    "name": "ðŸ”§ Trading Pair",
                    "value": f"`{symbol}`",
                    "inline": True,
                },
                {
                    "name": "ðŸ¤– AI Engine",
                    "value": "`Mystic AI Mutator v2.0`",
                    "inline": True,
                },
                {
                    "name": "â° Promotion Time",
                    "value": f"<t:{int(datetime.now().timestamp())}:F>",
                    "inline": True,
                },
            ],
            "timestamp": datetime.now().isoformat(),
            "footer": {
                "text": "Mystic Trading Platform - AI Evolution System",
                "icon_url": ("https://img.icons8.com/color/96/000000/crystal-ball.png"),
            },
            "thumbnail": {
                "url": ("https://img.icons8.com/color/96/000000/artificial-intelligence.png")
            },
        }

        payload = {
            "embeds": [embed],
            "username": "Mystic AI Mutator",
            "avatar_url": ("https://img.icons8.com/color/96/000000/artificial-intelligence.png"),
        }

        return self._send_discord_message(payload)

    def send_mutation_cycle_alert(self, cycle_results: dict[str, Any]) -> bool:
        """
        Send enhanced mutation cycle alert with detailed statistics

        Args:
            cycle_results: Results from mutation cycle

        Returns:
            True if alert was sent successfully, False otherwise
        """
        if not self.enabled:
            return False

        total_mutations = cycle_results.get("total_mutations", 0)
        promoted_mutations = cycle_results.get("promoted_mutations", 0)
        mutations = cycle_results.get("mutations", [])
        errors = cycle_results.get("errors", [])

        # Only send if there were mutations
        if total_mutations == 0:
            return False

        # Calculate statistics
        promotion_rate = promoted_mutations / total_mutations if total_mutations > 0 else 0

        # Calculate average profit from mutations
        profits = [
            m.get("backtest_results", {}).get("total_profit", 0)
            for m in mutations
            if m.get("backtest_results")
        ]
        avg_profit = sum(profits) / len(profits) if profits else 0

        # Determine color based on performance
        if promotion_rate > 0.3:  # 30% promotion rate
            color = 0x00FF00  # Green
            cycle_emoji = "ðŸŽ‰"
        elif promotion_rate > 0.1:  # 10% promotion rate
            color = 0x00FF88  # Light green
            cycle_emoji = "ðŸ“ˆ"
        elif promotion_rate > 0:  # Any promotions
            color = 0xFFFF00  # Yellow
            cycle_emoji = "ðŸ“Š"
        else:
            color = 0xFF8800  # Orange
            cycle_emoji = "âš ï¸"

        # Get best performing mutation
        best_mutation = None
        if mutations:
            best_mutation = max(
                mutations,
                key=lambda x: x.get("backtest_results", {}).get("total_profit", 0),
            )

        embed = {
            "title": f"{cycle_emoji} AI Mutation Cycle Complete",
            "description": (
                f"AI mutation cycle completed with **{total_mutations}** strategies tested and **{promoted_mutations}** promoted."
            ),
            "color": color,
            "fields": [
                {
                    "name": "ðŸ§¬ Total Mutations",
                    "value": f"`{total_mutations}`",
                    "inline": True,
                },
                {
                    "name": "âœ… Promoted Strategies",
                    "value": f"`{promoted_mutations}`",
                    "inline": True,
                },
                {
                    "name": "ðŸ“ˆ Promotion Rate",
                    "value": f"`{promotion_rate:.1%}`",
                    "inline": True,
                },
                {
                    "name": "ðŸ’° Average Profit",
                    "value": f"`${avg_profit:.2f}`",
                    "inline": True,
                },
                {
                    "name": "ðŸ¤– AI Generations",
                    "value": (
                        f"`{len([m for m in mutations if m.get('source') == 'ai_generation'])}`"
                    ),
                    "inline": True,
                },
                {
                    "name": "ðŸ”„ Mutations",
                    "value": (f"`{len([m for m in mutations if m.get('source') == 'mutation'])}`"),
                    "inline": True,
                },
            ],
            "timestamp": datetime.now().isoformat(),
            "footer": {
                "text": "Mystic Trading Platform - AI Evolution System",
                "icon_url": ("https://img.icons8.com/color/96/000000/crystal-ball.png"),
            },
        }

        # Add best performing mutation if available
        if best_mutation and best_mutation.get("backtest_results"):
            best_results = best_mutation["backtest_results"]
            embed["fields"].append(
                {
                    "name": "ðŸ† Best Performer",
                    "value": (
                        f"**{best_mutation.get('strategy_name', 'Unknown')}**\nProfit: `${best_results.get('total_profit', 0):.2f}` | Win Rate: {best_results.get('win_rate', 0):.1%}"
                    ),
                    "inline": False,
                }
            )

        # Add error summary if any
        if errors:
            error_summary = f"**{len(errors)} errors** occurred during the cycle."
            embed["fields"].append({"name": "âš ï¸ Errors", "value": error_summary, "inline": False})

        payload = {
            "embeds": [embed],
            "username": "Mystic AI Mutator",
            "avatar_url": ("https://img.icons8.com/color/96/000000/artificial-intelligence.png"),
        }

        return self._send_discord_message(payload)

    def send_system_startup_alert(self, system_info: dict[str, Any]) -> bool:
        """
        Send system startup alert

        Args:
            system_info: System information

        Returns:
            True if alert was sent successfully, False otherwise
        """
        if not self.enabled:
            return False

        embed = {
            "title": "ðŸš€ Mystic AI Trading System Started",
            "description": (
                "The AI mutation system has been successfully initialized and is now running."
            ),
            "color": 0x00FF00,  # Green
            "fields": [
                {
                    "name": "ðŸ¤– AI Engine",
                    "value": "`Mystic AI Mutator v2.0`",
                    "inline": True,
                },
                {
                    "name": "ðŸ“Š Live Data",
                    "value": "`Binance US`",
                    "inline": True,
                },
                {
                    "name": "ðŸ”„ Mutation Interval",
                    "value": f"`{system_info.get('cycle_interval', 300)}s`",
                    "inline": True,
                },
                {
                    "name": "ðŸŽ¯ Risk Level",
                    "value": f"`{system_info.get('risk_level', 'MODERATE')}`",
                    "inline": True,
                },
                {
                    "name": "ðŸ“ˆ AI Generation",
                    "value": (
                        "`Enabled`"
                        if system_info.get("enable_ai_generation", True)
                        else "`Disabled`"
                    ),
                    "inline": True,
                },
                {
                    "name": "ðŸ”§ Auto Trading",
                    "value": "`Connected`",
                    "inline": True,
                },
            ],
            "timestamp": datetime.now().isoformat(),
            "footer": {
                "text": "Mystic Trading Platform - AI Evolution System",
                "icon_url": ("https://img.icons8.com/color/96/000000/crystal-ball.png"),
            },
            "thumbnail": {"url": "https://img.icons8.com/color/96/000000/rocket.png"},
        }

        payload = {
            "embeds": [embed],
            "username": "Mystic AI Mutator",
            "avatar_url": ("https://img.icons8.com/color/96/000000/artificial-intelligence.png"),
        }

        return self._send_discord_message(payload)

    def send_performance_summary_alert(self, performance_data: dict[str, Any]) -> bool:
        """
        Send periodic performance summary alert

        Args:
            performance_data: Performance summary data

        Returns:
            True if alert was sent successfully, False otherwise
        """
        if not self.enabled:
            return False

        total_mutations = performance_data.get("total_mutations", 0)
        promoted_mutations = performance_data.get("promoted_mutations", 0)
        success_rate = performance_data.get("success_rate", 0)
        average_profit = performance_data.get("average_profit", 0)
        best_strategy = performance_data.get("best_strategy", {})

        # Determine color based on success rate
        if success_rate > 0.3:
            color = 0x00FF00  # Green
            summary_emoji = "ðŸŽ‰"
        elif success_rate > 0.1:
            color = 0x00FF88  # Light green
            summary_emoji = "ðŸ“ˆ"
        else:
            color = 0xFFFF00  # Yellow
            summary_emoji = "ðŸ“Š"

        embed = {
            "title": f"{summary_emoji} AI System Performance Summary",
            "description": ("Periodic performance summary of the AI mutation system."),
            "color": color,
            "fields": [
                {
                    "name": "ðŸ§¬ Total Mutations",
                    "value": f"`{total_mutations}`",
                    "inline": True,
                },
                {
                    "name": "âœ… Promoted Strategies",
                    "value": f"`{promoted_mutations}`",
                    "inline": True,
                },
                {
                    "name": "ðŸ“ˆ Success Rate",
                    "value": f"`{success_rate:.1%}`",
                    "inline": True,
                },
                {
                    "name": "ðŸ’° Average Profit",
                    "value": f"`${average_profit:.2f}`",
                    "inline": True,
                },
            ],
            "timestamp": datetime.now().isoformat(),
            "footer": {
                "text": "Mystic Trading Platform - AI Evolution System",
                "icon_url": ("https://img.icons8.com/color/96/000000/crystal-ball.png"),
            },
        }

        # Add best strategy if available
        if best_strategy and best_strategy.get("name"):
            embed["fields"].append(
                {
                    "name": "ðŸ† Best Strategy",
                    "value": (
                        f"**{best_strategy['name']}**\nProfit: `${best_strategy.get('profit', 0):.2f}` | Win Rate: {best_strategy.get('win_rate', 0):.1%}"
                    ),
                    "inline": False,
                }
            )

        payload = {
            "embeds": [embed],
            "username": "Mystic AI Mutator",
            "avatar_url": ("https://img.icons8.com/color/96/000000/artificial-intelligence.png"),
        }

        return self._send_discord_message(payload)

    def send_error_alert(self, error_message: str, context: str = "AI Mutation System") -> bool:
        """
        Send alert for errors in the mutation system

        Args:
            error_message: The error message
            context: Context where the error occurred

        Returns:
            True if alert was sent successfully, False otherwise
        """
        if not self.enabled:
            return False

        embed = {
            "title": "âš ï¸ AI Mutation Error",
            "description": f"An error occurred in the **{context}**",
            "color": 0xFF0000,  # Red
            "fields": [
                {
                    "name": "ðŸš¨ Error",
                    "value": f"```{error_message[:1000]}```",  # Limit length
                    "inline": False,
                }
            ],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "footer": {"text": "Mystic Trading Platform - Error Alert"},
        }

        payload = {"embeds": [embed]}

        return self._send_discord_message(payload)

    def send_system_status_alert(self, status: str, details: dict[str, Any]) -> bool:
        """
        Send system status alert

        Args:
            status: Status message (e.g., "Online", "Offline", "Maintenance")
            details: Additional status details

        Returns:
            True if alert was sent successfully, False otherwise
        """
        if not self.enabled:
            return False

        # Determine color based on status
        status_colors = {
            "online": 0x00FF00,
            "offline": 0xFF0000,
            "maintenance": 0xFFFF00,
            "warning": 0xFFA500,
        }

        color = status_colors.get(status.lower(), 0x808080)

        embed = {
            "title": f"ðŸ”§ System Status: {status.upper()}",
            "description": "Mystic AI Mutation System Status Update",
            "color": color,
            "fields": [
                {
                    "name": "ðŸ“Š Active Strategies",
                    "value": f"`{details.get('active_strategies', 0)}`",
                    "inline": True,
                },
                {
                    "name": "ðŸ§¬ Total Mutations",
                    "value": f"`{details.get('total_mutations', 0)}`",
                    "inline": True,
                },
                {
                    "name": "âœ… Success Rate",
                    "value": f"`{details.get('success_rate', 0):.1%}`",
                    "inline": True,
                },
            ],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "footer": {"text": "Mystic Trading Platform - System Status"},
        }

        payload = {"embeds": [embed]}

        return self._send_discord_message(payload)

    def _send_discord_message(self, payload: dict[str, Any]) -> bool:
        """
        Send message to Discord webhook

        Args:
            payload: Discord webhook payload

        Returns:
            True if successful, False otherwise
        """
        try:
            response = requests.post(
                self.webhook_url,
                data=json.dumps(payload),
                headers={"Content-Type": "application/json"},
                timeout=10,
            )

            if response.status_code == 204:
                logger.debug("Discord message sent successfully")
                return True
            else:
                logger.error(f"Discord webhook failed: {response.status_code} - {response.text}")
                return False

        except Exception as e:
            logger.error(f"Failed to send Discord message: {e}")
            return False


# Global instance
discord_notifier = DiscordNotifier()


# Legacy functions for backward compatibility
def send_discord_alert(strategy_name: str, results: dict[str, Any]) -> bool:
    """Simple Discord alert function for backward compatibility"""
    return discord_notifier.send_strategy_promoted_alert(strategy_name, results)

