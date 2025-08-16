# alerts.py
import requests
import os
import logging
from dotenv import load_dotenv
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")


def send_discord_alert(message: str, embed_data: Optional[Dict[str, Any]] = None) -> bool:
    """
    Send a Discord alert via webhook

    Args:
        message: Text message to send
        embed_data: Optional embed data for rich formatting

    Returns:
        bool: True if successful, False otherwise
    """
    if not DISCORD_WEBHOOK_URL:
        logger.warning("Discord webhook URL not configured")
        return False

    try:
        payload = {"content": message}

        if embed_data:
            embed = {
                "title": embed_data.get("title", "Trading Alert"),
                "description": embed_data.get("description", ""),
                "color": embed_data.get("color", 0x00FF00),  # Green by default
                "fields": embed_data.get("fields", []),
                "timestamp": embed_data.get("timestamp", ""),
                "footer": embed_data.get("footer", {"text": "Mystic AI Trading System"}),
            }
            payload["embeds"] = [embed]

        response = requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=10)
        response.raise_for_status()

        logger.info(f"Discord alert sent successfully: {message[:50]}...")
        return True

    except Exception as e:
        logger.error(f"Discord alert failed: {e}")
        return False


def alert_strategy_mutation(mutation_info: Dict[str, Any]) -> bool:
    """
    Send alert for new strategy mutation

    Args:
        mutation_info: Information about the mutation

    Returns:
        bool: True if successful
    """
    message = "ðŸ§¬ **New Strategy Mutation Created!**\n"
    message += f"**Name:** {mutation_info['name']}\n"
    message += f"**Parent:** {mutation_info['parent_strategy']}\n"
    message += f"**Parent Win Rate:** {mutation_info['parent_win_rate']:.1%}\n"
    message += f"**Parent Avg Profit:** {mutation_info['parent_avg_profit']:.2f}"

    embed_data = {
        "title": "Strategy Evolution",
        "description": (f"New mutation created from {mutation_info['parent_strategy']}"),
        "color": 0x00FF00,  # Green
        "fields": [
            {
                "name": "Mutation Name",
                "value": mutation_info["name"],
                "inline": True,
            },
            {
                "name": "Parent Strategy",
                "value": mutation_info["parent_strategy"],
                "inline": True,
            },
            {
                "name": "Parent Performance",
                "value": (
                    f"Win Rate: {mutation_info['parent_win_rate']:.1%}\nAvg Profit: {mutation_info['parent_avg_profit']:.2f}"
                ),
                "inline": False,
            },
        ],
    }

    return send_discord_alert(message, embed_data)


def alert_strategy_deactivation(strategy_info: Dict[str, Any]) -> bool:
    """
    Send alert for strategy deactivation

    Args:
        strategy_info: Information about the deactivated strategy

    Returns:
        bool: True if successful
    """
    message = "ðŸ’€ **Strategy Deactivated**\n"
    message += f"**Name:** {strategy_info['name']}\n"
    message += f"**Win Rate:** {strategy_info['win_rate']:.1%}\n"
    message += f"**Avg Profit:** {strategy_info['avg_profit']:.2f}\n"
    message += f"**Reason:** {strategy_info['reason']}"

    embed_data = {
        "title": "Strategy Deactivation",
        "description": (f"Strategy {strategy_info['name']} has been deactivated"),
        "color": 0xFF0000,  # Red
        "fields": [
            {
                "name": "Strategy Name",
                "value": strategy_info["name"],
                "inline": True,
            },
            {
                "name": "Performance",
                "value": (
                    f"Win Rate: {strategy_info['win_rate']:.1%}\nAvg Profit: {strategy_info['avg_profit']:.2f}"
                ),
                "inline": True,
            },
            {
                "name": "Deactivation Reason",
                "value": strategy_info["reason"],
                "inline": False,
            },
        ],
    }

    return send_discord_alert(message, embed_data)


def alert_trade_execution(trade_info: Dict[str, Any]) -> bool:
    """
    Send alert for trade execution

    Args:
        trade_info: Information about the trade

    Returns:
        bool: True if successful
    """
    success_emoji = "âœ…" if trade_info.get("success", False) else "âŒ"
    profit_emoji = "ðŸ“ˆ" if trade_info.get("profit", 0) > 0 else "ðŸ“‰"

    message = f"{success_emoji} **Trade Executed**\n"
    message += f"**Coin:** {trade_info['coin']}\n"
    message += f"**Strategy:** {trade_info['strategy_name']}\n"
    message += f"**Entry:** {trade_info['entry_price']:.2f}\n"
    message += f"**Exit:** {trade_info['exit_price']:.2f}\n"
    message += f"**Profit:** {profit_emoji} {trade_info['profit']:.2f}"

    color = 0x00FF00 if trade_info.get("success", False) else 0xFF0000

    embed_data = {
        "title": "Trade Execution",
        "description": f"Trade completed for {trade_info['coin']}",
        "color": color,
        "fields": [
            {
                "name": "Trading Pair",
                "value": trade_info["coin"],
                "inline": True,
            },
            {
                "name": "Strategy",
                "value": trade_info["strategy_name"],
                "inline": True,
            },
            {
                "name": "Entry Price",
                "value": f"{trade_info['entry_price']:.2f}",
                "inline": True,
            },
            {
                "name": "Exit Price",
                "value": f"{trade_info['exit_price']:.2f}",
                "inline": True,
            },
            {
                "name": "Profit/Loss",
                "value": f"{trade_info['profit']:.2f}",
                "inline": True,
            },
            {
                "name": "Success",
                "value": ("âœ… Yes" if trade_info.get("success", False) else "âŒ No"),
                "inline": True,
            },
        ],
    }

    return send_discord_alert(message, embed_data)


def alert_daily_summary(summary_data: Dict[str, Any]) -> bool:
    """
    Send daily trading summary

    Args:
        summary_data: Daily summary data

    Returns:
        bool: True if successful
    """
    message = "ðŸ“Š **Daily Trading Summary**\n"
    message += f"**Total Trades:** {summary_data.get('total_trades', 0)}\n"
    message += f"**Win Rate:** {summary_data.get('win_rate', 0):.1%}\n"
    message += f"**Total Profit:** {summary_data.get('total_profit', 0):.2f}\n"
    message += f"**Active Strategies:** {summary_data.get('active_strategies', 0)}"

    embed_data = {
        "title": "Daily Trading Summary",
        "description": "End of day trading performance report",
        "color": 0x0099FF,  # Blue
        "fields": [
            {
                "name": "Total Trades",
                "value": str(summary_data.get("total_trades", 0)),
                "inline": True,
            },
            {
                "name": "Win Rate",
                "value": f"{summary_data.get('win_rate', 0):.1%}",
                "inline": True,
            },
            {
                "name": "Total Profit",
                "value": f"{summary_data.get('total_profit', 0):.2f}",
                "inline": True,
            },
            {
                "name": "Active Strategies",
                "value": str(summary_data.get("active_strategies", 0)),
                "inline": True,
            },
            {
                "name": "Top Performer",
                "value": summary_data.get("top_performer", "N/A"),
                "inline": True,
            },
            {
                "name": "Worst Performer",
                "value": summary_data.get("worst_performer", "N/A"),
                "inline": True,
            },
        ],
    }

    return send_discord_alert(message, embed_data)


def alert_evolution_cycle(evolution_data: Dict[str, Any]) -> bool:
    """
    Send alert for evolution cycle completion

    Args:
        evolution_data: Evolution cycle results

    Returns:
        bool: True if successful
    """
    message = "ðŸ§¬ **Evolution Cycle Completed**\n"
    message += f"**New Strategies:** {evolution_data.get('total_new_strategies', 0)}\n"
    message += f"**Mutations:** {evolution_data.get('mutations_created', 0)}\n"
    message += f"**Crossovers:** {evolution_data.get('crossovers_created', 0)}\n"
    message += f"**Random:** {evolution_data.get('random_strategies_created', 0)}\n"
    message += f"**Active Population:** {evolution_data.get('population_stats', {}).get('active_strategies', 0)}"

    embed_data = {
        "title": "Strategy Evolution Cycle",
        "description": "New strategies have been created through evolution",
        "color": 0x9932CC,  # Purple
        "fields": [
            {
                "name": "New Strategies Created",
                "value": str(evolution_data.get("total_new_strategies", 0)),
                "inline": True,
            },
            {
                "name": "Mutations",
                "value": str(evolution_data.get("mutations_created", 0)),
                "inline": True,
            },
            {
                "name": "Crossovers",
                "value": str(evolution_data.get("crossovers_created", 0)),
                "inline": True,
            },
            {
                "name": "Random Strategies",
                "value": str(evolution_data.get("random_strategies_created", 0)),
                "inline": True,
            },
            {
                "name": "Active Population",
                "value": str(
                    evolution_data.get("population_stats", {}).get("active_strategies", 0)
                ),
                "inline": True,
            },
            {
                "name": "Deactivated",
                "value": str(len(evolution_data.get("deactivated_strategies", []))),
                "inline": True,
            },
        ],
    }

    return send_discord_alert(message, embed_data)


def alert_system_health(health_data: Dict[str, Any]) -> bool:
    """
    Send system health alert

    Args:
        health_data: System health information

    Returns:
        bool: True if successful
    """
    status = health_data.get("status", "unknown")
    status_emoji = "âœ…" if status == "healthy" else "âš ï¸" if status == "warning" else "âŒ"

    message = f"{status_emoji} **System Health Check**\n"
    message += f"**Status:** {status.title()}\n"
    message += f"**Database:** {health_data.get('database', 'unknown')}\n"
    message += f"**API Connections:** {health_data.get('api_connections', 'unknown')}\n"
    message += f"**Active Bots:** {health_data.get('active_bots', 0)}"

    color_map = {
        "healthy": 0x00FF00,  # Green
        "warning": 0xFFFF00,  # Yellow
        "error": 0xFF0000,  # Red
    }

    embed_data = {
        "title": "System Health Status",
        "description": f"Current system status: {status.title()}",
        "color": color_map.get(status, 0x808080),  # Gray for unknown
        "fields": [
            {"name": "Status", "value": status.title(), "inline": True},
            {
                "name": "Database",
                "value": health_data.get("database", "Unknown"),
                "inline": True,
            },
            {
                "name": "API Connections",
                "value": health_data.get("api_connections", "Unknown"),
                "inline": True,
            },
            {
                "name": "Active Bots",
                "value": str(health_data.get("active_bots", 0)),
                "inline": True,
            },
            {
                "name": "Last Update",
                "value": health_data.get("last_update", "Unknown"),
                "inline": True,
            },
        ],
    }

    return send_discord_alert(message, embed_data)


def test_discord_connection() -> bool:
    """
    Test Discord webhook connection

    Returns:
        bool: True if successful
    """
    test_message = "ðŸ§ª **Discord Integration Test**\nThis is a test message from your Mystic AI Trading System."

    success = send_discord_alert(test_message)
    if success:
        logger.info("Discord connection test successful")
    else:
        logger.error("Discord connection test failed")

    return success


