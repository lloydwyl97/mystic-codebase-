"""
Chart Profit - Profit Tracker Chart

Renders profit charts from trade logs
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt

logger = logging.getLogger("chart_profit")


def plot_profits(output_file: str = "profit_chart.png") -> bool:
    """Plot profit chart from trade data"""
    try:
        # Load trade data
        trades = load_trade_data()
        if not trades:
            logger.warning("No trade data available for plotting")
            return False

        # Prepare data for plotting
        dates = []
        values = []
        cumulative_pnl = []
        current_pnl = 0

        for trade in trades:
            try:
                # Parse timestamp
                if isinstance(trade["timestamp"], str):
                    date = datetime.fromisoformat(trade["timestamp"].replace("Z", "+00:00"))
                else:
                    date = trade["timestamp"]

                # Calculate value
                if trade["type"] == "BUY":
                    value = float(trade["amount"]) * float(trade["price"])
                    current_pnl -= value
                elif trade["type"] == "SELL":
                    value = float(trade["amount"]) * float(trade["price"])
                    current_pnl += value

                dates.append(date)
                values.append(value)
                cumulative_pnl.append(current_pnl)

            except Exception as e:
                logger.error(f"Error processing trade: {e}")
                continue

        if not dates:
            logger.warning("No valid trade data for plotting")
            return False

        # Create the plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Plot 1: Individual trade values
        ax1.plot(dates, values, "o-", alpha=0.7, linewidth=1)
        ax1.set_title("Individual Trade Values", fontsize=14, fontweight="bold")
        ax1.set_ylabel("Trade Value (USD)", fontsize=12)
        ax1.grid(True, alpha=0.3)

        # Format x-axis
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax1.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

        # Plot 2: Cumulative P&L
        ax2.plot(dates, cumulative_pnl, "g-", linewidth=2, label="Cumulative P&L")
        ax2.axhline(y=0, color="r", linestyle="--", alpha=0.5, label="Break-even")
        ax2.set_title("Cumulative Profit & Loss", fontsize=14, fontweight="bold")
        ax2.set_ylabel("P&L (USD)", fontsize=12)
        ax2.set_xlabel("Date", fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # Format x-axis
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax2.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"âœ… Profit chart saved to {output_file}")
        return True

    except Exception as e:
        logger.error(f"âŒ Error plotting profits: {e}")
        return False


def load_trade_data() -> list[dict[str, Any]]:
    """Load trade data from files"""
    try:
        trades = []

        # Load from active trades
        if os.path.exists("active_trades.json"):
            with open("active_trades.json") as f:
                active_trades = json.load(f)
                for symbol, trade in active_trades.items():
                    trades.append(
                        {
                            "symbol": symbol,
                            "type": "BUY",
                            "amount": trade["amount"],
                            "price": trade["buy_price"],
                            "timestamp": trade["timestamp"],
                        }
                    )

        # Load from trade history
        if os.path.exists("trade_history.json"):
            with open("trade_history.json") as f:
                history = json.load(f)
                trades.extend(history)

        # Sort by timestamp
        trades.sort(key=lambda x: x["timestamp"])
        return trades

    except Exception as e:
        logger.error(f"âŒ Error loading trade data: {e}")
        return []


def plot_market_performance(
    output_file: str = "market_performance.png",
) -> bool:
    """Plot market performance comparison"""
    try:
        from backend.ai.poller import get_cache

        cache = get_cache()
        if not cache.coingecko:
            logger.warning("No market data available")
            return False

        # Prepare data
        symbols = []
        prices = []
        changes = []

        for coin_id, data in cache.coingecko.items():
            if data.get("price") and data.get("price_change_24h") is not None:
                symbols.append(data["symbol"])
                prices.append(data["price"])
                changes.append(data["price_change_24h"])

        if not symbols:
            logger.warning("No valid market data for plotting")
            return False

        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot 1: Price distribution
        ax1.hist(prices, bins=20, alpha=0.7, color="skyblue", edgecolor="black")
        ax1.set_title("Price Distribution", fontsize=14, fontweight="bold")
        ax1.set_xlabel("Price (USD)", fontsize=12)
        ax1.set_ylabel("Number of Coins", fontsize=12)
        ax1.grid(True, alpha=0.3)

        # Plot 2: 24h Change distribution
        ax2.hist(changes, bins=20, alpha=0.7, color="lightgreen", edgecolor="black")
        ax2.set_title("24h Price Change Distribution", fontsize=14, fontweight="bold")
        ax2.set_xlabel("24h Change (%)", fontsize=12)
        ax2.set_ylabel("Number of Coins", fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.axvline(x=0, color="red", linestyle="--", alpha=0.7)

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"âœ… Market performance chart saved to {output_file}")
        return True

    except Exception as e:
        logger.error(f"âŒ Error plotting market performance: {e}")
        return False


def generate_trading_report() -> dict[str, Any]:
    """Generate comprehensive trading report"""
    try:
        from backend.ai.trade_tracker import (
            get_active_trades,
            get_trade_history,
            get_trade_summary,
        )

        summary = get_trade_summary()
        active_trades = get_active_trades()
        history = get_trade_history(100)

        # Calculate additional metrics
        total_trades = len(history)
        buy_trades = len([t for t in history if t["type"] == "BUY"])
        sell_trades = len([t for t in history if t["type"] == "SELL"])

        # Calculate average trade size
        trade_sizes = [
            float(t["amount"]) * float(t["price"]) for t in history if t["type"] == "BUY"
        ]
        avg_trade_size = sum(trade_sizes) / len(trade_sizes) if trade_sizes else 0

        report = {
            "summary": summary,
            "active_trades_count": len(active_trades),
            "total_trades": total_trades,
            "buy_trades": buy_trades,
            "sell_trades": sell_trades,
            "avg_trade_size": round(avg_trade_size, 2),
            "completion_rate": (round(sell_trades / buy_trades * 100, 2) if buy_trades > 0 else 0),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

        return report

    except Exception as e:
        logger.error(f"âŒ Error generating trading report: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    # Generate charts when run directly
    plot_profits()
    plot_market_performance()
    print("Charts generated successfully!")

