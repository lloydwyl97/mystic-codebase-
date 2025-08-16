import sqlite3
from datetime import datetime

import matplotlib.pyplot as plt


def plot_trades(symbol="ETHUSDT", db_path="simulation_trades.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.execute(
        """
        SELECT timestamp, price, simulated_profit FROM simulated_trades
        WHERE symbol = ? ORDER BY timestamp
    """,
        (symbol,),
    )

    rows = cursor.fetchall()
    if not rows:
        print("[Chart] No trades to plot.")
        return

    times = [datetime.fromisoformat(row[0]) for row in rows]
    prices = [row[1] for row in rows]
    profits = [row[2] for row in rows]

    colors = ["green" if p > 0 else "red" for p in profits]

    plt.figure(figsize=(12, 6))
    plt.scatter(times, prices, c=colors, label=symbol, alpha=0.7)
    plt.plot(times, prices, alpha=0.2)
    plt.title(f"Trade Chart for {symbol}")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("trade_chart.png")
    print("[Chart] Chart saved to trade_chart.png")


def plot_performance_over_time(db_path="simulation_trades.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.execute(
        """
        SELECT timestamp, simulated_profit FROM simulated_trades
        ORDER BY timestamp
    """
    )

    rows = cursor.fetchall()
    if not rows:
        print("[Chart] No performance data to plot.")
        return

    times = [datetime.fromisoformat(row[0]) for row in rows]
    profits = [row[1] for row in rows]
    cumulative = []
    total = 0
    for profit in profits:
        total += profit
        cumulative.append(total)

    plt.figure(figsize=(12, 6))
    plt.plot(times, cumulative, label="Cumulative Profit")
    plt.title("AI Trading Performance Over Time")
    plt.xlabel("Time")
    plt.ylabel("Cumulative Profit ($)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("performance_chart.png")
    print("[Chart] Performance chart saved to performance_chart.png")


