from notifier import send_alert
from simulation_logger import SimulationLogger
from trade_memory import TradeMemory


def send_daily_summary():
    logger = SimulationLogger()
    memory = TradeMemory()

    summary = logger.get_summary()
    best = memory.get_best_trade()
    worst = memory.get_worst_trade()
    stats = memory.get_trade_stats()

    message = (
        f"ðŸ“Š DAILY AI SUMMARY\n"
        f"Total Trades: {summary['total_trades']}\n"
        f"Avg Profit: ${summary['avg_profit']:.4f}\n"
        f"Total Profit: ${summary['total_profit']:.4f}\n"
        f"Win Rate: {(stats.get('winning_trades', 0) / max(summary['total_trades'], 1) * 100):.1f}%\n\n"
        f"ðŸ”¥ Best Trade: {best.get('symbol', 'N/A')} | ${best.get('simulated_profit', 0):.2f} @ {best.get('timestamp', 'N/A')}\n"
        f"ðŸ’€ Worst Trade: {worst.get('symbol', 'N/A')} | ${worst.get('simulated_profit', 0):.2f} @ {worst.get('timestamp', 'N/A')}"
    )

    send_alert(message)
    print("[DailySummary] Sent.")


def get_performance_metrics():
    logger = SimulationLogger()
    memory = TradeMemory()

    summary = logger.get_summary()
    stats = memory.get_trade_stats()

    return {
        "summary": summary,
        "stats": stats,
        "best_trade": memory.get_best_trade(),
        "worst_trade": memory.get_worst_trade(),
    }


