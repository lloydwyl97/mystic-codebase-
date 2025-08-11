import json
import sqlite3
from datetime import datetime


def export_trade_history(output_file="trade_history.json"):
    conn = sqlite3.connect("simulation_trades.db")
    cursor = conn.execute("""
        SELECT id, symbol, side, amount, price, timestamp, strategy, portfolio_id, status
        FROM simulated_trades
        ORDER BY timestamp DESC
    """)
    rows = [dict(zip([col[0] for col in cursor.description], row)) for row in cursor.fetchall()]
    with open(output_file, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"[Export] Trade history exported to {output_file}")


def export_performance_report(output_file="performance_report.json"):
    from ai_self_rating import get_ai_health_report
    from daily_summary import get_performance_metrics

    metrics = get_performance_metrics()
    health = get_ai_health_report()

    report = {
        "export_timestamp": datetime.timezone.utcnow().isoformat(),
        "performance_metrics": metrics,
        "ai_health": health,
        "summary": {
            "total_trades": metrics["summary"]["total_trades"],
            "total_profit": metrics["summary"]["total_profit"],
            "avg_profit": metrics["summary"]["avg_profit"],
            "ai_score": health["rating"]["ai_score"],
            "ai_rating": health["rating"]["rating"],
        },
    }

    with open(output_file, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[Export] Performance report exported to {output_file}")


def export_csv_trades(output_file="trades.csv"):
    conn = sqlite3.connect("simulation_trades.db")
    cursor = conn.execute("""
        SELECT id, symbol, side, amount, price, timestamp, strategy, portfolio_id, status
        FROM simulated_trades
        ORDER BY timestamp DESC
    """)

    with open(output_file, "w") as f:
        # Write header
        columns = [col[0] for col in cursor.description]
        f.write(",".join(columns) + "\n")

        # Write data
        for row in cursor.fetchall():
            f.write(",".join(str(val) for val in row) + "\n")

    print(f"[Export] CSV trades exported to {output_file}")
