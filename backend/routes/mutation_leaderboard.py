import sqlite3

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter()


@router.get("/mutation-leaderboard", response_class=HTMLResponse)
def leaderboard():
    conn = sqlite3.connect("simulation_trades.db")
    cursor = conn.cursor()
    cursor.execute(
        """
    SELECT file, simulated_profit, win_rate, num_trades, timestamp, promoted
    FROM strategy_mutations
    WHERE promoted = 1
    ORDER BY simulated_profit DESC
    LIMIT 25
    """
    )
    rows = cursor.fetchall()
    conn.close()

    html = "<h2>ðŸ† Promoted Strategy Leaderboard</h2><table border='1' cellpadding='5'>"
    html += "<tr><th>Rank</th><th>File</th><th>Profit</th><th>Win Rate</th><th>Trades</th><th>Promoted At</th><th>Status</th></tr>"
    for i, row in enumerate(rows, 1):
        status = "âœ… Live" if i == 1 else "—"
        html += f"<tr><td>{i}</td><td>{row[0]}</td><td>{row[1]}</td><td>{row[2]}</td><td>{row[3]}</td><td>{row[4]}</td><td>{status}</td></tr>"
    html += "</table>"
    return html


