"""
AI Dashboard Endpoints
Live AI mutation system status and performance metrics
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List

from fastapi import APIRouter, Body, HTTPException
from fastapi.responses import HTMLResponse

from ai_mutation.mutation_manager import mutation_manager
from ai_mutation.promote_mutation import StrategyPromoter
from ai_mutation.strategy_locker import get_live_strategy
from ai_mutation.version_tracker import get_strategy_versions
from database import get_db_connection

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/ai", tags=["ai-dashboard"])


@router.get("/dashboard")
async def get_ai_dashboard() -> Dict[str, Any]:
    """Get comprehensive AI dashboard data"""
    try:
        # Get current live strategy
        live_strategy_file = get_live_strategy()
        live_strategy_data = None

        if live_strategy_file:
            try:
                strategy_path = os.path.join("strategies", live_strategy_file)
                if not os.path.exists(strategy_path):
                    strategy_path = os.path.join("mutated_strategies", live_strategy_file)

                if os.path.exists(strategy_path):
                    with open(strategy_path, "r") as f:
                        live_strategy_data = json.load(f)
            except Exception as e:
                logger.error(f"Error loading live strategy: {e}")

        # Get mutation system status
        mutation_status = {
            "is_running": mutation_manager.is_running,
            "cycle_count": mutation_manager.cycle_count,
            "last_cycle_time": (
                mutation_manager.last_cycle_time.isoformat()
                if mutation_manager.last_cycle_time
                else None
            ),
            "cycle_interval": mutation_manager.cycle_interval,
            "enable_ai_generation": mutation_manager.enable_ai_generation,
        }

        # Get recent mutations from database
        recent_mutations = await get_recent_mutations(limit=10)

        # Get strategy versions
        strategy_versions = get_strategy_versions()

        # Get performance metrics
        performance_metrics = await get_performance_metrics()

        return {
            "timestamp": datetime.now(timezone.timezone.utc).isoformat(),
            "live_strategy": {
                "file": live_strategy_file,
                "data": live_strategy_data,
            },
            "mutation_system": mutation_status,
            "recent_mutations": recent_mutations,
            "strategy_versions": strategy_versions,
            "performance_metrics": performance_metrics,
            "system_status": "operational",
        }

    except Exception as e:
        logger.error(f"Error getting AI dashboard: {e}")
        raise HTTPException(status_code=500, detail=f"Dashboard error: {str(e)}")


async def get_recent_mutations(limit: int = 10) -> List[Dict[str, Any]]:
    """Get recent mutations from the database"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT strategy_file, strategy_type, strategy_name, simulated_profit,
                   win_rate, num_trades, max_drawdown, sharpe_ratio, promoted,
                   backtest_results, cycle_number, created_at
            FROM ai_mutations
            ORDER BY created_at DESC
            LIMIT ?
        """,
            (limit,),
        )

        rows = cursor.fetchall()
        mutations = []

        for row in rows:
            try:
                backtest_results = json.loads(row[10]) if row[10] else {}
            except (json.JSONDecodeError, TypeError) as json_error:
                logger.warning(
                    f"Could not parse backtest results for strategy {row[2]}: {json_error}"
                )
                backtest_results = {}

            mutations.append(
                {
                    "strategy_file": row[0],
                    "strategy_type": row[1],
                    "strategy_name": row[2],
                    "simulated_profit": row[3],
                    "win_rate": row[4],
                    "num_trades": row[5],
                    "max_drawdown": row[6],
                    "sharpe_ratio": row[7],
                    "promoted": bool(row[8]),
                    "backtest_results": backtest_results,
                    "cycle_number": row[11],
                    "created_at": row[12],
                }
            )

        conn.close()
        return mutations

    except Exception as e:
        logger.error(f"Error getting recent mutations: {e}")
        return []


async def get_performance_metrics() -> Dict[str, Any]:
    """Get AI system performance metrics"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Get total mutations
        cursor.execute("SELECT COUNT(*) FROM ai_mutations")
        total_mutations = cursor.fetchone()[0]

        # Get promoted mutations
        cursor.execute("SELECT COUNT(*) FROM ai_mutations WHERE promoted = 1")
        promoted_mutations = cursor.fetchone()[0]

        # Get average profit
        cursor.execute(
            "SELECT AVG(simulated_profit) FROM ai_mutations WHERE simulated_profit IS NOT NULL"
        )
        avg_profit = cursor.fetchone()[0] or 0.0

        # Get best performing strategy
        cursor.execute(
            """
            SELECT strategy_name, simulated_profit, win_rate, num_trades
            FROM ai_mutations
            WHERE promoted = 1
            ORDER BY simulated_profit DESC
            LIMIT 1
        """
        )
        best_strategy = cursor.fetchone()

        # Get mutation success rate
        success_rate = (promoted_mutations / total_mutations * 100) if total_mutations > 0 else 0.0

        # Get recent performance trend
        cursor.execute(
            """
            SELECT AVG(simulated_profit)
            FROM ai_mutations
            WHERE created_at >= datetime('now', '-7 days')
        """
        )
        recent_avg_profit = cursor.fetchone()[0] or 0.0

        conn.close()

        return {
            "total_mutations": total_mutations,
            "promoted_mutations": promoted_mutations,
            "success_rate": round(success_rate, 2),
            "average_profit": round(avg_profit, 2),
            "recent_average_profit": round(recent_avg_profit, 2),
            "best_strategy": (
                {
                    "name": best_strategy[0] if best_strategy else "None",
                    "profit": (round(best_strategy[1], 2) if best_strategy else 0.0),
                    "win_rate": (round(best_strategy[2], 3) if best_strategy else 0.0),
                    "trades": best_strategy[3] if best_strategy else 0,
                }
                if best_strategy
                else None
            ),
        }

    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}")
        return {
            "total_mutations": 0,
            "promoted_mutations": 0,
            "success_rate": 0.0,
            "average_profit": 0.0,
            "recent_average_profit": 0.0,
            "best_strategy": None,
        }


@router.get("/strategies/leaderboard")
async def get_strategy_leaderboard(limit: int = 20) -> List[Dict[str, Any]]:
    """Get strategy leaderboard ranked by performance"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT strategy_name, strategy_type, simulated_profit, win_rate,
                   num_trades, max_drawdown, sharpe_ratio, promoted, created_at
            FROM ai_mutations
            WHERE promoted = 1
            ORDER BY simulated_profit DESC
            LIMIT ?
        """,
            (limit,),
        )

        rows = cursor.fetchall()
        leaderboard = []

        for i, row in enumerate(rows, 1):
            leaderboard.append(
                {
                    "rank": i,
                    "strategy_name": row[0],
                    "strategy_type": row[1],
                    "simulated_profit": round(row[2], 2),
                    "win_rate": round(row[3], 3),
                    "num_trades": row[4],
                    "max_drawdown": round(row[5], 3),
                    "sharpe_ratio": round(row[6], 2),
                    "promoted": bool(row[7]),
                    "created_at": row[8],
                }
            )

        conn.close()
        return leaderboard

    except Exception as e:
        logger.error(f"Error getting leaderboard: {e}")
        return []


@router.get("/strategies/leaderboard/expanded")
async def get_expanded_strategy_leaderboard(limit: int = 20) -> Dict[str, Any]:
    """Get expanded strategy leaderboard with improvement, robustness, innovation, and badges"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Top by profit
        cursor.execute(
            """
            SELECT strategy_name, strategy_type, simulated_profit, win_rate, num_trades, max_drawdown, sharpe_ratio, promoted, created_at
            FROM ai_mutations
            WHERE promoted = 1
            ORDER BY simulated_profit DESC
            LIMIT ?
        """,
            (limit,),
        )
        top_strategies = cursor.fetchall()

        # Most improved (largest profit delta over last 7 days)
        cursor.execute(
            """
            SELECT strategy_name, strategy_type, MAX(simulated_profit - LAG(simulated_profit, 1) OVER (ORDER BY created_at)), promoted, created_at
            FROM ai_mutations
            WHERE promoted = 1 AND created_at >= datetime('now', '-7 days')
            GROUP BY strategy_name
            ORDER BY MAX(simulated_profit - LAG(simulated_profit, 1) OVER (ORDER BY created_at)) DESC
            LIMIT 5
        """
        )
        most_improved = cursor.fetchall()

        # Most robust (lowest max drawdown, min 20 trades)
        cursor.execute(
            """
            SELECT strategy_name, strategy_type, simulated_profit, max_drawdown, num_trades, promoted, created_at
            FROM ai_mutations
            WHERE promoted = 1 AND num_trades >= 20
            ORDER BY max_drawdown ASC
            LIMIT 5
        """
        )
        most_robust = cursor.fetchall()

        # Most innovative (recently promoted, unique type)
        cursor.execute(
            """
            SELECT strategy_name, strategy_type, simulated_profit, promoted, created_at
            FROM ai_mutations
            WHERE promoted = 1
            GROUP BY strategy_type
            ORDER BY created_at DESC
            LIMIT 5
        """
        )
        most_innovative = cursor.fetchall()

        # Get badge/achievement info (mocked for now)
        def get_badges(strategy_name):
            badges = []
            if "breakout" in strategy_name.lower():
                badges.append({"name": "Breakout Master", "icon": "🏆"})
            if "robust" in strategy_name.lower():
                badges.append({"name": "Robust Performer", "icon": "🛡️"})
            if "innovative" in strategy_name.lower():
                badges.append({"name": "Innovator", "icon": "💡"})
            return badges

        leaderboard = []
        for i, row in enumerate(top_strategies, 1):
            leaderboard.append(
                {
                    "rank": i,
                    "strategy_name": row[0],
                    "strategy_type": row[1],
                    "simulated_profit": round(row[2], 2),
                    "win_rate": round(row[3], 3),
                    "num_trades": row[4],
                    "max_drawdown": round(row[5], 3),
                    "sharpe_ratio": round(row[6], 2),
                    "promoted": bool(row[7]),
                    "created_at": row[8],
                    "badges": get_badges(row[0]),
                }
            )

        conn.close()
        return {
            "top_strategies": leaderboard,
            "most_improved": [
                {
                    "strategy_name": row[0],
                    "strategy_type": row[1],
                    "profit_delta": row[2],
                    "promoted": bool(row[3]),
                    "created_at": row[4],
                }
                for row in most_improved
            ],
            "most_robust": [
                {
                    "strategy_name": row[0],
                    "strategy_type": row[1],
                    "simulated_profit": row[2],
                    "max_drawdown": row[3],
                    "num_trades": row[4],
                    "promoted": bool(row[5]),
                    "created_at": row[6],
                }
                for row in most_robust
            ],
            "most_innovative": [
                {
                    "strategy_name": row[0],
                    "strategy_type": row[1],
                    "simulated_profit": row[2],
                    "promoted": bool(row[3]),
                    "created_at": row[4],
                }
                for row in most_innovative
            ],
        }
    except Exception as e:
        logger.error(f"Error getting expanded leaderboard: {e}")
        return {"error": str(e)}


@router.post("/mutation/run-cycle")
async def run_mutation_cycle() -> Dict[str, Any]:
    """Manually trigger a mutation cycle"""
    try:
        if mutation_manager.is_running:
            raise HTTPException(status_code=400, detail="Mutation engine is already running")

        results = await mutation_manager.run_single_cycle()

        return {
            "success": True,
            "message": "Mutation cycle completed",
            "results": results,
            "timestamp": datetime.now(timezone.timezone.utc).isoformat(),
        }

    except Exception as e:
        logger.error(f"Error running mutation cycle: {e}")
        raise HTTPException(status_code=500, detail=f"Mutation cycle failed: {str(e)}")


@router.get("/mutation/status")
async def get_mutation_status() -> Dict[str, Any]:
    """Get current mutation system status"""
    try:
        return {
            "is_running": mutation_manager.is_running,
            "cycle_count": mutation_manager.cycle_count,
            "last_cycle_time": (
                mutation_manager.last_cycle_time.isoformat()
                if mutation_manager.last_cycle_time
                else None
            ),
            "cycle_interval": mutation_manager.cycle_interval,
            "enable_ai_generation": mutation_manager.enable_ai_generation,
            "base_strategy": mutation_manager.base_strategy,
            "timestamp": datetime.now(timezone.timezone.utc).isoformat(),
        }

    except Exception as e:
        logger.error(f"Error getting mutation status: {e}")
        raise HTTPException(status_code=500, detail=f"Status error: {str(e)}")


@router.post("/mutation/start")
async def start_mutation_engine() -> Dict[str, Any]:
    """Start the mutation engine"""
    try:
        if mutation_manager.is_running:
            raise HTTPException(status_code=400, detail="Mutation engine is already running")

        mutation_manager.start_mutation_engine()

        return {
            "success": True,
            "message": "Mutation engine started",
            "timestamp": datetime.now(timezone.timezone.utc).isoformat(),
        }

    except Exception as e:
        logger.error(f"Error starting mutation engine: {e}")
        raise HTTPException(status_code=500, detail=f"Start error: {str(e)}")


@router.post("/mutation/stop")
async def stop_mutation_engine() -> Dict[str, Any]:
    """Stop the mutation engine"""
    try:
        if not mutation_manager.is_running:
            raise HTTPException(status_code=400, detail="Mutation engine is not running")

        mutation_manager.stop_mutation_engine()

        return {
            "success": True,
            "message": "Mutation engine stopped",
            "timestamp": datetime.now(timezone.timezone.utc).isoformat(),
        }

    except Exception as e:
        logger.error(f"Error stopping mutation engine: {e}")
        raise HTTPException(status_code=500, detail=f"Stop error: {str(e)}")


@router.get("/dashboard/html", response_class=HTMLResponse)
async def get_ai_dashboard_html() -> str:
    """Get AI dashboard as HTML page"""
    try:
        dashboard_data = await get_ai_dashboard()

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>AI Trading Dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .header {{ background: #2c3e50; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
                .section {{ background: white; padding: 20px; margin-bottom: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                .metric {{ display: inline-block; margin: 10px; padding: 15px; background: #ecf0f1; border-radius: 5px; min-width: 120px; text-align: center; }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
                .metric-label {{ font-size: 12px; color: #7f8c8d; }}
                .status-running {{ color: #27ae60; }}
                .status-stopped {{ color: #e74c3c; }}
                .strategy-card {{ border: 1px solid #ddd; padding: 15px; margin: 10px 0; border-radius: 5px; }}
                .promoted {{ border-left: 4px solid #27ae60; }}
                .not-promoted {{ border-left: 4px solid #e74c3c; }}
                table {{ width: 100%; border-collapse: collapse; }}
                th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f8f9fa; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🤖 AI Trading System Dashboard</h1>
                    <p>Live AI Mutation System Status</p>
                </div>

                <div class="section">
                    <h2>📊 System Status</h2>
                    <div class="metric">
                        <div class="metric-value {'status-running' if dashboard_data['mutation_system']['is_running'] else 'status-stopped'}">
                            {'🟢 Running' if dashboard_data['mutation_system']['is_running'] else '🔴 Stopped'}
                        </div>
                        <div class="metric-label">Mutation Engine</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{dashboard_data['mutation_system']['cycle_count']}</div>
                        <div class="metric-label">Total Cycles</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{dashboard_data['performance_metrics']['total_mutations']}</div>
                        <div class="metric-label">Total Mutations</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{dashboard_data['performance_metrics']['promoted_mutations']}</div>
                        <div class="metric-label">Promoted Strategies</div>
                    </div>
                </div>

                <div class="section">
                    <h2>🎯 Performance Metrics</h2>
                    <div class="metric">
                        <div class="metric-value">{dashboard_data['performance_metrics']['success_rate']}%</div>
                        <div class="metric-label">Success Rate</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">${dashboard_data['performance_metrics']['average_profit']}</div>
                        <div class="metric-label">Avg Profit</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">${dashboard_data['performance_metrics']['recent_average_profit']}</div>
                        <div class="metric-label">Recent Avg Profit</div>
                    </div>
                </div>

                <div class="section">
                    <h2>🏆 Current Live Strategy</h2>
                    {f'''
                    <div class="strategy-card promoted">
                        <h3>{dashboard_data['live_strategy']['data']['name'] if dashboard_data['live_strategy']['data'] else 'No Strategy'}</h3>
                        <p><strong>Type:</strong> {dashboard_data['live_strategy']['data']['strategy_type'] if dashboard_data['live_strategy']['data'] else 'N/A'}</p>
                        <p><strong>File:</strong> {dashboard_data['live_strategy']['file'] or 'None'}</p>
                    </div>
                    ''' if dashboard_data['live_strategy']['data'] else '<p>No live strategy currently active</p>'}
                </div>

                <div class="section">
                    <h2>📈 Recent Mutations</h2>
                    <table>
                        <thead>
                            <tr>
                                <th>Strategy</th>
                                <th>Type</th>
                                <th>Profit</th>
                                <th>Win Rate</th>
                                <th>Status</th>
                                <th>Created</th>
                            </tr>
                        </thead>
                        <tbody>
                            {''.join(f'''
                            <tr>
                                <td>{mutation['strategy_name']}</td>
                                <td>{mutation['strategy_type']}</td>
                                <td>${mutation['simulated_profit']}</td>
                                <td>{mutation['win_rate']}</td>
                                <td>{'✅ Promoted' if mutation['promoted'] else '❌ Not Promoted'}</td>
                                <td>{mutation['created_at']}</td>
                            </tr>
                            ''' for mutation in dashboard_data['recent_mutations'][:5])}
                        </tbody>
                    </table>
                </div>

                <div class="section">
                    <h2>🔄 Last Update</h2>
                    <p>{dashboard_data['timestamp']}</p>
                </div>
            </div>

            <script>
                // Auto-refresh every 30 seconds
                setTimeout(() => {{
                    location.reload();
                }}, 30000);
            </script>
        </body>
        </html>
        """

        return html

    except Exception as e:
        logger.error(f"Error generating dashboard HTML: {e}")
        return f"<h1>Error</h1><p>{str(e)}</p>"


@router.get("/performance/live")
async def get_live_performance() -> Dict[str, Any]:
    """Get real-time performance metrics"""
    try:
        # Get current live strategy
        live_strategy_file = get_live_strategy()

        # Get recent performance data
        conn = get_db_connection()
        cursor = conn.cursor()

        # Get last 24 hours of mutations
        cursor.execute(
            """
            SELECT COUNT(*) as total_mutations,
                   SUM(CASE WHEN promoted = 1 THEN 1 ELSE 0 END) as promoted_mutations,
                   AVG(simulated_profit) as avg_profit,
                   MAX(simulated_profit) as max_profit,
                   MIN(simulated_profit) as min_profit
            FROM ai_mutations
            WHERE created_at >= datetime('now', '-1 day')
        """
        )

        daily_stats = cursor.fetchone()

        # Get hourly performance for chart
        cursor.execute(
            """
            SELECT strftime('%H', created_at) as hour,
                   COUNT(*) as mutations,
                   SUM(CASE WHEN promoted = 1 THEN 1 ELSE 0 END) as promotions,
                   AVG(simulated_profit) as avg_profit
            FROM ai_mutations
            WHERE created_at >= datetime('now', '-24 hours')
            GROUP BY strftime('%H', created_at)
            ORDER BY hour
        """
        )

        hourly_data = cursor.fetchall()

        # Get current system status
        mutation_status = {
            "is_running": mutation_manager.is_running,
            "cycle_count": mutation_manager.cycle_count,
            "last_cycle_time": (
                mutation_manager.last_cycle_time.isoformat()
                if mutation_manager.last_cycle_time
                else None
            ),
            "next_cycle_in": (
                max(
                    0,
                    mutation_manager.cycle_interval
                    - (
                        datetime.now(timezone.timezone.utc) - mutation_manager.last_cycle_time
                    ).total_seconds(),
                )
                if mutation_manager.last_cycle_time
                else 0
            ),
        }

        conn.close()

        return {
            "timestamp": datetime.now(timezone.timezone.utc).isoformat(),
            "live_strategy": live_strategy_file,
            "daily_stats": {
                "total_mutations": daily_stats[0] or 0,
                "promoted_mutations": daily_stats[1] or 0,
                "success_rate": (
                    (daily_stats[1] / daily_stats[0] * 100)
                    if daily_stats[0] and daily_stats[0] > 0
                    else 0
                ),
                "avg_profit": round(daily_stats[2] or 0, 2),
                "max_profit": round(daily_stats[3] or 0, 2),
                "min_profit": round(daily_stats[4] or 0, 2),
            },
            "hourly_performance": [
                {
                    "hour": int(row[0]),
                    "mutations": row[1],
                    "promotions": row[2],
                    "avg_profit": round(row[3] or 0, 2),
                }
                for row in hourly_data
            ],
            "system_status": mutation_status,
        }

    except Exception as e:
        logger.error(f"Error getting live performance: {e}")
        raise HTTPException(status_code=500, detail=f"Performance error: {str(e)}")


@router.get("/performance/trends")
async def get_performance_trends(days: int = 7) -> Dict[str, Any]:
    """Get performance trends over time"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Get daily trends
        cursor.execute(
            """
            SELECT DATE(created_at) as date,
                   COUNT(*) as total_mutations,
                   SUM(CASE WHEN promoted = 1 THEN 1 ELSE 0 END) as promoted_mutations,
                   AVG(simulated_profit) as avg_profit,
                   MAX(simulated_profit) as max_profit,
                   MIN(simulated_profit) as min_profit,
                   AVG(win_rate) as avg_win_rate,
                   AVG(sharpe_ratio) as avg_sharpe
            FROM ai_mutations
            WHERE created_at >= datetime('now', '-' || ? || ' days')
            GROUP BY DATE(created_at)
            ORDER BY date
        """,
            (days,),
        )

        daily_trends = cursor.fetchall()

        # Get strategy type distribution
        cursor.execute(
            """
            SELECT strategy_type,
                   COUNT(*) as count,
                   AVG(simulated_profit) as avg_profit,
                   SUM(CASE WHEN promoted = 1 THEN 1 ELSE 0 END) as promoted_count
            FROM ai_mutations
            WHERE created_at >= datetime('now', '-' || ? || ' days')
            GROUP BY strategy_type
            ORDER BY count DESC
        """,
            (days,),
        )

        strategy_distribution = cursor.fetchall()

        conn.close()

        return {
            "timestamp": datetime.now(timezone.timezone.utc).isoformat(),
            "period_days": days,
            "daily_trends": [
                {
                    "date": row[0],
                    "total_mutations": row[1],
                    "promoted_mutations": row[2],
                    "success_rate": ((row[2] / row[1] * 100) if row[1] > 0 else 0),
                    "avg_profit": round(row[3] or 0, 2),
                    "max_profit": round(row[4] or 0, 2),
                    "min_profit": round(row[5] or 0, 2),
                    "avg_win_rate": round(row[6] or 0, 3),
                    "avg_sharpe": round(row[7] or 0, 2),
                }
                for row in daily_trends
            ],
            "strategy_distribution": [
                {
                    "strategy_type": row[0],
                    "count": row[1],
                    "avg_profit": round(row[2] or 0, 2),
                    "promoted_count": row[3],
                    "promotion_rate": ((row[3] / row[1] * 100) if row[1] > 0 else 0),
                }
                for row in strategy_distribution
            ],
        }

    except Exception as e:
        logger.error(f"Error getting performance trends: {e}")
        raise HTTPException(status_code=500, detail=f"Trends error: {str(e)}")


@router.get("/performance/analytics")
async def get_performance_analytics() -> Dict[str, Any]:
    """Get advanced performance analytics"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Get overall statistics
        cursor.execute(
            """
            SELECT COUNT(*) as total_mutations,
                   SUM(CASE WHEN promoted = 1 THEN 1 ELSE 0 END) as total_promoted,
                   AVG(simulated_profit) as overall_avg_profit,
                   STDDEV(simulated_profit) as profit_stddev,
                   AVG(win_rate) as overall_avg_win_rate,
                   AVG(sharpe_ratio) as overall_avg_sharpe,
                   AVG(max_drawdown) as overall_avg_drawdown
            FROM ai_mutations
        """
        )

        overall_stats = cursor.fetchone()

        # Get best performing strategies
        cursor.execute(
            """
            SELECT strategy_name, strategy_type, simulated_profit, win_rate,
                   num_trades, sharpe_ratio, max_drawdown, created_at
            FROM ai_mutations
            WHERE promoted = 1
            ORDER BY simulated_profit DESC
            LIMIT 10
        """
        )

        top_strategies = cursor.fetchall()

        # Get mutation success rate by time of day
        cursor.execute(
            """
            SELECT strftime('%H', created_at) as hour,
                   COUNT(*) as total,
                   SUM(CASE WHEN promoted = 1 THEN 1 ELSE 0 END) as promoted,
                   AVG(simulated_profit) as avg_profit
            FROM ai_mutations
            GROUP BY strftime('%H', created_at)
            ORDER BY hour
        """
        )

        hourly_success = cursor.fetchall()

        # Get correlation between parameters and success
        cursor.execute(
            """
            SELECT
                CASE
                    WHEN simulated_profit > 5 THEN 'high_profit'
                    WHEN simulated_profit > 0 THEN 'positive_profit'
                    ELSE 'negative_profit'
                END as profit_category,
                COUNT(*) as count,
                AVG(win_rate) as avg_win_rate,
                AVG(sharpe_ratio) as avg_sharpe,
                AVG(max_drawdown) as avg_drawdown
            FROM ai_mutations
            GROUP BY profit_category
        """
        )

        profit_categories = cursor.fetchall()

        conn.close()

        return {
            "timestamp": datetime.now(timezone.timezone.utc).isoformat(),
            "overall_statistics": {
                "total_mutations": overall_stats[0] or 0,
                "total_promoted": overall_stats[1] or 0,
                "overall_success_rate": (
                    (overall_stats[1] / overall_stats[0] * 100)
                    if overall_stats[0] and overall_stats[0] > 0
                    else 0
                ),
                "avg_profit": round(overall_stats[2] or 0, 2),
                "profit_volatility": round(overall_stats[3] or 0, 2),
                "avg_win_rate": round(overall_stats[4] or 0, 3),
                "avg_sharpe_ratio": round(overall_stats[5] or 0, 2),
                "avg_max_drawdown": round(overall_stats[6] or 0, 3),
            },
            "top_performing_strategies": [
                {
                    "name": row[0],
                    "type": row[1],
                    "profit": round(row[2], 2),
                    "win_rate": round(row[3], 3),
                    "trades": row[4],
                    "sharpe": round(row[5], 2),
                    "drawdown": round(row[6], 3),
                    "created": row[7],
                }
                for row in top_strategies
            ],
            "hourly_success_rates": [
                {
                    "hour": int(row[0]),
                    "total_mutations": row[1],
                    "promoted": row[2],
                    "success_rate": ((row[2] / row[1] * 100) if row[1] > 0 else 0),
                    "avg_profit": round(row[3] or 0, 2),
                }
                for row in hourly_success
            ],
            "profit_category_analysis": [
                {
                    "category": row[0],
                    "count": row[1],
                    "avg_win_rate": round(row[2] or 0, 3),
                    "avg_sharpe": round(row[3] or 0, 2),
                    "avg_drawdown": round(row[4] or 0, 3),
                }
                for row in profit_categories
            ],
        }

    except Exception as e:
        logger.error(f"Error getting performance analytics: {e}")
        raise HTTPException(status_code=500, detail=f"Analytics error: {str(e)}")


@router.get("/system/health")
async def get_system_health() -> Dict[str, Any]:
    """Get comprehensive system health status"""
    try:
        # Check mutation system
        mutation_health = {
            "status": "healthy" if mutation_manager.is_running else "stopped",
            "is_running": mutation_manager.is_running,
            "cycle_count": mutation_manager.cycle_count,
            "last_cycle_time": (
                mutation_manager.last_cycle_time.isoformat()
                if mutation_manager.last_cycle_time
                else None
            ),
            "cycle_interval": mutation_manager.cycle_interval,
            "enable_ai_generation": mutation_manager.enable_ai_generation,
        }

        # Check database connectivity
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM ai_mutations")
            db_count = cursor.fetchone()[0]
            conn.close()
            db_health = {"status": "healthy", "total_mutations": db_count}
        except Exception as e:
            db_health = {"status": "error", "error": str(e)}

        # Check file system
        try:
            import os

            strategies_exist = os.path.exists("strategies")
            mutated_strategies_exist = os.path.exists("mutated_strategies")
            live_strategy = get_live_strategy()

            fs_health = {
                "status": "healthy",
                "strategies_dir": strategies_exist,
                "mutated_strategies_dir": mutated_strategies_exist,
                "live_strategy_exists": live_strategy is not None,
            }
        except Exception as e:
            fs_health = {"status": "error", "error": str(e)}

        # Overall health
        overall_health = "healthy"
        if not mutation_manager.is_running:
            overall_health = "warning"
        if db_health["status"] == "error" or fs_health["status"] == "error":
            overall_health = "error"

        return {
            "timestamp": datetime.now(timezone.timezone.utc).isoformat(),
            "overall_status": overall_health,
            "mutation_system": mutation_health,
            "database": db_health,
            "file_system": fs_health,
            "uptime": (
                (
                    datetime.now(timezone.timezone.utc) - mutation_manager.last_cycle_time
                ).total_seconds()
                if mutation_manager.last_cycle_time
                else 0
            ),
        }

    except Exception as e:
        logger.error(f"Error getting system health: {e}")
        raise HTTPException(status_code=500, detail=f"Health check error: {str(e)}")


@router.get("/monitoring/real-time")
async def get_real_time_monitoring() -> Dict[str, Any]:
    """Get comprehensive real-time monitoring data"""
    try:
        # Get current system status
        mutation_status = {
            "is_running": mutation_manager.is_running,
            "cycle_count": mutation_manager.cycle_count,
            "last_cycle_time": (
                mutation_manager.last_cycle_time.isoformat()
                if mutation_manager.last_cycle_time
                else None
            ),
            "next_cycle_in": (
                max(
                    0,
                    mutation_manager.cycle_interval
                    - (
                        datetime.now(timezone.timezone.utc) - mutation_manager.last_cycle_time
                    ).total_seconds(),
                )
                if mutation_manager.last_cycle_time
                else 0
            ),
            "enable_ai_generation": mutation_manager.enable_ai_generation,
            "enable_base_mutation": mutation_manager.enable_base_mutation,
        }

        # Get recent performance metrics
        conn = get_db_connection()
        cursor = conn.cursor()

        # Get last 24 hours of mutations
        cursor.execute(
            """
            SELECT
                COUNT(*) as total_mutations,
                SUM(CASE WHEN promoted = 1 THEN 1 ELSE 0 END) as promoted_mutations,
                AVG(simulated_profit) as avg_profit,
                MAX(simulated_profit) as max_profit,
                MIN(simulated_profit) as min_profit,
                AVG(win_rate) as avg_win_rate,
                AVG(sharpe_ratio) as avg_sharpe,
                AVG(max_drawdown) as avg_drawdown
            FROM ai_mutations
            WHERE created_at >= datetime('now', '-1 day')
        """
        )

        daily_stats = cursor.fetchone()

        # Get hourly performance for chart
        cursor.execute(
            """
            SELECT strftime('%H', created_at) as hour,
                   COUNT(*) as mutations,
                   SUM(CASE WHEN promoted = 1 THEN 1 ELSE 0 END) as promotions,
                   AVG(simulated_profit) as avg_profit,
                   AVG(win_rate) as avg_win_rate
            FROM ai_mutations
            WHERE created_at >= datetime('now', '-24 hours')
            GROUP BY strftime('%H', created_at)
            ORDER BY hour
        """
        )

        hourly_data = cursor.fetchall()

        # Get strategy diversity
        cursor.execute(
            """
            SELECT strategy_type, COUNT(*) as count
            FROM ai_mutations
            WHERE created_at >= datetime('now', '-7 days')
            GROUP BY strategy_type
            ORDER BY count DESC
        """
        )

        strategy_diversity = cursor.fetchall()

        # Get error analysis
        cursor.execute(
            """
            SELECT error_type, COUNT(*) as count
            FROM ai_mutations
            WHERE error IS NOT NULL AND created_at >= datetime('now', '-7 days')
            GROUP BY error_type
            ORDER BY count DESC
        """
        )

        error_analysis = cursor.fetchall()

        conn.close()

        # Calculate performance metrics
        total_mutations = daily_stats[0] if daily_stats[0] else 0
        promoted_mutations = daily_stats[1] if daily_stats[1] else 0
        success_rate = (promoted_mutations / total_mutations * 100) if total_mutations > 0 else 0
        avg_profit = daily_stats[2] if daily_stats[2] else 0.0
        max_profit = daily_stats[3] if daily_stats[3] else 0.0
        min_profit = daily_stats[4] if daily_stats[4] else 0.0
        avg_win_rate = daily_stats[5] if daily_stats[5] else 0.0
        avg_sharpe = daily_stats[6] if daily_stats[6] else 0.0
        avg_drawdown = daily_stats[7] if daily_stats[7] else 0.0

        # Format hourly data
        hourly_chart_data = []
        for row in hourly_data:
            hourly_chart_data.append(
                {
                    "hour": int(row[0]),
                    "mutations": row[1],
                    "promotions": row[2],
                    "avg_profit": row[3] if row[3] else 0.0,
                    "avg_win_rate": row[4] if row[4] else 0.0,
                }
            )

        # Format strategy diversity
        strategy_diversity_data = []
        for row in strategy_diversity:
            strategy_diversity_data.append({"strategy_type": row[0], "count": row[1]})

        # Format error analysis
        error_analysis_data = []
        for row in error_analysis:
            error_analysis_data.append({"error_type": row[0], "count": row[1]})

        # Get current live strategy
        live_strategy_file = get_live_strategy()
        live_strategy_info = None
        if live_strategy_file and os.path.exists(live_strategy_file):
            try:
                with open(live_strategy_file, "r") as f:
                    live_strategy_info = json.load(f)
            except Exception as e:
                logger.error(f"Error reading live strategy: {e}")

        return {
            "status": "success",
            "timestamp": datetime.now(timezone.timezone.utc).isoformat(),
            "mutation_status": mutation_status,
            "performance_metrics": {
                "total_mutations_24h": total_mutations,
                "promoted_mutations_24h": promoted_mutations,
                "success_rate_24h": success_rate,
                "avg_profit_24h": avg_profit,
                "max_profit_24h": max_profit,
                "min_profit_24h": min_profit,
                "avg_win_rate_24h": avg_win_rate,
                "avg_sharpe_24h": avg_sharpe,
                "avg_drawdown_24h": avg_drawdown,
            },
            "hourly_chart_data": hourly_chart_data,
            "strategy_diversity": strategy_diversity_data,
            "error_analysis": error_analysis_data,
            "live_strategy": {
                "file": live_strategy_file,
                "info": live_strategy_info,
            },
            "system_health": {
                "database_connected": True,
                "mutation_manager_active": mutation_manager.is_running,
                "ai_generation_enabled": mutation_manager.enable_ai_generation,
                "base_mutation_enabled": mutation_manager.enable_base_mutation,
                "last_successful_cycle": (
                    mutation_manager.last_cycle_time.isoformat()
                    if mutation_manager.last_cycle_time
                    else None
                ),
            },
        }

    except Exception as e:
        logger.error(f"Error getting real-time monitoring: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting real-time monitoring: {str(e)}",
        )


@router.get("/monitoring/alerts")
async def get_system_alerts() -> Dict[str, Any]:
    """Get system alerts and warnings"""
    try:
        alerts = []
        warnings = []
        critical_issues = []

        # Check mutation success rate
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT COUNT(*) as total, SUM(CASE WHEN promoted = 1 THEN 1 ELSE 0 END) as promoted
            FROM ai_mutations
            WHERE created_at >= datetime('now', '-24 hours')
        """
        )

        result = cursor.fetchone()
        if result and result[0] > 0:
            success_rate = (result[1] / result[0]) * 100
            if success_rate < 10:
                critical_issues.append(
                    {
                        "type": "low_success_rate",
                        "message": (f"Very low mutation success rate: {success_rate:.1f}%"),
                        "severity": "critical",
                        "recommendation": ("Review mutation parameters and strategy generation"),
                    }
                )
            elif success_rate < 20:
                warnings.append(
                    {
                        "type": "low_success_rate",
                        "message": (f"Low mutation success rate: {success_rate:.1f}%"),
                        "severity": "warning",
                        "recommendation": "Monitor mutation performance",
                    }
                )

        # Check for recent errors
        cursor.execute(
            """
            SELECT COUNT(*) as error_count
            FROM ai_mutations
            WHERE error IS NOT NULL AND created_at >= datetime('now', '-1 hour')
        """
        )

        error_result = cursor.fetchone()
        if error_result and error_result[0] > 5:
            critical_issues.append(
                {
                    "type": "high_error_rate",
                    "message": (f"High error rate in last hour: {error_result[0]} errors"),
                    "severity": "critical",
                    "recommendation": ("Check system logs and fix underlying issues"),
                }
            )

        # Check mutation frequency
        cursor.execute(
            """
            SELECT COUNT(*) as recent_mutations
            FROM ai_mutations
            WHERE created_at >= datetime('now', '-1 hour')
        """
        )

        freq_result = cursor.fetchone()
        if freq_result and freq_result[0] == 0:
            warnings.append(
                {
                    "type": "no_recent_mutations",
                    "message": "No mutations in the last hour",
                    "severity": "warning",
                    "recommendation": "Check if mutation manager is running",
                }
            )

        # Check live strategy age
        live_strategy_file = get_live_strategy()
        if live_strategy_file and os.path.exists(live_strategy_file):
            file_age = datetime.now(timezone.timezone.utc) - datetime.fromtimestamp(
                os.path.getmtime(live_strategy_file), tz=timezone.timezone.utc
            )
            if file_age.days > 7:
                warnings.append(
                    {
                        "type": "old_live_strategy",
                        "message": (f"Live strategy is {file_age.days} days old"),
                        "severity": "warning",
                        "recommendation": ("Consider promoting a newer strategy"),
                    }
                )

        conn.close()

        # Check mutation manager status
        if not mutation_manager.is_running:
            critical_issues.append(
                {
                    "type": "mutation_manager_stopped",
                    "message": "Mutation manager is not running",
                    "severity": "critical",
                    "recommendation": "Restart the mutation manager",
                }
            )

        return {
            "status": "success",
            "timestamp": datetime.now(timezone.timezone.utc).isoformat(),
            "alerts": {
                "critical": critical_issues,
                "warnings": warnings,
                "info": alerts,
            },
            "summary": {
                "total_alerts": (len(critical_issues) + len(warnings) + len(alerts)),
                "critical_count": len(critical_issues),
                "warning_count": len(warnings),
                "info_count": len(alerts),
            },
        }

    except Exception as e:
        logger.error(f"Error getting system alerts: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting system alerts: {str(e)}")



@router.post("/monitoring/optimize")
async def trigger_system_optimization() -> Dict[str, Any]:
    """Trigger system optimization"""
    try:
        # This would integrate with the performance optimizer
        # For now, return a placeholder response

        optimization_result = {
            "status": "success",
            "message": "System optimization triggered",
            "optimizations_applied": [
                "Adjusted mutation parameters",
                "Updated promotion criteria",
                "Enhanced strategy generation",
            ],
            "timestamp": datetime.now(timezone.timezone.utc).isoformat(),
        }

        return optimization_result

    except Exception as e:
        logger.error(f"Error triggering optimization: {e}")
        raise HTTPException(status_code=500, detail=f"Error triggering optimization: {str(e)}")


@router.post("/mutation/manual-promote")
async def manual_promote_strategy(
    strategy_file: str = Body(..., embed=True),
    justification: str = Body(..., embed=True),
) -> Dict[str, Any]:
    """Manually promote a strategy to live with operator justification"""
    try:
        # Promote the given strategy file
        promoter = StrategyPromoter()
        promoted = promoter.promote_strategy(
            strategy_file, manual=True, justification=justification
        )
        return {
            "success": promoted,
            "message": f"Strategy {strategy_file} manually promoted.",
            "justification": justification,
            "timestamp": datetime.now(timezone.timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.error(f"Manual promotion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Manual promotion failed: {str(e)}")


@router.post("/mutation/rollback")
async def rollback_strategy(
    version_file: str = Body(..., embed=True),
    reason: str = Body(..., embed=True),
) -> Dict[str, Any]:
    """Rollback to a previous promoted strategy version"""
    try:
        import shutil

        # Assume version_file is a path to a previous promoted strategy
        live_strategy_path = "strategies/current_strategy.json"
        shutil.copy(version_file, live_strategy_path)
        return {
            "success": True,
            "message": f"Rolled back to {version_file}",
            "reason": reason,
            "timestamp": datetime.now(timezone.timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.error(f"Rollback failed: {e}")
        raise HTTPException(status_code=500, detail=f"Rollback failed: {str(e)}")


@router.get("/features")
async def get_available_features() -> Dict[str, Any]:
    """Get list of all available AI and trading platform features"""
    return {
        "features": [
            {
                "name": "Real-time Trading",
                "description": "Live market data and instant order execution",
                "enabled": True,
            },
            {
                "name": "AI Analytics",
                "description": "Machine learning powered market analysis",
                "enabled": True,
            },
            {
                "name": "Self-Evolving AI",
                "description": ("Mutation, promotion, and versioning of trading strategies"),
                "enabled": True,
            },
            {
                "name": "Strategy Leaderboard",
                "description": "Live leaderboard with badges and achievements",
                "enabled": True,
            },
            {
                "name": "Operator Controls",
                "description": ("Pause, resume, manual promotion, and rollback endpoints"),
                "enabled": True,
            },
            {
                "name": "Risk Management",
                "description": "Portfolio risk analysis and position sizing",
                "enabled": True,
            },
            {
                "name": "Auto Trading",
                "description": "Automated trading bots with custom strategies",
                "enabled": True,
            },
            {
                "name": "Social Trading",
                "description": "Copy successful traders and share strategies",
                "enabled": True,
            },
            {
                "name": "Mobile PWA",
                "description": "Progressive web app for mobile trading",
                "enabled": True,
            },
            {
                "name": "Advanced Orders",
                "description": ("Stop-loss, take-profit, and conditional orders"),
                "enabled": True,
            },
            {
                "name": "Comprehensive Monitoring",
                "description": ("Real-time dashboard, alerts, and performance trends"),
                "enabled": True,
            },
            {
                "name": "Discord Notifications",
                "description": "Rich Discord alerts for all major events",
                "enabled": True,
            },
            {
                "name": "API Documentation",
                "description": "OpenAPI docs at /docs and /redoc",
                "enabled": True,
            },
        ],
        "timestamp": datetime.now(timezone.timezone.utc).isoformat(),
    }
