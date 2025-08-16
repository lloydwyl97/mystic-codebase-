import json
import os
import sqlite3
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

# Enhanced configuration
REAPER_DB = "./data/strategy_reaper.db"
REAPER_INTERVAL = 3600  # 1 hour
STRATEGY_DIR = "./strategies"
GENERATED_DIR = "./generated_modules"
PERFORMANCE_THRESHOLD = 0.1  # 10% minimum performance
MAX_STRATEGIES = 50  # Maximum number of strategies to keep
MIN_TRADES = 5  # Minimum trades for evaluation


class ReaperDatabase:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize reaper database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS strategy_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_name TEXT NOT NULL,
                strategy_type TEXT NOT NULL,
                total_trades INTEGER NOT NULL,
                win_rate REAL NOT NULL,
                avg_profit REAL NOT NULL,
                total_profit REAL NOT NULL,
                max_drawdown REAL NOT NULL,
                sharpe_ratio REAL NOT NULL,
                last_trade_date TEXT,
                created_date TEXT NOT NULL,
                status TEXT NOT NULL,
                performance_score REAL NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS reaper_actions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                action_type TEXT NOT NULL,
                strategy_name TEXT NOT NULL,
                reason TEXT NOT NULL,
                performance_data TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS strategy_lifecycle (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_name TEXT NOT NULL,
                phase TEXT NOT NULL,
                start_date TEXT NOT NULL,
                end_date TEXT,
                performance_metrics TEXT NOT NULL,
                notes TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        conn.commit()
        conn.close()

    def save_strategy_performance(self, performance: Dict):
        """Save strategy performance to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO strategy_performance
            (strategy_name, strategy_type, total_trades, win_rate, avg_profit, total_profit,
             max_drawdown, sharpe_ratio, last_trade_date, created_date, status, performance_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                performance["strategy_name"],
                performance["strategy_type"],
                performance["total_trades"],
                performance["win_rate"],
                performance["avg_profit"],
                performance["total_profit"],
                performance["max_drawdown"],
                performance["sharpe_ratio"],
                performance.get("last_trade_date"),
                performance["created_date"],
                performance["status"],
                performance["performance_score"],
            ),
        )

        conn.commit()
        conn.close()

    def save_reaper_action(self, action: Dict):
        """Save reaper action to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO reaper_actions
            (timestamp, action_type, strategy_name, reason, performance_data)
            VALUES (?, ?, ?, ?, ?)
        """,
            (
                action["timestamp"],
                action["action_type"],
                action["strategy_name"],
                action["reason"],
                json.dumps(action["performance_data"]),
            ),
        )

        conn.commit()
        conn.close()

    def save_lifecycle_event(self, lifecycle: Dict):
        """Save strategy lifecycle event"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO strategy_lifecycle
            (strategy_name, phase, start_date, end_date, performance_metrics, notes)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (
                lifecycle["strategy_name"],
                lifecycle["phase"],
                lifecycle["start_date"],
                lifecycle.get("end_date"),
                json.dumps(lifecycle["performance_metrics"]),
                lifecycle.get("notes"),
            ),
        )

        conn.commit()
        conn.close()

    def get_latest_strategy_performance(self, strategy_name: str) -> Optional[Dict]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT strategy_name, total_trades, win_rate, avg_profit, total_profit, "
            "max_drawdown, sharpe_ratio, performance_score, status, created_date "
            "FROM strategy_performance WHERE strategy_name = ? ORDER BY timestamp DESC LIMIT 1",
            (strategy_name,),
        )
        row = cursor.fetchone()
        if row:
            return {
                "strategy_name": row[0],
                "total_trades": row[1],
                "win_rate": row[2],
                "avg_profit": row[3],
                "total_profit": row[4],
                "max_drawdown": row[5],
                "sharpe_ratio": row[6],
                "performance_score": row[7],
                "status": row[8],
                "created_date": row[9],
            }
        return None


def get_strategy_files() -> List[str]:
    """Get all strategy files from directories"""
    strategy_files = []

    # Get from strategies directory
    if os.path.exists(STRATEGY_DIR):
        for file in os.listdir(STRATEGY_DIR):
            if file.endswith(".json"):
                strategy_files.append(os.path.join(STRATEGY_DIR, file))

    # Get from generated modules directory
    if os.path.exists(GENERATED_DIR):
        for file in os.listdir(GENERATED_DIR):
            if file.endswith(".py"):
                strategy_files.append(os.path.join(GENERATED_DIR, file))

    return strategy_files


def analyze_strategy_performance(strategy_file: str) -> Optional[Dict]:
    """Analyze strategy performance using live data from the database"""
    strategy_name = os.path.basename(strategy_file)
    db = ReaperDatabase(REAPER_DB)
    performance = db.get_latest_strategy_performance(strategy_name)
    if performance:
        return performance
    else:
        # No live data available for this strategy
        return None


def calculate_adaptive_thresholds(performances: List[Dict]) -> Dict:
    """Calculate adaptive thresholds based on current performance distribution"""
    if not performances:
        return {
            "performance_threshold": PERFORMANCE_THRESHOLD,
            "min_trades": MIN_TRADES,
            "max_strategies": MAX_STRATEGIES,
        }

    # Calculate statistics
    scores = [p["performance_score"] for p in performances]
    trades = [p["total_trades"] for p in performances]

    # Adaptive thresholds
    performance_threshold = max(PERFORMANCE_THRESHOLD, np.percentile(scores, 25))
    min_trades = max(MIN_TRADES, int(np.percentile(trades, 10)))
    max_strategies = min(MAX_STRATEGIES, len(performances) + 10)

    return {
        "performance_threshold": performance_threshold,
        "min_trades": min_trades,
        "max_strategies": max_strategies,
    }


def evaluate_strategy_lifecycle(performance: Dict) -> str:
    """Evaluate strategy lifecycle phase"""
    score = performance["performance_score"]
    trades = performance["total_trades"]
    age_days = (
        datetime.timezone.utcnow() - datetime.fromisoformat(performance["created_date"])
    ).days

    if score > 0.8 and trades > 20:
        return "mature"
    elif score > 0.6 and trades > 10:
        return "developing"
    elif score > 0.4 and trades > 5:
        return "testing"
    elif age_days > 30 and score < 0.3:
        return "declining"
    else:
        return "new"


def should_reap_strategy(performance: Dict, thresholds: Dict) -> Tuple[bool, str]:
    """Determine if strategy should be reaped"""
    reasons = []

    # Check performance threshold
    if performance["performance_score"] < thresholds["performance_threshold"]:
        reasons.append(f"Low performance score: {performance['performance_score']:.3f}")

    # Check minimum trades
    if performance["total_trades"] < thresholds["min_trades"]:
        reasons.append(f"Insufficient trades: {performance['total_trades']}")

    # Check for extreme underperformance
    if performance["win_rate"] < 0.2 and performance["total_trades"] > 10:
        reasons.append(f"Very low win rate: {performance['win_rate']:.3f}")

    # Check for excessive drawdown
    if performance["max_drawdown"] > 0.5:
        reasons.append(f"Excessive drawdown: {performance['max_drawdown']:.3f}")

    # Check age for underperforming strategies
    age_days = (
        datetime.timezone.utcnow() - datetime.fromisoformat(performance["created_date"])
    ).days
    if age_days > 60 and performance["performance_score"] < 0.4:
        reasons.append(f"Old and underperforming: {age_days} days")

    should_reap = len(reasons) > 0
    reason = "; ".join(reasons) if reasons else "Strategy performing well"

    return should_reap, reason


def reap_strategy(strategy_file: str, reason: str, performance: Dict):
    """Reap (delete) underperforming strategy"""
    try:
        # Create backup before deletion
        backup_dir = "./strategy_backups"
        os.makedirs(backup_dir, exist_ok=True)

        backup_file = os.path.join(
            backup_dir,
            f"reaped_{os.path.basename(strategy_file)}_{datetime.timezone.utcnow().strftime('%Y%m%d_%H%M%S')}",
        )

        # Copy file to backup
        import shutil

        shutil.copy2(strategy_file, backup_file)

        # Delete original file
        os.remove(strategy_file)

        print(f"[Reaper] Reaped strategy: {os.path.basename(strategy_file)}")
        print(f"[Reaper] Reason: {reason}")
        print(f"[Reaper] Backup saved: {backup_file}")

        return True

    except Exception as e:
        print(f"[Reaper] Error reaping strategy {strategy_file}: {e}")
        return False


def optimize_strategy_count(performances: List[Dict], max_strategies: int) -> List[str]:
    """Optimize strategy count by keeping best performers"""
    if len(performances) <= max_strategies:
        return []

    # Sort by performance score
    sorted_performances = sorted(performances, key=lambda x: x["performance_score"], reverse=True)

    # Keep top performers
    sorted_performances[:max_strategies]
    reap_strategies = sorted_performances[max_strategies:]

    return [p["strategy_name"] for p in reap_strategies]


def reap_strategies_enhanced():
    """Enhanced strategy reaping with all features"""
    try:
        db = ReaperDatabase(REAPER_DB)

        # Get all strategy files
        strategy_files = get_strategy_files()
        print(f"[Reaper] Found {len(strategy_files)} strategy files")

        # Analyze performance of all strategies
        performances = []
        for strategy_file in strategy_files:
            performance = analyze_strategy_performance(strategy_file)
            if performance:
                performances.append(performance)
                # db.save_strategy_performance(performance) # This line is removed as per the edit hint

        if not performances:
            print("[Reaper] No strategies to evaluate")
            return

        # Calculate adaptive thresholds
        thresholds = calculate_adaptive_thresholds(performances)
        print(
            f"[Reaper] Adaptive thresholds: Performance={thresholds['performance_threshold']:.3f}, Min trades={thresholds['min_trades']}"
        )

        # Evaluate each strategy
        reaped_count = 0
        for performance in performances:
            strategy_file = None

            # Find corresponding file
            for file in strategy_files:
                if os.path.basename(file) == performance["strategy_name"]:
                    strategy_file = file
                    break

            if not strategy_file:
                continue

            # Evaluate lifecycle
            evaluate_strategy_lifecycle(performance)

            # Check if should be reaped
            should_reap, reason = should_reap_strategy(performance, thresholds)

            if should_reap:
                # Reap the strategy
                if reap_strategy(strategy_file, reason, performance):
                    reaped_count += 1

                    # Save reaper action
                    action = {
                        "timestamp": datetime.timezone.utcnow().isoformat(),
                        "action_type": "reap",
                        "strategy_name": performance["strategy_name"],
                        "reason": reason,
                        "performance_data": performance,
                    }
                    db.save_reaper_action(action)

                    # Save lifecycle event
                    lifecycle = {
                        "strategy_name": performance["strategy_name"],
                        "phase": "reaped",
                        "start_date": performance["created_date"],
                        "end_date": datetime.timezone.utcnow().isoformat(),
                        "performance_metrics": performance,
                        "notes": f"Reaped due to: {reason}",
                    }
                    db.save_lifecycle_event(lifecycle)

        # Optimize strategy count
        if len(performances) > thresholds["max_strategies"]:
            strategies_to_reap = optimize_strategy_count(performances, thresholds["max_strategies"])
            print(
                f"[Reaper] Optimizing strategy count: {len(strategies_to_reap)} strategies to reap"
            )

            for strategy_name in strategies_to_reap:
                # Find and reap strategy
                for file in strategy_files:
                    if os.path.basename(file) == strategy_name:
                        performance = next(
                            p for p in performances if p["strategy_name"] == strategy_name
                        )
                        reason = f"Strategy count optimization (ranked {len(performances) - len(strategies_to_reap) + 1}/{len(performances)})"

                        if reap_strategy(file, reason, performance):
                            reaped_count += 1

                            # Save action
                            action = {
                                "timestamp": (datetime.timezone.utcnow().isoformat()),
                                "action_type": "optimize",
                                "strategy_name": strategy_name,
                                "reason": reason,
                                "performance_data": performance,
                            }
                            db.save_reaper_action(action)
                        break

        # Print summary
        print(f"[Reaper] Reaping complete: {reaped_count} strategies reaped")
        print(f"[Reaper] Remaining strategies: {len(strategy_files) - reaped_count}")

        # Performance summary
        if performances:
            avg_score = np.mean([p["performance_score"] for p in performances])
            avg_trades = np.mean([p["total_trades"] for p in performances])
            print(f"[Reaper] Average performance score: {avg_score:.3f}")
            print(f"[Reaper] Average trades per strategy: {avg_trades:.1f}")

    except Exception as e:
        print(f"[Reaper] Enhanced reaping error: {e}")


# Main execution loop
while True:
    reap_strategies_enhanced()
    time.sleep(REAPER_INTERVAL)



