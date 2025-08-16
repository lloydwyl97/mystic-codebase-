"""
Mutation Manager for AI Strategy Evolution

Manages the AI mutation system, including strategy evolution cycles,
performance tracking, and system status monitoring.
"""

import asyncio
import json
import logging
import sqlite3
from datetime import datetime, timezone
from typing import Dict, List, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class MutationManager:
    """Manages AI strategy mutation and evolution cycles"""

    def __init__(self):
        self.is_running = False
        self.cycle_count = 0
        self.last_cycle_time = None
        self.cycle_interval = 300  # 5 minutes
        self.enable_ai_generation = True
        self.enable_base_mutation = True
        self.base_strategy = "breakout_ai.json"
        self.mutation_task = None
        self.db_path = "simulation_trades.db"

        # Initialize database
        self._init_database()

    def _init_database(self):
        """Initialize the mutation database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Create ai_mutations table if it doesn't exist
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS ai_mutations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_file TEXT NOT NULL,
                    strategy_type TEXT,
                    strategy_name TEXT,
                    simulated_profit REAL,
                    win_rate REAL,
                    num_trades INTEGER,
                    max_drawdown REAL,
                    sharpe_ratio REAL,
                    promoted INTEGER DEFAULT 0,
                    backtest_results TEXT,
                    cycle_number INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            conn.commit()
            conn.close()
            logger.info("âœ… Mutation database initialized")

        except Exception as e:
            logger.error(f"âŒ Error initializing mutation database: {e}")

    def start_mutation_engine(self):
        """Start the mutation engine"""
        if not self.is_running:
            self.is_running = True
            self.mutation_task = asyncio.create_task(self._mutation_loop())
            logger.info("ðŸš€ Mutation engine started")

    def stop_mutation_engine(self):
        """Stop the mutation engine"""
        if self.is_running:
            self.is_running = False
            if self.mutation_task:
                self.mutation_task.cancel()
            logger.info("ðŸ›‘ Mutation engine stopped")

    async def _mutation_loop(self):
        """Main mutation loop"""
        while self.is_running:
            try:
                await self.run_single_cycle()
                await asyncio.sleep(self.cycle_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"âŒ Error in mutation loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying

    async def run_single_cycle(self) -> Dict[str, Any]:
        """Run a single mutation cycle"""
        try:
            self.cycle_count += 1
            self.last_cycle_time = datetime.now(timezone.utc)

            logger.info(f"ðŸ”„ Running mutation cycle {self.cycle_count}")

            # Generate new strategies
            new_strategies = []

            if self.enable_ai_generation:
                ai_strategies = await self._generate_ai_strategies()
                new_strategies.extend(ai_strategies)

            if self.enable_base_mutation:
                base_strategies = await self._mutate_base_strategies()
                new_strategies.extend(base_strategies)

            # Evaluate strategies
            evaluated_strategies = []
            for strategy in new_strategies:
                evaluation = await self._evaluate_strategy(strategy)
                evaluated_strategies.append(evaluation)

            # Save to database
            for strategy in evaluated_strategies:
                self._save_mutation(strategy)

            # Promote best strategies
            await self._promote_best_strategies()

            logger.info(f"âœ… Mutation cycle {self.cycle_count} completed")
            return {
                "cycle_number": self.cycle_count,
                "strategies_generated": len(new_strategies),
                "strategies_evaluated": len(evaluated_strategies),
                "timestamp": self.last_cycle_time.isoformat(),
            }

        except Exception as e:
            logger.error(f"âŒ Error in mutation cycle: {e}")
            return {"error": str(e)}

    async def _generate_ai_strategies(self) -> List[Dict[str, Any]]:
        """Generate new AI strategies"""
        strategies = []

        # Generate different types of strategies
        strategy_types = ["breakout", "mean_reversion", "momentum", "scalping"]

        for strategy_type in strategy_types:
            strategy = {
                "strategy_type": strategy_type,
                "strategy_name": f"{strategy_type}_ai_v{self.cycle_count}",
                "strategy_file": (f"{strategy_type}_ai_v{self.cycle_count}.json"),
                "created_by": "ai_mutation",
                "ai_version": "v1.0.0",
                "parent": self.base_strategy,
                "description": f"AI-generated {strategy_type} strategy",
                "parameters": self._generate_strategy_parameters(strategy_type),
            }
            strategies.append(strategy)

        return strategies

    async def _mutate_base_strategies(self) -> List[Dict[str, Any]]:
        """Mutate existing base strategies"""
        strategies = []

        # Load and mutate base strategy
        base_path = Path("strategies") / self.base_strategy
        if base_path.exists():
            try:
                with open(base_path, "r") as f:
                    base_data = json.load(f)

                # Create mutated version
                mutated_strategy = {
                    "strategy_type": base_data.get("strategy_type", "breakout"),
                    "strategy_name": (
                        f"mutated_{self.base_strategy.replace('.json', '')}_v{self.cycle_count}"
                    ),
                    "strategy_file": (
                        f"mutated_{self.base_strategy.replace('.json', '')}_v{self.cycle_count}.json"
                    ),
                    "created_by": "ai_mutation",
                    "ai_version": "v1.0.0",
                    "parent": self.base_strategy,
                    "description": f"Mutated version of {self.base_strategy}",
                    "parameters": self._mutate_parameters(base_data.get("parameters", {})),
                }
                strategies.append(mutated_strategy)

            except Exception as e:
                logger.error(f"âŒ Error mutating base strategy: {e}")

        return strategies

    def _generate_strategy_parameters(self, strategy_type: str) -> Dict[str, Any]:
        """Generate parameters for a strategy type"""
        if strategy_type == "breakout":
            return {
                "lookback_period": 20,
                "breakout_threshold": 0.02,
                "stop_loss": 0.05,
                "take_profit": 0.10,
                "volume_threshold": 1.5,
            }
        elif strategy_type == "mean_reversion":
            return {
                "sma_period": 50,
                "std_deviation": 2.0,
                "entry_threshold": 0.03,
                "exit_threshold": 0.01,
            }
        elif strategy_type == "momentum":
            return {
                "rsi_period": 14,
                "rsi_oversold": 30,
                "rsi_overbought": 70,
                "momentum_period": 10,
            }
        else:  # scalping
            return {
                "ema_fast": 9,
                "ema_slow": 21,
                "profit_target": 0.005,
                "stop_loss": 0.003,
            }

    def _mutate_parameters(self, base_params: Dict[str, Any]) -> Dict[str, Any]:
        """Mutate existing parameters"""
        import random

        mutated_params = base_params.copy()

        # Randomly adjust some parameters
        for key, value in mutated_params.items():
            if isinstance(value, (int, float)):
                # Add/subtract up to 20% of the value
                variation = value * 0.2 * random.uniform(-1, 1)
                mutated_params[key] = value + variation

                # Ensure reasonable bounds
                if key.endswith("_period") and mutated_params[key] < 1:
                    mutated_params[key] = 1
                elif key.endswith("_threshold") and mutated_params[key] < 0:
                    mutated_params[key] = 0.001

        return mutated_params

    async def _evaluate_strategy(self, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a strategy's performance"""
        # Simulate backtesting
        import random

        # Simulate performance metrics
        simulated_profit = random.uniform(-0.15, 0.25)  # -15% to +25%
        win_rate = random.uniform(0.3, 0.7)  # 30% to 70%
        num_trades = random.randint(10, 100)
        max_drawdown = random.uniform(0.05, 0.20)  # 5% to 20%
        sharpe_ratio = random.uniform(-1.0, 2.0)

        return {
            **strategy,
            "simulated_profit": simulated_profit,
            "win_rate": win_rate,
            "num_trades": num_trades,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "promoted": False,
            "cycle_number": self.cycle_count,
            "backtest_results": json.dumps(
                {
                    "trades": [],
                    "equity_curve": [],
                    "metrics": {
                        "profit": simulated_profit,
                        "win_rate": win_rate,
                        "num_trades": num_trades,
                        "max_drawdown": max_drawdown,
                        "sharpe_ratio": sharpe_ratio,
                    },
                }
            ),
        }

    def _save_mutation(self, strategy: Dict[str, Any]):
        """Save mutation to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO ai_mutations (
                    strategy_file, strategy_type, strategy_name, simulated_profit,
                    win_rate, num_trades, max_drawdown, sharpe_ratio, promoted,
                    backtest_results, cycle_number, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    strategy["strategy_file"],
                    strategy["strategy_type"],
                    strategy["strategy_name"],
                    strategy["simulated_profit"],
                    strategy["win_rate"],
                    strategy["num_trades"],
                    strategy["max_drawdown"],
                    strategy["sharpe_ratio"],
                    strategy["promoted"],
                    strategy["backtest_results"],
                    strategy["cycle_number"],
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"âŒ Error saving mutation: {e}")

    async def _promote_best_strategies(self):
        """Promote the best performing strategies"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Find best strategies from recent cycles
            cursor.execute(
                """
                SELECT id, strategy_file, simulated_profit, win_rate, sharpe_ratio
                FROM ai_mutations
                WHERE promoted = 0 AND cycle_number >= ?
                ORDER BY simulated_profit DESC, sharpe_ratio DESC
                LIMIT 3
            """,
                (max(1, self.cycle_count - 5),),
            )

            best_strategies = cursor.fetchall()

            for (
                strategy_id,
                strategy_file,
                profit,
                win_rate,
                sharpe,
            ) in best_strategies:
                # Promote if meets criteria
                if profit > 0.05 and win_rate > 0.5 and sharpe > 0.5:
                    cursor.execute(
                        """
                        UPDATE ai_mutations
                        SET promoted = 1
                        WHERE id = ?
                    """,
                        (strategy_id,),
                    )

                    logger.info(f"ðŸŽ‰ Promoted strategy: {strategy_file}")

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"âŒ Error promoting strategies: {e}")

    def get_mutation_stats(self) -> Dict[str, Any]:
        """Get mutation system statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
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

            # Get success rate
            success_rate = (
                (promoted_mutations / total_mutations * 100) if total_mutations > 0 else 0.0
            )

            conn.close()

            return {
                "total_mutations": total_mutations,
                "promoted_mutations": promoted_mutations,
                "success_rate": round(success_rate, 2),
                "average_profit": round(avg_profit, 4),
                "cycle_count": self.cycle_count,
                "is_running": self.is_running,
            }

        except Exception as e:
            logger.error(f"âŒ Error getting mutation stats: {e}")
            return {
                "total_mutations": 0,
                "promoted_mutations": 0,
                "success_rate": 0.0,
                "average_profit": 0.0,
                "cycle_count": self.cycle_count,
                "is_running": self.is_running,
            }

    def get_recent_mutations(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent mutations"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT strategy_file, strategy_type, strategy_name, simulated_profit,
                       win_rate, num_trades, promoted, created_at
                FROM ai_mutations
                ORDER BY created_at DESC
                LIMIT ?
            """,
                (limit,),
            )

            rows = cursor.fetchall()
            conn.close()

            mutations = []
            for row in rows:
                mutations.append(
                    {
                        "strategy_file": row[0],
                        "strategy_type": row[1],
                        "strategy_name": row[2],
                        "profit": row[3],
                        "win_rate": row[4],
                        "num_trades": row[5],
                        "promoted": bool(row[6]),
                        "timestamp": row[7],
                    }
                )

            return mutations

        except Exception as e:
            logger.error(f"âŒ Error getting recent mutations: {e}")
            return []


# Singleton instance
mutation_manager = MutationManager()


