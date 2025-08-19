# hyper_tuner.py
"""
Hyperparameter Optimization Engine for AI Trading Strategies
Auto-tunes strategy parameters to maximize profit, win rate, and Sharpe ratio.
Built for Windows 11 Home + PowerShell + Docker.
"""

import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any

# Import your existing modules
from strat_versions import save_strategy_version

logger = logging.getLogger(__name__)


class HyperparameterTuner:
    """
    Advanced hyperparameter optimization engine with multiple algorithms.
    """

    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.best_configs = []
        self.optimization_history = []

    def generate_random_config(self, strategy_type: str = "rsi_ema_breakout") -> dict[str, Any]:
        """
        Generate random configuration for different strategy types.

        Args:
            strategy_type: Type of strategy to generate config for

        Returns:
            Dict with random parameters
        """
        if strategy_type == "rsi_ema_breakout":
            return {
                "rsi_period": random.randint(5, 30),
                "rsi_oversold": random.randint(20, 35),
                "rsi_overbought": random.randint(65, 80),
                "ema_fast": random.randint(5, 20),
                "ema_slow": random.randint(21, 100),
                "breakout_threshold": round(random.uniform(0.01, 0.05), 4),
                "volume_multiplier": round(random.uniform(1.0, 3.0), 2),
                "stop_loss_pct": round(random.uniform(0.02, 0.08), 4),
                "take_profit_pct": round(random.uniform(0.03, 0.12), 4),
            }
        elif strategy_type == "bollinger_bands":
            return {
                "bb_period": random.randint(10, 50),
                "bb_std": round(random.uniform(1.5, 3.0), 2),
                "volume_threshold": round(random.uniform(1.0, 2.5), 2),
                "rsi_period": random.randint(10, 25),
                "rsi_oversold": random.randint(25, 40),
                "rsi_overbought": random.randint(60, 75),
            }
        elif strategy_type == "macd_crossover":
            return {
                "macd_fast": random.randint(8, 15),
                "macd_slow": random.randint(20, 35),
                "macd_signal": random.randint(5, 15),
                "volume_filter": round(random.uniform(1.0, 2.0), 2),
                "trend_strength": round(random.uniform(0.5, 1.5), 2),
            }
        else:
            # Generic config
            return {
                "period_1": random.randint(5, 50),
                "period_2": random.randint(10, 100),
                "threshold": round(random.uniform(0.01, 0.10), 4),
                "multiplier": round(random.uniform(0.5, 3.0), 2),
            }

    def mutate_config(
        self, base_config: dict[str, Any], mutation_rate: float = 0.3
    ) -> dict[str, Any]:
        """
        Create a mutated version of an existing configuration.

        Args:
            base_config: Base configuration to mutate
            mutation_rate: Probability of mutating each parameter

        Returns:
            Mutated configuration
        """
        mutated = base_config.copy()

        for key, value in mutated.items():
            if random.random() < mutation_rate:
                if isinstance(value, int):
                    # Mutate integer parameters
                    if "period" in key.lower():
                        mutated[key] = max(1, value + random.randint(-5, 5))
                    else:
                        mutated[key] = max(1, value + random.randint(-2, 2))
                elif isinstance(value, float):
                    # Mutate float parameters
                    if "threshold" in key.lower() or "pct" in key.lower():
                        mutated[key] = max(0.001, value * random.uniform(0.8, 1.2))
                    else:
                        mutated[key] = max(0.1, value * random.uniform(0.7, 1.3))

        return mutated

    def crossover_configs(self, config1: dict[str, Any], config2: dict[str, Any]) -> dict[str, Any]:
        """
        Create a new configuration by combining two parent configurations.

        Args:
            config1: First parent configuration
            config2: Second parent configuration

        Returns:
            Combined configuration
        """
        child = {}

        for key in config1:
            if key in config2:
                if random.random() < 0.5:
                    child[key] = config1[key]
                else:
                    child[key] = config2[key]

                # Add some random variation
                if isinstance(child[key], (int, float)):
                    if random.random() < 0.2:  # 20% chance of small mutation
                        if isinstance(child[key], int):
                            child[key] = max(1, child[key] + random.randint(-1, 1))
                        else:
                            child[key] = max(0.001, child[key] * random.uniform(0.95, 1.05))
            else:
                child[key] = config1[key]

        return child

    def evaluate_config(
        self, config: dict[str, Any], strategy_type: str = "rsi_ema_breakout"
    ) -> dict[str, Any]:
        """
        Evaluate a configuration using backtesting or live performance.

        Args:
            config: Configuration to evaluate
            strategy_type: Type of strategy

        Returns:
            Evaluation results
        """
        try:
            # For now, simulate evaluation with random performance
            # In production, this would run actual backtesting or use live data

            # Simulate realistic performance based on config quality
            base_profit = 100.0

            # Adjust based on parameter quality
            if "rsi_period" in config:
                if 10 <= config["rsi_period"] <= 20:
                    base_profit *= 1.2  # Good RSI period
                elif config["rsi_period"] < 5 or config["rsi_period"] > 30:
                    base_profit *= 0.5  # Poor RSI period

            if "ema_fast" in config and "ema_slow" in config:
                if config["ema_fast"] < config["ema_slow"]:
                    base_profit *= 1.1  # Logical EMA relationship
                else:
                    base_profit *= 0.3  # Illogical EMA relationship

            # Add randomness to simulate real market conditions
            profit = base_profit * random.uniform(0.5, 1.5)
            win_rate = random.uniform(0.4, 0.8)
            trades_count = random.randint(5, 50)

            # Calculate Sharpe ratio (simplified)
            sharpe_ratio = (profit / max(trades_count, 1)) / max(profit * 0.1, 1)

            return {
                "config": config,
                "total_profit": round(profit, 2),
                "win_rate": round(win_rate, 4),
                "trades_count": trades_count,
                "sharpe_ratio": round(sharpe_ratio, 4),
                "evaluation_time": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            logger.error(f"Error evaluating config: {e}")
            return {
                "config": config,
                "total_profit": -1000.0,
                "win_rate": 0.0,
                "trades_count": 0,
                "sharpe_ratio": 0.0,
                "error": str(e),
            }

    def run_random_search(
        self,
        strategy_type: str = "rsi_ema_breakout",
        rounds: int = 100,
        save_best: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Run random search optimization.

        Args:
            strategy_type: Type of strategy to optimize
            rounds: Number of random configurations to try
            save_best: Whether to save the best configuration

        Returns:
            List of best configurations found
        """
        print(f"ðŸ§¬ Starting Random Search Optimization for {strategy_type}")
        print(f"ðŸ“Š Testing {rounds} random configurations...")

        best_configs = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Generate and submit all configurations
            future_to_config = {
                executor.submit(
                    self.evaluate_config,
                    self.generate_random_config(strategy_type),
                    strategy_type,
                ): i
                for i in range(rounds)
            }

            # Process results as they complete
            for future in as_completed(future_to_config):
                result = future.result()
                trial_num = future_to_config[future] + 1

                print(
                    f"[TRIAL {trial_num:3d}] Profit: ${result['total_profit']:8.2f} | "
                    f"Win Rate: {result['win_rate']:.1%} | "
                    f"Trades: {result['trades_count']:2d} | "
                    f"Sharpe: {result['sharpe_ratio']:.3f}"
                )

                # Track best configurations
                if result["total_profit"] > 0:
                    best_configs.append(result)
                    best_configs.sort(key=lambda x: x["total_profit"], reverse=True)
                    best_configs = best_configs[:10]  # Keep top 10

        print("\nðŸ† Random Search Complete!")
        print(f"âœ… Found {len(best_configs)} profitable configurations")

        if best_configs and save_best:
            best_config = best_configs[0]
            config_name = (
                f"{strategy_type}_optimized_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
            )
            save_strategy_version(config_name, best_config)
            print(f"ðŸ’¾ Saved best config as: {config_name}")

        return best_configs

    def run_genetic_optimization(
        self,
        strategy_type: str = "rsi_ema_breakout",
        population_size: int = 20,
        generations: int = 10,
        mutation_rate: float = 0.3,
        save_best: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Run genetic algorithm optimization.

        Args:
            strategy_type: Type of strategy to optimize
            population_size: Size of the population
            generations: Number of generations to evolve
            mutation_rate: Probability of mutation
            save_best: Whether to save the best configuration

        Returns:
            List of best configurations found
        """
        print(f"ðŸ§¬ Starting Genetic Optimization for {strategy_type}")
        print(f"ðŸ‘¥ Population: {population_size} | Generations: {generations}")

        # Initialize population
        population = []
        for _ in range(population_size):
            config = self.generate_random_config(strategy_type)
            result = self.evaluate_config(config, strategy_type)
            population.append(result)

        best_configs = []

        for generation in range(generations):
            print(f"\nðŸ”„ Generation {generation + 1}/{generations}")

            # Sort population by fitness (profit)
            population.sort(key=lambda x: x["total_profit"], reverse=True)

            # Keep track of best
            generation_best = population[0]
            best_configs.append(generation_best)

            print(
                f"ðŸ† Best: ${generation_best['total_profit']:.2f} | "
                f"Win Rate: {generation_best['win_rate']:.1%} | "
                f"Sharpe: {generation_best['sharpe_ratio']:.3f}"
            )

            # Create new population
            new_population = []

            # Elitism: Keep top 20%
            elite_count = max(1, population_size // 5)
            new_population.extend(population[:elite_count])

            # Generate rest through crossover and mutation
            while len(new_population) < population_size:
                # Select parents (tournament selection)
                parent1 = random.choice(population[: population_size // 2])
                parent2 = random.choice(population[: population_size // 2])

                # Crossover
                child_config = self.crossover_configs(parent1["config"], parent2["config"])

                # Mutation
                if random.random() < mutation_rate:
                    child_config = self.mutate_config(child_config, mutation_rate)

                # Evaluate child
                child_result = self.evaluate_config(child_config, strategy_type)
                new_population.append(child_result)

            population = new_population

        print("\nðŸ† Genetic Optimization Complete!")
        print(f"âœ… Best configuration: ${best_configs[-1]['total_profit']:.2f}")

        if best_configs and save_best:
            best_config = best_configs[-1]
            config_name = (
                f"{strategy_type}_genetic_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
            )
            save_strategy_version(config_name, best_config)
            print(f"ðŸ’¾ Saved best config as: {config_name}")

        return best_configs

    def run_bayesian_optimization(
        self,
        strategy_type: str = "rsi_ema_breakout",
        rounds: int = 50,
        save_best: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Run Bayesian optimization (simplified version).

        Args:
            strategy_type: Type of strategy to optimize
            rounds: Number of optimization rounds
            save_best: Whether to save the best configuration

        Returns:
            List of best configurations found
        """
        print(f"ðŸ§¬ Starting Bayesian Optimization for {strategy_type}")
        print(f"ðŸ“Š Running {rounds} optimization rounds...")

        best_configs = []
        explored_configs = []

        for round_num in range(rounds):
            # Generate config based on exploration vs exploitation
            if round_num < rounds // 3:
                # Exploration phase: random configs
                config = self.generate_random_config(strategy_type)
            else:
                # Exploitation phase: mutate best configs
                if best_configs:
                    base_config = random.choice(best_configs[:3])["config"]
                    config = self.mutate_config(base_config, mutation_rate=0.2)
                else:
                    config = self.generate_random_config(strategy_type)

            # Evaluate
            result = self.evaluate_config(config, strategy_type)
            explored_configs.append(result)

            print(
                f"[ROUND {round_num + 1:2d}] Profit: ${result['total_profit']:8.2f} | "
                f"Win Rate: {result['win_rate']:.1%} | "
                f"Sharpe: {result['sharpe_ratio']:.3f}"
            )

            # Track best
            if result["total_profit"] > 0:
                best_configs.append(result)
                best_configs.sort(key=lambda x: x["total_profit"], reverse=True)
                best_configs = best_configs[:10]

        print("\nðŸ† Bayesian Optimization Complete!")

        if not best_configs:
            print("âŒ No profitable configurations found.")
            return best_configs

        print(f"âœ… Best configuration: ${best_configs[0]['total_profit']:.2f}")

        if best_configs and save_best:
            best_config = best_configs[0]
            config_name = (
                f"{strategy_type}_bayesian_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
            )
            save_strategy_version(config_name, best_config)
            print(f"ðŸ’¾ Saved best config as: {config_name}")

        return best_configs

    def optimize_strategy(
        self,
        strategy_type: str = "rsi_ema_breakout",
        method: str = "genetic",
        **kwargs,
    ) -> dict[str, Any]:
        """
        Main optimization method that runs the specified algorithm.

        Args:
            strategy_type: Type of strategy to optimize
            method: Optimization method ('random', 'genetic', 'bayesian')
            **kwargs: Additional parameters for the optimization method

        Returns:
            Best configuration found
        """
        start_time = time.time()

        if method == "random":
            best_configs = self.run_random_search(strategy_type, **kwargs)
        elif method == "genetic":
            best_configs = self.run_genetic_optimization(strategy_type, **kwargs)
        elif method == "bayesian":
            best_configs = self.run_bayesian_optimization(strategy_type, **kwargs)
        else:
            raise ValueError(f"Unknown optimization method: {method}")

        optimization_time = time.time() - start_time

        if not best_configs:
            print("âŒ No profitable configurations found.")
            return None

        best_config = best_configs[0]
        print("\nðŸŽ¯ Optimization Summary:")
        print(f"â±ï¸  Time: {optimization_time:.1f} seconds")
        print(f"ðŸ’° Best Profit: ${best_config['total_profit']:.2f}")
        print(f"ðŸ“ˆ Win Rate: {best_config['win_rate']:.1%}")
        print(f"ðŸ“Š Sharpe Ratio: {best_config['sharpe_ratio']:.3f}")
        print(f"ðŸ”„ Trades: {best_config['trades_count']}")

        return best_config


# Convenience functions
def optimize_rsi_ema_breakout(method: str = "genetic", rounds: int = 50) -> dict[str, Any]:
    """Optimize RSI + EMA + Breakout strategy"""
    tuner = HyperparameterTuner()
    return tuner.optimize_strategy("rsi_ema_breakout", method, rounds=rounds)


def optimize_bollinger_bands(method: str = "genetic", rounds: int = 50) -> dict[str, Any]:
    """Optimize Bollinger Bands strategy"""
    tuner = HyperparameterTuner()
    return tuner.optimize_strategy("bollinger_bands", method, rounds=rounds)


def optimize_macd_crossover(method: str = "genetic", rounds: int = 50) -> dict[str, Any]:
    """Optimize MACD Crossover strategy"""
    tuner = HyperparameterTuner()
    return tuner.optimize_strategy("macd_crossover", method, rounds=rounds)


# Example usage
if __name__ == "__main__":
    print("ðŸ§¬ Hyperparameter Optimization Engine")
    print("=" * 50)

    # Test different optimization methods
    methods = ["random", "genetic", "bayesian"]

    for method in methods:
        print(f"\nðŸ”§ Testing {method.upper()} optimization...")
        try:
            result = optimize_rsi_ema_breakout(method, rounds=20)
            if result:
                print(f"âœ… {method.upper()} completed successfully")
        except Exception as e:
            print(f"âŒ {method.upper()} failed: {e}")

    print("\nðŸŽ‰ Optimization testing complete!")

