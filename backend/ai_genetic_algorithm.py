"""
Genetic Algorithm Engine for Strategy Evolution
Advanced genetic algorithm implementation for trading strategy optimization
"""

import asyncio
import json
import os
import redis
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Tuple
import random
import copy
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load environment variables
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))


@dataclass
class StrategyGene:
    """Individual strategy gene representation"""

    id: str
    parameters: Dict[str, Any]
    fitness: float = 0.0
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    mutation_count: int = 0
    crossover_count: int = 0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    performance_history: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "parameters": self.parameters,
            "fitness": self.fitness,
            "generation": self.generation,
            "parent_ids": self.parent_ids,
            "mutation_count": self.mutation_count,
            "crossover_count": self.crossover_count,
            "created_at": self.created_at,
            "performance_history": self.performance_history,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StrategyGene":
        """Create from dictionary"""
        return cls(**data)


class GeneticAlgorithmEngine:
    def __init__(self):
        """Initialize Genetic Algorithm Engine"""
        self.redis_client = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            db=int(os.getenv("REDIS_DB", 0)),
            decode_responses=True,
        )
        self.running = False
        self.population: List[StrategyGene] = []
        self.generation = 0
        self.best_fitness = 0.0
        self.evolution_history = []

        # Genetic algorithm parameters
        self.population_size = 50
        self.mutation_rate = 0.1
        self.crossover_rate = 0.8
        self.elite_size = 5
        self.max_generations = 100
        self.fitness_threshold = 0.8

        # Strategy parameter ranges
        self.parameter_ranges = {
            "rsi_period": (10, 30),
            "rsi_oversold": (20, 40),
            "rsi_overbought": (60, 80),
            "sma_short": (5, 25),
            "sma_long": (20, 100),
            "macd_fast": (8, 16),
            "macd_slow": (20, 32),
            "macd_signal": (5, 15),
            "bb_period": (10, 30),
            "bb_std": (1.5, 3.0),
            "volume_sma": (10, 30),
            "stop_loss": (0.02, 0.10),
            "take_profit": (0.05, 0.20),
            "position_size": (0.05, 0.25),
            "max_positions": (1, 5),
        }

    async def start(self):
        """Start the Genetic Algorithm Engine"""
        print("ðŸ§¬ Starting Genetic Algorithm Engine...")
        self.running = True

        # Initialize population
        await self.initialize_population()

        # Start evolution process
        await self.evolve_population()

    async def initialize_population(self):
        """Initialize the initial population"""
        print(f"ðŸŽ¯ Initializing population of {self.population_size} strategies...")

        self.population = []
        for i in range(self.population_size):
            gene = self.create_random_gene(f"GENE_{self.generation}_{i}")
            self.population.append(gene)

        # Store initial population
        await self.store_population()
        print(f"âœ… Initialized {len(self.population)} strategies")

    def create_random_gene(self, gene_id: str) -> StrategyGene:
        """Create a random strategy gene"""
        parameters = {}

        for param_name, (min_val, max_val) in self.parameter_ranges.items():
            if isinstance(min_val, int) and isinstance(max_val, int):
                parameters[param_name] = random.randint(min_val, max_val)
            else:
                parameters[param_name] = round(random.uniform(min_val, max_val), 3)

        return StrategyGene(id=gene_id, parameters=parameters, generation=self.generation)

    async def evolve_population(self):
        """Main evolution loop"""
        print("ðŸ”„ Starting population evolution...")

        while self.running and self.generation < self.max_generations:
            try:
                print(f"\nðŸ§¬ Generation {self.generation + 1}/{self.max_generations}")

                # Evaluate fitness
                await self.evaluate_population()

                # Check termination conditions
                if self.best_fitness >= self.fitness_threshold:
                    print(f"ðŸŽ¯ Target fitness reached: {self.best_fitness:.4f}")
                    break

                # Create next generation
                await self.create_next_generation()

                # Store evolution data
                await self.store_evolution_data()

                # Wait before next generation
                await asyncio.sleep(60)  # 1 minute between generations

            except Exception as e:
                print(f"âŒ Error in evolution: {e}")
                await asyncio.sleep(300)

    async def evaluate_population(self):
        """Evaluate fitness of all individuals in population"""
        print("ðŸ“Š Evaluating population fitness...")

        evaluation_tasks = []
        for gene in self.population:
            task = self.evaluate_gene(gene)
            evaluation_tasks.append(task)

        # Run evaluations concurrently
        results = await asyncio.gather(*evaluation_tasks, return_exceptions=True)

        # Update fitness scores
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"âŒ Evaluation error for {self.population[i].id}: {result}")
                self.population[i].fitness = 0.0
            else:
                self.population[i].fitness = result

        # Sort population by fitness
        self.population.sort(key=lambda x: x.fitness, reverse=True)

        # Update best fitness
        if self.population:
            self.best_fitness = self.population[0].fitness

        print(f"ðŸ† Best fitness: {self.best_fitness:.4f}")
        print(f"ðŸ“ˆ Average fitness: {np.mean([g.fitness for g in self.population]):.4f}")

    async def evaluate_gene(self, gene: StrategyGene) -> float:
        """Evaluate fitness of a single gene"""
        try:
            # Simulate backtest with gene parameters
            performance = await self.simulate_backtest(gene.parameters)

            # Calculate fitness score
            fitness = self.calculate_fitness(performance)

            # Store performance history
            gene.performance_history.append(
                {
                    "generation": self.generation,
                    "performance": performance,
                    "fitness": fitness,
                    "timestamp": datetime.now().isoformat(),
                }
            )

            return fitness

        except Exception as e:
            print(f"Error evaluating gene {gene.id}: {e}")
            return 0.0

    async def simulate_backtest(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate backtest with given parameters"""
        try:
            # Generate historical data
            data = self.generate_test_data()

            # Apply strategy with parameters
            trades = self.apply_strategy(data, parameters)

            # Calculate performance metrics
            performance = self.calculate_performance_metrics(trades, data)

            return performance

        except Exception as e:
            print(f"Error in backtest simulation: {e}")
            return {
                "total_return": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "total_trades": 0,
            }

    def generate_test_data(self) -> pd.DataFrame:
        """Fetch real historical data for backtesting from Binance API"""
        try:
            import requests

            symbol = "BTCUSDT"
            url = f"https://api.binance.us/api/v3/klines?symbol={symbol}&interval=1h&limit=500"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            klines = response.json()
            closes = [float(k[4]) for k in klines]
            highs = [float(k[2]) for k in klines]
            lows = [float(k[3]) for k in klines]
            opens = [float(k[1]) for k in klines]
            volumes = [float(k[5]) for k in klines]
            df = pd.DataFrame(
                {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes}
            )
            return df
        except Exception as e:
            print(f"Error fetching historical data: {e}")
            return pd.DataFrame()

    def apply_strategy(
        self, data: pd.DataFrame, parameters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Apply trading strategy with given parameters"""
        trades = []
        position = None

        # Calculate indicators
        data["rsi"] = self.calculate_rsi(data["close"], parameters.get("rsi_period", 14))
        data["sma_short"] = data["close"].rolling(window=parameters.get("sma_short", 10)).mean()
        data["sma_long"] = data["close"].rolling(window=parameters.get("sma_long", 50)).mean()
        data["macd"] = self.calculate_macd(
            data["close"],
            parameters.get("macd_fast", 12),
            parameters.get("macd_slow", 26),
        )
        data["macd_signal"] = data["macd"].rolling(window=parameters.get("macd_signal", 9)).mean()
        data["volume_sma"] = data["volume"].rolling(window=parameters.get("volume_sma", 20)).mean()

        bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(
            data["close"],
            parameters.get("bb_period", 20),
            parameters.get("bb_std", 2),
        )
        data["bb_upper"] = bb_upper
        data["bb_lower"] = bb_lower

        # Strategy logic
        for i in range(100, len(data)):  # Start after enough data for indicators
            current_price = data["close"].iloc[i]

            # Entry conditions
            if position is None:  # No position
                # Buy signal
                if (
                    data["rsi"].iloc[i] < parameters.get("rsi_oversold", 30)
                    and data["sma_short"].iloc[i] > data["sma_long"].iloc[i]
                    and data["macd"].iloc[i] > data["macd_signal"].iloc[i]
                    and data["volume"].iloc[i] > data["volume_sma"].iloc[i] * 1.2
                ):

                    position = {
                        "entry_price": current_price,
                        "entry_time": data.index[i],
                        "size": parameters.get("position_size", 0.1),
                    }

            elif position is not None:  # Have position
                # Exit conditions
                stop_loss = position["entry_price"] * (1 - parameters.get("stop_loss", 0.05))
                take_profit = position["entry_price"] * (1 + parameters.get("take_profit", 0.10))

                if (
                    current_price <= stop_loss
                    or current_price >= take_profit
                    or data["rsi"].iloc[i] > parameters.get("rsi_overbought", 70)
                ):

                    # Close position
                    exit_price = current_price
                    pnl = (exit_price - position["entry_price"]) / position["entry_price"]

                    trades.append(
                        {
                            "entry_time": position["entry_time"],
                            "exit_time": data.index[i],
                            "entry_price": position["entry_price"],
                            "exit_price": exit_price,
                            "pnl": pnl,
                            "size": position["size"],
                        }
                    )

                    position = None

        return trades

    def calculate_performance_metrics(
        self, trades: List[Dict[str, Any]], data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Calculate performance metrics from trades"""
        if not trades:
            return {
                "total_return": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "total_trades": 0,
            }

        # Calculate metrics
        total_return = sum(trade["pnl"] for trade in trades)
        winning_trades = [t for t in trades if t["pnl"] > 0]
        losing_trades = [t for t in trades if t["pnl"] < 0]

        win_rate = len(winning_trades) / len(trades) if trades else 0

        total_profit = sum(t["pnl"] for t in winning_trades)
        total_loss = abs(sum(t["pnl"] for t in losing_trades))
        profit_factor = total_profit / total_loss if total_loss > 0 else float("inf")

        # Calculate Sharpe ratio (simplified)
        returns = [t["pnl"] for t in trades]
        sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0

        # Calculate max drawdown
        cumulative_returns = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = cumulative_returns - running_max
        max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0

        return {
            "total_return": total_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "total_trades": len(trades),
        }

    def calculate_fitness(self, performance: Dict[str, Any]) -> float:
        """Calculate fitness score from performance metrics"""
        try:
            # Weighted fitness calculation
            total_return_weight = 0.4
            sharpe_weight = 0.3
            win_rate_weight = 0.2
            drawdown_weight = 0.1

            # Normalize metrics
            total_return_score = min(performance["total_return"] * 10, 1.0)  # Cap at 10% return
            sharpe_score = min(max(performance["sharpe_ratio"] / 2, 0), 1.0)  # Normalize Sharpe
            win_rate_score = performance["win_rate"]
            drawdown_score = max(0, 1 + performance["max_drawdown"])  # Penalize drawdown

            # Calculate weighted fitness
            fitness = (
                total_return_score * total_return_weight
                + sharpe_score * sharpe_weight
                + win_rate_score * win_rate_weight
                + drawdown_score * drawdown_weight
            )

            return max(0, fitness)  # Ensure non-negative

        except Exception as e:
            print(f"Error calculating fitness: {e}")
            return 0.0

    async def create_next_generation(self):
        """Create the next generation using genetic operators"""
        print("ðŸ”„ Creating next generation...")

        new_population = []

        # Elitism: Keep best individuals
        elite = self.population[: self.elite_size]
        new_population.extend(elite)

        # Generate rest of population through crossover and mutation
        while len(new_population) < self.population_size:
            if random.random() < self.crossover_rate:
                # Crossover
                parent1 = self.select_parent()
                parent2 = self.select_parent()
                child = self.crossover(parent1, parent2)
            else:
                # Mutation
                parent = self.select_parent()
                child = self.mutate(parent)

            new_population.append(child)

        # Update population
        self.population = new_population[: self.population_size]
        self.generation += 1

        # Update generation numbers
        for gene in self.population:
            if gene.generation < self.generation:
                gene.generation = self.generation

        print(f"âœ… Created generation {self.generation} with {len(self.population)} individuals")

    def select_parent(self) -> StrategyGene:
        """Select parent using tournament selection"""
        tournament_size = 3
        tournament = random.sample(self.population, tournament_size)
        return max(tournament, key=lambda x: x.fitness)

    def crossover(self, parent1: StrategyGene, parent2: StrategyGene) -> StrategyGene:
        """Perform crossover between two parents"""
        child_params = {}

        for param_name in self.parameter_ranges.keys():
            if random.random() < 0.5:
                child_params[param_name] = parent1.parameters[param_name]
            else:
                child_params[param_name] = parent2.parameters[param_name]

        child = StrategyGene(
            id=f"GENE_{self.generation}_{len(self.population)}",
            parameters=child_params,
            generation=self.generation,
            parent_ids=[parent1.id, parent2.id],
            crossover_count=1,
        )

        parent1.crossover_count += 1
        parent2.crossover_count += 1

        return child

    def mutate(self, parent: StrategyGene) -> StrategyGene:
        """Perform mutation on parent"""
        child_params = copy.deepcopy(parent.parameters)

        # Mutate random parameters
        for param_name in self.parameter_ranges.keys():
            if random.random() < self.mutation_rate:
                min_val, max_val = self.parameter_ranges[param_name]

                if isinstance(min_val, int) and isinstance(max_val, int):
                    child_params[param_name] = random.randint(min_val, max_val)
                else:
                    child_params[param_name] = round(random.uniform(min_val, max_val), 3)

        child = StrategyGene(
            id=f"GENE_{self.generation}_{len(self.population)}",
            parameters=child_params,
            generation=self.generation,
            parent_ids=[parent.id],
            mutation_count=1,
        )

        parent.mutation_count += 1

        return child

    async def store_population(self):
        """Store current population in Redis"""
        try:
            population_data = [gene.to_dict() for gene in self.population]
            self.redis_client.set("genetic_population", json.dumps(population_data), ex=86400)

            # Store best individual
            if self.population:
                best_gene = self.population[0]
                self.redis_client.set("best_strategy", json.dumps(best_gene.to_dict()), ex=86400)

        except Exception as e:
            print(f"Error storing population: {e}")

    async def store_evolution_data(self):
        """Store evolution history data"""
        try:
            generation_data = {
                "generation": self.generation,
                "best_fitness": self.best_fitness,
                "average_fitness": np.mean([g.fitness for g in self.population]),
                "population_size": len(self.population),
                "timestamp": datetime.now().isoformat(),
            }

            self.evolution_history.append(generation_data)

            # Store in Redis
            self.redis_client.set(
                "evolution_history",
                json.dumps(self.evolution_history),
                ex=86400,
            )

        except Exception as e:
            print(f"Error storing evolution data: {e}")

    # Technical indicator calculations
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        return ema_fast - ema_slow

    def calculate_bollinger_bands(
        self, prices: pd.Series, period: int = 20, std_dev: float = 2
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, sma, lower

    async def stop(self):
        """Stop the Genetic Algorithm Engine"""
        print("ðŸ›‘ Stopping Genetic Algorithm Engine...")
        self.running = False

        # Store final population
        await self.store_population()


async def main():
    """Main function"""
    ga_engine = GeneticAlgorithmEngine()

    try:
        await ga_engine.start()
    except KeyboardInterrupt:
        print("ðŸ›‘ Received interrupt signal")
    except Exception as e:
        print(f"âŒ Error in main: {e}")
    finally:
        await ga_engine.stop()


if __name__ == "__main__":
    asyncio.run(main())


