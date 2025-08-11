import numpy as np
import time
import json
import sqlite3
import random
from datetime import datetime
from typing import Dict, List, Tuple, Any
import ast
import hashlib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enhanced configuration
TRAINER_DB = "./data/mutation_trainer.db"
TRAINER_INTERVAL = 7200  # 2 hours
POPULATION_SIZE = 50
GENERATIONS = 10
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.8
ELITE_SIZE = 5


class TrainerDatabase:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize trainer database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS genetic_population (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                generation INTEGER NOT NULL,
                individual_id TEXT NOT NULL,
                strategy_code TEXT NOT NULL,
                fitness_score REAL NOT NULL,
                mutation_history TEXT NOT NULL,
                parent_ids TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS evolution_progress (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                generation INTEGER NOT NULL,
                avg_fitness REAL NOT NULL,
                best_fitness REAL NOT NULL,
                worst_fitness REAL NOT NULL,
                diversity_score REAL NOT NULL,
                convergence_rate REAL NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS mutation_operations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                operation_type TEXT NOT NULL,
                parent_id TEXT NOT NULL,
                child_id TEXT NOT NULL,
                mutation_details TEXT NOT NULL,
                fitness_improvement REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        conn.commit()
        conn.close()

    def save_individual(self, individual: Dict):
        """Save genetic individual to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO genetic_population
            (generation, individual_id, strategy_code, fitness_score, mutation_history, parent_ids)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (
                individual["generation"],
                individual["individual_id"],
                individual["strategy_code"],
                individual["fitness_score"],
                json.dumps(individual["mutation_history"]),
                json.dumps(individual.get("parent_ids", [])),
            ),
        )

        conn.commit()
        conn.close()

    def save_evolution_progress(self, progress: Dict):
        """Save evolution progress to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO evolution_progress
            (generation, avg_fitness, best_fitness, worst_fitness, diversity_score, convergence_rate)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (
                progress["generation"],
                progress["avg_fitness"],
                progress["best_fitness"],
                progress["worst_fitness"],
                progress["diversity_score"],
                progress["convergence_rate"],
            ),
        )

        conn.commit()
        conn.close()

    def save_mutation_operation(self, operation: Dict):
        """Save mutation operation to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO mutation_operations
            (timestamp, operation_type, parent_id, child_id, mutation_details, fitness_improvement)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (
                operation["timestamp"],
                operation["operation_type"],
                operation["parent_id"],
                operation["child_id"],
                json.dumps(operation["mutation_details"]),
                operation.get("fitness_improvement", 0),
            ),
        )

        conn.commit()
        conn.close()


def generate_base_strategy() -> str:
    """Generate a base strategy template"""
    return """
def base_strategy(df):
    import pandas as pd

    # Calculate basic indicators
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()

    # Calculate RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # Generate signals
    df['signal'] = 0
    df.loc[(df['close'] > df['sma_20']) & (df['rsi'] < 70), 'signal'] = 1
    df.loc[(df['close'] < df['sma_20']) & (df['rsi'] > 30), 'signal'] = -1

    return df
"""


def parse_strategy_code(code: str) -> Dict[str, Any]:
    """Parse strategy code into components"""
    try:
        tree = ast.parse(code)
        components = {
            "imports": [],
            "indicators": [],
            "conditions": [],
            "signals": [],
        }

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                components["imports"].append([alias.name for alias in node.names])
            elif isinstance(node, ast.ImportFrom):
                components["imports"].append(
                    f"{node.module}.{[alias.name for alias in node.names]}"
                )
            elif isinstance(node, ast.Call):
                if hasattr(node, "func") and hasattr(node.func, "id"):
                    components["indicators"].append(node.func.id)

        return components
    except Exception as e:
        logger.error(f"Error parsing strategy code: {e}")
        return {
            "imports": [],
            "indicators": [],
            "conditions": [],
            "signals": [],
        }


def mutate_strategy_code(code: str, mutation_type: str = "random") -> str:
    """Apply mutations to strategy code"""
    try:
        lines = code.split("\n")
        mutated_lines = lines.copy()

        if mutation_type == "parameter":
            # Mutate parameters
            for i, line in enumerate(mutated_lines):
                if "rolling(window=" in line:
                    # Mutate window size
                    old_window = int(line.split("window=")[1].split(")")[0])
                    new_window = max(5, min(100, old_window + random.randint(-10, 10)))
                    mutated_lines[i] = line.replace(f"window={old_window}", f"window={new_window}")
                elif "rsi() <" in line or "rsi() >" in line:
                    # Mutate RSI thresholds
                    if "<" in line:
                        threshold = int(line.split("<")[1].split(",")[0])
                        new_threshold = max(20, min(80, threshold + random.randint(-10, 10)))
                        mutated_lines[i] = line.replace(f"< {threshold}", f"< {new_threshold}")
                    elif ">" in line:
                        threshold = int(line.split(">")[1].split(",")[0])
                        new_threshold = max(20, min(80, threshold + random.randint(-10, 10)))
                        mutated_lines[i] = line.replace(f"> {threshold}", f"> {new_threshold}")

        elif mutation_type == "indicator":
            # Add or remove indicators
            indicator_options = [
                "ema_fast = df['close'].ewm(span=12).mean(); ema_slow = df['close'].ewm(span=26).mean(); df['macd'] = ema_fast - ema_slow",
                "sma = df['close'].rolling(window=20).mean(); std = df['close'].rolling(window=20).std(); df['bb_upper'] = sma + (std * 2)",
                "sma = df['close'].rolling(window=20).mean(); std = df['close'].rolling(window=20).std(); df['bb_lower'] = sma - (std * 2)",
                "lowest_low = df['low'].rolling(window=14).min(); highest_high = df['high'].rolling(window=14).max(); df['stoch'] = 100 * ((df['close'] - lowest_low) / (highest_high - lowest_low))",
            ]

            if random.random() < 0.5:
                # Add indicator
                insert_pos = random.randint(5, len(mutated_lines) - 5)
                new_indicator = random.choice(indicator_options)
                mutated_lines.insert(insert_pos, f"    {new_indicator}")
            else:
                # Remove indicator (if exists)
                for i, line in enumerate(mutated_lines):
                    if any(
                        indicator in line
                        for indicator in [
                            "macd",
                            "bb_upper",
                            "bb_lower",
                            "stoch",
                        ]
                    ):
                        if random.random() < 0.3:
                            mutated_lines[i] = f"    # {line.strip()}"  # Comment out

        elif mutation_type == "condition":
            # Mutate signal conditions
            for i, line in enumerate(mutated_lines):
                if "signal" in line and ("=" in line or ">" in line or "<" in line):
                    if random.random() < 0.3:
                        # Add additional condition
                        if "&" in line:
                            mutated_lines[i] = line.replace("&", "|")  # Change AND to OR
                        elif "|" in line:
                            mutated_lines[i] = line.replace("|", "&")  # Change OR to AND
                        else:
                            # Add volume condition
                            mutated_lines[i] = line.replace(
                                "signal'] = 1",
                                "signal'] = 1 & (df['volume'] > df['volume'].rolling(20).mean())",
                            )

        return "\n".join(mutated_lines)

    except Exception as e:
        logger.error(f"Mutation error: {e}")
        return code


def crossover_strategies(parent1_code: str, parent2_code: str) -> Tuple[str, str]:
    """Perform crossover between two parent strategies"""
    try:
        lines1 = parent1_code.split("\n")
        lines2 = parent2_code.split("\n")

        # Find crossover points
        min_len = min(len(lines1), len(lines2))
        if min_len < 10:
            return parent1_code, parent2_code

        crossover_point = random.randint(5, min_len - 5)

        # Create children
        child1_lines = lines1[:crossover_point] + lines2[crossover_point:]
        child2_lines = lines2[:crossover_point] + lines1[crossover_point:]

        # Clean up function structure
        child1_code = "\n".join(child1_lines)
        child2_code = "\n".join(child2_lines)

        # Ensure proper indentation
        child1_code = fix_indentation(child1_code)
        child2_code = fix_indentation(child2_code)

        return child1_code, child2_code

    except Exception as e:
        logger.error(f"Crossover error: {e}")
        return parent1_code, parent2_code


def fix_indentation(code: str) -> str:
    """Fix indentation in strategy code"""
    lines = code.split("\n")
    fixed_lines = []

    for line in lines:
        if line.strip().startswith("def "):
            fixed_lines.append(line.strip())
        elif line.strip().startswith("import ") or line.strip().startswith("from "):
            fixed_lines.append(line.strip())
        elif line.strip().startswith("#"):
            fixed_lines.append(line.strip())
        elif line.strip():
            fixed_lines.append("    " + line.strip())
        else:
            fixed_lines.append("")

    return "\n".join(fixed_lines)


def evaluate_fitness(strategy_code: str) -> float:
    """Evaluate fitness of a strategy"""
    try:
        # Parse strategy components
        components = parse_strategy_code(strategy_code)

        # Calculate fitness based on multiple factors
        fitness_score = 0.0

        # Code complexity (prefer simpler strategies)
        complexity_penalty = len(strategy_code.split("\n")) * 0.01
        fitness_score -= complexity_penalty

        # Indicator diversity (reward variety)
        unique_indicators = len(set(components["indicators"]))
        fitness_score += unique_indicators * 0.1

        # Code validity (syntax check)
        try:
            ast.parse(strategy_code)
            fitness_score += 0.5  # Valid syntax bonus
        except SyntaxError as e:
            logger.warning(f"Invalid syntax in strategy code: {e}")
            fitness_score -= 1.0  # Invalid syntax penalty

        # Strategy completeness
        if "signal" in strategy_code:
            fitness_score += 0.3
        if "def " in strategy_code:
            fitness_score += 0.2

        # Random performance simulation
        performance_bonus = random.uniform(-0.5, 1.0)
        fitness_score += performance_bonus

        return max(0.0, fitness_score)  # Ensure non-negative

    except Exception as e:
        logger.error(f"Fitness evaluation error: {e}")
        return 0.0


def create_individual(generation: int, strategy_code: str = None) -> Dict:
    """Create a genetic individual"""
    if strategy_code is None:
        strategy_code = generate_base_strategy()

    individual_id = hashlib.md5(strategy_code.encode()).hexdigest()[:8]
    fitness_score = evaluate_fitness(strategy_code)

    return {
        "generation": generation,
        "individual_id": individual_id,
        "strategy_code": strategy_code,
        "fitness_score": fitness_score,
        "mutation_history": [],
        "parent_ids": [],
    }


def select_parents(population: List[Dict], tournament_size: int = 3) -> Tuple[Dict, Dict]:
    """Select parents using tournament selection"""
    if len(population) < 2:
        return population[0], population[0]

    # Tournament selection
    tournament1 = random.sample(population, min(tournament_size, len(population)))
    tournament2 = random.sample(population, min(tournament_size, len(population)))

    parent1 = max(tournament1, key=lambda x: x["fitness_score"])
    parent2 = max(tournament2, key=lambda x: x["fitness_score"])

    return parent1, parent2


def evolve_population(population: List[Dict], generation: int) -> List[Dict]:
    """Evolve population using genetic operators"""
    new_population = []

    # Elitism: keep best individuals
    sorted_population = sorted(population, key=lambda x: x["fitness_score"], reverse=True)
    elite = sorted_population[:ELITE_SIZE]
    new_population.extend(elite)

    # Generate rest of population
    while len(new_population) < POPULATION_SIZE:
        if random.random() < CROSSOVER_RATE:
            # Crossover
            parent1, parent2 = select_parents(population)
            child1_code, child2_code = crossover_strategies(
                parent1["strategy_code"], parent2["strategy_code"]
            )

            child1 = create_individual(generation, child1_code)
            child2 = create_individual(generation, child2_code)

            child1["parent_ids"] = [
                parent1["individual_id"],
                parent2["individual_id"],
            ]
            child2["parent_ids"] = [
                parent1["individual_id"],
                parent2["individual_id"],
            ]

            new_population.extend([child1, child2])
        else:
            # Mutation
            parent = random.choice(population)
            mutation_types = ["parameter", "indicator", "condition"]
            mutation_type = random.choice(mutation_types)

            mutated_code = mutate_strategy_code(parent["strategy_code"], mutation_type)
            child = create_individual(generation, mutated_code)
            child["parent_ids"] = [parent["individual_id"]]
            child["mutation_history"] = parent["mutation_history"] + [mutation_type]

            new_population.append(child)

    return new_population[:POPULATION_SIZE]


def calculate_population_metrics(population: List[Dict]) -> Dict:
    """Calculate population metrics"""
    fitness_scores = [ind["fitness_score"] for ind in population]

    return {
        "avg_fitness": np.mean(fitness_scores),
        "best_fitness": np.max(fitness_scores),
        "worst_fitness": np.min(fitness_scores),
        "diversity_score": np.std(fitness_scores),
        "convergence_rate": (
            1.0 - (np.std(fitness_scores) / np.mean(fitness_scores))
            if np.mean(fitness_scores) > 0
            else 0
        ),
    }


def train_mutations_enhanced():
    """Enhanced mutation training with all features"""
    try:
        db = TrainerDatabase(TRAINER_DB)

        # Initialize population
        population = []
        for _ in range(POPULATION_SIZE):
            individual = create_individual(0)
            population.append(individual)
            db.save_individual(individual)

        logger.info(f"[Trainer] Initialized population of {POPULATION_SIZE} individuals")

        # Evolution loop
        for generation in range(GENERATIONS):
            logger.info(f"[Trainer] Starting generation {generation + 1}/{GENERATIONS}")

            # Calculate population metrics
            metrics = calculate_population_metrics(population)
            metrics["generation"] = generation + 1
            db.save_evolution_progress(metrics)

            logger.info(f"[Trainer] Generation {generation + 1} metrics:")
            logger.info(f"[Trainer] Avg fitness: {metrics['avg_fitness']:.3f}")
            logger.info(f"[Trainer] Best fitness: {metrics['best_fitness']:.3f}")
            logger.info(f"[Trainer] Diversity: {metrics['diversity_score']:.3f}")

            # Evolve population
            new_population = evolve_population(population, generation + 1)

            # Save new individuals
            for individual in new_population:
                db.save_individual(individual)

            # Track mutations
            for individual in new_population:
                if individual["parent_ids"]:
                    for parent_id in individual["parent_ids"]:
                        operation = {
                            "timestamp": (datetime.timezone.utcnow().isoformat()),
                            "operation_type": (
                                "mutation" if len(individual["parent_ids"]) == 1 else "crossover"
                            ),
                            "parent_id": parent_id,
                            "child_id": individual["individual_id"],
                            "mutation_details": {
                                "mutation_history": individual["mutation_history"],
                                "fitness_score": individual["fitness_score"],
                            },
                            "fitness_improvement": (0),  # Would calculate based on parent fitness
                        }
                        db.save_mutation_operation(operation)

            population = new_population

            # Check for convergence
            if metrics["convergence_rate"] > 0.9:
                logger.info(f"[Trainer] Population converged at generation {generation + 1}")
                break

        # Final results
        best_individual = max(population, key=lambda x: x["fitness_score"])
        logger.info("[Trainer] Training complete!")
        logger.info(f"[Trainer] Best individual ID: {best_individual['individual_id']}")
        logger.info(f"[Trainer] Best fitness score: {best_individual['fitness_score']:.3f}")
        logger.info(
            f"[Trainer] Best strategy code length: {len(best_individual['strategy_code'])} characters"
        )

        # Save best strategy
        with open(
            f"./strategies/best_evolved_strategy_{datetime.timezone.utcnow().strftime('%Y%m%d_%H%M%S')}.py",
            "w",
        ) as f:
            f.write(best_individual["strategy_code"])

        logger.info("[Trainer] Best strategy saved to file")

    except Exception as e:
        logger.error(f"[Trainer] Enhanced training error: {e}")


# Main execution loop
while True:
    train_mutations_enhanced()
    time.sleep(TRAINER_INTERVAL)
