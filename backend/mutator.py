# mutator.py
import random
import uuid
import logging
from typing import List, Dict, Any, Optional
from db_logger import register_strategy, get_session
from reward_engine import get_top_performers
from models import Strategy
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

DB_URL = "sqlite:////data/trading_memory.db"
engine = create_engine(DB_URL, echo=False)
SessionLocal = sessionmaker(bind=engine)


def mutate_top_strategies(top_n: int = 3, mutation_count: int = 2) -> List[Dict[str, Any]]:
    """
    Create mutations of top performing strategies

    Args:
        top_n: Number of top strategies to mutate
        mutation_count: Number of mutations per strategy

    Returns:
        List of created mutations
    """
    top_strategies = get_top_performers(top_n=top_n, min_trades=5)
    mutations_created = []

    logger.info(f"Selected {len(top_strategies)} top strategies for mutation")

    for strat in top_strategies:
        for i in range(mutation_count):
            mutation_name = f"{strat['name']}_mut_{uuid.uuid4().hex[:4]}"
            mutation_desc = generate_mutation_description(strat["name"], strat)

            strategy_id = register_strategy(mutation_name, mutation_desc)

            if strategy_id:
                mutation_info = {
                    "id": strategy_id,
                    "name": mutation_name,
                    "description": mutation_desc,
                    "parent_strategy": strat["name"],
                    "parent_win_rate": strat["win_rate"],
                    "parent_avg_profit": strat["avg_profit"],
                }
                mutations_created.append(mutation_info)
                logger.info(f"Created mutation: {mutation_name} from {strat['name']}")

    logger.info(f"Created {len(mutations_created)} strategy mutations")
    return mutations_created


def generate_mutation_description(parent_name: str, parent_stats: Dict[str, Any]) -> str:
    """
    Generate a description for a mutated strategy

    Args:
        parent_name: Name of parent strategy
        parent_stats: Performance stats of parent strategy

    Returns:
        str: Generated description
    """
    mutations = [
        f"Enhanced version of {parent_name} with improved entry timing",
        f"Optimized {parent_name} with reduced risk parameters",
        f"Advanced {parent_name} with volume confirmation",
        f"Refined {parent_name} with momentum filters",
        f"Evolved {parent_name} with trend strength indicators",
        f"Improved {parent_name} with volatility adjustments",
        f"Enhanced {parent_name} with support/resistance levels",
        f"Optimized {parent_name} with RSI divergence",
        f"Advanced {parent_name} with MACD crossovers",
        f"Refined {parent_name} with Bollinger Band signals",
    ]

    base_mutation = random.choice(mutations)

    # Add performance context
    win_rate = parent_stats.get("win_rate", 0.0)
    avg_profit = parent_stats.get("avg_profit", 0.0)

    if win_rate > 0.6:
        performance_note = f" Based on high-performing parent (Win Rate: {win_rate:.1%}, Avg Profit: {avg_profit:.2f})"
    elif win_rate > 0.4:
        performance_note = f" Based on moderate-performing parent (Win Rate: {win_rate:.1%}, Avg Profit: {avg_profit:.2f})"
    else:
        performance_note = (
            f" Based on parent strategy (Win Rate: {win_rate:.1%}, Avg Profit: {avg_profit:.2f})"
        )

    return base_mutation + performance_note


def crossover_strategies(strategy1_id: int, strategy2_id: int) -> Optional[int]:
    """
    Create a new strategy by combining two existing strategies

    Args:
        strategy1_id: First strategy ID
        strategy2_id: Second strategy ID

    Returns:
        int: New strategy ID if successful, None otherwise
    """
    session = get_session()
    try:
        strat1 = session.query(Strategy).filter_by(id=strategy1_id).first()
        strat2 = session.query(Strategy).filter_by(id=strategy2_id).first()

        if not strat1 or not strat2:
            logger.error("One or both strategies not found for crossover")
            return None

        # Create crossover name and description
        crossover_name = f"Cross_{strat1.name}_{strat2.name}_{uuid.uuid4().hex[:4]}"
        crossover_desc = (
            f"Hybrid strategy combining {strat1.name} and {strat2.name}. "
            f"Takes best elements from both strategies: "
            f"{strat1.name} (Win Rate: {strat1.win_rate:.1%}, Avg Profit: {strat1.avg_profit:.2f}) and "
            f"{strat2.name} (Win Rate: {strat2.win_rate:.1%}, Avg Profit: {strat2.avg_profit:.2f})"
        )

        strategy_id = register_strategy(crossover_name, crossover_desc)

        if strategy_id:
            logger.info(f"Created crossover strategy: {crossover_name}")

        return strategy_id

    except Exception as e:
        logger.error(f"Failed to create crossover strategy: {e}")
        return None
    finally:
        session.close()


def create_random_strategy() -> Optional[int]:
    """
    Create a completely random strategy for exploration

    Returns:
        int: New strategy ID if successful, None otherwise
    """
    strategy_templates = [
        {
            "name": f"Random_EMA_{uuid.uuid4().hex[:4]}",
            "description": ("Random EMA crossover strategy with dynamic timeframes"),
        },
        {
            "name": f"Random_RSI_{uuid.uuid4().hex[:4]}",
            "description": ("Random RSI strategy with custom overbought/oversold levels"),
        },
        {
            "name": f"Random_BB_{uuid.uuid4().hex[:4]}",
            "description": ("Random Bollinger Bands strategy with volume confirmation"),
        },
        {
            "name": f"Random_MACD_{uuid.uuid4().hex[:4]}",
            "description": "Random MACD strategy with signal line crossovers",
        },
        {
            "name": f"Random_Vol_{uuid.uuid4().hex[:4]}",
            "description": ("Random volatility breakout strategy with ATR filters"),
        },
    ]

    template = random.choice(strategy_templates)
    strategy_id = register_strategy(template["name"], template["description"])

    if strategy_id:
        logger.info(f"Created random strategy: {template['name']}")

    return strategy_id


def evolve_strategy_population(
    mutation_rate: float = 0.3,
    crossover_rate: float = 0.2,
    random_rate: float = 0.1,
) -> Dict[str, Any]:
    """
    Evolve the entire strategy population

    Args:
        mutation_rate: Probability of mutating top strategies
        crossover_rate: Probability of creating crossovers
        random_rate: Probability of creating random strategies

    Returns:
        Dict with evolution results
    """
    evolution_results = {
        "mutations_created": 0,
        "crossovers_created": 0,
        "random_strategies_created": 0,
        "total_new_strategies": 0,
        "details": [],
    }

    # Create mutations
    if random.random() < mutation_rate:
        mutations = mutate_top_strategies(top_n=3, mutation_count=2)
        evolution_results["mutations_created"] = len(mutations)
        evolution_results["details"].extend([{"type": "mutation", "info": m} for m in mutations])

    # Create crossovers
    if random.random() < crossover_rate:
        top_strategies = get_top_performers(top_n=5, min_trades=5)
        if len(top_strategies) >= 2:
            # Create a few crossovers
            for _ in range(min(2, len(top_strategies) // 2)):
                strat1 = random.choice(top_strategies)
                strat2 = random.choice(top_strategies)
                if strat1["id"] != strat2["id"]:
                    crossover_id = crossover_strategies(strat1["id"], strat2["id"])
                    if crossover_id:
                        evolution_results["crossovers_created"] += 1
                        evolution_results["details"].append(
                            {
                                "type": "crossover",
                                "info": {
                                    "id": crossover_id,
                                    "parent1": strat1["name"],
                                    "parent2": strat2["name"],
                                },
                            }
                        )

    # Create random strategies
    if random.random() < random_rate:
        for _ in range(random.randint(1, 2)):
            random_id = create_random_strategy()
            if random_id:
                evolution_results["random_strategies_created"] += 1
                evolution_results["details"].append({"type": "random", "info": {"id": random_id}})

    evolution_results["total_new_strategies"] = (
        evolution_results["mutations_created"]
        + evolution_results["crossovers_created"]
        + evolution_results["random_strategies_created"]
    )

    logger.info(
        f"Evolution completed: {evolution_results['total_new_strategies']} new strategies created"
    )
    return evolution_results


def cleanup_poor_strategies(
    max_strategies: int = 50, min_win_rate: float = 0.3
) -> List[Dict[str, Any]]:
    """
    Clean up poor performing strategies to keep population manageable

    Args:
        max_strategies: Maximum number of strategies to keep
        min_win_rate: Minimum win rate to keep strategy

    Returns:
        List of deactivated strategies
    """
    session = get_session()
    try:
        # Get all strategies ordered by performance
        strategies = (
            session.query(Strategy)
            .filter(Strategy.is_active)
            .order_by(Strategy.win_rate.desc(), Strategy.avg_profit.desc())
            .all()
        )

        deactivated = []

        # Deactivate strategies that exceed max count or have poor performance
        for i, strat in enumerate(strategies):
            should_deactivate = False

            # Deactivate if too many strategies
            if i >= max_strategies:
                should_deactivate = True

            # Deactivate if poor performance and enough trades
            elif (
                strat.trades_executed >= 10
                and strat.win_rate < min_win_rate
                and strat.avg_profit < 0
            ):
                should_deactivate = True

            if should_deactivate:
                strat.is_active = False
                strat.updated_at = datetime.now(timezone.utc)
                deactivated.append(
                    {
                        "id": strat.id,
                        "name": strat.name,
                        "win_rate": strat.win_rate,
                        "avg_profit": strat.avg_profit,
                        "trades_executed": strat.trades_executed,
                        "reason": (
                            "population_limit" if i >= max_strategies else "poor_performance"
                        ),
                    }
                )

        session.commit()
        logger.info(f"Deactivated {len(deactivated)} poor performing strategies")
        return deactivated

    except Exception as e:
        session.rollback()
        logger.error(f"Failed to cleanup poor strategies: {e}")
        return []
    finally:
        session.close()


def run_evolution_cycle() -> Dict[str, Any]:
    """
    Run a complete evolution cycle (can be called by scheduler)

    Returns:
        Dict with evolution results
    """
    logger.info("Starting evolution cycle...")

    # Evolve population
    evolution_results = evolve_strategy_population()

    # Cleanup poor strategies
    deactivated = cleanup_poor_strategies()

    # Get current population stats
    session = get_session()
    try:
        total_strategies = session.query(Strategy).count()
        active_strategies = session.query(Strategy).filter_by(is_active=True).count()

        evolution_results["population_stats"] = {
            "total_strategies": total_strategies,
            "active_strategies": active_strategies,
            "deactivated_strategies": len(deactivated),
        }

        evolution_results["deactivated_strategies"] = deactivated

    finally:
        session.close()

    logger.info(
        f"Evolution cycle completed. Active strategies: {evolution_results['population_stats']['active_strategies']}"
    )
    return evolution_results


if __name__ == "__main__":
    import time
    import signal
    import sys

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info("ðŸ§¬ Strategy Mutator Service Starting...")

    def signal_handler(signum, frame):
        logger.info("ðŸ›‘ Received shutdown signal, stopping strategy mutator...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        logger.info("âœ… Strategy Mutator Service is running")
        logger.info("ðŸ”„ Will run evolution cycles every 30 minutes")

        cycle_count = 0

        # Keep the service running and perform evolution cycles
        while True:
            try:
                cycle_count += 1
                logger.info(f"ðŸ”„ Starting evolution cycle #{cycle_count}")

                results = run_evolution_cycle()

                logger.info(f"âœ… Evolution cycle #{cycle_count} completed:")
                logger.info(f"   - New strategies created: {results['total_new_strategies']}")
                logger.info(
                    f"   - Active strategies: {results['population_stats']['active_strategies']}"
                )
                logger.info(
                    f"   - Strategies deactivated: {len(results['deactivated_strategies'])}"
                )

                # Wait 30 minutes before next cycle
                logger.info("â° Waiting 30 minutes before next evolution cycle...")
                time.sleep(1800)  # 30 minutes

            except Exception as e:
                logger.error(f"âŒ Evolution cycle #{cycle_count} failed: {e}")
                logger.info("â° Waiting 5 minutes before retrying...")
                time.sleep(300)  # 5 minutes

    except KeyboardInterrupt:
        logger.info("ðŸ›‘ Strategy Mutator Service stopped by user")
    except Exception as e:
        logger.error(f"âŒ Strategy Mutator Service failed: {e}")
        sys.exit(1)


