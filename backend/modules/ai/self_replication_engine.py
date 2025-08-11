"""
Self-Replication Engine for Mystic AI Trading Platform
Evolves trading strategies using genetic algorithms and agent performance optimization.
"""

import logging
import random
import numpy as np
import copy
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
import sys
import os

# Add backend to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from modules.ai.persistent_cache import PersistentCache
from modules.ai.signal_engine import SignalEngine

logger = logging.getLogger(__name__)


class TradingAgent:
    def __init__(self, agent_id: str, parameters: Dict[str, Any]):
        """Initialize a trading agent with specific parameters"""
        self.agent_id = agent_id
        self.parameters = parameters
        self.performance = {
            "total_pnl": 0.0,
            "win_rate": 0.0,
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
            "last_updated": datetime.now(timezone.utc).isoformat()
        }
        self.generation = 0
        self.parent_id = None

    def mutate_parameters(self, mutation_rate: float = 0.1) -> Dict[str, Any]:
        """Mutate agent parameters slightly"""
        mutated = copy.deepcopy(self.parameters)

        for key, value in mutated.items():
            if isinstance(value, (int, float)):
                # Add random noise
                noise = random.gauss(0, mutation_rate * abs(value))
                mutated[key] = value + noise

                # Ensure parameters stay within reasonable bounds
                if key == "rsi_oversold":
                    mutated[key] = max(10, min(40, mutated[key]))
                elif key == "rsi_overbought":
                    mutated[key] = max(60, min(90, mutated[key]))
                elif key == "ema_fast":
                    mutated[key] = max(5, min(20, mutated[key]))
                elif key == "ema_slow":
                    mutated[key] = max(20, min(50, mutated[key]))
                elif key == "risk_factor":
                    mutated[key] = max(0.1, min(2.0, mutated[key]))
                elif key == "confidence_threshold":
                    mutated[key] = max(0.3, min(0.9, mutated[key]))

        return mutated

    def calculate_fitness(self) -> float:
        """Calculate agent fitness score based on performance"""
        try:
            # Base fitness on PnL and win rate
            pnl_weight = 0.6
            win_rate_weight = 0.3
            sharpe_weight = 0.1

            # Normalize PnL (assume max PnL of 1000 for normalization)
            normalized_pnl = min(self.performance["total_pnl"] / 1000.0, 1.0)

            # Calculate fitness
            fitness = (
                pnl_weight * normalized_pnl +
                win_rate_weight * self.performance["win_rate"] +
                sharpe_weight * max(0, self.performance["sharpe_ratio"])
            )

            return max(0.0, fitness)

        except Exception as e:
            logger.error(f"Failed to calculate fitness for agent {self.agent_id}: {e}")
            return 0.0

    def update_performance(self, trade_result: Dict[str, Any]):
        """Update agent performance with trade result"""
        try:
            pnl = trade_result.get("pnl", 0.0)
            self.performance["total_pnl"] += pnl

            if trade_result.get("success", False):
                self.performance["total_trades"] += 1

                if pnl > 0:
                    self.performance["winning_trades"] += 1
                else:
                    self.performance["losing_trades"] += 1

            # Update win rate
            if self.performance["total_trades"] > 0:
                self.performance["win_rate"] = (
                    self.performance["winning_trades"] / self.performance["total_trades"]
                )

            # Update max drawdown
            if pnl < 0:
                self.performance["max_drawdown"] = min(
                    self.performance["max_drawdown"], pnl
                )

            # Update Sharpe ratio (simplified calculation)
            if self.performance["total_trades"] > 0:
                avg_return = self.performance["total_pnl"] / self.performance["total_trades"]
                self.performance["sharpe_ratio"] = avg_return / max(1.0, abs(self.performance["max_drawdown"]))

            self.performance["last_updated"] = datetime.now(timezone.utc).isoformat()

        except Exception as e:
            logger.error(f"Failed to update performance for agent {self.agent_id}: {e}")


class SelfReplicationEngine:
    def __init__(self):
        """Initialize the self-replication engine"""
        self.agents: Dict[str, TradingAgent] = {}
        self.generation_size = 20
        self.elite_size = 4
        self.crossover_rate = 0.7
        self.mutation_rate = 0.1
        self.evolution_interval = 100  # Evolve every 100 trades
        self.trade_count = 0
        
        # Dashboard tracking attributes
        self.spawned_versions = 0
        self.active_instances = 0
        self.last_replication_time = datetime.now(timezone.utc).isoformat()
        self.replication_success_rate = 0.0

        # Initialize population
        self._initialize_population()

        logger.info("✅ SelfReplicationEngine initialized")

    def _initialize_population(self):
        """Initialize the initial agent population"""
        try:
            for i in range(self.generation_size):
                # Create random parameters for each agent
                parameters = {
                    "rsi_oversold": random.uniform(20, 35),
                    "rsi_overbought": random.uniform(65, 80),
                    "ema_fast": random.randint(8, 16),
                    "ema_slow": random.randint(20, 40),
                    "risk_factor": random.uniform(0.5, 1.5),
                    "confidence_threshold": random.uniform(0.4, 0.8)
                }

                agent = TradingAgent(f"agent_gen0_{i}", parameters)
                self.agents[agent.agent_id] = agent

            logger.info(f"✅ Initialized population with {self.generation_size} agents")

        except Exception as e:
            logger.error(f"Failed to initialize population: {e}")

    def _select_parents(self) -> List[TradingAgent]:
        """Select parents for breeding using tournament selection"""
        try:
            tournament_size = 3
            parents = []

            for _ in range(self.generation_size - self.elite_size):
                # Select tournament participants
                tournament = random.sample(list(self.agents.values()), tournament_size)

                # Select winner based on fitness
                winner = max(tournament, key=lambda agent: agent.calculate_fitness())
                parents.append(winner)

            return parents

        except Exception as e:
            logger.error(f"Failed to select parents: {e}")
            return []

    def _crossover_parameters(self, parent1: TradingAgent, parent2: TradingAgent) -> Dict[str, Any]:
        """Perform crossover between two parent agents"""
        try:
            child_params = {}

            for key in parent1.parameters.keys():
                if random.random() < self.crossover_rate:
                    # Crossover: take parameter from either parent
                    child_params[key] = random.choice([
                        parent1.parameters[key],
                        parent2.parameters[key]
                    ])
                else:
                    # Average of both parents
                    child_params[key] = (
                        parent1.parameters[key] + parent2.parameters[key]
                    ) / 2

            return child_params

        except Exception as e:
            logger.error(f"Failed to crossover parameters: {e}")
            return parent1.parameters.copy()

    def _evolve_generation(self):
        """Evolve to the next generation"""
        try:
            # Sort agents by fitness
            sorted_agents = sorted(
                self.agents.values(),
                key=lambda agent: agent.calculate_fitness(),
                reverse=True
            )

            # Keep elite agents
            elite_agents = sorted_agents[:self.elite_size]

            # Select parents for breeding
            parents = self._select_parents()

            # Create new generation
            new_agents = {}
            generation = max(agent.generation for agent in self.agents.values()) + 1

            # Add elite agents to new generation
            for agent in elite_agents:
                new_agent = TradingAgent(
                    f"agent_gen{generation}_{len(new_agents)}",
                    agent.parameters.copy()
                )
                new_agent.generation = generation
                new_agent.parent_id = agent.agent_id
                new_agents[new_agent.agent_id] = new_agent

            # Breed new agents
            while len(new_agents) < self.generation_size:
                # Select two parents
                parent1, parent2 = random.sample(parents, 2)

                # Create child through crossover
                child_params = self._crossover_parameters(parent1, parent2)

                # Mutate child
                child_params = self._mutate_parameters(child_params)

                # Create child agent
                child_agent = TradingAgent(
                    f"agent_gen{generation}_{len(new_agents)}",
                    child_params
                )
                child_agent.generation = generation
                child_agent.parent_id = f"{parent1.agent_id}_{parent2.agent_id}"

                new_agents[child_agent.agent_id] = child_agent

            # Replace old generation
            self.agents = new_agents

            logger.info(f"✅ Evolved to generation {generation} with {len(new_agents)} agents")

        except Exception as e:
            logger.error(f"Failed to evolve generation: {e}")

    def _mutate_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Mutate parameters with random noise"""
        try:
            mutated = copy.deepcopy(parameters)

            for key, value in mutated.items():
                if isinstance(value, (int, float)):
                    # Add random noise
                    noise = random.gauss(0, self.mutation_rate * abs(value))
                    mutated[key] = value + noise

                    # Ensure parameters stay within reasonable bounds
                    if key == "rsi_oversold":
                        mutated[key] = max(10, min(40, mutated[key]))
                    elif key == "rsi_overbought":
                        mutated[key] = max(60, min(90, mutated[key]))
                    elif key == "ema_fast":
                        mutated[key] = max(5, min(20, mutated[key]))
                    elif key == "ema_slow":
                        mutated[key] = max(20, min(50, mutated[key]))
                    elif key == "risk_factor":
                        mutated[key] = max(0.1, min(2.0, mutated[key]))
                    elif key == "confidence_threshold":
                        mutated[key] = max(0.3, min(0.9, mutated[key]))

            return mutated

        except Exception as e:
            logger.error(f"Failed to mutate parameters: {e}")
            return parameters

    def _evaluate_agent(self, agent: TradingAgent, symbol: str) -> Dict[str, Any]:
        """Evaluate an agent's performance on historical data"""
        try:
            # Get price history from cache
            cache = PersistentCache()
            price_history = cache.get_price_history('aggregated', symbol, limit=100)

            if not price_history or len(price_history) < 50:
                return {
                    "agent_id": agent.agent_id,
                    "success": False,
                    "reason": "Insufficient price data"
                }

            # Simulate trading with agent parameters
            trades = []
            position = 0
            entry_price = 0

            for i in range(20, len(price_history)):  # Start after enough data for indicators
                current_price = float(price_history[i]["price"])
                signal = self._generate_agent_signal(agent, price_history[:i+1])

                if signal == "BUY" and position <= 0:
                    position = 1
                    entry_price = current_price
                elif signal == "SELL" and position >= 0:
                    position = -1
                    entry_price = current_price
                elif signal == "CLOSE" and position != 0:
                    # Close position
                    pnl = (current_price - entry_price) * position
                    trades.append({
                        "success": True,
                        "pnl": pnl,
                        "entry_price": entry_price,
                        "exit_price": current_price,
                        "position": position
                    })
                    position = 0

            # Close any remaining position
            if position != 0:
                current_price = float(price_history[-1]["price"])
                pnl = (current_price - entry_price) * position
                trades.append({
                    "success": True,
                    "pnl": pnl,
                    "entry_price": entry_price,
                    "exit_price": current_price,
                    "position": position
                })

            # Update agent performance
            for trade in trades:
                agent.update_performance(trade)

            return {
                "agent_id": agent.agent_id,
                "success": True,
                "trades": trades,
                "total_pnl": sum(trade["pnl"] for trade in trades),
                "trade_count": len(trades)
            }

        except Exception as e:
            logger.error(f"Failed to evaluate agent {agent.agent_id}: {e}")
            return {
                "agent_id": agent.agent_id,
                "success": False,
                "error": str(e)
            }

    def _generate_agent_signal(self, agent: TradingAgent, price_history: List[Dict]) -> str:
        """Generate trading signal using agent parameters"""
        try:
            if len(price_history) < 30:
                return "HOLD"

            # Extract prices
            prices = [float(p["price"]) for p in price_history]

            # Calculate indicators using agent parameters
            rsi_period = int(agent.parameters.get("rsi_period", 14))
            ema_fast = int(agent.parameters.get("ema_fast", 12))
            ema_slow = int(agent.parameters.get("ema_slow", 26))

            rsi_values = self._calculate_rsi(prices, rsi_period)
            ema_fast_values = self._calculate_ema(prices, ema_fast)
            ema_slow_values = self._calculate_ema(prices, ema_slow)

            if len(rsi_values) == 0 or len(ema_fast_values) == 0 or len(ema_slow_values) == 0:
                return "HOLD"

            # Get current values
            current_rsi = rsi_values[-1]
            current_ema_fast = ema_fast_values[-1]
            current_ema_slow = ema_slow_values[-1]
            current_price = prices[-1]

            # Generate signal based on agent parameters
            rsi_oversold = agent.parameters.get("rsi_oversold", 30)
            rsi_overbought = agent.parameters.get("rsi_overbought", 70)
            confidence_threshold = agent.parameters.get("confidence_threshold", 0.6)

            # Calculate signal confidence based on price volatility
            price_volatility = abs(current_price - prices[-2]) / prices[-2] if len(prices) > 1 else 0.0
            signal_confidence = max(0.0, min(1.0, 1.0 - price_volatility))

            # RSI signals with confidence threshold
            if current_rsi < rsi_oversold and signal_confidence >= confidence_threshold:
                return "BUY"
            elif current_rsi > rsi_overbought and signal_confidence >= confidence_threshold:
                return "SELL"

            # EMA crossover signals with confidence threshold
            if current_ema_fast > current_ema_slow and signal_confidence >= confidence_threshold:
                return "BUY"
            elif current_ema_fast < current_ema_slow and signal_confidence >= confidence_threshold:
                return "SELL"

            return "HOLD"

        except Exception as e:
            logger.error(f"Failed to generate signal for agent {agent.agent_id}: {e}")
            return "HOLD"

    def _calculate_rsi(self, prices: List[float], period: int = 14) -> List[float]:
        """Calculate RSI values"""
        try:
            if len(prices) < period + 1:
                return []

            rsi_values = []
            gains = []
            losses = []

            # Calculate initial gains and losses
            for i in range(1, len(prices)):
                change = prices[i] - prices[i-1]
                gains.append(max(change, 0))
                losses.append(max(-change, 0))

            # Calculate initial average gain and loss
            avg_gain = sum(gains[:period]) / period
            avg_loss = sum(losses[:period]) / period

            # Calculate first RSI
            if avg_loss == 0:
                rsi = 100
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            rsi_values.append(rsi)

            # Calculate remaining RSI values
            for i in range(period, len(gains)):
                avg_gain = (avg_gain * (period - 1) + gains[i]) / period
                avg_loss = (avg_loss * (period - 1) + losses[i]) / period

                if avg_loss == 0:
                    rsi = 100
                else:
                    rs = avg_gain / avg_loss
                    rsi = 100 - (100 / (1 + rs))
                rsi_values.append(rsi)

            return rsi_values

        except Exception as e:
            logger.error(f"Failed to calculate RSI: {e}")
            return []

    def _calculate_ema(self, prices: List[float], period: int) -> List[float]:
        """Calculate EMA values"""
        try:
            if len(prices) < period:
                return []

            ema_values = []
            multiplier = 2 / (period + 1)

            # First EMA is SMA
            sma = sum(prices[:period]) / period
            ema_values.append(sma)

            # Calculate remaining EMA values
            for i in range(period, len(prices)):
                ema = (prices[i] * multiplier) + (ema_values[-1] * (1 - multiplier))
                ema_values.append(ema)

            return ema_values

        except Exception as e:
            logger.error(f"Failed to calculate EMA: {e}")
            return []

    def train_agents(self, exchange: str, symbol: str) -> Dict[str, Any]:
        """Train agents on historical data"""
        try:
            evaluations = []

            # Evaluate all agents
            for agent in self.agents.values():
                evaluation = self._evaluate_agent(agent, symbol)
                evaluations.append(evaluation)

            # Update trade count
            self.trade_count += sum(eval.get("trade_count", 0) for eval in evaluations if eval.get("success"))

            # Evolve if enough trades have been made
            if self.trade_count >= self.evolution_interval:
                self._evolve_generation()
                self.trade_count = 0

            # Get best agent
            best_agent = self.get_best_agent()

            return {
                "success": True,
                "symbol": symbol,
                "exchange": exchange,
                "agents_evaluated": len(evaluations),
                "best_agent": best_agent,
                "generation": max(agent.generation for agent in self.agents.values()),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            logger.error(f"Failed to train agents: {e}")
            return {"success": False, "error": str(e)}

    def get_best_agent(self) -> Optional[Dict[str, Any]]:
        """Get the best performing agent"""
        try:
            if not self.agents:
                return None

            # Find agent with highest fitness
            best_agent = max(self.agents.values(), key=lambda agent: agent.calculate_fitness())

            return {
                "agent_id": best_agent.agent_id,
                "parameters": best_agent.parameters,
                "performance": best_agent.performance,
                "fitness": best_agent.calculate_fitness(),
                "generation": best_agent.generation
            }

        except Exception as e:
            logger.error(f"Failed to get best agent: {e}")
            return None

    def get_agent_population(self) -> Dict[str, Any]:
        """Get current agent population statistics"""
        try:
            if not self.agents:
                return {"success": False, "reason": "No agents available"}

            fitness_scores = [agent.calculate_fitness() for agent in self.agents.values()]
            performances = [agent.performance for agent in self.agents.values()]

            return {
                "success": True,
                "population_size": len(self.agents),
                "generation": max(agent.generation for agent in self.agents.values()),
                "average_fitness": np.mean(fitness_scores),
                "max_fitness": max(fitness_scores),
                "min_fitness": min(fitness_scores),
                "average_pnl": np.mean([p["total_pnl"] for p in performances]),
                "average_win_rate": np.mean([p["win_rate"] for p in performances]),
                "agents": [
                    {
                        "agent_id": agent.agent_id,
                        "fitness": agent.calculate_fitness(),
                        "performance": agent.performance,
                        "generation": agent.generation
                    }
                    for agent in self.agents.values()
                ]
            }

        except Exception as e:
            logger.error(f"Failed to get agent population: {e}")
            return {"success": False, "error": str(e)}

    def update_replication_stats(self, success: bool = True):
        """Update replication statistics for dashboard"""
        try:
            self.spawned_versions += 1
            self.active_instances = len(self.agents)
            self.last_replication_time = datetime.now(timezone.utc).isoformat()
            
            # Calculate success rate (simplified)
            if success:
                self.replication_success_rate = min(1.0, self.replication_success_rate + 0.1)
            else:
                self.replication_success_rate = max(0.0, self.replication_success_rate - 0.05)
                
        except Exception as e:
            logger.error(f"Failed to update replication stats: {e}")

    def get_replication_stats(self) -> Dict[str, Any]:
        """Get replication statistics for dashboard"""
        try:
            return {
                "spawned_versions": self.spawned_versions,
                "active_instances": self.active_instances,
                "last_replication_time": self.last_replication_time,
                "replication_success_rate": self.replication_success_rate
            }
        except Exception as e:
            logger.error(f"Failed to get replication stats: {e}")
            return {
                "spawned_versions": 0,
                "active_instances": 0,
                "last_replication_time": datetime.now(timezone.utc).isoformat(),
                "replication_success_rate": 0.0
            }

    def get_signal_engine_reference(self) -> SignalEngine:
        """Get reference to SignalEngine for integration"""
        try:
            # Create SignalEngine instance for integration
            signal_engine = SignalEngine()
            return signal_engine
        except Exception as e:
            logger.error(f"Failed to get SignalEngine reference: {e}")
            return None


# Global self-replication engine instance
self_replication_engine = SelfReplicationEngine()


def get_self_replication_engine() -> SelfReplicationEngine:
    """Get the global self-replication engine instance"""
    return self_replication_engine


if __name__ == "__main__":
    # Test the self-replication engine
    engine = SelfReplicationEngine()
    print(f"✅ SelfReplicationEngine initialized: {engine}")

    # Test agent training
    result = engine.train_agents('coinbase', 'BTC-USD')
    print(f"Training result: {result}")

    # Test best agent
    best_agent = engine.get_best_agent()
    print(f"Best agent: {best_agent}")

    # Test population stats
    population = engine.get_agent_population()
    print(f"Population stats: {population}")
