"""
Neural Mesh for Mystic AI Trading Platform
Maintains a distributed shared agent learning network for parameter sharing and strategy merging.
"""

import copy
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any

# Add backend to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from backend.modules.ai.persistent_cache import PersistentCache

logger = logging.getLogger(__name__)


class NeuralMesh:
    def __init__(self):
        """Initialize neural mesh with distributed learning parameters"""
        self.cache = PersistentCache()

        # Mesh configuration
        self.mesh_update_interval = 3600  # 1 hour
        self.min_trades_for_evaluation = 20
        self.performance_weight_decay = 0.95
        self.parameter_merge_rate = 0.1

        # Agent tracking
        self.registered_agents = {}
        self.mesh_state = {
            "global_parameters": {},
            "performance_weights": {},
            "last_update": None,
            "mesh_version": 0
        }

        # Parameter categories for sharing
        self.parameter_categories = {
            "rsi_thresholds": ["rsi_oversold", "rsi_overbought"],
            "ema_weights": ["ema_fast", "ema_slow"],
            "risk_parameters": ["risk_factor", "confidence_threshold"],
            "trade_parameters": ["take_profit_percentage", "stop_loss_percentage", "position_size_percentage"]
        }

        # Initialize mesh state
        self._initialize_mesh_state()

        logger.info("âœ… NeuralMesh initialized")

    def _initialize_mesh_state(self):
        """Initialize the global mesh state with default parameters"""
        try:
            # Default global parameters
            self.mesh_state["global_parameters"] = {
                "rsi_oversold": 30.0,
                "rsi_overbought": 70.0,
                "ema_fast": 12,
                "ema_slow": 26,
                "risk_factor": 1.0,
                "confidence_threshold": 0.6,
                "take_profit_percentage": 0.05,
                "stop_loss_percentage": 0.03,
                "position_size_percentage": 0.1
            }

            self.mesh_state["performance_weights"] = {}
            self.mesh_state["last_update"] = datetime.now(timezone.utc).isoformat()
            self.mesh_state["mesh_version"] = 0

            logger.info("âœ… Mesh state initialized with default parameters")

        except Exception as e:
            logger.error(f"Failed to initialize mesh state: {e}")

    def _get_agent_performance(self, agent_id: str) -> dict[str, Any]:
        """Get agent performance from cache"""
        try:
            # Get recent agent performance signals
            signals = self.cache.get_signals_by_type("AGENT_PERFORMANCE", limit=10)

            # Filter by agent_id
            agent_signals = [
                signal for signal in signals
                if signal.get("metadata", {}).get("agent_id") == agent_id
            ]

            if agent_signals:
                latest_performance = agent_signals[0].get("metadata", {}).get("performance", {})
                return {
                    "total_pnl": latest_performance.get("total_pnl", 0.0),
                    "win_rate": latest_performance.get("win_rate", 0.0),
                    "total_trades": latest_performance.get("total_trades", 0),
                    "sharpe_ratio": latest_performance.get("sharpe_ratio", 0.0),
                    "timestamp": agent_signals[0].get("timestamp")
                }

            return {
                "total_pnl": 0.0,
                "win_rate": 0.0,
                "total_trades": 0,
                "sharpe_ratio": 0.0,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            logger.error(f"Failed to get agent performance for {agent_id}: {e}")
            return {
                "total_pnl": 0.0,
                "win_rate": 0.0,
                "total_trades": 0,
                "sharpe_ratio": 0.0,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

    def _get_agent_parameters(self, agent_id: str) -> dict[str, Any]:
        """Get agent parameters from cache"""
        try:
            # Get recent agent parameter signals
            signals = self.cache.get_signals_by_type("AGENT_PARAMETERS", limit=5)

            # Filter by agent_id
            agent_signals = [
                signal for signal in signals
                if signal.get("metadata", {}).get("agent_id") == agent_id
            ]

            if agent_signals:
                return agent_signals[0].get("metadata", {}).get("parameters", {})

            # Return default parameters if none found
            return copy.deepcopy(self.mesh_state["global_parameters"])

        except Exception as e:
            logger.error(f"Failed to get agent parameters for {agent_id}: {e}")
            return copy.deepcopy(self.mesh_state["global_parameters"])

    def _calculate_performance_score(self, performance: dict[str, Any]) -> float:
        """Calculate performance score for parameter merging"""
        try:
            # Weight factors for different performance metrics
            pnl_weight = 0.4
            win_rate_weight = 0.3
            sharpe_weight = 0.2
            trade_count_weight = 0.1

            # Normalize PnL (assume max PnL of 1000 for normalization)
            normalized_pnl = min(performance.get("total_pnl", 0.0) / 1000.0, 1.0)

            # Normalize trade count (assume max trades of 100 for normalization)
            normalized_trades = min(performance.get("total_trades", 0) / 100.0, 1.0)

            # Calculate weighted score
            score = (
                pnl_weight * normalized_pnl +
                win_rate_weight * performance.get("win_rate", 0.0) +
                sharpe_weight * max(0, performance.get("sharpe_ratio", 0.0)) +
                trade_count_weight * normalized_trades
            )

            return max(0.0, min(1.0, score))

        except Exception as e:
            logger.error(f"Failed to calculate performance score: {e}")
            return 0.0

    def _merge_parameters(self, agent_parameters: dict[str, Any],
                         performance_score: float) -> dict[str, Any]:
        """Merge agent parameters with global parameters based on performance"""
        try:
            merged_parameters = copy.deepcopy(self.mesh_state["global_parameters"])

            # Merge each parameter category
            for category, parameters in self.parameter_categories.items():
                for param in parameters:
                    if param in agent_parameters:
                        # Weighted average based on performance
                        global_value = merged_parameters.get(param, 0.0)
                        agent_value = agent_parameters.get(param, 0.0)

                        # Use performance score as weight for agent parameters
                        merged_value = (
                            (1 - performance_score) * global_value +
                            performance_score * agent_value
                        )

                        merged_parameters[param] = merged_value

            return merged_parameters

        except Exception as e:
            logger.error(f"Failed to merge parameters: {e}")
            return copy.deepcopy(self.mesh_state["global_parameters"])

    def _evaluate_agent_trades(self, agent_id: str) -> dict[str, Any]:
        """Evaluate agent's trading performance"""
        try:
            # Get agent performance
            performance = self._get_agent_performance(agent_id)

            # Check if agent has enough trades for evaluation
            if performance.get("total_trades", 0) < self.min_trades_for_evaluation:
                return {
                    "agent_id": agent_id,
                    "evaluated": False,
                    "reason": f"Insufficient trades: {performance.get('total_trades', 0)} < {self.min_trades_for_evaluation}"
                }

            # Calculate performance score
            performance_score = self._calculate_performance_score(performance)

            # Get agent parameters
            agent_parameters = self._get_agent_parameters(agent_id)

            # Merge parameters
            merged_parameters = self._merge_parameters(agent_parameters, performance_score)

            return {
                "agent_id": agent_id,
                "evaluated": True,
                "performance": performance,
                "performance_score": performance_score,
                "agent_parameters": agent_parameters,
                "merged_parameters": merged_parameters,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            logger.error(f"Failed to evaluate agent {agent_id}: {e}")
            return {
                "agent_id": agent_id,
                "evaluated": False,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

    def _update_mesh_state(self, agent_evaluations: list[dict[str, Any]]) -> dict[str, Any]:
        """Update mesh state with agent evaluations"""
        try:
            # Filter successful evaluations
            successful_evaluations = [
                eval for eval in agent_evaluations
                if eval.get("evaluated", False)
            ]

            if not successful_evaluations:
                logger.warning("No successful agent evaluations for mesh update")
                return self.mesh_state

            # Calculate new global parameters
            total_weight = 0.0
            weighted_parameters = {}

            for evaluation in successful_evaluations:
                performance_score = evaluation.get("performance_score", 0.0)
                merged_parameters = evaluation.get("merged_parameters", {})

                # Accumulate weighted parameters
                for param, value in merged_parameters.items():
                    if param not in weighted_parameters:
                        weighted_parameters[param] = 0.0
                    weighted_parameters[param] += value * performance_score

                total_weight += performance_score

            # Calculate final global parameters
            if total_weight > 0:
                for param in weighted_parameters:
                    weighted_parameters[param] /= total_weight

                # Update global parameters
                self.mesh_state["global_parameters"].update(weighted_parameters)

            # Update performance weights
            for evaluation in successful_evaluations:
                agent_id = evaluation.get("agent_id")
                performance_score = evaluation.get("performance_score", 0.0)

                # Apply decay to existing weight
                existing_weight = self.mesh_state["performance_weights"].get(agent_id, 0.0)
                decayed_weight = existing_weight * self.performance_weight_decay

                # Update with new performance
                self.mesh_state["performance_weights"][agent_id] = max(decayed_weight, performance_score)

            logger.info(f"âœ… Mesh state updated with {len(successful_evaluations)} agent evaluations")

            return self.mesh_state

        except Exception as e:
            logger.error(f"Failed to update mesh state: {e}")
            return self.mesh_state

    def update_mesh(self) -> dict[str, Any]:
        """Update the neural mesh with latest agent data"""
        try:
            logger.info("ðŸ”„ Updating neural mesh...")

            # Get all registered agents
            agent_ids = list(self.registered_agents.keys())

            if not agent_ids:
                logger.warning("No registered agents for mesh update")
                return self.mesh_state

            # Evaluate all agents
            agent_evaluations = []
            for agent_id in agent_ids:
                evaluation = self._evaluate_agent_trades(agent_id)
                agent_evaluations.append(evaluation)

            # Update mesh state
            updated_mesh = self._update_mesh_state(agent_evaluations)

            # Store mesh state in cache
            mesh_id = f"neural_mesh_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
            self.cache.store_signal(
                signal_id=mesh_id,
                symbol="NEURAL_MESH",
                signal_type="MESH_UPDATE",
                confidence=len([e for e in agent_evaluations if e.get("evaluated", False)]) / len(agent_evaluations),
                strategy="distributed_learning",
                metadata={
                    "mesh_state": updated_mesh,
                    "agent_evaluations": agent_evaluations,
                    "update_timestamp": datetime.now(timezone.utc).isoformat()
                }
            )

            logger.info(f"âœ… Neural mesh updated: {len(agent_evaluations)} agents evaluated")

            return updated_mesh

        except Exception as e:
            logger.error(f"Failed to update mesh: {e}")
            return self.mesh_state

    def get_mesh_state(self) -> dict[str, Any]:
        """Get current mesh state"""
        try:
            return {
                "mesh_state": self.mesh_state,
                "registered_agents": list(self.registered_agents.keys()),
                "parameter_categories": self.parameter_categories,
                "configuration": {
                    "mesh_update_interval": self.mesh_update_interval,
                    "min_trades_for_evaluation": self.min_trades_for_evaluation,
                    "performance_weight_decay": self.performance_weight_decay,
                    "parameter_merge_rate": self.parameter_merge_rate
                },
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            logger.error(f"Failed to get mesh state: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

    def sync_agent(self, agent_id: str) -> dict[str, Any]:
        """Sync an agent with the current mesh state"""
        try:
            logger.info(f"ðŸ”„ Syncing agent {agent_id} with mesh state")

            # Register agent if not already registered
            if agent_id not in self.registered_agents:
                self.registered_agents[agent_id] = {
                    "registered_at": datetime.now(timezone.utc).isoformat(),
                    "last_sync": None,
                    "sync_count": 0
                }

            # Update agent sync info
            self.registered_agents[agent_id]["last_sync"] = datetime.now(timezone.utc).isoformat()
            self.registered_agents[agent_id]["sync_count"] += 1

            # Get current mesh parameters
            mesh_parameters = copy.deepcopy(self.mesh_state["global_parameters"])

            # Get agent's current performance weight
            performance_weight = self.mesh_state["performance_weights"].get(agent_id, 0.0)

            sync_response = {
                "agent_id": agent_id,
                "synced": True,
                "mesh_parameters": mesh_parameters,
                "performance_weight": performance_weight,
                "mesh_version": self.mesh_state["mesh_version"],
                "sync_timestamp": datetime.now(timezone.utc).isoformat()
            }

            logger.info(f"âœ… Agent {agent_id} synced with mesh state")

            return sync_response

        except Exception as e:
            logger.error(f"Failed to sync agent {agent_id}: {e}")
            return {
                "agent_id": agent_id,
                "synced": False,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

    def register_agent(self, agent_id: str, initial_parameters: dict[str, Any] | None = None) -> dict[str, Any]:
        """Register a new agent with the neural mesh"""
        try:
            logger.info(f"ðŸ“ Registering agent {agent_id} with neural mesh")

            # Register agent
            self.registered_agents[agent_id] = {
                "registered_at": datetime.now(timezone.utc).isoformat(),
                "last_sync": None,
                "sync_count": 0,
                "initial_parameters": initial_parameters or copy.deepcopy(self.mesh_state["global_parameters"])
            }

            # Initialize performance weight
            self.mesh_state["performance_weights"][agent_id] = 0.0

            registration_response = {
                "agent_id": agent_id,
                "registered": True,
                "mesh_parameters": copy.deepcopy(self.mesh_state["global_parameters"]),
                "mesh_version": self.mesh_state["mesh_version"],
                "registration_timestamp": datetime.now(timezone.utc).isoformat()
            }

            logger.info(f"âœ… Agent {agent_id} registered with neural mesh")

            return registration_response

        except Exception as e:
            logger.error(f"Failed to register agent {agent_id}: {e}")
            return {
                "agent_id": agent_id,
                "registered": False,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

    def get_mesh_history(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get neural mesh update history"""
        try:
            # Get recent mesh update signals from cache
            signals = self.cache.get_signals_by_type("MESH_UPDATE", limit=limit)

            return signals

        except Exception as e:
            logger.error(f"Failed to get mesh history: {e}")
            return []

    def get_mesh_status(self) -> dict[str, Any]:
        """Get current neural mesh status"""
        try:
            return {
                "service": "NeuralMesh",
                "status": "active",
                "registered_agents": len(self.registered_agents),
                "mesh_version": self.mesh_state["mesh_version"],
                "last_update": self.mesh_state["last_update"],
                "configuration": {
                    "mesh_update_interval": self.mesh_update_interval,
                    "min_trades_for_evaluation": self.min_trades_for_evaluation,
                    "performance_weight_decay": self.performance_weight_decay,
                    "parameter_merge_rate": self.parameter_merge_rate
                },
                "parameter_categories": self.parameter_categories
            }

        except Exception as e:
            logger.error(f"Failed to get mesh status: {e}")
            return {"success": False, "error": str(e)}


# Global neural mesh instance
neural_mesh = NeuralMesh()


def get_neural_mesh() -> NeuralMesh:
    """Get the global neural mesh instance"""
    return neural_mesh


if __name__ == "__main__":
    # Test the neural mesh
    mesh = NeuralMesh()
    print(f"âœ… NeuralMesh initialized: {mesh}")

    # Test mesh update
    mesh.update_mesh()
    print("Mesh updated")

    # Test mesh state
    state = mesh.get_mesh_state()
    print(f"Mesh state: {state}")

    # Test status
    status = mesh.get_mesh_status()
    print(f"Mesh status: {status['status']}")


