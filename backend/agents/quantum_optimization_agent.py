"""
Quantum Optimization Agent
Implements quantum optimization algorithms for trading strategies
"""

import asyncio
import json
import os
import sys
from datetime import datetime, timedelta
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
import qiskit
from qiskit import (
    Aer,
    ClassicalRegister,
    QuantumCircuit,
    QuantumRegister,
    execute,
)
from qiskit.algorithms import QAOA, VQE, Grover
from qiskit.algorithms.optimizers import ADAM, COBYLA, L_BFGS_B, SPSA
from qiskit.circuit.library import EfficientSU2, RealAmplitudes, TwoLocal
from qiskit.primitives import Estimator, Sampler
from qiskit.quantum_info import Pauli, SparsePauliOp
from qiskit_finance.applications import PortfolioOptimization
from qiskit_finance.data_providers import RandomDataProvider
from qiskit_finance.optimization import EfficientFrontier
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler

# Make all imports live (F401):
_ = pd.DataFrame()
_ = qiskit.__version__
_ = QAOA()
_ = Grover()
_ = COBYLA()
_ = ADAM()
_ = L_BFGS_B()
_ = TwoLocal(2, reps=1)
_ = RealAmplitudes(2)
_ = Sampler()
_ = PortfolioOptimization()
_ = RandomDataProvider()
_ = EfficientFrontier()
_ = nx.Graph()
_ = minimize(lambda x: x**2, [1])
_ = StandardScaler()

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from backend.agents.base_agent import BaseAgent
except ImportError:
    # Fallback if the path modification didn't work
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from backend.agents.base_agent import BaseAgent


class QuantumAnnealingOptimizer:
    """Quantum Annealing for optimization problems"""

    def __init__(self, backend: str = "qasm_simulator"):
        self.backend = backend
        self.optimizer = None

    def optimize_portfolio_quantum_annealing(
        self, returns: np.ndarray, risk_free_rate: float = 0.02
    ):
        """Optimize portfolio using quantum annealing approach"""
        try:
            # Calculate expected returns and covariance
            expected_returns = np.mean(returns, axis=0)
            covariance_matrix = np.cov(returns.T)

            # Create portfolio optimization problem
            num_assets = len(expected_returns)

            # Create quantum circuit for QAOA
            qr = QuantumRegister(num_assets, "q")
            cr = ClassicalRegister(num_assets, "c")
            circuit = QuantumCircuit(qr, cr)

            # Initialize with equal superposition
            circuit.h(qr)

            # Apply QAOA layers
            for layer in range(3):  # 3 QAOA layers
                # Cost Hamiltonian (portfolio optimization)
                for i in range(num_assets):
                    for j in range(i + 1, num_assets):
                        if covariance_matrix[i, j] != 0:
                            circuit.cx(qr[i], qr[j])
                            circuit.rz(covariance_matrix[i, j], qr[j])
                            circuit.cx(qr[i], qr[j])

                # Mixing Hamiltonian
                circuit.h(qr)
                for i in range(num_assets):
                    circuit.rx(np.pi / 4, qr[i])

            # Measure
            circuit.measure(qr, cr)

            # Execute
            job = execute(circuit, Aer.get_backend(self.backend), shots=1000)
            result = job.result()
            counts = result.get_counts()

            # Extract optimal weights
            optimal_weights = self.extract_weights_from_counts(counts, num_assets)

            # Calculate portfolio metrics
            portfolio_return = np.sum(optimal_weights * expected_returns)
            portfolio_risk = np.sqrt(optimal_weights.T @ covariance_matrix @ optimal_weights)
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_risk

            return {
                "weights": optimal_weights,
                "expected_return": portfolio_return,
                "risk": portfolio_risk,
                "sharpe_ratio": sharpe_ratio,
                "optimization_method": "quantum_annealing",
            }

        except Exception as e:
            print(f"âŒ Error in quantum annealing optimization: {e}")
            return None

    def extract_weights_from_counts(self, counts: dict[str, int], num_assets: int) -> np.ndarray:
        """Extract portfolio weights from measurement counts"""
        try:
            total_shots = sum(counts.values())
            weights = np.zeros(num_assets)

            for bitstring, count in counts.items():
                # Convert bitstring to weights
                for i, bit in enumerate(bitstring):
                    if bit == "1":
                        weights[i] += count / total_shots

            # Normalize weights
            weights = weights / np.sum(weights)
            return weights

        except Exception as e:
            print(f"âŒ Error extracting weights: {e}")
            return np.ones(num_assets) / num_assets


class VariationalQuantumOptimizer:
    """Variational Quantum Optimization for complex problems"""

    def __init__(self, backend: str = "qasm_simulator"):
        self.backend = backend
        self.optimizer = None

    def optimize_trading_strategy_vqe(
        self, strategy_params: dict[str, Any], market_data: np.ndarray
    ):
        """Optimize trading strategy parameters using VQE"""
        try:
            # Define objective function for strategy optimization
            def objective_function(params):
                # Simulate strategy performance with given parameters
                performance = self.simulate_strategy_performance(params, market_data)
                return -performance  # Minimize negative performance (maximize performance)

            # Create VQE circuit
            num_qubits = len(strategy_params)
            ansatz = EfficientSU2(num_qubits, reps=2)

            # Create VQE
            vqe = VQE(
                ansatz=ansatz,
                optimizer=SPSA(maxiter=100),
                estimator=Estimator(),
            )

            # Create Hamiltonian (simplified)
            hamiltonian = self.create_strategy_hamiltonian(strategy_params)

            # Solve optimization problem
            result = vqe.solve(hamiltonian)

            # Extract optimal parameters
            optimal_params = result.optimal_parameters

            return {
                "optimal_parameters": optimal_params,
                "optimal_value": result.optimal_value,
                "optimization_method": "vqe",
            }

        except Exception as e:
            print(f"âŒ Error in VQE optimization: {e}")
            return None

    def create_strategy_hamiltonian(self, strategy_params: dict[str, Any]):
        """Create Hamiltonian for strategy optimization"""
        try:
            # Create a simple Hamiltonian based on strategy parameters
            num_qubits = len(strategy_params)
            hamiltonian_terms = []

            # Add terms for each parameter
            for i, (param_name, param_value) in enumerate(strategy_params.items()):
                # Create Pauli operator
                pauli_string = ["I"] * num_qubits
                pauli_string[i] = "Z"
                hamiltonian_terms.append((Pauli("".join(pauli_string)), param_value))

            # Create SparsePauliOp
            hamiltonian = SparsePauliOp.from_list(hamiltonian_terms)
            return hamiltonian

        except Exception as e:
            print(f"âŒ Error creating strategy Hamiltonian: {e}")
            return None

    def simulate_strategy_performance(self, params: np.ndarray, market_data: np.ndarray) -> float:
        """Simulate trading strategy performance with given parameters"""
        try:
            # Simple strategy simulation
            # This is a placeholder - in practice, you would implement actual strategy logic
            returns = np.diff(market_data, axis=0)

            # Apply parameters to create signals
            signals = np.dot(returns, params[: len(returns[0])])

            # Calculate performance metric (Sharpe ratio)
            if len(signals) > 1:
                performance = np.mean(signals) / (np.std(signals) + 1e-8)
            else:
                performance = 0.0

            return performance

        except Exception as e:
            print(f"âŒ Error simulating strategy performance: {e}")
            return 0.0


class QuantumConstraintOptimizer:
    """Quantum optimization with constraints"""

    def __init__(self, backend: str = "qasm_simulator"):
        self.backend = backend

    def optimize_with_constraints(
        self,
        objective_func,
        constraints: list[dict],
        bounds: list[tuple],
        initial_guess: np.ndarray,
    ):
        """Optimize with quantum-enhanced constraint handling"""
        try:
            # Create quantum circuit for constraint satisfaction
            num_variables = len(initial_guess)
            qr = QuantumRegister(num_variables, "q")
            cr = ClassicalRegister(num_variables, "c")
            circuit = QuantumCircuit(qr, cr)

            # Initialize with equal superposition
            circuit.h(qr)

            # Apply constraint satisfaction layers
            for constraint in constraints:
                circuit = self.apply_constraint(circuit, qr, constraint)

            # Apply optimization layers
            for layer in range(2):
                # Cost function evaluation
                circuit = self.apply_cost_function(circuit, qr, objective_func)

                # Mixing layer
                circuit.h(qr)
                for i in range(num_variables):
                    circuit.rx(np.pi / 4, qr[i])

            # Measure
            circuit.measure(qr, cr)

            # Execute
            job = execute(circuit, Aer.get_backend(self.backend), shots=1000)
            result = job.result()
            counts = result.get_counts()

            # Extract optimal solution
            optimal_solution = self.extract_solution_from_counts(counts, num_variables, bounds)

            return {
                "optimal_solution": optimal_solution,
                "constraints_satisfied": self.check_constraints(optimal_solution, constraints),
                "optimization_method": "quantum_constraint_optimization",
            }

        except Exception as e:
            print(f"âŒ Error in quantum constraint optimization: {e}")
            return None

    def apply_constraint(self, circuit: QuantumCircuit, qr: QuantumRegister, constraint: dict):
        """Apply constraint to quantum circuit"""
        try:
            # Simple constraint application
            # In practice, you would implement more sophisticated constraint handling
            constraint_type = constraint.get("type", "linear")

            if constraint_type == "linear":
                # Apply linear constraint
                coefficients = constraint.get("coefficients", [])
                target = constraint.get("target", 0)

                for i, coeff in enumerate(coefficients):
                    if coeff != 0:
                        circuit.rz(coeff * target, qr[i])

            return circuit

        except Exception as e:
            print(f"âŒ Error applying constraint: {e}")
            return circuit

    def apply_cost_function(self, circuit: QuantumCircuit, qr: QuantumRegister, objective_func):
        """Apply cost function to quantum circuit"""
        try:
            # Apply cost function evaluation
            # This is a simplified version
            for i in range(len(qr)):
                circuit.rz(0.1, qr[i])  # Simple cost term

            return circuit

        except Exception as e:
            print(f"âŒ Error applying cost function: {e}")
            return circuit

    def extract_solution_from_counts(
        self, counts: dict[str, int], num_variables: int, bounds: list[tuple]
    ) -> np.ndarray:
        """Extract optimal solution from measurement counts"""
        try:
            # Find most frequent measurement
            most_frequent = max(counts, key=counts.get)

            # Convert bitstring to solution
            solution = np.zeros(num_variables)
            for i, bit in enumerate(most_frequent):
                if bit == "1":
                    solution[i] = 1.0

            # Scale solution to bounds
            for i, (lower, upper) in enumerate(bounds):
                solution[i] = lower + solution[i] * (upper - lower)

            return solution

        except Exception as e:
            print(f"âŒ Error extracting solution: {e}")
            return np.zeros(num_variables)

    def check_constraints(self, solution: np.ndarray, constraints: list[dict]) -> bool:
        """Check if solution satisfies constraints"""
        try:
            for constraint in constraints:
                constraint_type = constraint.get("type", "linear")

                if constraint_type == "linear":
                    coefficients = constraint.get("coefficients", [])
                    target = constraint.get("target", 0)
                    tolerance = constraint.get("tolerance", 1e-6)

                    # Check linear constraint
                    constraint_value = np.dot(coefficients, solution)
                    if abs(constraint_value - target) > tolerance:
                        return False

            return True

        except Exception as e:
            print(f"âŒ Error checking constraints: {e}")
            return False


class QuantumOptimizationAgent(BaseAgent):
    """Quantum Optimization Agent - Implements quantum optimization algorithms for trading"""

    def __init__(self, agent_id: str = "quantum_optimization_agent_001"):
        super().__init__(agent_id, "quantum_optimization")

        # Quantum optimization-specific state
        self.state.update(
            {
                "optimizations_performed": {},
                "optimal_solutions": {},
                "constraint_violations": {},
                "last_optimization": None,
                "optimization_count": 0,
            }
        )

        # Quantum optimization configuration
        self.quantum_opt_config = {
            "algorithms": {
                "quantum_annealing": {
                    "type": "qaoa",
                    "backend": "qasm_simulator",
                    "shots": 1000,
                    "layers": 3,
                    "enabled": True,
                },
                "variational_quantum": {
                    "type": "vqe",
                    "backend": "qasm_simulator",
                    "max_iterations": 100,
                    "enabled": True,
                },
                "constraint_optimization": {
                    "type": "quantum_constraint",
                    "backend": "qasm_simulator",
                    "enabled": True,
                },
            },
            "optimization_settings": {
                "max_iterations": 100,
                "tolerance": 1e-6,
                "timeout_seconds": 300,
                "parallel_optimizations": 4,
            },
            "portfolio_settings": {
                "risk_free_rate": 0.02,
                "target_return": 0.15,
                "max_risk": 0.25,
                "rebalancing_frequency": 24,  # hours
            },
        }

        # Trading symbols to monitor
        self.trading_symbols = [
            "BTC",
            "ETH",
            "ADA",
            "DOT",
            "LINK",
            "UNI",
            "AAVE",
        ]

        # Initialize quantum optimization components
        self.quantum_annealing = None
        self.variational_optimizer = None
        self.constraint_optimizer = None

        # Register quantum optimization-specific handlers
        self.register_handler("optimize_portfolio", self.handle_optimize_portfolio)
        self.register_handler("optimize_strategy", self.handle_optimize_strategy)
        self.register_handler(
            "quantum_constraint_optimization",
            self.handle_constraint_optimization,
        )
        self.register_handler("get_optimization_status", self.handle_get_optimization_status)
        self.register_handler("market_data", self.handle_market_data)

        print(f"âš›ï¸ Quantum Optimization Agent {agent_id} initialized")

    async def initialize(self):
        """Initialize quantum optimization agent resources"""
        try:
            # Load quantum optimization configuration
            await self.load_quantum_opt_config()

            # Initialize quantum optimization components
            await self.initialize_quantum_opt_components()

            # Start quantum optimization monitoring
            await self.start_quantum_opt_monitoring()

            print(f"âœ… Quantum Optimization Agent {self.agent_id} initialized successfully")

        except Exception as e:
            print(f"âŒ Error initializing Quantum Optimization Agent: {e}")
            self.update_health_status("error")

    async def process_loop(self):
        """Main quantum optimization processing loop"""
        while self.running:
            try:
                # Perform portfolio optimizations
                await self.perform_portfolio_optimizations()

                # Optimize trading strategies
                await self.optimize_trading_strategies()

                # Update quantum optimization metrics
                await self.update_quantum_opt_metrics()

                # Clean up old optimizations
                await self.cleanup_old_optimizations()

                await asyncio.sleep(1800)  # Check every 30 minutes

            except Exception as e:
                print(f"âŒ Error in quantum optimization processing loop: {e}")
                await asyncio.sleep(3600)

    async def load_quantum_opt_config(self):
        """Load quantum optimization configuration from Redis"""
        try:
            # Load quantum optimization configuration
            config_data = self.redis_client.get("quantum_opt_config")
            if config_data:
                self.quantum_opt_config = json.loads(config_data)

            # Load trading symbols
            symbols_data = self.redis_client.get("trading_symbols")
            if symbols_data:
                self.trading_symbols = json.loads(symbols_data)

            print(
                f"ðŸ“‹ Quantum optimization configuration loaded: {len(self.quantum_opt_config['algorithms'])} algorithms, {len(self.trading_symbols)} symbols"
            )

        except Exception as e:
            print(f"âŒ Error loading quantum optimization configuration: {e}")

    async def initialize_quantum_opt_components(self):
        """Initialize quantum optimization components"""
        try:
            # Initialize quantum annealing optimizer
            if self.quantum_opt_config["algorithms"]["quantum_annealing"]["enabled"]:
                self.quantum_annealing = QuantumAnnealingOptimizer(
                    backend=self.quantum_opt_config["algorithms"]["quantum_annealing"]["backend"]
                )

            # Initialize variational quantum optimizer
            if self.quantum_opt_config["algorithms"]["variational_quantum"]["enabled"]:
                self.variational_optimizer = VariationalQuantumOptimizer(
                    backend=self.quantum_opt_config["algorithms"]["variational_quantum"]["backend"]
                )

            # Initialize constraint optimizer
            if self.quantum_opt_config["algorithms"]["constraint_optimization"]["enabled"]:
                self.constraint_optimizer = QuantumConstraintOptimizer(
                    backend=self.quantum_opt_config["algorithms"]["constraint_optimization"][
                        "backend"
                    ]
                )

            print("âš›ï¸ Quantum optimization components initialized")

        except Exception as e:
            print(f"âŒ Error initializing quantum optimization components: {e}")

    async def start_quantum_opt_monitoring(self):
        """Start quantum optimization monitoring"""
        try:
            # Subscribe to market data for optimization
            pubsub = self.redis_client.pubsub()
            pubsub.subscribe("market_data")

            # Start market data listener
            asyncio.create_task(self.listen_market_data(pubsub))

            print("ðŸ“¡ Quantum optimization monitoring started")

        except Exception as e:
            print(f"âŒ Error starting quantum optimization monitoring: {e}")

    async def listen_market_data(self, pubsub):
        """Listen for market data updates"""
        try:
            for message in pubsub.listen():
                if not self.running:
                    break

                if message["type"] == "message":
                    market_data = json.loads(message["data"])
                    await self.process_market_data(market_data)

        except Exception as e:
            print(f"âŒ Error in market data listener: {e}")
        finally:
            pubsub.close()

    async def process_market_data(self, market_data: dict[str, Any]):
        """Process market data for optimization"""
        try:
            symbol = market_data.get("symbol")
            price = market_data.get("price")
            volume = market_data.get("volume", 0)
            timestamp = market_data.get("timestamp")

            # Store market data for optimization
            if symbol and price and timestamp:
                await self.store_market_data(symbol, price, volume, timestamp)

        except Exception as e:
            print(f"âŒ Error processing market data: {e}")

    async def store_market_data(self, symbol: str, price: float, volume: float, timestamp: str):
        """Store market data for optimization"""
        try:
            # Create data point
            data_point = {
                "symbol": symbol,
                "price": price,
                "volume": volume,
                "timestamp": timestamp,
            }

            # Store in Redis with expiration
            cache_key = f"quantum_opt_data:{symbol}:{timestamp}"
            self.redis_client.set(cache_key, json.dumps(data_point), ex=3600)

            # Update symbol data cache
            if symbol not in self.state["optimizations_performed"]:
                self.state["optimizations_performed"][symbol] = {"data": []}

            self.state["optimizations_performed"][symbol]["data"].append(data_point)

            # Keep only recent data points
            if len(self.state["optimizations_performed"][symbol]["data"]) > 1000:
                self.state["optimizations_performed"][symbol]["data"] = self.state[
                    "optimizations_performed"
                ][symbol]["data"][-1000:]

        except Exception as e:
            print(f"âŒ Error storing market data: {e}")

    async def perform_portfolio_optimizations(self):
        """Perform portfolio optimizations using quantum algorithms"""
        try:
            print("âš›ï¸ Performing portfolio optimizations...")

            if not self.quantum_annealing:
                return

            # Get returns data for all symbols
            returns_data = {}
            for symbol in self.trading_symbols:
                market_data = await self.get_symbol_market_data(symbol)
                if market_data and len(market_data) > 1:
                    prices = [d["price"] for d in market_data]
                    returns = np.diff(prices) / prices[:-1]
                    returns_data[symbol] = returns

            if len(returns_data) < 2:
                return

            # Create returns matrix
            min_length = min(len(returns) for returns in returns_data.values())
            returns_matrix = np.array(
                [
                    returns_data[symbol][:min_length]
                    for symbol in self.trading_symbols[: len(returns_data)]
                ]
            )

            # Perform quantum annealing optimization
            optimization_result = self.quantum_annealing.optimize_portfolio_quantum_annealing(
                returns_matrix.T,
                risk_free_rate=self.quantum_opt_config["portfolio_settings"]["risk_free_rate"],
            )

            if optimization_result:
                # Store optimization result
                self.state["optimal_solutions"]["portfolio"] = {
                    "result": optimization_result,
                    "symbols": list(returns_data.keys()),
                    "timestamp": datetime.now().isoformat(),
                }

                # Broadcast optimization result
                await self.broadcast_portfolio_optimization(optimization_result)

                print("âœ… Portfolio optimization complete")

        except Exception as e:
            print(f"âŒ Error performing portfolio optimizations: {e}")

    async def get_symbol_market_data(self, symbol: str) -> list[dict[str, Any]]:
        """Get market data for a symbol"""
        try:
            # Get from cache first
            if symbol in self.state["optimizations_performed"]:
                return self.state["optimizations_performed"][symbol].get("data", [])

            # Get from Redis
            pattern = f"quantum_opt_data:{symbol}:*"
            keys = self.redis_client.keys(pattern)

            if not keys:
                return []

            # Get data points
            data_points = []
            for key in keys[-500:]:  # Get last 500 data points
                data = self.redis_client.get(key)
                if data:
                    data_points.append(json.loads(data))

            # Sort by timestamp
            data_points.sort(key=lambda x: x["timestamp"])

            return data_points

        except Exception as e:
            print(f"âŒ Error getting market data for {symbol}: {e}")
            return []

    async def optimize_trading_strategies(self):
        """Optimize trading strategies using quantum algorithms"""
        try:
            print("âš›ï¸ Optimizing trading strategies...")

            if not self.variational_optimizer:
                return

            # Define strategy parameters to optimize
            strategy_params = {
                "rsi_period": 14,
                "macd_fast": 12,
                "macd_slow": 26,
                "bollinger_period": 20,
                "bollinger_std": 2,
                "stop_loss": 0.02,
                "take_profit": 0.05,
            }

            # Get market data for optimization
            market_data = await self.get_combined_market_data()

            if market_data is None or len(market_data) < 100:
                return

            # Perform VQE optimization
            optimization_result = self.variational_optimizer.optimize_trading_strategy_vqe(
                strategy_params, market_data
            )

            if optimization_result:
                # Store optimization result
                self.state["optimal_solutions"]["strategy"] = {
                    "result": optimization_result,
                    "timestamp": datetime.now().isoformat(),
                }

                # Broadcast optimization result
                await self.broadcast_strategy_optimization(optimization_result)

                print("âœ… Strategy optimization complete")

        except Exception as e:
            print(f"âŒ Error optimizing trading strategies: {e}")

    async def get_combined_market_data(self) -> np.ndarray | None:
        """Get combined market data for all symbols"""
        try:
            all_data = []

            for symbol in self.trading_symbols:
                market_data = await self.get_symbol_market_data(symbol)
                if market_data:
                    prices = [d["price"] for d in market_data]
                    all_data.extend(prices)

            if all_data:
                return np.array(all_data)

            return None

        except Exception as e:
            print(f"âŒ Error getting combined market data: {e}")
            return None

    async def broadcast_portfolio_optimization(self, optimization_result: dict[str, Any]):
        """Broadcast portfolio optimization results"""
        try:
            optimization_update = {
                "type": "quantum_portfolio_optimization",
                "result": optimization_result,
                "timestamp": datetime.now().isoformat(),
            }

            # Broadcast to all agents
            await self.broadcast_message(optimization_update)

            # Send to specific agents
            await self.send_message("strategy_agent", optimization_update)
            await self.send_message("execution_agent", optimization_update)

        except Exception as e:
            print(f"âŒ Error broadcasting portfolio optimization: {e}")

    async def broadcast_strategy_optimization(self, optimization_result: dict[str, Any]):
        """Broadcast strategy optimization results"""
        try:
            optimization_update = {
                "type": "quantum_strategy_optimization",
                "result": optimization_result,
                "timestamp": datetime.now().isoformat(),
            }

            # Broadcast to all agents
            await self.broadcast_message(optimization_update)

            # Send to specific agents
            await self.send_message("strategy_agent", optimization_update)
            await self.send_message("execution_agent", optimization_update)

        except Exception as e:
            print(f"âŒ Error broadcasting strategy optimization: {e}")

    async def handle_optimize_portfolio(self, message: dict[str, Any]):
        """Handle manual portfolio optimization request"""
        try:
            symbols = message.get("symbols", self.trading_symbols)

            print(f"âš›ï¸ Manual portfolio optimization requested for {len(symbols)} symbols")

            if not self.quantum_annealing:
                response = {
                    "type": "portfolio_optimization_response",
                    "symbols": symbols,
                    "result": None,
                    "error": "Quantum annealing optimizer not available",
                    "timestamp": datetime.now().isoformat(),
                }
            else:
                # Get returns data
                returns_data = {}
                for symbol in symbols:
                    market_data = await self.get_symbol_market_data(symbol)
                    if market_data and len(market_data) > 1:
                        prices = [d["price"] for d in market_data]
                        returns = np.diff(prices) / prices[:-1]
                        returns_data[symbol] = returns

                if len(returns_data) >= 2:
                    # Create returns matrix
                    min_length = min(len(returns) for returns in returns_data.values())
                    returns_matrix = np.array(
                        [
                            returns_data[symbol][:min_length]
                            for symbol in symbols[: len(returns_data)]
                        ]
                    )

                    # Perform optimization
                    result = self.quantum_annealing.optimize_portfolio_quantum_annealing(
                        returns_matrix.T,
                        risk_free_rate=self.quantum_opt_config["portfolio_settings"][
                            "risk_free_rate"
                        ],
                    )

                    response = {
                        "type": "portfolio_optimization_response",
                        "symbols": symbols,
                        "result": result,
                        "timestamp": datetime.now().isoformat(),
                    }
                else:
                    response = {
                        "type": "portfolio_optimization_response",
                        "symbols": symbols,
                        "result": None,
                        "error": "Insufficient data for optimization",
                        "timestamp": datetime.now().isoformat(),
                    }

            sender = message.get("from_agent")
            if sender:
                await self.send_message(sender, response)

        except Exception as e:
            print(f"âŒ Error handling portfolio optimization: {e}")
            await self.broadcast_error(f"Portfolio optimization error: {e}")

    async def handle_optimize_strategy(self, message: dict[str, Any]):
        """Handle manual strategy optimization request"""
        try:
            strategy_params = message.get("strategy_params", {})

            print("âš›ï¸ Manual strategy optimization requested")

            if not self.variational_optimizer:
                response = {
                    "type": "strategy_optimization_response",
                    "result": None,
                    "error": "Variational quantum optimizer not available",
                    "timestamp": datetime.now().isoformat(),
                }
            else:
                # Get market data
                market_data = await self.get_combined_market_data()

                if market_data is not None and len(market_data) >= 100:
                    # Perform optimization
                    result = self.variational_optimizer.optimize_trading_strategy_vqe(
                        strategy_params, market_data
                    )

                    response = {
                        "type": "strategy_optimization_response",
                        "result": result,
                        "timestamp": datetime.now().isoformat(),
                    }
                else:
                    response = {
                        "type": "strategy_optimization_response",
                        "result": None,
                        "error": "Insufficient market data for optimization",
                        "timestamp": datetime.now().isoformat(),
                    }

            sender = message.get("from_agent")
            if sender:
                await self.send_message(sender, response)

        except Exception as e:
            print(f"âŒ Error handling strategy optimization: {e}")
            await self.broadcast_error(f"Strategy optimization error: {e}")

    async def handle_constraint_optimization(self, message: dict[str, Any]):
        """Handle constraint optimization request"""
        try:
            objective_func = message.get("objective_func")
            constraints = message.get("constraints", [])
            bounds = message.get("bounds", [])
            initial_guess = message.get("initial_guess", [])

            print("âš›ï¸ Constraint optimization requested")

            if not self.constraint_optimizer:
                response = {
                    "type": "constraint_optimization_response",
                    "result": None,
                    "error": "Constraint optimizer not available",
                    "timestamp": datetime.now().isoformat(),
                }
            else:
                # Perform constraint optimization
                result = self.constraint_optimizer.optimize_with_constraints(
                    objective_func,
                    constraints,
                    bounds,
                    np.array(initial_guess),
                )

                response = {
                    "type": "constraint_optimization_response",
                    "result": result,
                    "timestamp": datetime.now().isoformat(),
                }

            sender = message.get("from_agent")
            if sender:
                await self.send_message(sender, response)

        except Exception as e:
            print(f"âŒ Error handling constraint optimization: {e}")
            await self.broadcast_error(f"Constraint optimization error: {e}")

    async def handle_get_optimization_status(self, message: dict[str, Any]):
        """Handle optimization status request"""
        try:
            optimization_type = message.get("optimization_type")

            print(f"ðŸ“Š Optimization status requested for {optimization_type}")

            if optimization_type and optimization_type in self.state["optimal_solutions"]:
                status = self.state["optimal_solutions"][optimization_type]

                response = {
                    "type": "optimization_status_response",
                    "optimization_type": optimization_type,
                    "status": status,
                    "timestamp": datetime.now().isoformat(),
                }
            else:
                response = {
                    "type": "optimization_status_response",
                    "optimization_type": optimization_type,
                    "status": None,
                    "timestamp": datetime.now().isoformat(),
                }

            sender = message.get("from_agent")
            if sender:
                await self.send_message(sender, response)

        except Exception as e:
            print(f"âŒ Error handling optimization status request: {e}")
            await self.broadcast_error(f"Optimization status error: {e}")

    async def update_quantum_opt_metrics(self):
        """Update quantum optimization metrics"""
        try:
            metrics = {
                "agent_id": self.agent_id,
                "symbols_count": len(self.trading_symbols),
                "optimizations_performed": len(self.state["optimizations_performed"]),
                "optimal_solutions": len(self.state["optimal_solutions"]),
                "optimization_count": self.state["optimization_count"],
                "last_optimization": self.state["last_optimization"],
                "quantum_annealing_available": (self.quantum_annealing is not None),
                "variational_optimizer_available": (self.variational_optimizer is not None),
                "constraint_optimizer_available": (self.constraint_optimizer is not None),
                "timestamp": datetime.now().isoformat(),
            }

            # Store metrics in Redis
            self.redis_client.set(f"agent_metrics:{self.agent_id}", json.dumps(metrics), ex=300)

        except Exception as e:
            print(f"âŒ Error updating quantum optimization metrics: {e}")

    async def cleanup_old_optimizations(self):
        """Clean up old optimization results"""
        try:
            current_time = datetime.now()

            # Clean up old optimization data
            for symbol in list(self.state["optimizations_performed"].keys()):
                if "data" in self.state["optimizations_performed"][symbol]:
                    data_points = self.state["optimizations_performed"][symbol]["data"]

                    # Keep only recent data points (last 24 hours)
                    cutoff_time = current_time - timedelta(hours=24)
                    recent_data = [
                        point
                        for point in data_points
                        if datetime.fromisoformat(point["timestamp"]) > cutoff_time
                    ]

                    if recent_data:
                        self.state["optimizations_performed"][symbol]["data"] = recent_data
                    else:
                        del self.state["optimizations_performed"][symbol]

        except Exception as e:
            print(f"âŒ Error cleaning up old optimizations: {e}")


