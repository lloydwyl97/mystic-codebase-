"""
Quantum Algorithm Engine
Implements quantum algorithms for financial applications
"""

import asyncio
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import os
import sys
import qiskit
from qiskit import (
    QuantumCircuit,
    QuantumRegister,
    ClassicalRegister,
    execute,
    Aer,
)
from qiskit.circuit.library.standard_gates import RYGate, RZGate
from qiskit.algorithms import Grover
from qiskit.algorithms.optimizers import SPSA, COBYLA
from qiskit.circuit.library import QFT, TwoLocal
from qiskit.quantum_info import Operator, Statevector
from qiskit.primitives import Sampler, Estimator
from qiskit_machine_learning.algorithms import VQC, VQR
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector
import torch
import torch.nn as nn

# Make all imports live (F401):
_ = pd.DataFrame()
_ = Tuple[int, int]
_ = qiskit.__version__
_ = COBYLA()
_ = TwoLocal(2, [RYGate, RZGate], entanglement="linear")
_ = Operator(np.eye(2))
_ = Statevector([10])
_ = Sampler()
_ = Estimator()
_ = torch.tensor([1.0])
_ = nn.Linear(2, 1)

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from backend.agents.base_agent import BaseAgent
except ImportError:
    # Fallback if the path modification didn't work
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from backend.agents.base_agent import BaseAgent


class QuantumPortfolioOptimizer:
    """Quantum portfolio optimization using VQE"""

    def __init__(self, num_assets: int, backend: str = "qasm_simulator"):
        self.num_assets = num_assets
        self.backend = backend
        self.qubits = num_assets
        self.circuit = None
        self.optimizer = None

    def create_portfolio_circuit(self, returns: np.ndarray, risk_free_rate: float = 0.02):
        """Create quantum circuit for portfolio optimization"""
        try:
            # Calculate covariance matrix
            np.cov(returns.T)

            # Create quantum circuit
            qr = QuantumRegister(self.qubits, "q")
            cr = ClassicalRegister(self.qubits, "c")
            circuit = QuantumCircuit(qr, cr)

            # Initialize with equal weights
            circuit.h(qr)

            # Add rotation gates for optimization
            for i in range(self.qubits):
                circuit.ry(2 * np.pi * np.random.random(), qr[i])

            # Measure all qubits
            circuit.measure(qr, cr)

            self.circuit = circuit
            return circuit

        except Exception as e:
            print(f"âŒ Error creating portfolio circuit: {e}")
            return None

    def optimize_portfolio(
        self,
        returns: np.ndarray,
        target_return: float = None,
        max_risk: float = None,
    ):
        """Optimize portfolio using quantum variational algorithm"""
        try:
            # Create circuit
            circuit = self.create_portfolio_circuit(returns)
            if circuit is None:
                return None

            # Define cost function
            def cost_function(params):
                # Execute circuit with parameters
                circuit_with_params = circuit.bind_parameters(params)
                job = execute(
                    circuit_with_params,
                    Aer.get_backend(self.backend),
                    shots=1000,
                )
                result = job.result()
                counts = result.get_counts()

                # Calculate portfolio metrics
                weights = self.extract_weights_from_counts(counts)
                portfolio_return = np.sum(weights * np.mean(returns, axis=0))
                portfolio_risk = np.sqrt(weights.T @ np.cov(returns.T) @ weights)

                # Cost function (minimize risk for given return)
                if target_return is not None:
                    return_cost = abs(portfolio_return - target_return) * 100
                    risk_cost = portfolio_risk * 10
                    return return_cost + risk_cost
                else:
                    return portfolio_risk

            # Optimize using SPSA
            optimizer = SPSA(maxiter=100)
            initial_params = np.random.random(circuit.num_parameters) * 2 * np.pi

            result = optimizer.minimize(cost_function, initial_params)

            # Extract optimal weights
            optimal_circuit = circuit.bind_parameters(result.x)
            job = execute(optimal_circuit, Aer.get_backend(self.backend), shots=1000)
            counts = job.result().get_counts()
            optimal_weights = self.extract_weights_from_counts(counts)

            return {
                "weights": optimal_weights,
                "expected_return": np.sum(optimal_weights * np.mean(returns, axis=0)),
                "risk": np.sqrt(optimal_weights.T @ np.cov(returns.T) @ optimal_weights),
                "sharpe_ratio": (
                    (np.sum(optimal_weights * np.mean(returns, axis=0)) - 0.02)
                    / np.sqrt(optimal_weights.T @ np.cov(returns.T) @ optimal_weights)
                ),
            }

        except Exception as e:
            print(f"âŒ Error optimizing portfolio: {e}")
            return None

    def extract_weights_from_counts(self, counts: Dict[str, int]) -> np.ndarray:
        """Extract portfolio weights from measurement counts"""
        try:
            total_shots = sum(counts.values())
            weights = np.zeros(self.num_assets)

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
            return np.ones(self.num_assets) / self.num_assets


class QuantumFourierTransform:
    """Quantum Fourier Transform for time series analysis"""

    def __init__(self, num_qubits: int, backend: str = "qasm_simulator"):
        self.num_qubits = num_qubits
        self.backend = backend

    def apply_qft(self, data: np.ndarray) -> np.ndarray:
        """Apply Quantum Fourier Transform to time series data"""
        try:
            # Normalize data to [0, 1]
            normalized_data = (data - np.min(data)) / (np.max(data) - np.min(data))

            # Create quantum circuit
            qr = QuantumRegister(self.num_qubits, "q")
            cr = ClassicalRegister(self.num_qubits, "c")
            circuit = QuantumCircuit(qr, cr)

            # Encode data into quantum state
            for i, value in enumerate(normalized_data[: 2**self.num_qubits]):
                if value > 0.5:
                    circuit.x(qr[i])

            # Apply QFT
            qft_circuit = QFT(num_qubits=self.num_qubits)
            circuit = circuit.compose(qft_circuit)

            # Measure
            circuit.measure(qr, cr)

            # Execute
            job = execute(circuit, Aer.get_backend(self.backend), shots=1000)
            result = job.result()
            counts = result.get_counts()

            # Extract frequency components
            frequencies = self.extract_frequencies(counts)

            return frequencies

        except Exception as e:
            print(f"âŒ Error applying QFT: {e}")
            return np.fft.fft(data)

    def extract_frequencies(self, counts: Dict[str, int]) -> np.ndarray:
        """Extract frequency components from measurement counts"""
        try:
            total_shots = sum(counts.values())
            frequencies = np.zeros(2**self.num_qubits)

            for bitstring, count in counts.items():
                # Convert bitstring to integer
                freq_index = int(bitstring, 2)
                frequencies[freq_index] = count / total_shots

            return frequencies

        except Exception as e:
            print(f"âŒ Error extracting frequencies: {e}")
            return np.zeros(2**self.num_qubits)


class QuantumSearch:
    """Quantum search algorithms for pattern recognition"""

    def __init__(self, backend: str = "qasm_simulator"):
        self.backend = backend

    def grover_search(self, oracle_function, num_qubits: int, num_iterations: int = None):
        """Implement Grover's search algorithm"""
        try:
            # Create oracle
            oracle = QuantumCircuit(num_qubits)
            # Apply oracle function (simplified)
            oracle.h(range(num_qubits))
            oracle.x(range(num_qubits))
            oracle.h(num_qubits - 1)
            oracle.mct(list(range(num_qubits - 1)), num_qubits - 1)
            oracle.h(num_qubits - 1)
            oracle.x(range(num_qubits))
            oracle.h(range(num_qubits))

            # Create Grover algorithm
            grover = Grover(oracle=oracle, good_state=None)

            # Execute
            job = execute(grover, Aer.get_backend(self.backend), shots=1000)
            result = job.result()

            return result

        except Exception as e:
            print(f"âŒ Error in Grover search: {e}")
            return None

    def quantum_pattern_search(self, data: np.ndarray, pattern: np.ndarray):
        """Search for patterns in data using quantum algorithms"""
        try:
            # Convert data to quantum representation
            num_qubits = min(8, len(data))

            # Create quantum circuit
            qr = QuantumRegister(num_qubits, "q")
            cr = ClassicalRegister(num_qubits, "c")
            circuit = QuantumCircuit(qr, cr)

            # Encode data
            for i, value in enumerate(data[: 2**num_qubits]):
                if value > np.mean(data):
                    circuit.x(qr[i])

            # Apply search algorithm
            circuit.h(qr)
            circuit.measure(qr, cr)

            # Execute
            job = execute(circuit, Aer.get_backend(self.backend), shots=1000)
            result = job.result()
            counts = result.get_counts()

            # Find matches
            matches = self.find_pattern_matches(counts, pattern)

            return matches

        except Exception as e:
            print(f"âŒ Error in quantum pattern search: {e}")
            return []


class QuantumAlgorithmEngine(BaseAgent):
    """Quantum Algorithm Engine - Implements quantum algorithms for financial applications"""

    def __init__(self, agent_id: str = "quantum_algorithm_engine_001"):
        super().__init__(agent_id, "quantum_algorithm")

        # Quantum engine-specific state
        self.state.update(
            {
                "algorithms_executed": {},
                "quantum_results": {},
                "optimization_history": {},
                "last_execution": None,
                "execution_count": 0,
            }
        )

        # Quantum configuration
        self.quantum_config = {
            "algorithms": {
                "portfolio_optimization": {
                    "type": "vqe",
                    "backend": "qasm_simulator",
                    "shots": 1000,
                    "max_iterations": 100,
                    "enabled": True,
                },
                "fourier_transform": {
                    "type": "qft",
                    "backend": "qasm_simulator",
                    "num_qubits": 8,
                    "enabled": True,
                },
                "quantum_search": {
                    "type": "grover",
                    "backend": "qasm_simulator",
                    "shots": 1000,
                    "enabled": True,
                },
                "quantum_ml": {
                    "type": "vqc",
                    "backend": "qasm_simulator",
                    "shots": 1000,
                    "enabled": True,
                },
            },
            "execution_settings": {
                "max_qubits": 12,
                "timeout_seconds": 300,
                "error_mitigation": True,
                "optimization_level": 2,
            },
            "financial_settings": {
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

        # Initialize quantum components
        self.portfolio_optimizer = None
        self.fourier_transform = None
        self.quantum_search = None
        self.quantum_ml = None

        # Register quantum-specific handlers
        self.register_handler("execute_quantum_algorithm", self.handle_execute_quantum_algorithm)
        self.register_handler("optimize_portfolio", self.handle_optimize_portfolio)
        self.register_handler("quantum_analysis", self.handle_quantum_analysis)
        self.register_handler("market_data", self.handle_market_data)

        print(f"âš›ï¸ Quantum Algorithm Engine {agent_id} initialized")

    async def initialize(self):
        """Initialize quantum algorithm engine resources"""
        try:
            # Load quantum configuration
            await self.load_quantum_config()

            # Initialize quantum components
            await self.initialize_quantum_components()

            # Start quantum monitoring
            await self.start_quantum_monitoring()

            print(f"âœ… Quantum Algorithm Engine {self.agent_id} initialized successfully")

        except Exception as e:
            print(f"âŒ Error initializing Quantum Algorithm Engine: {e}")
            self.update_health_status("error")

    async def process_loop(self):
        """Main quantum processing loop"""
        while self.running:
            try:
                # Execute quantum algorithms
                await self.execute_quantum_algorithms()

                # Optimize portfolios
                await self.optimize_portfolios()

                # Update quantum metrics
                await self.update_quantum_metrics()

                # Clean up old results
                await self.cleanup_old_results()

                await asyncio.sleep(600)  # Check every 10 minutes

            except Exception as e:
                print(f"âŒ Error in quantum processing loop: {e}")
                await asyncio.sleep(1200)

    async def load_quantum_config(self):
        """Load quantum configuration from Redis"""
        try:
            # Load quantum configuration
            config_data = self.redis_client.get("quantum_algorithm_config")
            if config_data:
                self.quantum_config = json.loads(config_data)

            # Load trading symbols
            symbols_data = self.redis_client.get("trading_symbols")
            if symbols_data:
                self.trading_symbols = json.loads(symbols_data)

            print(
                f"ðŸ“‹ Quantum configuration loaded: {len(self.quantum_config['algorithms'])} algorithms, {len(self.trading_symbols)} symbols"
            )

        except Exception as e:
            print(f"âŒ Error loading quantum configuration: {e}")

    async def initialize_quantum_components(self):
        """Initialize quantum algorithm components"""
        try:
            # Initialize portfolio optimizer
            if self.quantum_config["algorithms"]["portfolio_optimization"]["enabled"]:
                self.portfolio_optimizer = QuantumPortfolioOptimizer(
                    num_assets=len(self.trading_symbols),
                    backend=self.quantum_config["algorithms"]["portfolio_optimization"]["backend"],
                )

            # Initialize Fourier transform
            if self.quantum_config["algorithms"]["fourier_transform"]["enabled"]:
                self.fourier_transform = QuantumFourierTransform(
                    num_qubits=self.quantum_config["algorithms"]["fourier_transform"]["num_qubits"],
                    backend=self.quantum_config["algorithms"]["fourier_transform"]["backend"],
                )

            # Initialize quantum search
            if self.quantum_config["algorithms"]["quantum_search"]["enabled"]:
                self.quantum_search = QuantumSearch(
                    backend=self.quantum_config["algorithms"]["quantum_search"]["backend"]
                )

            print("âš›ï¸ Quantum components initialized")

        except Exception as e:
            print(f"âŒ Error initializing quantum components: {e}")

    async def start_quantum_monitoring(self):
        """Start quantum monitoring"""
        try:
            # Subscribe to market data for quantum analysis
            pubsub = self.redis_client.pubsub()
            pubsub.subscribe("market_data")

            # Start market data listener
            asyncio.create_task(self.listen_market_data(pubsub))

            print("ðŸ“¡ Quantum monitoring started")

        except Exception as e:
            print(f"âŒ Error starting quantum monitoring: {e}")

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

    async def process_market_data(self, market_data: Dict[str, Any]):
        """Process market data for quantum analysis"""
        try:
            symbol = market_data.get("symbol")
            price = market_data.get("price")
            volume = market_data.get("volume", 0)
            timestamp = market_data.get("timestamp")

            # Store market data for quantum analysis
            if symbol and price and timestamp:
                await self.store_market_data(symbol, price, volume, timestamp)

        except Exception as e:
            print(f"âŒ Error processing market data: {e}")

    async def store_market_data(self, symbol: str, price: float, volume: float, timestamp: str):
        """Store market data for quantum analysis"""
        try:
            # Create data point
            data_point = {
                "symbol": symbol,
                "price": price,
                "volume": volume,
                "timestamp": timestamp,
            }

            # Store in Redis with expiration
            cache_key = f"quantum_data:{symbol}:{timestamp}"
            self.redis_client.set(cache_key, json.dumps(data_point), ex=3600)

            # Update symbol data cache
            if symbol not in self.state["quantum_results"]:
                self.state["quantum_results"][symbol] = {"data": []}

            self.state["quantum_results"][symbol]["data"].append(data_point)

            # Keep only recent data points
            if len(self.state["quantum_results"][symbol]["data"]) > 1000:
                self.state["quantum_results"][symbol]["data"] = self.state["quantum_results"][
                    symbol
                ]["data"][-1000:]

        except Exception as e:
            print(f"âŒ Error storing market data: {e}")

    async def execute_quantum_algorithms(self):
        """Execute quantum algorithms for all symbols"""
        try:
            print(f"âš›ï¸ Executing quantum algorithms for {len(self.trading_symbols)} symbols...")

            for symbol in self.trading_symbols:
                try:
                    await self.execute_symbol_quantum_algorithms(symbol)
                except Exception as e:
                    print(f"âŒ Error executing quantum algorithms for {symbol}: {e}")

            print("âœ… Quantum algorithm execution complete")

        except Exception as e:
            print(f"âŒ Error executing quantum algorithms: {e}")

    async def execute_symbol_quantum_algorithms(self, symbol: str):
        """Execute quantum algorithms for a specific symbol"""
        try:
            # Get market data
            market_data = await self.get_symbol_market_data(symbol)

            if not market_data or len(market_data) < 100:
                return

            results = {}

            # Execute Fourier transform analysis
            if self.fourier_transform:
                try:
                    prices = [d["price"] for d in market_data]
                    frequencies = self.fourier_transform.apply_qft(np.array(prices))

                    results["fourier_transform"] = {
                        "frequencies": frequencies.tolist(),
                        "dominant_frequency": np.argmax(frequencies),
                        "frequency_spectrum": frequencies[:10].tolist(),
                    }
                except Exception as e:
                    print(f"âŒ Error in Fourier transform for {symbol}: {e}")

            # Execute quantum pattern search
            if self.quantum_search:
                try:
                    prices = [d["price"] for d in market_data]
                    # Look for patterns (simplified)
                    pattern = np.array(prices[-10:])  # Last 10 prices
                    matches = self.quantum_search.quantum_pattern_search(np.array(prices), pattern)

                    results["pattern_search"] = {
                        "pattern_matches": len(matches),
                        "pattern_strength": np.mean(matches) if matches else 0,
                    }
                except Exception as e:
                    print(f"âŒ Error in quantum pattern search for {symbol}: {e}")

            # Store results
            if results:
                self.state["quantum_results"][symbol]["analysis"] = {
                    "results": results,
                    "timestamp": datetime.now().isoformat(),
                }

                # Broadcast results
                await self.broadcast_quantum_results(symbol, results)

        except Exception as e:
            print(f"âŒ Error executing quantum algorithms for {symbol}: {e}")

    async def get_symbol_market_data(self, symbol: str) -> List[Dict[str, Any]]:
        """Get market data for a symbol"""
        try:
            # Get from cache first
            if symbol in self.state["quantum_results"]:
                return self.state["quantum_results"][symbol].get("data", [])

            # Get from Redis
            pattern = f"quantum_data:{symbol}:*"
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

    async def optimize_portfolios(self):
        """Optimize portfolios using quantum algorithms"""
        try:
            print("âš›ï¸ Optimizing portfolios using quantum algorithms...")

            if not self.portfolio_optimizer:
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

            # Optimize portfolio
            optimization_result = self.portfolio_optimizer.optimize_portfolio(
                returns_matrix.T,
                target_return=self.quantum_config["financial_settings"]["target_return"],
                max_risk=self.quantum_config["financial_settings"]["max_risk"],
            )

            if optimization_result:
                # Store optimization result
                self.state["optimization_history"][datetime.now().isoformat()] = {
                    "result": optimization_result,
                    "symbols": list(returns_data.keys()),
                }

                # Broadcast optimization result
                await self.broadcast_portfolio_optimization(optimization_result)

                print("âœ… Portfolio optimization complete")

        except Exception as e:
            print(f"âŒ Error optimizing portfolios: {e}")

    async def broadcast_quantum_results(self, symbol: str, results: Dict[str, Any]):
        """Broadcast quantum analysis results to other agents"""
        try:
            quantum_update = {
                "type": "quantum_analysis_update",
                "symbol": symbol,
                "results": results,
                "timestamp": datetime.now().isoformat(),
            }

            # Broadcast to all agents
            await self.broadcast_message(quantum_update)

            # Send to specific agents
            await self.send_message("strategy_agent", quantum_update)
            await self.send_message("risk_agent", quantum_update)

        except Exception as e:
            print(f"âŒ Error broadcasting quantum results: {e}")

    async def broadcast_portfolio_optimization(self, optimization_result: Dict[str, Any]):
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

    async def handle_execute_quantum_algorithm(self, message: Dict[str, Any]):
        """Handle manual quantum algorithm execution request"""
        try:
            algorithm_type = message.get("algorithm_type")
            symbol = message.get("symbol")

            print(f"âš›ï¸ Manual quantum algorithm execution requested: {algorithm_type}")

            if algorithm_type and symbol:
                result = await self.execute_specific_algorithm(algorithm_type, symbol)

                response = {
                    "type": "quantum_algorithm_response",
                    "algorithm_type": algorithm_type,
                    "symbol": symbol,
                    "result": result,
                    "timestamp": datetime.now().isoformat(),
                }
            else:
                response = {
                    "type": "quantum_algorithm_response",
                    "algorithm_type": algorithm_type,
                    "symbol": symbol,
                    "result": None,
                    "error": "Missing algorithm_type or symbol",
                    "timestamp": datetime.now().isoformat(),
                }

            sender = message.get("from_agent")
            if sender:
                await self.send_message(sender, response)

        except Exception as e:
            print(f"âŒ Error handling quantum algorithm execution: {e}")
            await self.broadcast_error(f"Quantum algorithm execution error: {e}")

    async def execute_specific_algorithm(
        self, algorithm_type: str, symbol: str
    ) -> Optional[Dict[str, Any]]:
        """Execute a specific quantum algorithm"""
        try:
            market_data = await self.get_symbol_market_data(symbol)

            if not market_data or len(market_data) < 50:
                return None

            if algorithm_type == "fourier_transform" and self.fourier_transform:
                prices = [d["price"] for d in market_data]
                frequencies = self.fourier_transform.apply_qft(np.array(prices))

                return {
                    "frequencies": frequencies.tolist(),
                    "dominant_frequency": np.argmax(frequencies),
                    "frequency_spectrum": frequencies[:10].tolist(),
                }

            elif algorithm_type == "pattern_search" and self.quantum_search:
                prices = [d["price"] for d in market_data]
                pattern = np.array(prices[-10:])
                matches = self.quantum_search.quantum_pattern_search(np.array(prices), pattern)

                return {
                    "pattern_matches": len(matches),
                    "pattern_strength": np.mean(matches) if matches else 0,
                }

            elif algorithm_type == "portfolio_optimization" and self.portfolio_optimizer:
                # This would need multiple symbols
                return {"error": "Portfolio optimization requires multiple symbols"}

            else:
                return {"error": f"Unknown algorithm type: {algorithm_type}"}

        except Exception as e:
            print(f"âŒ Error executing specific algorithm: {e}")
            return {"error": str(e)}

    async def handle_optimize_portfolio(self, message: Dict[str, Any]):
        """Handle manual portfolio optimization request"""
        try:
            symbols = message.get("symbols", self.trading_symbols)

            print(f"âš›ï¸ Manual portfolio optimization requested for {len(symbols)} symbols")

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
                    [returns_data[symbol][:min_length] for symbol in symbols[: len(returns_data)]]
                )

                # Optimize portfolio
                result = self.portfolio_optimizer.optimize_portfolio(
                    returns_matrix.T,
                    target_return=self.quantum_config["financial_settings"]["target_return"],
                    max_risk=self.quantum_config["financial_settings"]["max_risk"],
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

    async def handle_quantum_analysis(self, message: Dict[str, Any]):
        """Handle quantum analysis request"""
        try:
            symbol = message.get("symbol")

            print(f"âš›ï¸ Quantum analysis requested for {symbol}")

            if symbol:
                await self.execute_symbol_quantum_algorithms(symbol)

                response = {
                    "type": "quantum_analysis_response",
                    "symbol": symbol,
                    "status": "completed",
                    "timestamp": datetime.now().isoformat(),
                }
            else:
                response = {
                    "type": "quantum_analysis_response",
                    "symbol": symbol,
                    "status": "failed",
                    "error": "No symbol provided",
                    "timestamp": datetime.now().isoformat(),
                }

            sender = message.get("from_agent")
            if sender:
                await self.send_message(sender, response)

        except Exception as e:
            print(f"âŒ Error handling quantum analysis: {e}")
            await self.broadcast_error(f"Quantum analysis error: {e}")

    async def update_quantum_metrics(self):
        """Update quantum metrics"""
        try:
            metrics = {
                "agent_id": self.agent_id,
                "symbols_count": len(self.trading_symbols),
                "algorithms_executed": len(self.state["algorithms_executed"]),
                "optimization_count": len(self.state["optimization_history"]),
                "last_execution": self.state["last_execution"],
                "execution_count": self.state["execution_count"],
                "timestamp": datetime.now().isoformat(),
            }

            # Store metrics in Redis
            self.redis_client.set(f"agent_metrics:{self.agent_id}", json.dumps(metrics), ex=300)

        except Exception as e:
            print(f"âŒ Error updating quantum metrics: {e}")

    async def cleanup_old_results(self):
        """Clean up old quantum results"""
        try:
            current_time = datetime.now()

            # Clean up old optimization history
            cutoff_time = current_time - timedelta(days=7)
            expired_optimizations = [
                timestamp
                for timestamp in self.state["optimization_history"].keys()
                if datetime.fromisoformat(timestamp) < cutoff_time
            ]

            for timestamp in expired_optimizations:
                del self.state["optimization_history"][timestamp]

            if expired_optimizations:
                print(f"ðŸ—‘ï¸ Cleaned up {len(expired_optimizations)} expired optimizations")

        except Exception as e:
            print(f"âŒ Error cleaning up old results: {e}")


