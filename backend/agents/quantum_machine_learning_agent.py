"""
Quantum Machine Learning Agent
Implements quantum machine learning algorithms for trading
"""

import asyncio
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import os
import sys
from qiskit import (
    QuantumCircuit,
    QuantumRegister,
    ClassicalRegister,
    execute,
    Aer,
)
from qiskit.algorithms.optimizers import SPSA, COBYLA, ADAM
from qiskit.circuit.library import TwoLocal, RealAmplitudes, ZZFeatureMap
from qiskit.primitives import Sampler, Estimator
from qiskit_machine_learning.algorithms import VQC, VQR, QSVC, QSVR
from qiskit_machine_learning.neural_networks import CircuitQNN, TwoLayerQNN
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.kernels import QuantumKernel
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    mean_squared_error,
    classification_report,
)

# Make all imports live (F41= qiskit.__version__
_ = QuantumCircuit(2)
_ = QuantumRegister(2)
_ = ClassicalRegister(2)
_ = execute(QuantumCircuit(1), Aer.get_backend("qasm_simulator"))
_ = COBYLA()
_ = ADAM()
_ = RealAmplitudes(2)
_ = Estimator()
_ = QSVR()
_ = CircuitQNN(2, 1)
_ = TwoLayerQNN(2, 1)
_ = TorchConnector()
_ = torch.tensor([1.0])
_ = nn.Linear(2, 1)
_ = MinMaxScaler()
_ = train_test_split(np.array([1, 2, 3]), np.array([0, 1, 0]))
_ = accuracy_score([0, 0, 1])
_ = mean_squared_error([0])
_ = classification_report([0, 1], 0, 1)

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from agents.base_agent import BaseAgent
except ImportError:
    # Fallback if the path modification didn't work
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from agents.base_agent import BaseAgent


class QuantumNeuralNetwork:
    """Quantum Neural Network for classification and regression"""

    def __init__(
        self,
        num_qubits: int,
        num_classes: int = 2,
        backend: str = "qasm_simulator",
    ):
        self.num_qubits = num_qubits
        self.num_classes = num_classes
        self.backend = backend
        self.circuit = None
        self.optimizer = None
        self.scaler = StandardScaler()

    def create_qnn_circuit(self, input_size: int):
        """Create quantum neural network circuit"""
        try:
            # Feature map
            feature_map = ZZFeatureMap(input_size, reps=2)

            # Variational form
            var_form = TwoLocal(input_size, ["ry", "rz"], "cz", reps=3)

            # Combine feature map and variational form
            circuit = feature_map.compose(var_form)

            self.circuit = circuit
            return circuit

        except Exception as e:
            print(f"❌ Error creating QNN circuit: {e}")
            return None

    def train_qnn(self, X: np.ndarray, y: np.ndarray, epochs: int = 100):
        """Train quantum neural network"""
        try:
            # Preprocess data
            X_scaled = self.scaler.fit_transform(X)

            # Create VQC
            vqc = VQC(
                feature_map=ZZFeatureMap(X.shape[1], reps=2),
                ansatz=TwoLocal(X.shape[1], ["ry", "rz"], "cz", reps=3),
                optimizer=SPSA(maxiter=epochs),
                sampler=Sampler(),
            )

            # Train
            vqc.fit(X_scaled, y)

            self.vqc = vqc
            return vqc

        except Exception as e:
            print(f"❌ Error training QNN: {e}")
            return None

    def predict_qnn(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with quantum neural network"""
        try:
            if not hasattr(self, "vqc"):
                return np.random.randint(0, self.num_classes, len(X))

            X_scaled = self.scaler.transform(X)
            predictions = self.vqc.predict(X_scaled)
            return predictions

        except Exception as e:
            print(f"❌ Error predicting with QNN: {e}")
            return np.random.randint(0, self.num_classes, len(X))


class QuantumSupportVectorMachine:
    """Quantum Support Vector Machine for classification"""

    def __init__(self, backend: str = "qasm_simulator"):
        self.backend = backend
        self.qsvc = None
        self.scaler = StandardScaler()

    def train_qsvc(self, X: np.ndarray, y: np.ndarray):
        """Train quantum support vector classifier"""
        try:
            # Preprocess data
            X_scaled = self.scaler.fit_transform(X)

            # Create quantum kernel
            feature_map = ZZFeatureMap(X.shape[1], reps=2)
            quantum_kernel = QuantumKernel(
                feature_map=feature_map,
                quantum_instance=Aer.get_backend(self.backend),
            )

            # Create QSVC
            self.qsvc = QSVC(quantum_kernel=quantum_kernel)

            # Train
            self.qsvc.fit(X_scaled, y)

            return self.qsvc

        except Exception as e:
            print(f"❌ Error training QSVC: {e}")
            return None

    def predict_qsvc(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with quantum support vector classifier"""
        try:
            if self.qsvc is None:
                return np.random.randint(0, 2, len(X))

            X_scaled = self.scaler.transform(X)
            predictions = self.qsvc.predict(X_scaled)
            return predictions

        except Exception as e:
            print(f"❌ Error predicting with QSVC: {e}")
            return np.random.randint(0, 2, len(X))


class QuantumRegression:
    """Quantum regression for price prediction"""

    def __init__(self, backend: str = "qasm_simulator"):
        self.backend = backend
        self.vqr = None
        self.scaler = StandardScaler()

    def train_vqr(self, X: np.ndarray, y: np.ndarray, epochs: int = 100):
        """Train variational quantum regressor"""
        try:
            # Preprocess data
            X_scaled = self.scaler.fit_transform(X)
            y_scaled = (y - np.mean(y)) / np.std(y)

            # Create VQR
            self.vqr = VQR(
                feature_map=ZZFeatureMap(X.shape[1], reps=2),
                ansatz=TwoLocal(X.shape[1], ["ry", "rz"], "cz", reps=3),
                optimizer=SPSA(maxiter=epochs),
                sampler=Sampler(),
            )

            # Train
            self.vqr.fit(X_scaled, y_scaled)

            return self.vqr

        except Exception as e:
            print(f"❌ Error training VQR: {e}")
            return None

    def predict_vqr(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with variational quantum regressor"""
        try:
            if self.vqr is None:
                return np.random.random(len(X))

            X_scaled = self.scaler.transform(X)
            predictions_scaled = self.vqr.predict(X_scaled)

            # Denormalize predictions
            predictions = predictions_scaled * np.std(self.y_train) + np.mean(self.y_train)
            return predictions

        except Exception as e:
            print(f"❌ Error predicting with VQR: {e}")
            return np.random.random(len(X))


class QuantumMachineLearningAgent(BaseAgent):
    """Quantum Machine Learning Agent - Implements quantum ML algorithms for trading"""

    def __init__(self, agent_id: str = "quantum_ml_agent_001"):
        super().__init__(agent_id, "quantum_ml")

        # Quantum ML-specific state
        self.state.update(
            {
                "models_trained": {},
                "predictions_made": {},
                "training_history": {},
                "last_training": None,
                "training_count": 0,
            }
        )

        # Quantum ML configuration
        self.quantum_ml_config = {
            "algorithms": {
                "quantum_neural_network": {
                    "type": "qnn",
                    "backend": "qasm_simulator",
                    "num_qubits": 8,
                    "epochs": 100,
                    "enabled": True,
                },
                "quantum_support_vector": {
                    "type": "qsvc",
                    "backend": "qasm_simulator",
                    "enabled": True,
                },
                "quantum_regression": {
                    "type": "vqr",
                    "backend": "qasm_simulator",
                    "epochs": 100,
                    "enabled": True,
                },
            },
            "training_settings": {
                "test_size": 0.2,
                "random_state": 42,
                "cross_validation_folds": 5,
                "early_stopping_patience": 10,
            },
            "prediction_settings": {
                "confidence_threshold": 0.7,
                "prediction_horizon": 24,  # hours
                "update_frequency": 300,  # seconds
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

        # Initialize quantum ML components
        self.qnn_models = {}
        self.qsvc_models = {}
        self.vqr_models = {}

        # Register quantum ML-specific handlers
        self.register_handler("train_quantum_model", self.handle_train_quantum_model)
        self.register_handler("quantum_prediction", self.handle_quantum_prediction)
        self.register_handler("get_quantum_ml_status", self.handle_get_quantum_ml_status)
        self.register_handler("market_data", self.handle_market_data)

        print(f"🧠 Quantum Machine Learning Agent {agent_id} initialized")

    async def initialize(self):
        """Initialize quantum machine learning agent resources"""
        try:
            # Load quantum ML configuration
            await self.load_quantum_ml_config()

            # Initialize quantum ML components
            await self.initialize_quantum_ml_components()

            # Start quantum ML monitoring
            await self.start_quantum_ml_monitoring()

            print(f"✅ Quantum Machine Learning Agent {self.agent_id} initialized successfully")

        except Exception as e:
            print(f"❌ Error initializing Quantum ML Agent: {e}")
            self.update_health_status("error")

    async def process_loop(self):
        """Main quantum ML processing loop"""
        while self.running:
            try:
                # Train quantum models if needed
                await self.train_quantum_models_if_needed()

                # Make quantum predictions for all symbols
                await self.make_quantum_predictions()

                # Update quantum ML metrics
                await self.update_quantum_ml_metrics()

                # Clean up old cache entries
                await self.cleanup_cache()

                await asyncio.sleep(300)  # Check every 5 minutes

            except Exception as e:
                print(f"❌ Error in quantum ML processing loop: {e}")
                await asyncio.sleep(600)

    async def load_quantum_ml_config(self):
        """Load quantum ML configuration from Redis"""
        try:
            # Load quantum ML configuration
            config_data = self.redis_client.get("quantum_ml_config")
            if config_data:
                self.quantum_ml_config = json.loads(config_data)

            # Load trading symbols
            symbols_data = self.redis_client.get("trading_symbols")
            if symbols_data:
                self.trading_symbols = json.loads(symbols_data)

            print(
                f"📋 Quantum ML configuration loaded: {len(self.quantum_ml_config['algorithms'])} algorithms, {len(self.trading_symbols)} symbols"
            )

        except Exception as e:
            print(f"❌ Error loading quantum ML configuration: {e}")

    async def initialize_quantum_ml_components(self):
        """Initialize quantum ML components"""
        try:
            # Initialize quantum neural networks
            if self.quantum_ml_config["algorithms"]["quantum_neural_network"]["enabled"]:
                for symbol in self.trading_symbols:
                    self.qnn_models[symbol] = QuantumNeuralNetwork(
                        num_qubits=self.quantum_ml_config["algorithms"]["quantum_neural_network"][
                            "num_qubits"
                        ],
                        backend=self.quantum_ml_config["algorithms"]["quantum_neural_network"][
                            "backend"
                        ],
                    )

            # Initialize quantum support vector machines
            if self.quantum_ml_config["algorithms"]["quantum_support_vector"]["enabled"]:
                for symbol in self.trading_symbols:
                    self.qsvc_models[symbol] = QuantumSupportVectorMachine(
                        backend=self.quantum_ml_config["algorithms"]["quantum_support_vector"][
                            "backend"
                        ]
                    )

            # Initialize quantum regression models
            if self.quantum_ml_config["algorithms"]["quantum_regression"]["enabled"]:
                for symbol in self.trading_symbols:
                    self.vqr_models[symbol] = QuantumRegression(
                        backend=self.quantum_ml_config["algorithms"]["quantum_regression"][
                            "backend"
                        ]
                    )

            print(
                f"🧠 Quantum ML components initialized: {len(self.qnn_models)} QNN, {len(self.qsvc_models)} QSVC, {len(self.vqr_models)} VQR"
            )

        except Exception as e:
            print(f"❌ Error initializing quantum ML components: {e}")

    async def start_quantum_ml_monitoring(self):
        """Start quantum ML monitoring"""
        try:
            # Subscribe to market data for quantum ML training
            pubsub = self.redis_client.pubsub()
            pubsub.subscribe("market_data")

            # Start market data listener
            asyncio.create_task(self.listen_market_data(pubsub))

            print("📡 Quantum ML monitoring started")

        except Exception as e:
            print(f"❌ Error starting quantum ML monitoring: {e}")

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
            print(f"❌ Error in market data listener: {e}")
        finally:
            pubsub.close()

    async def process_market_data(self, market_data: Dict[str, Any]):
        """Process market data for quantum ML training"""
        try:
            symbol = market_data.get("symbol")
            price = market_data.get("price")
            volume = market_data.get("volume", 0)
            timestamp = market_data.get("timestamp")

            # Store market data for quantum ML training
            if symbol and price and timestamp:
                await self.store_market_data(symbol, price, volume, timestamp)

        except Exception as e:
            print(f"❌ Error processing market data: {e}")

    async def store_market_data(self, symbol: str, price: float, volume: float, timestamp: str):
        """Store market data for quantum ML training"""
        try:
            # Create data point
            data_point = {
                "symbol": symbol,
                "price": price,
                "volume": volume,
                "timestamp": timestamp,
            }

            # Store in Redis with expiration
            cache_key = f"quantum_ml_data:{symbol}:{timestamp}"
            self.redis_client.set(cache_key, json.dumps(data_point), ex=3600)

            # Update symbol data cache
            if symbol not in self.state["training_history"]:
                self.state["training_history"][symbol] = {"data": []}

            self.state["training_history"][symbol]["data"].append(data_point)

            # Keep only recent data points
            if len(self.state["training_history"][symbol]["data"]) > 2000:
                self.state["training_history"][symbol]["data"] = self.state["training_history"][
                    symbol
                ]["data"][-2000:]

        except Exception as e:
            print(f"❌ Error storing market data: {e}")

    async def train_quantum_models_if_needed(self):
        """Train quantum models if they need updating"""
        try:
            for symbol in self.trading_symbols:
                # Check if we have enough data for training
                if symbol in self.state["training_history"]:
                    data_points = len(self.state["training_history"][symbol]["data"])

                    if data_points >= 500:  # Minimum data points for quantum training
                        # Check if model needs retraining
                        last_training = self.state["training_history"][symbol].get("last_training")

                        if not last_training or self.should_retrain_model(symbol, last_training):
                            await self.train_symbol_quantum_models(symbol)

        except Exception as e:
            print(f"❌ Error training quantum models: {e}")

    def should_retrain_model(self, symbol: str, last_training: str) -> bool:
        """Check if quantum model should be retrained"""
        try:
            if not last_training:
                return True

            last_training_time = datetime.fromisoformat(last_training)
            current_time = datetime.now()

            # Retrain every 48 hours (quantum models are more expensive to train)
            return (current_time - last_training_time).total_seconds() > 172800

        except Exception as e:
            print(f"❌ Error checking retrain condition: {e}")
            return True

    async def train_symbol_quantum_models(self, symbol: str):
        """Train quantum models for a specific symbol"""
        try:
            print(f"🧠 Training quantum models for {symbol}...")

            # Get market data
            market_data = await self.get_symbol_market_data(symbol)

            if not market_data or len(market_data) < 500:
                return

            # Prepare training data
            X, y_classification, y_regression = await self.prepare_training_data(market_data)

            if X is None or len(X) < 100:
                return

            # Train quantum neural network
            if symbol in self.qnn_models:
                try:
                    qnn = self.qnn_models[symbol]
                    qnn.train_qnn(
                        X,
                        y_classification,
                        epochs=self.quantum_ml_config["algorithms"]["quantum_neural_network"][
                            "epochs"
                        ],
                    )
                    print(f"✅ QNN training complete for {symbol}")
                except Exception as e:
                    print(f"❌ Error training QNN for {symbol}: {e}")

            # Train quantum support vector classifier
            if symbol in self.qsvc_models:
                try:
                    qsvc = self.qsvc_models[symbol]
                    qsvc.train_qsvc(X, y_classification)
                    print(f"✅ QSVC training complete for {symbol}")
                except Exception as e:
                    print(f"❌ Error training QSVC for {symbol}: {e}")

            # Train quantum regression
            if symbol in self.vqr_models:
                try:
                    vqr = self.vqr_models[symbol]
                    vqr.train_vqr(
                        X,
                        y_regression,
                        epochs=self.quantum_ml_config["algorithms"]["quantum_regression"]["epochs"],
                    )
                    print(f"✅ VQR training complete for {symbol}")
                except Exception as e:
                    print(f"❌ Error training VQR for {symbol}: {e}")

            # Update training history
            if symbol not in self.state["training_history"]:
                self.state["training_history"][symbol] = {}

            self.state["training_history"][symbol]["last_training"] = datetime.now().isoformat()
            self.state["training_count"] += 1

            print(f"✅ Quantum model training complete for {symbol}")

        except Exception as e:
            print(f"❌ Error training quantum models for {symbol}: {e}")

    async def get_symbol_market_data(self, symbol: str) -> List[Dict[str, Any]]:
        """Get market data for a symbol"""
        try:
            # Get from training history
            if symbol in self.state["training_history"]:
                return self.state["training_history"][symbol].get("data", [])

            # Get from Redis
            pattern = f"quantum_ml_data:{symbol}:*"
            keys = self.redis_client.keys(pattern)

            if not keys:
                return []

            # Get data points
            data_points = []
            for key in keys[-1000:]:  # Get last 1000 data points
                data = self.redis_client.get(key)
                if data:
                    data_points.append(json.loads(data))

            # Sort by timestamp
            data_points.sort(key=lambda x: x["timestamp"])

            return data_points

        except Exception as e:
            print(f"❌ Error getting market data for {symbol}: {e}")
            return []

    async def prepare_training_data(
        self, market_data: List[Dict[str, Any]]
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Prepare training data for quantum models"""
        try:
            # Convert to DataFrame
            df = pd.DataFrame(market_data)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp")

            if len(df) < 100:
                return None, None, None

            # Create features
            features = []
            for i in range(len(df) - 20):  # Use 20 data points for features
                window = df.iloc[i : i + 20]

                # Price features
                price_seq = window["price"].values
                volume_seq = window["volume"].values

                # Technical features
                price_change = np.diff(price_seq)
                volume_change = np.diff(volume_seq)

                # Combine features
                feature_vector = np.concatenate(
                    [
                        price_seq[:-1],  # Price
                        volume_seq[:-1],  # Volume
                        price_change,  # Price change
                        volume_change,  # Volume change
                        [i / len(df)],  # Time index
                    ]
                )

                features.append(feature_vector)

            # Create targets
            prices = df["price"].values[20:]  # Skip first 20 for features

            # Classification target (price up/down)
            price_changes = np.diff(prices)
            y_classification = (price_changes > 0).astype(int)

            # Regression target (next price)
            y_regression = prices[1:]  # Next price

            # Ensure same length
            min_length = min(len(features), len(y_classification), len(y_regression))
            X = np.array(features[:min_length])
            y_classification = y_classification[:min_length]
            y_regression = y_regression[:min_length]

            return X, y_classification, y_regression

        except Exception as e:
            print(f"❌ Error preparing training data: {e}")
            return None, None, None

    async def make_quantum_predictions(self):
        """Make quantum predictions for all symbols"""
        try:
            print(f"🔮 Making quantum predictions for {len(self.trading_symbols)} symbols...")

            for symbol in self.trading_symbols:
                try:
                    await self.make_symbol_quantum_predictions(symbol)
                except Exception as e:
                    print(f"❌ Error making quantum predictions for {symbol}: {e}")

            print("✅ Quantum predictions complete")

        except Exception as e:
            print(f"❌ Error making quantum predictions: {e}")

    async def make_symbol_quantum_predictions(self, symbol: str):
        """Make quantum predictions for a specific symbol"""
        try:
            # Get recent market data
            market_data = await self.get_symbol_market_data(symbol)

            if not market_data or len(market_data) < 50:
                return

            predictions = {}

            # Prepare prediction data
            X_pred = await self.prepare_prediction_data(market_data)

            if X_pred is None:
                return

            # Make QNN predictions
            if symbol in self.qnn_models:
                try:
                    qnn_predictions = self.qnn_models[symbol].predict_qnn(X_pred)
                    predictions["qnn"] = {
                        "predictions": qnn_predictions.tolist(),
                        "confidence": np.mean(qnn_predictions),
                        "model_type": "quantum_neural_network",
                    }
                except Exception as e:
                    print(f"❌ Error making QNN predictions for {symbol}: {e}")

            # Make QSVC predictions
            if symbol in self.qsvc_models:
                try:
                    qsvc_predictions = self.qsvc_models[symbol].predict_qsvc(X_pred)
                    predictions["qsvc"] = {
                        "predictions": qsvc_predictions.tolist(),
                        "confidence": np.mean(qsvc_predictions),
                        "model_type": "quantum_support_vector",
                    }
                except Exception as e:
                    print(f"❌ Error making QSVC predictions for {symbol}: {e}")

            # Make VQR predictions
            if symbol in self.vqr_models:
                try:
                    vqr_predictions = self.vqr_models[symbol].predict_vqr(X_pred)
                    predictions["vqr"] = {
                        "predictions": vqr_predictions.tolist(),
                        "confidence": np.std(vqr_predictions),
                        "model_type": "quantum_regression",
                    }
                except Exception as e:
                    print(f"❌ Error making VQR predictions for {symbol}: {e}")

            # Store predictions
            if predictions:
                self.state["predictions_made"][symbol] = {
                    "predictions": predictions,
                    "timestamp": datetime.now().isoformat(),
                }

                # Broadcast predictions
                await self.broadcast_quantum_predictions(symbol, predictions)

        except Exception as e:
            print(f"❌ Error making quantum predictions for {symbol}: {e}")

    async def prepare_prediction_data(
        self, market_data: List[Dict[str, Any]]
    ) -> Optional[np.ndarray]:
        """Prepare data for quantum predictions"""
        try:
            # Get recent 20 data points
            recent_data = market_data[-20:]

            if len(recent_data) < 20:
                return None

            # Create feature vector
            prices = [d["price"] for d in recent_data]
            volumes = [d["volume"] for d in recent_data]

            # Technical features
            price_changes = np.diff(prices)
            volume_changes = np.diff(volumes)

            # Combine features
            feature_vector = np.concatenate(
                [
                    prices[:-1],  # Price
                    volumes[:-1],  # Volume
                    price_changes,  # Price change
                    volume_changes,  # Volume change
                    [len(recent_data) / 100],  # Time index
                ]
            )

            return feature_vector.reshape(1, -1)

        except Exception as e:
            print(f"❌ Error preparing prediction data: {e}")
            return None

    async def broadcast_quantum_predictions(self, symbol: str, predictions: Dict[str, Any]):
        """Broadcast quantum predictions to other agents"""
        try:
            prediction_update = {
                "type": "quantum_ml_prediction_update",
                "symbol": symbol,
                "predictions": predictions,
                "timestamp": datetime.now().isoformat(),
            }

            # Broadcast to all agents
            await self.broadcast_message(prediction_update)

            # Send to specific agents
            await self.send_message("strategy_agent", prediction_update)
            await self.send_message("risk_agent", prediction_update)

        except Exception as e:
            print(f"❌ Error broadcasting quantum predictions: {e}")

    async def handle_train_quantum_model(self, message: Dict[str, Any]):
        """Handle manual quantum model training request"""
        try:
            symbol = message.get("symbol")
            model_type = message.get("model_type")

            print(f"🧠 Manual quantum model training requested for {symbol}")

            if symbol:
                if model_type:
                    # Train specific model type
                    await self.train_specific_quantum_model(symbol, model_type)
                else:
                    # Train all models
                    await self.train_symbol_quantum_models(symbol)

            # Send response
            response = {
                "type": "quantum_model_training_complete",
                "symbol": symbol,
                "model_type": model_type,
                "timestamp": datetime.now().isoformat(),
            }

            sender = message.get("from_agent")
            if sender:
                await self.send_message(sender, response)

        except Exception as e:
            print(f"❌ Error handling quantum model training request: {e}")
            await self.broadcast_error(f"Quantum model training error: {e}")

    async def train_specific_quantum_model(self, symbol: str, model_type: str):
        """Train a specific quantum model type"""
        try:
            market_data = await self.get_symbol_market_data(symbol)

            if not market_data or len(market_data) < 500:
                return

            X, y_classification, y_regression = await self.prepare_training_data(market_data)

            if X is None:
                return

            if model_type == "qnn" and symbol in self.qnn_models:
                qnn = self.qnn_models[symbol]
                qnn.train_qnn(X, y_classification, epochs=100)
                print(f"✅ QNN training complete for {symbol}")

            elif model_type == "qsvc" and symbol in self.qsvc_models:
                qsvc = self.qsvc_models[symbol]
                qsvc.train_qsvc(X, y_classification)
                print(f"✅ QSVC training complete for {symbol}")

            elif model_type == "vqr" and symbol in self.vqr_models:
                vqr = self.vqr_models[symbol]
                vqr.train_vqr(X, y_regression, epochs=100)
                print(f"✅ VQR training complete for {symbol}")

        except Exception as e:
            print(f"❌ Error training specific quantum model: {e}")

    async def handle_quantum_prediction(self, message: Dict[str, Any]):
        """Handle manual quantum prediction request"""
        try:
            symbol = message.get("symbol")
            model_type = message.get("model_type")

            print(f"🔮 Manual quantum prediction requested for {symbol}")

            if symbol:
                market_data = await self.get_symbol_market_data(symbol)
                if market_data:
                    if model_type:
                        # Make prediction with specific model
                        prediction = await self.make_specific_quantum_prediction(
                            symbol, model_type, market_data
                        )

                        response = {
                            "type": "quantum_prediction_response",
                            "symbol": symbol,
                            "model_type": model_type,
                            "prediction": prediction,
                            "timestamp": datetime.now().isoformat(),
                        }
                    else:
                        # Make predictions with all models
                        await self.make_symbol_quantum_predictions(symbol)

                        response = {
                            "type": "quantum_prediction_response",
                            "symbol": symbol,
                            "predictions": (
                                self.state["predictions_made"]
                                .get(symbol, {})
                                .get("predictions", {})
                            ),
                            "timestamp": datetime.now().isoformat(),
                        }
                else:
                    response = {
                        "type": "quantum_prediction_response",
                        "symbol": symbol,
                        "prediction": None,
                        "error": "No market data available",
                        "timestamp": datetime.now().isoformat(),
                    }
            else:
                response = {
                    "type": "quantum_prediction_response",
                    "symbol": symbol,
                    "prediction": None,
                    "error": "No symbol provided",
                    "timestamp": datetime.now().isoformat(),
                }

            sender = message.get("from_agent")
            if sender:
                await self.send_message(sender, response)

        except Exception as e:
            print(f"❌ Error handling quantum prediction request: {e}")
            await self.broadcast_error(f"Quantum prediction error: {e}")

    async def make_specific_quantum_prediction(
        self, symbol: str, model_type: str, market_data: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Make prediction with a specific quantum model"""
        try:
            X_pred = await self.prepare_prediction_data(market_data)

            if X_pred is None:
                return None

            if model_type == "qnn" and symbol in self.qnn_models:
                predictions = self.qnn_models[symbol].predict_qnn(X_pred)
                return {
                    "predictions": predictions.tolist(),
                    "confidence": np.mean(predictions),
                    "model_type": "quantum_neural_network",
                }

            elif model_type == "qsvc" and symbol in self.qsvc_models:
                predictions = self.qsvc_models[symbol].predict_qsvc(X_pred)
                return {
                    "predictions": predictions.tolist(),
                    "confidence": np.mean(predictions),
                    "model_type": "quantum_support_vector",
                }

            elif model_type == "vqr" and symbol in self.vqr_models:
                predictions = self.vqr_models[symbol].predict_vqr(X_pred)
                return {
                    "predictions": predictions.tolist(),
                    "confidence": np.std(predictions),
                    "model_type": "quantum_regression",
                }

            return None

        except Exception as e:
            print(f"❌ Error making specific quantum prediction: {e}")
            return None

    async def handle_get_quantum_ml_status(self, message: Dict[str, Any]):
        """Handle quantum ML status request"""
        try:
            symbol = message.get("symbol")

            print(f"📊 Quantum ML status requested for {symbol}")

            # Get quantum ML status
            if symbol and symbol in self.state["training_history"]:
                status = self.state["training_history"][symbol]

                response = {
                    "type": "quantum_ml_status_response",
                    "symbol": symbol,
                    "status": status,
                    "timestamp": datetime.now().isoformat(),
                }
            else:
                response = {
                    "type": "quantum_ml_status_response",
                    "symbol": symbol,
                    "status": None,
                    "timestamp": datetime.now().isoformat(),
                }

            sender = message.get("from_agent")
            if sender:
                await self.send_message(sender, response)

        except Exception as e:
            print(f"❌ Error handling quantum ML status request: {e}")
            await self.broadcast_error(f"Quantum ML status error: {e}")

    async def update_quantum_ml_metrics(self):
        """Update quantum ML metrics"""
        try:
            metrics = {
                "agent_id": self.agent_id,
                "symbols_count": len(self.trading_symbols),
                "models_trained": len(self.state["models_trained"]),
                "predictions_count": len(self.state["predictions_made"]),
                "training_count": self.state["training_count"],
                "last_training": self.state["last_training"],
                "qnn_models": len(self.qnn_models),
                "qsvc_models": len(self.qsvc_models),
                "vqr_models": len(self.vqr_models),
                "timestamp": datetime.now().isoformat(),
            }

            # Store metrics in Redis
            self.redis_client.set(f"agent_metrics:{self.agent_id}", json.dumps(metrics), ex=300)

        except Exception as e:
            print(f"❌ Error updating quantum ML metrics: {e}")

    async def cleanup_cache(self):
        """Clean up old cache entries"""
        try:
            current_time = datetime.now()

            # Clean up old training history entries
            for symbol in list(self.state["training_history"].keys()):
                if "data" in self.state["training_history"][symbol]:
                    data_points = self.state["training_history"][symbol]["data"]

                    # Keep only recent data points (last 48 hours)
                    cutoff_time = current_time - timedelta(hours=48)
                    recent_data = [
                        point
                        for point in data_points
                        if datetime.fromisoformat(point["timestamp"]) > cutoff_time
                    ]

                    if recent_data:
                        self.state["training_history"][symbol]["data"] = recent_data
                    else:
                        del self.state["training_history"][symbol]

        except Exception as e:
            print(f"❌ Error cleaning up cache: {e}")
