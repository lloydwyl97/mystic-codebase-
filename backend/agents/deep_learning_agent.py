"""
Deep Learning Agent
Handles deep learning models for price prediction and pattern recognition
"""

import asyncio
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import joblib

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.agents.base_agent import BaseAgent


class LSTMModel(nn.Module):
    """LSTM model for time series prediction"""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int,
        dropout: float = 0.2,
    ):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out


class CNNModel(nn.Module):
    """CNN model for pattern recognition"""

    def __init__(self, input_channels: int, num_classes: int):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)

        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(128 * 12, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))

        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


class DeepLearningAgent(BaseAgent):
    """Deep Learning Agent - Uses neural networks for prediction and analysis"""

    def __init__(self, agent_id: str = "deep_learning_agent_001"):
        super().__init__(agent_id, "deep_learning")

        # Deep learning-specific state
        self.state.update(
            {
                "models_trained": {},
                "predictions_made": {},
                "model_cache": {},
                "training_history": {},
                "last_training": None,
                "training_count": 0,
            }
        )

        # Model configuration
        self.model_config = {
            "models": {
                "lstm_price_predictor": {
                    "type": "lstm",
                    "input_size": 10,
                    "hidden_size": 128,
                    "num_layers": 3,
                    "output_size": 1,
                    "sequence_length": 60,
                    "enabled": True,
                },
                "cnn_pattern_recognizer": {
                    "type": "cnn",
                    "input_channels": 5,
                    "num_classes": 14,
                    "sequence_length": 100,
                    "enabled": True,
                },
                "transformer_forecaster": {
                    "type": "transformer",
                    "input_size": 10,
                    "hidden_size": 256,
                    "num_layers": 6,
                    "num_heads": 8,
                    "output_size": 1,
                    "enabled": False,  # Future enhancement
                },
            },
            "training_settings": {
                "batch_size": 32,
                "learning_rate": 0.001,
                "epochs": 100,
                "validation_split": 0.2,
                "early_stopping_patience": 10,
                "min_data_points": 1000,
            },
            "prediction_settings": {
                "prediction_horizon": 24,  # hours
                "confidence_threshold": 0.7,
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

        # Initialize models and scalers
        self.models = {}
        self.scalers = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Register deep learning-specific handlers
        self.register_handler("train_model", self.handle_train_model)
        self.register_handler("make_prediction", self.handle_make_prediction)
        self.register_handler("get_model_status", self.handle_get_model_status)
        self.register_handler("market_data", self.handle_market_data)

        print(f"ðŸ§  Deep Learning Agent {agent_id} initialized on {self.device}")

    async def initialize(self):
        """Initialize deep learning agent resources"""
        try:
            # Load model configuration
            await self.load_model_config()

            # Initialize models
            await self.initialize_models()

            # Load pre-trained models if available
            await self.load_pretrained_models()

            # Start model monitoring
            await self.start_model_monitoring()

            print(f"âœ… Deep Learning Agent {self.agent_id} initialized successfully")

        except Exception as e:
            print(f"âŒ Error initializing Deep Learning Agent: {e}")
            self.update_health_status("error")

    async def process_loop(self):
        """Main deep learning processing loop"""
        while self.running:
            try:
                # Train models if needed
                await self.train_models_if_needed()

                # Make predictions for all symbols
                await self.make_all_predictions()

                # Update model metrics
                await self.update_model_metrics()

                # Clean up old cache entries
                await self.cleanup_cache()

                await asyncio.sleep(300)  # Check every 5 minutes

            except Exception as e:
                print(f"âŒ Error in deep learning processing loop: {e}")
                await asyncio.sleep(600)

    async def load_model_config(self):
        """Load model configuration from Redis"""
        try:
            # Load model configuration
            config_data = self.redis_client.get("deep_learning_config")
            if config_data:
                self.model_config = json.loads(config_data)

            # Load trading symbols
            symbols_data = self.redis_client.get("trading_symbols")
            if symbols_data:
                self.trading_symbols = json.loads(symbols_data)

            print(
                f"ðŸ“‹ Model configuration loaded: {len(self.model_config['models'])} models, {len(self.trading_symbols)} symbols"
            )

        except Exception as e:
            print(f"âŒ Error loading model configuration: {e}")

    async def initialize_models(self):
        """Initialize deep learning models"""
        try:
            for model_name, model_config in self.model_config["models"].items():
                if model_config["enabled"]:
                    model = await self.create_model(model_name, model_config)
                    if model:
                        self.models[model_name] = model

                        # Initialize scaler for this model
                        self.scalers[model_name] = MinMaxScaler()

            print(f"ðŸ§  Models initialized: {len(self.models)} models")

        except Exception as e:
            print(f"âŒ Error initializing models: {e}")

    async def create_model(
        self, model_name: str, model_config: Dict[str, Any]
    ) -> Optional[nn.Module]:
        """Create a deep learning model"""
        try:
            model_type = model_config["type"]

            if model_type == "lstm":
                model = LSTMModel(
                    input_size=model_config["input_size"],
                    hidden_size=model_config["hidden_size"],
                    num_layers=model_config["num_layers"],
                    output_size=model_config["output_size"],
                )
            elif model_type == "cnn":
                model = CNNModel(
                    input_channels=model_config["input_channels"],
                    num_classes=model_config["num_classes"],
                )
            else:
                print(f"âŒ Unknown model type: {model_type}")
                return None

            model = model.to(self.device)
            print(f"âœ… Created {model_type} model: {model_name}")

            return model

        except Exception as e:
            print(f"âŒ Error creating model {model_name}: {e}")
            return None

    async def load_pretrained_models(self):
        """Load pre-trained models from disk"""
        try:
            models_dir = os.path.join(os.path.dirname(__file__), "..", "models")

            for model_name in self.models.keys():
                model_path = os.path.join(models_dir, f"{model_name}.pth")
                scaler_path = os.path.join(models_dir, f"{model_name}_scaler.pkl")

                if os.path.exists(model_path):
                    # Load model weights
                    self.models[model_name].load_state_dict(
                        torch.load(model_path, map_location=self.device)
                    )
                    self.models[model_name].eval()

                    # Load scaler
                    if os.path.exists(scaler_path):
                        self.scalers[model_name] = joblib.load(scaler_path)

                    print(f"âœ… Loaded pre-trained model: {model_name}")

        except Exception as e:
            print(f"âŒ Error loading pre-trained models: {e}")

    async def start_model_monitoring(self):
        """Start model monitoring"""
        try:
            # Subscribe to market data for model training
            pubsub = self.redis_client.pubsub()
            pubsub.subscribe("market_data")

            # Start market data listener
            asyncio.create_task(self.listen_market_data(pubsub))

            print("ðŸ“¡ Model monitoring started")

        except Exception as e:
            print(f"âŒ Error starting model monitoring: {e}")

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
        """Process market data for model training"""
        try:
            symbol = market_data.get("symbol")
            price = market_data.get("price")
            volume = market_data.get("volume", 0)
            timestamp = market_data.get("timestamp")

            # Store market data for model training
            if symbol and price and timestamp:
                await self.store_market_data(symbol, price, volume, timestamp)

        except Exception as e:
            print(f"âŒ Error processing market data: {e}")

    async def store_market_data(self, symbol: str, price: float, volume: float, timestamp: str):
        """Store market data for model training"""
        try:
            # Create data point
            data_point = {
                "symbol": symbol,
                "price": price,
                "volume": volume,
                "timestamp": timestamp,
            }

            # Store in Redis with expiration
            cache_key = f"dl_data:{symbol}:{timestamp}"
            self.redis_client.set(cache_key, json.dumps(data_point), ex=3600)

            # Update symbol data cache
            if symbol not in self.state["model_cache"]:
                self.state["model_cache"][symbol] = []

            self.state["model_cache"][symbol].append(data_point)

            # Keep only recent data points
            if len(self.state["model_cache"][symbol]) > 2000:
                self.state["model_cache"][symbol] = self.state["model_cache"][symbol][-2000:]

        except Exception as e:
            print(f"âŒ Error storing market data: {e}")

    async def train_models_if_needed(self):
        """Train models if they need updating"""
        try:
            for symbol in self.trading_symbols:
                # Check if we have enough data for training
                if symbol in self.state["model_cache"]:
                    data_points = len(self.state["model_cache"][symbol])

                    if data_points >= self.model_config["training_settings"]["min_data_points"]:
                        # Check if model needs retraining
                        last_training = (
                            self.state["training_history"].get(symbol, {}).get("last_training")
                        )

                        if not last_training or self.should_retrain_model(symbol, last_training):
                            await self.train_symbol_models(symbol)

        except Exception as e:
            print(f"âŒ Error training models: {e}")

    def should_retrain_model(self, symbol: str, last_training: str) -> bool:
        """Check if model should be retrained"""
        try:
            if not last_training:
                return True

            last_training_time = datetime.fromisoformat(last_training)
            current_time = datetime.now()

            # Retrain every 24 hours
            return (current_time - last_training_time).total_seconds() > 86400

        except Exception as e:
            print(f"âŒ Error checking retrain condition: {e}")
            return True

    async def train_symbol_models(self, symbol: str):
        """Train models for a specific symbol"""
        try:
            print(f"ðŸ‹ï¸ Training models for {symbol}...")

            # Get market data
            market_data = await self.get_symbol_market_data(symbol)

            if (
                not market_data
                or len(market_data) < self.model_config["training_settings"]["min_data_points"]
            ):
                return

            # Train each model
            for model_name, model in self.models.items():
                try:
                    await self.train_model(symbol, model_name, model, market_data)
                except Exception as e:
                    print(f"âŒ Error training {model_name} for {symbol}: {e}")

            # Update training history
            if symbol not in self.state["training_history"]:
                self.state["training_history"][symbol] = {}

            self.state["training_history"][symbol]["last_training"] = datetime.now().isoformat()
            self.state["training_count"] += 1

            print(f"âœ… Model training complete for {symbol}")

        except Exception as e:
            print(f"âŒ Error training models for {symbol}: {e}")

    async def get_symbol_market_data(self, symbol: str) -> List[Dict[str, Any]]:
        """Get market data for a symbol"""
        try:
            # Get from cache first
            if symbol in self.state["model_cache"]:
                return self.state["model_cache"][symbol]

            # Get from Redis
            pattern = f"dl_data:{symbol}:*"
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
            print(f"âŒ Error getting market data for {symbol}: {e}")
            return []

    async def train_model(
        self,
        symbol: str,
        model_name: str,
        model: nn.Module,
        market_data: List[Dict[str, Any]],
    ):
        """Train a specific model"""
        try:
            self.model_config["models"][model_name]
            training_config = self.model_config["training_settings"]

            # Prepare data
            X, y = await self.prepare_training_data(symbol, model_name, market_data)

            if X is None or y is None:
                return

            # Split data
            split_idx = int(len(X) * (1 - training_config["validation_split"]))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]

            # Create data loaders
            train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
            val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))

            train_loader = DataLoader(
                train_dataset,
                batch_size=training_config["batch_size"],
                shuffle=True,
            )
            val_loader = DataLoader(val_dataset, batch_size=training_config["batch_size"])

            # Setup training
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=training_config["learning_rate"])

            # Training loop
            best_val_loss = float("inf")
            patience_counter = 0

            for epoch in range(training_config["epochs"]):
                # Training phase
                model.train()
                train_loss = 0
                for batch_X, batch_y in train_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()

                # Validation phase
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                        outputs = model(batch_X)
                        val_loss += criterion(outputs, batch_y).item()

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0

                    # Save best model
                    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
                    os.makedirs(models_dir, exist_ok=True)
                    torch.save(
                        model.state_dict(),
                        os.path.join(models_dir, f"{model_name}_{symbol}.pth"),
                    )

                    # Save scaler
                    scaler_path = os.path.join(models_dir, f"{model_name}_{symbol}_scaler.pkl")
                    joblib.dump(self.scalers[model_name], scaler_path)
                else:
                    patience_counter += 1

                if patience_counter >= training_config["early_stopping_patience"]:
                    print(f"ðŸ›‘ Early stopping at epoch {epoch}")
                    break

                if epoch % 10 == 0:
                    print(
                        f"ðŸ“Š Epoch {epoch}: Train Loss: {train_loss/len(train_loader):.6f}, Val Loss: {val_loss/len(val_loader):.6f}"
                    )

            # Update training history
            if symbol not in self.state["training_history"]:
                self.state["training_history"][symbol] = {}

            self.state["training_history"][symbol][model_name] = {
                "last_training": datetime.now().isoformat(),
                "best_val_loss": best_val_loss,
                "epochs_trained": epoch + 1,
            }

            print(f"âœ… {model_name} training complete for {symbol}")

        except Exception as e:
            print(f"âŒ Error training {model_name} for {symbol}: {e}")

    async def prepare_training_data(
        self, symbol: str, model_name: str, market_data: List[Dict[str, Any]]
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Prepare training data for a model"""
        try:
            model_config = self.model_config["models"][model_name]
            model_type = model_config["type"]

            # Convert to DataFrame
            df = pd.DataFrame(market_data)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp")

            if model_type == "lstm":
                return await self.prepare_lstm_data(df, model_config)
            elif model_type == "cnn":
                return await self.prepare_cnn_data(df, model_config)
            else:
                return None, None

        except Exception as e:
            print(f"âŒ Error preparing training data: {e}")
            return None, None

    async def prepare_lstm_data(
        self, df: pd.DataFrame, model_config: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for LSTM model"""
        try:
            sequence_length = model_config["sequence_length"]

            # Create features
            features = []
            for i in range(len(df) - sequence_length):
                sequence = df.iloc[i : i + sequence_length]

                # Price features
                price_seq = sequence["price"].values
                volume_seq = sequence["volume"].values

                # Technical features
                price_change = np.diff(price_seq)
                volume_change = np.diff(volume_seq)

                # Combine features
                feature_seq = np.column_stack(
                    [
                        price_seq[:-1],  # Price
                        volume_seq[:-1],  # Volume
                        price_change,  # Price change
                        volume_change,  # Volume change
                        np.arange(len(price_change)),  # Time index
                    ]
                )

                features.append(feature_seq)

            # Create targets (next price)
            targets = df["price"].values[sequence_length:]

            # Normalize features
            features_array = np.array(features)
            scaler = self.scalers.get("lstm_price_predictor", MinMaxScaler())
            features_normalized = scaler.fit_transform(
                features_array.reshape(-1, features_array.shape[-1])
            ).reshape(features_array.shape)

            self.scalers["lstm_price_predictor"] = scaler

            return features_normalized, targets

        except Exception as e:
            print(f"âŒ Error preparing LSTM data: {e}")
            return None, None

    async def prepare_cnn_data(
        self, df: pd.DataFrame, model_config: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for CNN model"""
        try:
            sequence_length = model_config["sequence_length"]

            # Create features
            features = []
            labels = []  # Pattern labels (simplified for now)

            for i in range(len(df) - sequence_length):
                sequence = df.iloc[i : i + sequence_length]

                # Price features
                price_seq = sequence["price"].values
                volume_seq = sequence["volume"].values

                # Technical features
                price_change = np.diff(price_seq)
                volume_change = np.diff(volume_seq)

                # Combine features
                feature_seq = np.column_stack(
                    [
                        price_seq[:-1],  # Price
                        volume_seq[:-1],  # Volume
                        price_change,  # Price change
                        volume_change,  # Volume change
                        np.arange(len(price_change)),  # Time index
                    ]
                )

                features.append(feature_seq.T)  # Transpose for CNN

                # Simple pattern label (0-13 for different patterns)
                # This is a simplified approach - in production, you'd use actual pattern detection
                label = i % 14
                labels.append(label)

            # Normalize features
            features_array = np.array(features)
            scaler = self.scalers.get("cnn_pattern_recognizer", MinMaxScaler())
            features_normalized = scaler.fit_transform(
                features_array.reshape(-1, features_array.shape[-1])
            ).reshape(features_array.shape)

            self.scalers["cnn_pattern_recognizer"] = scaler

            return features_normalized, np.array(labels)

        except Exception as e:
            print(f"âŒ Error preparing CNN data: {e}")
            return None, None

    async def make_all_predictions(self):
        """Make predictions for all symbols"""
        try:
            print(f"ðŸ”® Making predictions for {len(self.trading_symbols)} symbols...")

            for symbol in self.trading_symbols:
                try:
                    await self.make_symbol_predictions(symbol)
                except Exception as e:
                    print(f"âŒ Error making predictions for {symbol}: {e}")

            print("âœ… Predictions complete")

        except Exception as e:
            print(f"âŒ Error making all predictions: {e}")

    async def make_symbol_predictions(self, symbol: str):
        """Make predictions for a specific symbol"""
        try:
            # Get recent market data
            market_data = await self.get_symbol_market_data(symbol)

            if not market_data or len(market_data) < 100:
                return

            predictions = {}

            # Make predictions with each model
            for model_name, model in self.models.items():
                try:
                    prediction = await self.make_model_prediction(
                        symbol, model_name, model, market_data
                    )
                    if prediction:
                        predictions[model_name] = prediction
                except Exception as e:
                    print(f"âŒ Error making prediction with {model_name} for {symbol}: {e}")

            # Store predictions
            if predictions:
                self.state["predictions_made"][symbol] = {
                    "predictions": predictions,
                    "timestamp": datetime.now().isoformat(),
                }

                # Broadcast predictions
                await self.broadcast_predictions(symbol, predictions)

        except Exception as e:
            print(f"âŒ Error making predictions for {symbol}: {e}")

    async def make_model_prediction(
        self,
        symbol: str,
        model_name: str,
        model: nn.Module,
        market_data: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Make prediction with a specific model"""
        try:
            model_config = self.model_config["models"][model_name]
            model_type = model_config["type"]

            # Prepare input data
            if model_type == "lstm":
                X = await self.prepare_lstm_prediction_data(market_data, model_config)
            elif model_type == "cnn":
                X = await self.prepare_cnn_prediction_data(market_data, model_config)
            else:
                return None

            if X is None:
                return None

            # Make prediction
            model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X).unsqueeze(0).to(self.device)
                prediction = model(X_tensor)
                prediction_value = prediction.cpu().numpy()[0][0]

            # Calculate confidence (simplified)
            confidence = 0.8  # In production, you'd calculate actual confidence

            return {
                "value": float(prediction_value),
                "confidence": confidence,
                "model_type": model_type,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            print(f"âŒ Error making prediction with {model_name}: {e}")
            return None

    async def prepare_lstm_prediction_data(
        self, market_data: List[Dict[str, Any]], model_config: Dict[str, Any]
    ) -> Optional[np.ndarray]:
        """Prepare data for LSTM prediction"""
        try:
            sequence_length = model_config["sequence_length"]

            if len(market_data) < sequence_length:
                return None

            # Get recent sequence
            recent_data = market_data[-sequence_length:]

            # Create features
            price_seq = [d["price"] for d in recent_data]
            volume_seq = [d["volume"] for d in recent_data]

            # Technical features
            price_change = np.diff(price_seq)
            volume_change = np.diff(volume_seq)

            # Combine features
            feature_seq = np.column_stack(
                [
                    price_seq[:-1],  # Price
                    volume_seq[:-1],  # Volume
                    price_change,  # Price change
                    volume_change,  # Volume change
                    np.arange(len(price_change)),  # Time index
                ]
            )

            # Normalize features
            scaler = self.scalers.get("lstm_price_predictor")
            if scaler:
                feature_normalized = scaler.transform(feature_seq)
                return feature_normalized

            return feature_seq

        except Exception as e:
            print(f"âŒ Error preparing LSTM prediction data: {e}")
            return None

    async def prepare_cnn_prediction_data(
        self, market_data: List[Dict[str, Any]], model_config: Dict[str, Any]
    ) -> Optional[np.ndarray]:
        """Prepare data for CNN prediction"""
        try:
            sequence_length = model_config["sequence_length"]

            if len(market_data) < sequence_length:
                return None

            # Get recent sequence
            recent_data = market_data[-sequence_length:]

            # Create features
            price_seq = [d["price"] for d in recent_data]
            volume_seq = [d["volume"] for d in recent_data]

            # Technical features
            price_change = np.diff(price_seq)
            volume_change = np.diff(volume_seq)

            # Combine features
            feature_seq = np.column_stack(
                [
                    price_seq[:-1],  # Price
                    volume_seq[:-1],  # Volume
                    price_change,  # Price change
                    volume_change,  # Volume change
                    np.arange(len(price_change)),  # Time index
                ]
            )

            # Normalize features
            scaler = self.scalers.get("cnn_pattern_recognizer")
            if scaler:
                feature_normalized = scaler.transform(feature_seq)
                return feature_normalized.T.reshape(1, 5, -1)  # Reshape for CNN

            return feature_seq.T.reshape(1, 5, -1)

        except Exception as e:
            print(f"âŒ Error preparing CNN prediction data: {e}")
            return None

    async def broadcast_predictions(self, symbol: str, predictions: Dict[str, Any]):
        """Broadcast predictions to other agents"""
        try:
            prediction_update = {
                "type": "deep_learning_prediction_update",
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
            print(f"âŒ Error broadcasting predictions: {e}")

    async def handle_train_model(self, message: Dict[str, Any]):
        """Handle manual model training request"""
        try:
            symbol = message.get("symbol")
            model_name = message.get("model_name")

            print(f"ðŸ‹ï¸ Manual model training requested for {symbol}")

            if symbol:
                if model_name and model_name in self.models:
                    # Train specific model
                    market_data = await self.get_symbol_market_data(symbol)
                    if market_data:
                        await self.train_model(
                            symbol,
                            model_name,
                            self.models[model_name],
                            market_data,
                        )
                else:
                    # Train all models
                    await self.train_symbol_models(symbol)

            # Send response
            response = {
                "type": "model_training_complete",
                "symbol": symbol,
                "model_name": model_name,
                "timestamp": datetime.now().isoformat(),
            }

            sender = message.get("from_agent")
            if sender:
                await self.send_message(sender, response)

        except Exception as e:
            print(f"âŒ Error handling model training request: {e}")
            await self.broadcast_error(f"Model training error: {e}")

    async def handle_make_prediction(self, message: Dict[str, Any]):
        """Handle manual prediction request"""
        try:
            symbol = message.get("symbol")
            model_name = message.get("model_name")

            print(f"ðŸ”® Manual prediction requested for {symbol}")

            if symbol:
                market_data = await self.get_symbol_market_data(symbol)
                if market_data:
                    if model_name and model_name in self.models:
                        # Make prediction with specific model
                        prediction = await self.make_model_prediction(
                            symbol,
                            model_name,
                            self.models[model_name],
                            market_data,
                        )

                        response = {
                            "type": "prediction_response",
                            "symbol": symbol,
                            "model_name": model_name,
                            "prediction": prediction,
                            "timestamp": datetime.now().isoformat(),
                        }
                    else:
                        # Make predictions with all models
                        await self.make_symbol_predictions(symbol)

                        response = {
                            "type": "prediction_response",
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
                        "type": "prediction_response",
                        "symbol": symbol,
                        "prediction": None,
                        "error": "No market data available",
                        "timestamp": datetime.now().isoformat(),
                    }
            else:
                response = {
                    "type": "prediction_response",
                    "symbol": symbol,
                    "prediction": None,
                    "error": "No symbol provided",
                    "timestamp": datetime.now().isoformat(),
                }

            sender = message.get("from_agent")
            if sender:
                await self.send_message(sender, response)

        except Exception as e:
            print(f"âŒ Error handling prediction request: {e}")
            await self.broadcast_error(f"Prediction error: {e}")

    async def handle_get_model_status(self, message: Dict[str, Any]):
        """Handle model status request"""
        try:
            symbol = message.get("symbol")

            print(f"ðŸ“Š Model status requested for {symbol}")

            # Get model status
            if symbol and symbol in self.state["training_history"]:
                status = self.state["training_history"][symbol]

                response = {
                    "type": "model_status_response",
                    "symbol": symbol,
                    "status": status,
                    "timestamp": datetime.now().isoformat(),
                }
            else:
                response = {
                    "type": "model_status_response",
                    "symbol": symbol,
                    "status": None,
                    "timestamp": datetime.now().isoformat(),
                }

            sender = message.get("from_agent")
            if sender:
                await self.send_message(sender, response)

        except Exception as e:
            print(f"âŒ Error handling model status request: {e}")
            await self.broadcast_error(f"Model status error: {e}")

    async def update_model_metrics(self):
        """Update model metrics"""
        try:
            metrics = {
                "agent_id": self.agent_id,
                "symbols_count": len(self.trading_symbols),
                "models_count": len(self.models),
                "predictions_count": len(self.state["predictions_made"]),
                "training_count": self.state["training_count"],
                "last_training": self.state["last_training"],
                "cache_size": len(self.state["model_cache"]),
                "device": str(self.device),
                "timestamp": datetime.now().isoformat(),
            }

            # Store metrics in Redis
            self.redis_client.set(f"agent_metrics:{self.agent_id}", json.dumps(metrics), ex=300)

        except Exception as e:
            print(f"âŒ Error updating model metrics: {e}")

    async def cleanup_cache(self):
        """Clean up old cache entries"""
        try:
            current_time = datetime.now()

            # Clean up old model cache entries
            for symbol in list(self.state["model_cache"].keys()):
                data_points = self.state["model_cache"][symbol]

                # Keep only recent data points (last 48 hours)
                cutoff_time = current_time - timedelta(hours=48)
                recent_data = [
                    point
                    for point in data_points
                    if datetime.fromisoformat(point["timestamp"]) > cutoff_time
                ]

                if recent_data:
                    self.state["model_cache"][symbol] = recent_data
                else:
                    del self.state["model_cache"][symbol]

        except Exception as e:
            print(f"âŒ Error cleaning up cache: {e}")

    async def handle_market_data(self, message: Dict[str, Any]):
        """Handle market data message"""
        try:
            market_data = message.get("market_data", {})
            print(f"ðŸ“Š Deep Learning Agent received market data for {len(market_data)} symbols")
            
            # Process market data
            await self.process_market_data(market_data)
            
            # Store market data for each symbol
            for symbol, data in market_data.items():
                if symbol in self.trading_symbols:
                    price = data.get("price", 0)
                    volume = data.get("volume", 0)
                    timestamp = data.get("timestamp", datetime.now().isoformat())
                    
                    await self.store_market_data(symbol, price, volume, timestamp)
            
            # Check if models need retraining
            await self.train_models_if_needed()
            
            # Make predictions with new data
            await self.make_all_predictions()
            
        except Exception as e:
            print(f"âŒ Error handling market data: {e}")
            await self.broadcast_error(f"Market data handling error: {e}")


