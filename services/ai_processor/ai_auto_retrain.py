"""
AI Auto-Retrain Service
Automatic model retraining and optimization system
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Any

import joblib  # type: ignore[reportMissingTypeStubs]
import numpy as np
import pandas as pd
import redis
import torch
import torch.nn as nn
import torch.optim as optim
from dotenv import load_dotenv
from numpy.typing import NDArray
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import MinMaxScaler

from utils.redis_helpers import to_str, to_str_list

# Configure logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))


class AutoRetrainService:
    def __init__(self):
        """Initialize Auto-Retrain Service"""
        self.redis_client = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            db=int(os.getenv("REDIS_DB", 0)),
            decode_responses=True,
        )
        self.running = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Retraining parameters
        self.retrain_threshold = 0.05  # 5% performance degradation
        self.min_data_points = 1000
        self.retrain_interval_hours = 24
        self.performance_window_days = 7

        # Model configurations
        self.model_configs = {
            "lstm": {
                "input_size": 10,
                "hidden_size": 128,
                "num_layers": 3,
                "output_size": 3,
                "sequence_length": 60,
            },
            "transformer": {
                "input_size": 10,
                "d_model": 128,
                "nhead": 8,
                "num_layers": 4,
                "output_size": 3,
                "sequence_length": 60,
            },
        }

    async def start(self):
        """Start the Auto-Retrain Service"""
        logger.info("🔄 Starting Auto-Retrain Service...")
        self.running = True

        # Start monitoring and retraining
        await self.monitor_and_retrain()

    async def monitor_and_retrain(self):
        """Monitor model performance and trigger retraining"""
        logger.info("👀 Monitoring model performance...")

        while self.running:
            try:
                # Check all active models
                active_models = to_str_list(self.redis_client.lrange("ai_strategies", 0, -1))

                for model_id in active_models:
                    await self.check_model_performance(model_id)

                # Check for retraining requests
                retrain_request = to_str(self.redis_client.lpop("retrain_queue"))
                if retrain_request:
                    request_data = json.loads(retrain_request)
                    await self.process_retrain_request(request_data)

                await asyncio.sleep(300)  # Check every 5 minutes

            except Exception as e:
                logger.error(f"❌ Error in monitoring: {e}")
                await asyncio.sleep(600)

    async def check_model_performance(self, model_id: str):
        """Check if model needs retraining"""
        try:
            model_data = self.redis_client.get(f"ai_strategy:{model_id}")
            if not model_data:
                return

            model = json.loads(model_data)

            # Get recent performance data
            recent_performance = await self.get_recent_performance(model_id)

            if not recent_performance:
                return

            # Check if performance has degraded
            if await self.should_retrain(model, recent_performance):
                logger.info(f"🔄 Model {model_id} needs retraining")

                # Add to retrain queue
                retrain_request = {
                    "model_id": model_id,
                    "reason": "performance_degradation",
                    "current_performance": recent_performance,
                    "timestamp": datetime.now().isoformat(),
                }

                self.redis_client.lpush("retrain_queue", json.dumps(retrain_request))

        except Exception as e:
            logger.error(f"Error checking model performance: {e}")

    async def get_recent_performance(self, model_id: str) -> dict[str, Any] | None:
        """Get recent performance data for model"""
        try:
            # Get performance data from last 7 days
            end_date = datetime.now()
            end_date - timedelta(days=self.performance_window_days)

            # This would typically come from a database or Redis
            # For now, we'll simulate performance data
            performance_data = {
                "accuracy": np.random.uniform(0.6, 0.8),
                "total_return": np.random.uniform(-0.1, 0.2),
                "sharpe_ratio": np.random.uniform(0.5, 1.5),
                "win_rate": np.random.uniform(0.4, 0.7),
                "total_trades": np.random.randint(10, 100),
            }

            return performance_data

        except Exception as e:
            logger.error(f"Error getting recent performance: {e}")
            return None

    async def should_retrain(
        self, model: dict[str, Any], recent_performance: dict[str, Any]
    ) -> bool:
        """Determine if model should be retrained"""
        try:
            # Get baseline performance
            baseline_performance = model.get("performance", {})

            if not baseline_performance:
                return False

            # Check accuracy degradation
            baseline_acc = baseline_performance.get("accuracy", 0)
            current_acc = recent_performance.get("accuracy", 0)

            if baseline_acc - current_acc > self.retrain_threshold:
                return True

            # Check return degradation
            baseline_return = baseline_performance.get("total_return", 0)
            current_return = recent_performance.get("total_return", 0)

            if baseline_return - current_return > self.retrain_threshold:
                return True

            # Check if enough time has passed since last retrain
            last_retrain = model.get("last_retrain")
            if last_retrain:
                last_retrain_time = datetime.fromisoformat(last_retrain)
                time_since_retrain = datetime.now() - last_retrain_time

                if (
                    time_since_retrain.total_seconds()
                    > self.retrain_interval_hours * 3600
                ):
                    return True

            return False

        except Exception as e:
            logger.error(f"Error checking if should retrain: {e}")
            return False

    async def process_retrain_request(self, request_data: dict[str, Any]):
        """Process retraining request"""
        try:
            model_id = request_data.get("model_id")
            reason = request_data.get("reason", "unknown")

            logger.info(f"🔄 Processing retrain request for {model_id} - Reason: {reason}")

            # Get model data
            model_data = self.redis_client.get(f"ai_strategy:{model_id}")
            if not model_data:
                logger.warning(f"❌ Model {model_id} not found")
                return

            model = json.loads(model_data)

            # Retrain model
            retrained_model = await self.retrain_model(model)

            if retrained_model:
                # Update model
                await self.update_model(model_id, retrained_model)

                # Notify versioning service
                self.redis_client.lpush("new_models_queue", json.dumps(retrained_model))

                logger.info(f"✅ Successfully retrained model {model_id}")
            else:
                logger.warning(f"❌ Failed to retrain model {model_id}")

        except Exception as e:
            logger.error(f"Error processing retrain request: {e}")

    async def retrain_model(self, model: dict[str, Any]) -> dict[str, Any] | None:
        """Retrain a model with new data"""
        try:
            model_type = model.get("type", "lstm")
            symbol = model.get("symbol", "BTC/USDT")

            logger.info(f"🔄 Retraining {model_type} model for {symbol}")

            # Get new training data
            new_data = await self.get_training_data(symbol)

            if new_data.empty:
                logger.warning("❌ No new training data available")
                return None

            # Prepare features
            features = self.prepare_features(new_data)

            if len(features[0]) == 0:
                logger.warning("❌ Failed to prepare features")
                return None

            # Train new model
            new_model, new_scaler = await self.train_model(
                model_type, features, model.get("parameters", {})
            )

            if new_model is None:
                logger.warning("❌ Failed to train new model")
                return None

            # Evaluate new model
            performance = await self.evaluate_model(new_model, new_scaler, new_data)

            # Create retrained model object
            retrained_model = {
                "id": (
                    f"{model['id']}_RETRAINED_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                ),
                "name": f"{model['name']} (Retrained)",
                "type": model_type,
                "symbol": symbol,
                "model_type": model_type,
                "status": "ACTIVE",
                "created_at": datetime.now().isoformat(),
                "parameters": model.get("parameters", {}),
                "performance": performance,
                "model_path": (
                    f"models/{model_type}_{symbol.replace('/', '_')}_retrained_{datetime.now().strftime('%Y%m%d%H%M%S')}.pth"
                ),
                "scaler_path": (
                    f"scalers/{model_type}_{symbol.replace('/', '_')}_retrained_{datetime.now().strftime('%Y%m%d%H%M%S')}.pkl"
                ),
                "parent_model": model["id"],
                "retrain_reason": "performance_degradation",
                "last_retrain": datetime.now().isoformat(),
            }

            # Save new model and scaler
            await self.save_model(new_model, new_scaler, retrained_model)

            return retrained_model

        except Exception as e:
            logger.error(f"Error retraining model: {e}")
            return None

    async def get_training_data(self, symbol: str) -> pd.DataFrame:
        """Get training data for retraining"""
        try:
            # Generate realistic historical data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)
            dates = pd.date_range(start=start_date, end=end_date, freq="H")

            np.random.seed(42)
            initial_price = (
                45000 if "BTC" in symbol else 3000 if "ETH" in symbol else 100
            )

            # Generate price data with trends and volatility
            returns = np.random.normal(0.0001, 0.02, len(dates))
            prices = [initial_price]

            for ret in returns[1:]:
                new_price = prices[-1] * (1 + ret)
                prices.append(max(new_price, 1))

            # Create OHLCV data
            data = pd.DataFrame(
                {
                    "timestamp": dates,
                    "open": prices,
                    "high": [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
                    "low": [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
                    "close": prices,
                    "volume": np.random.uniform(1000000, 5000000, len(dates)),
                }
            )

            # Ensure High >= Low
            data["high"] = data[["open", "high", "close"]].max(axis=1)
            data["low"] = data[["open", "low", "close"]].min(axis=1)

            data.set_index("timestamp", inplace=True)
            return data

        except Exception as e:
            logger.error(f"Error getting training data: {e}")
            return pd.DataFrame()

    def prepare_features(self, data: pd.DataFrame) -> tuple[NDArray[Any], NDArray[Any]]:
        """Prepare features for model training"""
        try:
            # Calculate technical indicators
            data["sma_20"] = data["close"].rolling(window=20).mean()
            data["sma_50"] = data["close"].rolling(window=50).mean()
            data["rsi"] = self.calculate_rsi(data["close"])
            data["macd"] = self.calculate_macd(data["close"])
            data["bb_upper"], data["bb_middle"], data["bb_lower"] = (
                self.calculate_bollinger_bands(data["close"])
            )
            data["volume_sma"] = data["volume"].rolling(window=20).mean()
            data["price_change"] = data["close"].pct_change()
            data["volatility"] = data["price_change"].rolling(window=20).std()

            # Create features
            feature_columns = [
                "close",
                "volume",
                "sma_20",
                "sma_50",
                "rsi",
                "macd",
                "bb_upper",
                "bb_middle",
                "bb_lower",
                "volatility",
            ]

            features = data[feature_columns].fillna(0).values

            # Create labels (simplified: 0=Hold, 1=Buy, 2=Sell)
            labels = np.zeros(len(features))

            # Simple labeling logic
            for i in range(60, len(features)):
                future_return = (
                    (data["close"].iloc[i + 1] - data["close"].iloc[i])
                    / data["close"].iloc[i]
                    if i + 1 < len(data)
                    else 0
                )

                if future_return > 0.01:  # 1% gain
                    labels[i] = 1  # Buy
                elif future_return < -0.01:  # 1% loss
                    labels[i] = 2  # Sell
                else:
                    labels[i] = 0  # Hold

            return features, labels

        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return np.array([]), np.array([])

    async def train_model(
        self,
        strategy_type: str,
        features: NDArray[Any],
        parameters: dict[str, Any] | None = None,
    ) -> tuple[nn.Module | None, MinMaxScaler | None]:
        """Train the AI model"""
        try:
            if len(features) == 0:
                return None, None

            # Prepare data
            X, y = features

            # Normalize features
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X)

            # Create sequences
            sequence_length = self.model_configs[strategy_type]["sequence_length"]
            X_sequences, y_sequences = self.create_sequences(
                X_scaled, y, sequence_length
            )

            if len(X_sequences) == 0:
                return None, None

            # Split data
            split_idx = int(0.8 * len(X_sequences))
            X_train, X_test = X_sequences[:split_idx], X_sequences[split_idx:]
            y_train, y_test = y_sequences[:split_idx], y_sequences[split_idx:]

            # Create model
            if strategy_type == "lstm":
                config = self.model_configs["lstm"]
                model = nn.LSTM(
                    input_size=config["input_size"],
                    hidden_size=config["hidden_size"],
                    num_layers=config["num_layers"],
                    batch_first=True,
                    dropout=0.2,
                )
                model.fc = nn.Linear(config["hidden_size"], config["output_size"])
            elif strategy_type == "transformer":
                config = self.model_configs["transformer"]
                model = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(config["d_model"], config["nhead"]),
                    config["num_layers"],
                )
                model.input_projection = nn.Linear(
                    config["input_size"], config["d_model"]
                )
                model.output_projection = nn.Linear(
                    config["d_model"], config["output_size"]
                )
            else:
                return None, None

            model.to(self.device)

            # Training parameters
            learning_rate = (
                parameters.get("learning_rate", 0.001) if parameters else 0.001
            )
            epochs = parameters.get("epochs", 50) if parameters else 50
            parameters.get("batch_size", 32) if parameters else 32

            # Train model
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

            # Convert to tensors
            X_train_tensor = torch.FloatTensor(X_train).to(self.device)
            y_train_tensor = torch.LongTensor(y_train).to(self.device)
            X_test_tensor = torch.FloatTensor(X_test).to(self.device)
            y_test_tensor = torch.LongTensor(y_test).to(self.device)

            # Training loop
            model.train()
            for epoch in range(epochs):
                optimizer.zero_grad()
                outputs = model(X_train_tensor)
                loss = criterion(outputs, y_train_tensor)
                loss.backward()
                optimizer.step()

                if epoch % 10 == 0:
                    logger.info(f"Epoch {epoch}, Loss: {loss.item():.4f}")

            # Evaluate model
            model.eval()
            with torch.no_grad():
                test_outputs = model(X_test_tensor)
                test_predictions = torch.argmax(test_outputs, dim=1)
                accuracy = accuracy_score(y_test_tensor.cpu(), test_predictions.cpu())
                logger.info(f"Retrained model accuracy: {accuracy:.4f}")

            return model, scaler

        except Exception as e:
            logger.error(f"Error training model: {e}")
            return None, None

    def create_sequences(
        self, X: NDArray[Any], y: NDArray[Any], sequence_length: int
    ) -> tuple[NDArray[Any], NDArray[Any]]:
        """Create sequences for time series prediction"""
        try:
            X_sequences, y_sequences = [], []

            for i in range(sequence_length, len(X)):
                X_sequences.append(X[i - sequence_length : i])
                y_sequences.append(y[i])

            return np.array(X_sequences), np.array(y_sequences)

        except Exception as e:
            logger.error(f"Error creating sequences: {e}")
            return np.array([]), np.array([])

    async def evaluate_model(
        self, model: nn.Module, scaler: MinMaxScaler, data: pd.DataFrame
    ) -> dict[str, Any]:
        """Evaluate model performance"""
        try:
            # Prepare test data
            features = self.prepare_features(data)
            if len(features[0]) == 0:
                return {}

            X, y = features
            X_scaled = scaler.transform(X)

            # Create sequences
            sequence_length = 60
            X_sequences, y_sequences = self.create_sequences(
                X_scaled, y, sequence_length
            )

            if len(X_sequences) == 0:
                return {}

            # Test model
            model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_sequences).to(self.device)
                outputs = model(X_tensor)
                predictions = torch.argmax(outputs, dim=1)

                # Calculate metrics
                accuracy = accuracy_score(y_sequences, predictions.cpu())
                precision = precision_score(
                    y_sequences,
                    predictions.cpu(),
                    average="weighted",
                    zero_division=0,
                )
                recall = recall_score(
                    y_sequences,
                    predictions.cpu(),
                    average="weighted",
                    zero_division=0,
                )
                f1 = f1_score(
                    y_sequences,
                    predictions.cpu(),
                    average="weighted",
                    zero_division=0,
                )

            return {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "total_return": 0.0,  # Would be calculated from backtest
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
            }

        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            return {}

    async def save_model(
        self,
        model: nn.Module,
        scaler: MinMaxScaler,
        model_data: dict[str, Any],
    ):
        """Save model and scaler"""
        try:
            # Create directories if they don't exist
            os.makedirs("models", exist_ok=True)
            os.makedirs("scalers", exist_ok=True)

            # Save model
            torch.save(model.state_dict(), model_data["model_path"])

            # Save scaler
            joblib.dump(scaler, model_data["scaler_path"])

            logger.info(f"✅ Saved retrained model: {model_data['model_path']}")

        except Exception as e:
            logger.error(f"Error saving model: {e}")

    async def update_model(self, model_id: str, new_model_data: dict[str, Any]):
        """Update model in Redis"""
        try:
            # Update model data
            self.redis_client.set(
                f"ai_strategy:{model_id}", json.dumps(new_model_data), ex=86400
            )

            # Update last retrain time
            self.redis_client.set(
                f"last_retrain:{model_id}",
                datetime.now().isoformat(),
                ex=86400,
            )

            # Broadcast model metrics update
            await self.broadcast_model_metrics()

            # Broadcast retrain status update
            await self.broadcast_retrain_status()

        except Exception as e:
            logger.error(f"Error updating model: {e}")

    async def broadcast_model_metrics(self):
        """Broadcast model performance metrics"""
        try:
            # Get all active models
            active_models = to_str_list(self.redis_client.lrange("ai_strategies", 0, -1))
            models_data = []

            for model_id in active_models:
                model_data = self.redis_client.get(f"ai_strategy:{model_id}")
                if model_data:
                    model = json.loads(model_data)
                    models_data.append(model)

            # Create metrics payload
            metrics_payload = {
                "models": models_data,
                "timestamp": datetime.now().isoformat(),
            }

            # Store in Redis for dashboard access
            self.redis_client.set("model_metrics", json.dumps(metrics_payload), ex=300)

            # Publish to Redis channel
            self.redis_client.publish("model_metrics", json.dumps(metrics_payload))

        except Exception as e:
            logger.error(f"Error broadcasting model metrics: {e}")

    async def broadcast_retrain_status(self):
        """Broadcast retrain status and queue"""
        try:
            # Get retrain queue
            queue_data = to_str_list(self.redis_client.lrange("retrain_queue", 0, -1))
            queue = [json.loads(item) for item in queue_data]

            # Get current retrain status
            status = {
                "currently_retraining": None,
                "retrain_progress": 0.0,
                "estimated_completion": None,
                "timestamp": datetime.now().isoformat(),
            }

            # Check if any model is currently retraining
            for item in queue:
                if item.get("status") == "retraining":
                    status["currently_retraining"] = item["model_id"]
                    status["retrain_progress"] = item.get("progress", 0.0)
                    status["estimated_completion"] = item.get("estimated_completion")
                    break

            # Create status payload
            status_payload = {
                "queue": queue,
                "status": status,
                "timestamp": datetime.now().isoformat(),
            }

            # Store in Redis for dashboard access
            self.redis_client.set("retrain_status", json.dumps(status_payload), ex=300)

            # Publish to Redis channel
            self.redis_client.publish("retrain_status", json.dumps(status_payload))

        except Exception as e:
            logger.error(f"Error broadcasting retrain status: {e}")

    # Technical indicator calculations
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def calculate_macd(
        self, prices: pd.Series, fast: int = 12, slow: int = 26
    ) -> pd.Series:
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        return ema_fast - ema_slow

    def calculate_bollinger_bands(
        self, prices: pd.Series, period: int = 20, std_dev: float = 2
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, sma, lower

    async def stop(self):
        """Stop the Auto-Retrain Service"""
        logger.info("🛑 Stopping Auto-Retrain Service...")
        self.running = False


async def main():
    """Main function"""
    retrain_service = AutoRetrainService()

    try:
        await retrain_service.start()
    except KeyboardInterrupt:
        logger.info("🛑 Received interrupt signal")
    except Exception as e:
        logger.error(f"❌ Error in main: {e}")
    finally:
        await retrain_service.stop()


if __name__ == "__main__":
    asyncio.run(main())
