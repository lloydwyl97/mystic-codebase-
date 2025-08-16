﻿"""
AI Strategy Generator Service
Advanced neural network-based strategy generation and signal optimization
"""

import asyncio
import json
import os
import redis
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import joblib
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="AI Strategy Generator", version="1.0.0")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return JSONResponse(
        status_code=200,
        content={
            "status": "healthy",
            "service": "ai-strategy-generator",
            "timestamp": datetime.now().isoformat(),
        },
    )


@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "AI Strategy Generator is running"}


class LSTMPredictor(nn.Module):
    """LSTM Neural Network for price prediction"""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int,
        dropout: float = 0.2,
    ):
        super(LSTMPredictor, self).__init__()
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


class TransformerPredictor(nn.Module):
    """Transformer-based model for market analysis"""

    def __init__(
        self,
        input_size: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        output_size: int,
    ):
        super(TransformerPredictor, self).__init__()
        self.d_model = d_model
        self.input_projection = nn.Linear(input_size, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model * 4),
            num_layers,
        )
        self.output_projection = nn.Linear(d_model, output_size)

    def forward(self, x):
        x = self.input_projection(x)
        x = x.permute(1, 0, 2)  # (seq_len, batch, d_model)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # (batch, seq_len, d_model)
        x = self.output_projection(x[:, -1, :])  # Take last sequence output
        return x


class AIStrategyGenerator:
    def __init__(self):
        """Initialize AI Strategy Generator"""
        self.redis_client = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            db=int(os.getenv("REDIS_DB", 0)),
            decode_responses=True,
        )
        self.running = False
        self.models = {}
        self.scalers = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Model configurations
        self.model_configs = {
            "lstm": {
                "input_size": 10,
                "hidden_size": 128,
                "num_layers": 3,
                "output_size": 3,  # Buy, Hold, Sell
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
        """Start the AI Strategy Generator"""
        print("ðŸš€ Starting AI Strategy Generator...")
        self.running = True

        # Start strategy generation
        await self.generate_strategies()

    async def generate_strategies(self):
        """Generate AI strategies continuously"""
        print("ðŸ§  Starting AI strategy generation...")

        while self.running:
            try:
                # Check for strategy generation requests
                request = self.redis_client.lpop("ai_strategy_queue")

                if request:
                    request_data = json.loads(request)
                    await self.process_strategy_request(request_data)
                else:
                    # Generate periodic strategies
                    await self.generate_periodic_strategies()

                await asyncio.sleep(300)  # Check every 5 minutes

            except Exception as e:
                print(f"âŒ Error in strategy generation: {e}")
                await asyncio.sleep(600)

    async def process_strategy_request(self, request_data: Dict[str, Any]):
        """Process individual strategy generation request"""
        try:
            strategy_type = request_data.get("type", "lstm")
            symbol = request_data.get("symbol", "BTC/USDT")
            parameters = request_data.get("parameters", {})

            print(f"ðŸŽ¯ Generating {strategy_type} strategy for {symbol}")

            # Generate strategy
            strategy = await self.create_ai_strategy(strategy_type, symbol, parameters)

            if strategy:
                # Store strategy
                await self.store_strategy(strategy)

                # Publish strategy
                await self.publish_strategy(strategy)

                print(f"âœ… Generated strategy: {strategy['id']}")

        except Exception as e:
            print(f"âŒ Error processing strategy request: {e}")

    async def generate_periodic_strategies(self):
        """Generate strategies periodically"""
        try:
            symbols = ["BTC/USDT", "ETH/USDT", "ADA/USDT", "SOL/USDT"]
            strategy_types = ["lstm", "transformer"]

            for symbol in symbols:
                for strategy_type in strategy_types:
                    # Check if we need a new strategy
                    if await self.should_generate_strategy(symbol, strategy_type):
                        strategy = await self.create_ai_strategy(strategy_type, symbol)
                        if strategy:
                            await self.store_strategy(strategy)
                            await self.publish_strategy(strategy)
                            print(f"âœ… Generated periodic strategy: {strategy['id']}")

        except Exception as e:
            print(f"âŒ Error in periodic strategy generation: {e}")

    async def create_ai_strategy(
        self,
        strategy_type: str,
        symbol: str,
        parameters: Dict[str, Any] = None,
    ) -> Optional[Dict[str, Any]]:
        """Create an AI strategy"""
        try:
            # Get historical data
            data = await self.get_historical_data(symbol)

            if data.empty:
                return None

            # Prepare features
            features = self.prepare_features(data)

            # Train model
            model, scaler = await self.train_model(strategy_type, features, parameters)

            if model is None:
                return None

            # Generate strategy configuration
            strategy_config = self.generate_strategy_config(strategy_type, parameters)

            # Create strategy object
            strategy = {
                "id": (f"AI_{strategy_type.upper()}_{datetime.now().strftime('%Y%m%d%H%M%S')}"),
                "name": f"AI {strategy_type.upper()} Strategy",
                "type": strategy_type,
                "symbol": symbol,
                "model_type": strategy_type,
                "status": "ACTIVE",
                "created_at": datetime.now().isoformat(),
                "parameters": strategy_config,
                "performance": {
                    "accuracy": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1_score": 0.0,
                },
                "model_path": (
                    f"models/{strategy_type}_{symbol.replace('/', '_')}_{datetime.now().strftime('%Y%m%d%H%M%S')}.pth"
                ),
                "scaler_path": (
                    f"scalers/{strategy_type}_{symbol.replace('/', '_')}_{datetime.now().strftime('%Y%m%d%H%M%S')}.pkl"
                ),
            }

            # Save model and scaler
            await self.save_model(model, scaler, strategy)

            return strategy

        except Exception as e:
            print(f"âŒ Error creating AI strategy: {e}")
            return None

    async def get_historical_data(self, symbol: str) -> pd.DataFrame:
        """Get historical data for training"""
        try:
            # Generate realistic historical data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)
            dates = pd.date_range(start=start_date, end=end_date, freq="H")

            np.random.seed(42)
            initial_price = 45000 if "BTC" in symbol else 3000 if "ETH" in symbol else 100

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
            print(f"Error getting historical data: {e}")
            return pd.DataFrame()

    def prepare_features(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features for model training"""
        try:
            # Calculate technical indicators
            data["sma_20"] = data["close"].rolling(window=20).mean()
            data["sma_50"] = data["close"].rolling(window=50).mean()
            data["rsi"] = self.calculate_rsi(data["close"])
            data["macd"] = self.calculate_macd(data["close"])
            data["bb_upper"], data["bb_middle"], data["bb_lower"] = self.calculate_bollinger_bands(
                data["close"]
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
                    (data["close"].iloc[i + 1] - data["close"].iloc[i]) / data["close"].iloc[i]
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
            print(f"Error preparing features: {e}")
            return np.array([]), np.array([])

    async def train_model(
        self,
        strategy_type: str,
        features: np.ndarray,
        parameters: Dict[str, Any] = None,
    ) -> Tuple[Optional[nn.Module], Optional[MinMaxScaler]]:
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
            X_sequences, y_sequences = self.create_sequences(X_scaled, y, sequence_length)

            if len(X_sequences) == 0:
                return None, None

            # Split data
            split_idx = int(0.8 * len(X_sequences))
            X_train, X_test = X_sequences[:split_idx], X_sequences[split_idx:]
            y_train, y_test = y_sequences[:split_idx], y_sequences[split_idx:]

            # Create model
            if strategy_type == "lstm":
                config = self.model_configs["lstm"]
                model = LSTMPredictor(
                    input_size=config["input_size"],
                    hidden_size=config["hidden_size"],
                    num_layers=config["num_layers"],
                    output_size=config["output_size"],
                )
            elif strategy_type == "transformer":
                config = self.model_configs["transformer"]
                model = TransformerPredictor(
                    input_size=config["input_size"],
                    d_model=config["d_model"],
                    nhead=config["nhead"],
                    num_layers=config["num_layers"],
                    output_size=config["output_size"],
                )
            else:
                return None, None

            model.to(self.device)

            # Training parameters
            learning_rate = parameters.get("learning_rate", 0.001) if parameters else 0.001
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
                    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

            # Evaluate model
            model.eval()
            with torch.no_grad():
                test_outputs = model(X_test_tensor)
                test_predictions = torch.argmax(test_outputs, dim=1)
                accuracy = accuracy_score(y_test_tensor.cpu(), test_predictions.cpu())
                print(f"Model accuracy: {accuracy:.4f}")

            return model, scaler

        except Exception as e:
            print(f"Error training model: {e}")
            return None, None

    def create_sequences(
        self, X: np.ndarray, y: np.ndarray, sequence_length: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for time series prediction"""
        try:
            X_sequences, y_sequences = [], []

            for i in range(sequence_length, len(X)):
                X_sequences.append(X[i - sequence_length : i])
                y_sequences.append(y[i])

            return np.array(X_sequences), np.array(y_sequences)

        except Exception as e:
            print(f"Error creating sequences: {e}")
            return np.array([]), np.array([])

    def generate_strategy_config(
        self, strategy_type: str, parameters: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Generate strategy configuration"""
        config = {
            "model_type": strategy_type,
            "sequence_length": self.model_configs[strategy_type]["sequence_length"],
            "confidence_threshold": (
                parameters.get("confidence_threshold", 0.7) if parameters else 0.7
            ),
            "position_size": (parameters.get("position_size", 0.1) if parameters else 0.1),
            "stop_loss": (parameters.get("stop_loss", 0.05) if parameters else 0.05),
            "take_profit": (parameters.get("take_profit", 0.1) if parameters else 0.1),
            "max_positions": (parameters.get("max_positions", 3) if parameters else 3),
        }

        return config

    async def save_model(self, model: nn.Module, scaler: MinMaxScaler, strategy: Dict[str, Any]):
        """Save model and scaler"""
        try:
            # Create directories if they don't exist
            os.makedirs("models", exist_ok=True)
            os.makedirs("scalers", exist_ok=True)

            # Save model
            torch.save(model.state_dict(), strategy["model_path"])

            # Save scaler
            joblib.dump(scaler, strategy["scaler_path"])

            print(f"âœ… Saved model: {strategy['model_path']}")

        except Exception as e:
            print(f"Error saving model: {e}")

    async def should_generate_strategy(self, symbol: str, strategy_type: str) -> bool:
        """Check if we should generate a new strategy"""
        try:
            # Check last strategy generation time
            last_generation = self.redis_client.get(
                f"last_strategy_generation:{symbol}:{strategy_type}"
            )

            if last_generation is None:
                return True

            last_time = datetime.fromisoformat(last_generation)
            time_diff = datetime.now() - last_time

            # Generate new strategy every 24 hours
            return time_diff.total_seconds() > 86400

        except Exception as e:
            print(f"Error checking strategy generation: {e}")
            return False

    async def store_strategy(self, strategy: Dict[str, Any]):
        """Store strategy in Redis"""
        try:
            self.redis_client.set(f"ai_strategy:{strategy['id']}", json.dumps(strategy), ex=86400)
            self.redis_client.lpush("ai_strategies", strategy["id"])
            self.redis_client.ltrim("ai_strategies", 0, 99)  # Keep only 100 strategies

            # Update last generation time
            self.redis_client.set(
                f"last_strategy_generation:{strategy['symbol']}:{strategy['type']}",
                datetime.now().isoformat(),
                ex=86400,
            )

        except Exception as e:
            print(f"Error storing strategy: {e}")

    async def publish_strategy(self, strategy: Dict[str, Any]):
        """Publish strategy to Redis channels"""
        try:
            self.redis_client.publish("ai_strategies", json.dumps(strategy))
        except Exception as e:
            print(f"Error publishing strategy: {e}")

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
        """Stop the AI Strategy Generator"""
        print("ðŸ›‘ Stopping AI Strategy Generator...")
        self.running = False


async def main():
    """Main function"""
    generator = AIStrategyGenerator()

    # Start the strategy generator in the background
    async def run_generator():
        try:
            await generator.start()
        except Exception as e:
            print(f"âŒ Error in strategy generator: {e}")

    # Create background task
    background_task = asyncio.create_task(run_generator())

    # Get port from environment
    port = int(os.getenv("SERVICE_PORT", 8002))

    # Start FastAPI server
    config = uvicorn.Config(app=app, host="0.0.0.0", port=port, log_level="info")
    server = uvicorn.Server(config)

    try:
        await server.serve()
    except KeyboardInterrupt:
        print("ðŸ›‘ Received interrupt signal")
    except Exception as e:
        print(f"âŒ Error in main: {e}")
    finally:
        await generator.stop()
        background_task.cancel()
        try:
            await background_task
        except asyncio.CancelledError:
            pass


if __name__ == "__main__":
    asyncio.run(main())


