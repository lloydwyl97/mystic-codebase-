"""
Reinforcement Learning Agent
Handles reinforcement learning for trading strategy optimization
"""

import asyncio
import json
import os
import random
import sys
from collections import deque
from datetime import datetime, timedelta
from typing import Any

import joblib  # type: ignore[reportMissingTypeStubs]
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

# Make all imports live (F401_ = Tuple[int, int]

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.agents.base_agent import BaseAgent


class DQNetwork(nn.Module):
    """Deep Q-Network for trading"""

    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        super(DQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


class PolicyNetwork(nn.Module):
    """Policy Network for Actor-Critic methods"""

    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return torch.softmax(x, dim=-1)


class ValueNetwork(nn.Module):
    """Value Network for Actor-Critic methods"""

    def __init__(self, state_size: int, hidden_size: int = 128):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


class TradingEnvironment:
    """Trading environment for reinforcement learning"""

    def __init__(
        self,
        data: pd.DataFrame,
        initial_balance: float = 10000.0,
        transaction_fee: float = 0.001,
    ):
        self.data = data
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        self.reset()

    def reset(self):
        """Reset environment to initial state"""
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0
        self.current_price = self.data.iloc[0]["price"]
        return self._get_state()

    def step(self, action: int):
        """Take action and return new state, reward, done"""
        # Actions: 0 = hold, 1 = buy, 2 = sell
        reward = 0
        done = False

        if self.current_step >= len(self.data) - 1:
            done = True
            return self._get_state(), reward, done

        self.current_price = self.data.iloc[self.current_step]["price"]
        next_price = self.data.iloc[self.current_step + 1]["price"]

        if action == 1:  # Buy
            if self.balance > 0:
                shares_to_buy = self.balance / (self.current_price * (1 + self.transaction_fee))
                self.shares_held += shares_to_buy
                self.balance -= shares_to_buy * self.current_price * (1 + self.transaction_fee)

        elif action == 2:  # Sell
            if self.shares_held > 0:
                self.total_shares_sold += self.shares_held
                self.total_sales_value += (
                    self.shares_held * self.current_price * (1 - self.transaction_fee)
                )
                self.balance += self.shares_held * self.current_price * (1 - self.transaction_fee)
                self.shares_held = 0

        # Calculate reward (price change)
        price_change = (next_price - self.current_price) / self.current_price
        reward = price_change

        self.current_step += 1

        return self._get_state(), reward, done

    def _get_state(self):
        """Get current state representation"""
        if self.current_step >= len(self.data):
            return np.zeros(10)

        # Price features
        current_price = self.data.iloc[self.current_step]["price"]
        price_change = (
            self.data.iloc[self.current_step]["price"]
            - self.data.iloc[max(0, self.current_step - 1)]["price"]
        )

        # Volume features
        current_volume = self.data.iloc[self.current_step]["volume"]
        volume_change = (
            self.data.iloc[self.current_step]["volume"]
            - self.data.iloc[max(0, self.current_step - 1)]["volume"]
        )

        # Technical indicators (simplified)
        if self.current_step >= 20:
            sma_20 = self.data.iloc[self.current_step - 20 : self.current_step]["price"].mean()
            price_vs_sma = (current_price - sma_20) / sma_20
        else:
            price_vs_sma = 0

        if self.current_step >= 50:
            sma_50 = self.data.iloc[self.current_step - 50 : self.current_step]["price"].mean()
            price_vs_sma_50 = (current_price - sma_50) / sma_50
        else:
            price_vs_sma_50 = 0

        # Portfolio state
        portfolio_value = self.balance + (self.shares_held * current_price)
        portfolio_return = (portfolio_value - self.initial_balance) / self.initial_balance

        # State vector
        state = np.array(
            [
                current_price / 1000,  # Normalized price
                price_change / 100,  # Normalized price change
                current_volume / 1000000,  # Normalized volume
                volume_change / 100000,  # Normalized volume change
                price_vs_sma,  # Price vs 20-day SMA
                price_vs_sma_50,  # Price vs 50-day SMA
                self.balance / self.initial_balance,  # Cash ratio
                self.shares_held / 100,  # Shares held
                portfolio_return,  # Portfolio return
                self.current_step / len(self.data),  # Progress
            ]
        )

        return state


class ReinforcementLearningAgent(BaseAgent):
    """Reinforcement Learning Agent - Uses RL for trading strategy optimization"""

    def __init__(self, agent_id: str = "reinforcement_learning_agent_001"):
        super().__init__(agent_id, "reinforcement_learning")

        # RL-specific state
        self.state.update(
            {
                "models_trained": {},
                "strategies_generated": {},
                "training_history": {},
                "last_training": None,
                "training_count": 0,
            }
        )

        # RL configuration
        self.rl_config = {
            "algorithms": {
                "dqn": {
                    "type": "deep_q_learning",
                    "state_size": 10,
                    "action_size": 3,
                    "hidden_size": 128,
                    "learning_rate": 0.001,
                    "gamma": 0.95,
                    "epsilon": 1.0,
                    "epsilon_min": 0.01,
                    "epsilon_decay": 0.995,
                    "memory_size": 10000,
                    "batch_size": 32,
                    "enabled": True,
                },
                "actor_critic": {
                    "type": "actor_critic",
                    "state_size": 10,
                    "action_size": 3,
                    "hidden_size": 128,
                    "learning_rate": 0.001,
                    "gamma": 0.95,
                    "enabled": True,
                },
            },
            "training_settings": {
                "episodes": 1000,
                "max_steps": 1000,
                "min_data_points": 500,
                "validation_split": 0.2,
                "early_stopping_patience": 50,
            },
            "strategy_settings": {
                "confidence_threshold": 0.7,
                "min_trades": 10,
                "max_position_size": 0.5,
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

        # Initialize models and environments
        self.models = {}
        self.environments = {}
        self.memories = {}
        self.scalers = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Register RL-specific handlers
        self.register_handler("train_rl_model", self.handle_train_rl_model)
        self.register_handler("generate_strategy", self.handle_generate_strategy)
        self.register_handler("get_rl_status", self.handle_get_rl_status)
        self.register_handler("market_data", self.handle_market_data)

        print(f"ðŸŽ¯ Reinforcement Learning Agent {agent_id} initialized on {self.device}")

    async def initialize(self):
        """Initialize reinforcement learning agent resources"""
        try:
            # Load RL configuration
            await self.load_rl_config()

            # Initialize models
            await self.initialize_models()

            # Load pre-trained models if available
            await self.load_pretrained_models()

            # Start RL monitoring
            await self.start_rl_monitoring()

            print(f"âœ… Reinforcement Learning Agent {self.agent_id} initialized successfully")

        except Exception as e:
            print(f"âŒ Error initializing Reinforcement Learning Agent: {e}")
            self.update_health_status("error")

    async def process_loop(self):
        """Main RL processing loop"""
        while self.running:
            try:
                # Train models if needed
                await self.train_models_if_needed()

                # Generate strategies for all symbols
                await self.generate_all_strategies()

                # Update RL metrics
                await self.update_rl_metrics()

                # Clean up old cache entries
                await self.cleanup_cache()

                await asyncio.sleep(300)  # Check every 5 minutes

            except Exception as e:
                print(f"âŒ Error in RL processing loop: {e}")
                await asyncio.sleep(600)

    async def load_rl_config(self):
        """Load RL configuration from Redis"""
        try:
            # Load RL configuration
            config_data = self.redis_client.get("reinforcement_learning_config")
            if config_data:
                self.rl_config = json.loads(config_data)

            # Load trading symbols
            symbols_data = self.redis_client.get("trading_symbols")
            if symbols_data:
                self.trading_symbols = json.loads(symbols_data)

            print(
                f"ðŸ“‹ RL configuration loaded: {len(self.rl_config['algorithms'])} algorithms, {len(self.trading_symbols)} symbols"
            )

        except Exception as e:
            print(f"âŒ Error loading RL configuration: {e}")

    async def initialize_models(self):
        """Initialize reinforcement learning models"""
        try:
            for algorithm_name, algorithm_config in self.rl_config["algorithms"].items():
                if algorithm_config["enabled"]:
                    model = await self.create_rl_model(algorithm_name, algorithm_config)
                    if model:
                        self.models[algorithm_name] = model

                        # Initialize memory for DQN
                        if algorithm_config["type"] == "deep_q_learning":
                            self.memories[algorithm_name] = deque(
                                maxlen=algorithm_config["memory_size"]
                            )

                        # Initialize scaler
                        self.scalers[algorithm_name] = StandardScaler()

            print(f"ðŸ§  RL models initialized: {len(self.models)} models")

        except Exception as e:
            print(f"âŒ Error initializing RL models: {e}")

    async def create_rl_model(
        self, algorithm_name: str, algorithm_config: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Create a reinforcement learning model"""
        try:
            algorithm_type = algorithm_config["type"]

            if algorithm_type == "deep_q_learning":
                # Create DQN model
                q_network = DQNetwork(
                    state_size=algorithm_config["state_size"],
                    action_size=algorithm_config["action_size"],
                    hidden_size=algorithm_config["hidden_size"],
                ).to(self.device)

                target_network = DQNetwork(
                    state_size=algorithm_config["state_size"],
                    action_size=algorithm_config["action_size"],
                    hidden_size=algorithm_config["hidden_size"],
                ).to(self.device)

                target_network.load_state_dict(q_network.state_dict())

                model = {
                    "type": "dqn",
                    "q_network": q_network,
                    "target_network": target_network,
                    "optimizer": optim.Adam(
                        q_network.parameters(),
                        lr=algorithm_config["learning_rate"],
                    ),
                    "config": algorithm_config,
                }

            elif algorithm_type == "actor_critic":
                # Create Actor-Critic model
                policy_network = PolicyNetwork(
                    state_size=algorithm_config["state_size"],
                    action_size=algorithm_config["action_size"],
                    hidden_size=algorithm_config["hidden_size"],
                ).to(self.device)

                value_network = ValueNetwork(
                    state_size=algorithm_config["state_size"],
                    hidden_size=algorithm_config["hidden_size"],
                ).to(self.device)

                model = {
                    "type": "actor_critic",
                    "policy_network": policy_network,
                    "value_network": value_network,
                    "policy_optimizer": optim.Adam(
                        policy_network.parameters(),
                        lr=algorithm_config["learning_rate"],
                    ),
                    "value_optimizer": optim.Adam(
                        value_network.parameters(),
                        lr=algorithm_config["learning_rate"],
                    ),
                    "config": algorithm_config,
                }

            else:
                print(f"âŒ Unknown algorithm type: {algorithm_type}")
                return None

            print(f"âœ… Created {algorithm_type} model: {algorithm_name}")
            return model

        except Exception as e:
            print(f"âŒ Error creating RL model {algorithm_name}: {e}")
            return None

    async def load_pretrained_models(self):
        """Load pre-trained RL models from disk"""
        try:
            models_dir = os.path.join(os.path.dirname(__file__), "..", "models")

            for algorithm_name in self.models.keys():
                model_path = os.path.join(models_dir, f"{algorithm_name}.pth")
                scaler_path = os.path.join(models_dir, f"{algorithm_name}_scaler.pkl")

                if os.path.exists(model_path):
                    # Load model weights
                    model = self.models[algorithm_name]
                    checkpoint = torch.load(model_path, map_location=self.device)

                    if model["type"] == "dqn":
                        model["q_network"].load_state_dict(checkpoint["q_network"])
                        model["target_network"].load_state_dict(checkpoint["target_network"])
                    elif model["type"] == "actor_critic":
                        model["policy_network"].load_state_dict(checkpoint["policy_network"])
                        model["value_network"].load_state_dict(checkpoint["value_network"])

                    # Load scaler
                    if os.path.exists(scaler_path):
                        self.scalers[algorithm_name] = joblib.load(scaler_path)

                    print(f"âœ… Loaded pre-trained RL model: {algorithm_name}")

        except Exception as e:
            print(f"âŒ Error loading pre-trained RL models: {e}")

    async def start_rl_monitoring(self):
        """Start RL monitoring"""
        try:
            # Subscribe to market data for RL training
            pubsub = self.redis_client.pubsub()
            pubsub.subscribe("market_data")

            # Start market data listener
            asyncio.create_task(self.listen_market_data(pubsub))

            print("ðŸ“¡ RL monitoring started")

        except Exception as e:
            print(f"âŒ Error starting RL monitoring: {e}")

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
        """Process market data for RL training"""
        try:
            symbol = market_data.get("symbol")
            price = market_data.get("price")
            volume = market_data.get("volume", 0)
            timestamp = market_data.get("timestamp")

            # Store market data for RL training
            if symbol and price and timestamp:
                await self.store_market_data(symbol, price, volume, timestamp)

        except Exception as e:
            print(f"âŒ Error processing market data: {e}")

    async def store_market_data(self, symbol: str, price: float, volume: float, timestamp: str):
        """Store market data for RL training"""
        try:
            # Create data point
            data_point = {
                "symbol": symbol,
                "price": price,
                "volume": volume,
                "timestamp": timestamp,
            }

            # Store in Redis with expiration
            cache_key = f"rl_data:{symbol}:{timestamp}"
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
            print(f"âŒ Error storing market data: {e}")

    async def train_models_if_needed(self):
        """Train RL models if they need updating"""
        try:
            for symbol in self.trading_symbols:
                # Check if we have enough data for training
                if symbol in self.state["training_history"]:
                    data_points = len(self.state["training_history"][symbol]["data"])

                    if data_points >= self.rl_config["training_settings"]["min_data_points"]:
                        # Check if model needs retraining
                        last_training = self.state["training_history"][symbol].get("last_training")

                        if not last_training or self.should_retrain_model(symbol, last_training):
                            await self.train_symbol_models(symbol)

        except Exception as e:
            print(f"âŒ Error training RL models: {e}")

    def should_retrain_model(self, symbol: str, last_training: str) -> bool:
        """Check if RL model should be retrained"""
        try:
            if not last_training:
                return True

            last_training_time = datetime.fromisoformat(last_training)
            current_time = datetime.now()

            # Retrain every 12 hours
            return (current_time - last_training_time).total_seconds() > 43200

        except Exception as e:
            print(f"âŒ Error checking retrain condition: {e}")
            return True

    async def train_symbol_models(self, symbol: str):
        """Train RL models for a specific symbol"""
        try:
            print(f"ðŸ‹ï¸ Training RL models for {symbol}...")

            # Get market data
            market_data = await self.get_symbol_market_data(symbol)

            if (
                not market_data
                or len(market_data) < self.rl_config["training_settings"]["min_data_points"]
            ):
                return

            # Create trading environment
            df = pd.DataFrame(market_data)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp")

            environment = TradingEnvironment(df)
            self.environments[symbol] = environment

            # Train each model
            for algorithm_name, model in self.models.items():
                try:
                    await self.train_rl_model(symbol, algorithm_name, model, environment)
                except Exception as e:
                    print(f"âŒ Error training {algorithm_name} for {symbol}: {e}")

            # Update training history
            if symbol not in self.state["training_history"]:
                self.state["training_history"][symbol] = {}

            self.state["training_history"][symbol]["last_training"] = datetime.now().isoformat()
            self.state["training_count"] += 1

            print(f"âœ… RL model training complete for {symbol}")

        except Exception as e:
            print(f"âŒ Error training RL models for {symbol}: {e}")

    async def get_symbol_market_data(self, symbol: str) -> list[dict[str, Any]]:
        """Get market data for a symbol"""
        try:
            # Get from training history
            if symbol in self.state["training_history"]:
                return self.state["training_history"][symbol].get("data", [])

            # Get from Redis
            pattern = f"rl_data:{symbol}:*"
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

    async def train_rl_model(
        self,
        symbol: str,
        algorithm_name: str,
        model: dict[str, Any],
        environment: TradingEnvironment,
    ):
        """Train a specific RL model"""
        try:
            algorithm_type = model["type"]
            training_config = self.rl_config["training_settings"]

            if algorithm_type == "dqn":
                await self.train_dqn(symbol, algorithm_name, model, environment, training_config)
            elif algorithm_type == "actor_critic":
                await self.train_actor_critic(
                    symbol, algorithm_name, model, environment, training_config
                )

        except Exception as e:
            print(f"âŒ Error training RL model {algorithm_name}: {e}")

    async def train_dqn(
        self,
        symbol: str,
        algorithm_name: str,
        model: dict[str, Any],
        environment: TradingEnvironment,
        training_config: dict[str, Any],
    ):
        """Train DQN model"""
        try:
            q_network = model["q_network"]
            target_network = model["target_network"]
            optimizer = model["optimizer"]
            config = model["config"]
            memory = self.memories[algorithm_name]

            best_reward = float("-inf")
            patience_counter = 0

            for episode in range(training_config["episodes"]):
                state = environment.reset()
                total_reward = 0

                for step in range(training_config["max_steps"]):
                    # Epsilon-greedy action selection
                    if random.random() < config["epsilon"]:
                        action = random.randint(0, config["action_size"] - 1)
                    else:
                        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                        q_values = q_network(state_tensor)
                        action = q_values.argmax().item()

                    # Take action
                    next_state, reward, done = environment.step(action)

                    # Store experience
                    memory.append((state, action, reward, next_state, done))

                    # Train if enough samples
                    if len(memory) >= config["batch_size"]:
                        await self.train_dqn_step(
                            q_network,
                            target_network,
                            optimizer,
                            memory,
                            config,
                        )

                    state = next_state
                    total_reward += reward

                    if done:
                        break

                # Update target network
                if episode % 10 == 0:
                    target_network.load_state_dict(q_network.state_dict())

                # Decay epsilon
                config["epsilon"] = max(
                    config["epsilon_min"],
                    config["epsilon"] * config["epsilon_decay"],
                )

                # Early stopping
                if total_reward > best_reward:
                    best_reward = total_reward
                    patience_counter = 0

                    # Save best model
                    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
                    os.makedirs(models_dir, exist_ok=True)
                    torch.save(
                        {
                            "q_network": q_network.state_dict(),
                            "target_network": target_network.state_dict(),
                            "config": config,
                        },
                        os.path.join(models_dir, f"{algorithm_name}_{symbol}.pth"),
                    )
                else:
                    patience_counter += 1

                if patience_counter >= training_config["early_stopping_patience"]:
                    print(f"ðŸ›‘ Early stopping at episode {episode}")
                    break

                if episode % 100 == 0:
                    print(
                        f"ðŸ“Š Episode {episode}: Total Reward: {total_reward:.2f}, Epsilon: {config['epsilon']:.3f}"
                    )

            # Update training history
            if symbol not in self.state["training_history"]:
                self.state["training_history"][symbol] = {}

            self.state["training_history"][symbol][algorithm_name] = {
                "last_training": datetime.now().isoformat(),
                "best_reward": best_reward,
                "episodes_trained": episode + 1,
            }

            print(f"âœ… DQN training complete for {symbol}")

        except Exception as e:
            print(f"âŒ Error training DQN for {symbol}: {e}")

    async def train_dqn_step(
        self,
        q_network: nn.Module,
        target_network: nn.Module,
        optimizer: optim.Optimizer,
        memory: deque,
        config: dict[str, Any],
    ):
        """Train DQN for one step"""
        try:
            # Sample batch
            batch = random.sample(memory, config["batch_size"])
            states, actions, rewards, next_states, dones = zip(*batch, strict=False)

            # Convert to tensors
            states = torch.FloatTensor(states).to(self.device)
            actions = torch.LongTensor(actions).to(self.device)
            rewards = torch.FloatTensor(rewards).to(self.device)
            next_states = torch.FloatTensor(next_states).to(self.device)
            dones = torch.BoolTensor(dones).to(self.device)

            # Current Q values
            current_q_values = q_network(states).gather(1, actions.unsqueeze(1))

            # Next Q values
            next_q_values = target_network(next_states).max(1)[0].detach()
            target_q_values = rewards + (config["gamma"] * next_q_values * ~dones)

            # Loss
            loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        except Exception as e:
            print(f"âŒ Error in DQN training step: {e}")

    async def train_actor_critic(
        self,
        symbol: str,
        algorithm_name: str,
        model: dict[str, Any],
        environment: TradingEnvironment,
        training_config: dict[str, Any],
    ):
        """Train Actor-Critic model"""
        try:
            policy_network = model["policy_network"]
            value_network = model["value_network"]
            policy_optimizer = model["policy_optimizer"]
            value_optimizer = model["value_optimizer"]
            config = model["config"]

            best_reward = float("-inf")
            patience_counter = 0

            for episode in range(training_config["episodes"]):
                state = environment.reset()
                total_reward = 0

                for step in range(training_config["max_steps"]):
                    # Get action probabilities
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    action_probs = policy_network(state_tensor)

                    # Sample action
                    action_dist = torch.distributions.Categorical(action_probs)
                    action = action_dist.sample()

                    # Take action
                    next_state, reward, done = environment.step(action.item())

                    # Get value estimate
                    value = value_network(state_tensor)

                    # Calculate advantage (simplified)
                    advantage = reward - value.item()

                    # Policy loss
                    policy_loss = -action_dist.log_prob(action) * advantage

                    # Value loss
                    value_loss = nn.MSELoss()(value, torch.tensor([reward]).to(self.device))

                    # Backward pass
                    policy_optimizer.zero_grad()
                    policy_loss.backward()
                    policy_optimizer.step()

                    value_optimizer.zero_grad()
                    value_loss.backward()
                    value_optimizer.step()

                    state = next_state
                    total_reward += reward

                    if done:
                        break

                # Early stopping
                if total_reward > best_reward:
                    best_reward = total_reward
                    patience_counter = 0

                    # Save best model
                    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
                    os.makedirs(models_dir, exist_ok=True)
                    torch.save(
                        {
                            "policy_network": policy_network.state_dict(),
                            "value_network": value_network.state_dict(),
                            "config": config,
                        },
                        os.path.join(models_dir, f"{algorithm_name}_{symbol}.pth"),
                    )
                else:
                    patience_counter += 1

                if patience_counter >= training_config["early_stopping_patience"]:
                    print(f"ðŸ›‘ Early stopping at episode {episode}")
                    break

                if episode % 100 == 0:
                    print(f"ðŸ“Š Episode {episode}: Total Reward: {total_reward:.2f}")

            # Update training history
            if symbol not in self.state["training_history"]:
                self.state["training_history"][symbol] = {}

            self.state["training_history"][symbol][algorithm_name] = {
                "last_training": datetime.now().isoformat(),
                "best_reward": best_reward,
                "episodes_trained": episode + 1,
            }

            print(f"âœ… Actor-Critic training complete for {symbol}")

        except Exception as e:
            print(f"âŒ Error training Actor-Critic for {symbol}: {e}")

    async def generate_all_strategies(self):
        """Generate strategies for all symbols"""
        try:
            print(f"ðŸŽ¯ Generating strategies for {len(self.trading_symbols)} symbols...")

            for symbol in self.trading_symbols:
                try:
                    await self.generate_symbol_strategy(symbol)
                except Exception as e:
                    print(f"âŒ Error generating strategy for {symbol}: {e}")

            print("âœ… Strategy generation complete")

        except Exception as e:
            print(f"âŒ Error generating all strategies: {e}")

    async def generate_symbol_strategy(self, symbol: str):
        """Generate strategy for a specific symbol"""
        try:
            # Get recent market data
            market_data = await self.get_symbol_market_data(symbol)

            if not market_data or len(market_data) < 100:
                return

            strategies = {}

            # Generate strategies with each model
            for algorithm_name, model in self.models.items():
                try:
                    strategy = await self.generate_model_strategy(
                        symbol, algorithm_name, model, market_data
                    )
                    if strategy:
                        strategies[algorithm_name] = strategy
                except Exception as e:
                    print(f"âŒ Error generating strategy with {algorithm_name} for {symbol}: {e}")

            # Store strategies
            if strategies:
                self.state["strategies_generated"][symbol] = {
                    "strategies": strategies,
                    "timestamp": datetime.now().isoformat(),
                }

                # Broadcast strategies
                await self.broadcast_strategies(symbol, strategies)

        except Exception as e:
            print(f"âŒ Error generating strategy for {symbol}: {e}")

    async def generate_model_strategy(
        self,
        symbol: str,
        algorithm_name: str,
        model: dict[str, Any],
        market_data: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """Generate strategy with a specific model"""
        try:
            algorithm_type = model["type"]

            # Create environment for testing
            df = pd.DataFrame(market_data)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp")

            environment = TradingEnvironment(df)

            # Test strategy
            state = environment.reset()
            actions = []
            rewards = []

            for step in range(min(100, len(df) - 1)):  # Test for 100 steps
                if algorithm_type == "dqn":
                    q_network = model["q_network"]
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    q_values = q_network(state_tensor)
                    action = q_values.argmax().item()

                elif algorithm_type == "actor_critic":
                    policy_network = model["policy_network"]
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    action_probs = policy_network(state_tensor)
                    action = action_probs.argmax().item()

                actions.append(action)
                state, reward, done = environment.step(action)
                rewards.append(reward)

                if done:
                    break

            # Calculate strategy metrics
            total_reward = sum(rewards)
            action_distribution = {
                0: actions.count(0),
                1: actions.count(1),
                2: actions.count(2),
            }

            # Determine strategy type
            if action_distribution[1] > action_distribution[2]:
                strategy_type = "bullish"
            elif action_distribution[2] > action_distribution[1]:
                strategy_type = "bearish"
            else:
                strategy_type = "neutral"

            return {
                "type": strategy_type,
                "total_reward": total_reward,
                "action_distribution": action_distribution,
                "confidence": min(abs(total_reward) / 10, 1.0),  # Normalize confidence
                "algorithm": algorithm_type,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            print(f"âŒ Error generating strategy with {algorithm_name}: {e}")
            return None

    async def broadcast_strategies(self, symbol: str, strategies: dict[str, Any]):
        """Broadcast strategies to other agents"""
        try:
            strategy_update = {
                "type": "rl_strategy_update",
                "symbol": symbol,
                "strategies": strategies,
                "timestamp": datetime.now().isoformat(),
            }

            # Broadcast to all agents
            await self.broadcast_message(strategy_update)

            # Send to specific agents
            await self.send_message("strategy_agent", strategy_update)
            await self.send_message("execution_agent", strategy_update)

        except Exception as e:
            print(f"âŒ Error broadcasting strategies: {e}")

    async def handle_train_rl_model(self, message: dict[str, Any]):
        """Handle manual RL model training request"""
        try:
            symbol = message.get("symbol")
            algorithm_name = message.get("algorithm_name")

            print(f"ðŸ‹ï¸ Manual RL model training requested for {symbol}")

            if symbol:
                if algorithm_name and algorithm_name in self.models:
                    # Train specific model
                    market_data = await self.get_symbol_market_data(symbol)
                    if market_data:
                        df = pd.DataFrame(market_data)
                        df["timestamp"] = pd.to_datetime(df["timestamp"])
                        df = df.sort_values("timestamp")

                        environment = TradingEnvironment(df)
                        await self.train_rl_model(
                            symbol,
                            algorithm_name,
                            self.models[algorithm_name],
                            environment,
                        )
                else:
                    # Train all models
                    await self.train_symbol_models(symbol)

            # Send response
            response = {
                "type": "rl_model_training_complete",
                "symbol": symbol,
                "algorithm_name": algorithm_name,
                "timestamp": datetime.now().isoformat(),
            }

            sender = message.get("from_agent")
            if sender:
                await self.send_message(sender, response)

        except Exception as e:
            print(f"âŒ Error handling RL model training request: {e}")
            await self.broadcast_error(f"RL model training error: {e}")

    async def handle_generate_strategy(self, message: dict[str, Any]):
        """Handle manual strategy generation request"""
        try:
            symbol = message.get("symbol")
            algorithm_name = message.get("algorithm_name")

            print(f"ðŸŽ¯ Manual strategy generation requested for {symbol}")

            if symbol:
                market_data = await self.get_symbol_market_data(symbol)
                if market_data:
                    if algorithm_name and algorithm_name in self.models:
                        # Generate strategy with specific model
                        strategy = await self.generate_model_strategy(
                            symbol,
                            algorithm_name,
                            self.models[algorithm_name],
                            market_data,
                        )

                        response = {
                            "type": "strategy_generation_response",
                            "symbol": symbol,
                            "algorithm_name": algorithm_name,
                            "strategy": strategy,
                            "timestamp": datetime.now().isoformat(),
                        }
                    else:
                        # Generate strategies with all models
                        await self.generate_symbol_strategy(symbol)

                        response = {
                            "type": "strategy_generation_response",
                            "symbol": symbol,
                            "strategies": (
                                self.state["strategies_generated"]
                                .get(symbol, {})
                                .get("strategies", {})
                            ),
                            "timestamp": datetime.now().isoformat(),
                        }
                else:
                    response = {
                        "type": "strategy_generation_response",
                        "symbol": symbol,
                        "strategy": None,
                        "error": "No market data available",
                        "timestamp": datetime.now().isoformat(),
                    }
            else:
                response = {
                    "type": "strategy_generation_response",
                    "symbol": symbol,
                    "strategy": None,
                    "error": "No symbol provided",
                    "timestamp": datetime.now().isoformat(),
                }

            sender = message.get("from_agent")
            if sender:
                await self.send_message(sender, response)

        except Exception as e:
            print(f"âŒ Error handling strategy generation request: {e}")
            await self.broadcast_error(f"Strategy generation error: {e}")

    async def handle_get_rl_status(self, message: dict[str, Any]):
        """Handle RL status request"""
        try:
            symbol = message.get("symbol")

            print(f"ðŸ“Š RL status requested for {symbol}")

            # Get RL status
            if symbol and symbol in self.state["training_history"]:
                status = self.state["training_history"][symbol]

                response = {
                    "type": "rl_status_response",
                    "symbol": symbol,
                    "status": status,
                    "timestamp": datetime.now().isoformat(),
                }
            else:
                response = {
                    "type": "rl_status_response",
                    "symbol": symbol,
                    "status": None,
                    "timestamp": datetime.now().isoformat(),
                }

            sender = message.get("from_agent")
            if sender:
                await self.send_message(sender, response)

        except Exception as e:
            print(f"âŒ Error handling RL status request: {e}")
            await self.broadcast_error(f"RL status error: {e}")

    async def update_rl_metrics(self):
        """Update RL metrics"""
        try:
            metrics = {
                "agent_id": self.agent_id,
                "symbols_count": len(self.trading_symbols),
                "models_count": len(self.models),
                "strategies_count": len(self.state["strategies_generated"]),
                "training_count": self.state["training_count"],
                "last_training": self.state["last_training"],
                "device": str(self.device),
                "timestamp": datetime.now().isoformat(),
            }

            # Store metrics in Redis
            self.redis_client.set(f"agent_metrics:{self.agent_id}", json.dumps(metrics), ex=300)

        except Exception as e:
            print(f"âŒ Error updating RL metrics: {e}")

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
            print(f"âŒ Error cleaning up cache: {e}")

    async def handle_market_data(self, message: dict[str, Any]):
        """Handle market data message"""
        try:
            market_data = message.get("market_data", {})
            print(f"ðŸ“Š Reinforcement Learning Agent received market data for {len(market_data)} symbols")
            
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
            
            # Generate strategies with new data
            await self.generate_all_strategies()
            
        except Exception as e:
            print(f"âŒ Error handling market data: {e}")
            await self.broadcast_error(f"Market data handling error: {e}")


