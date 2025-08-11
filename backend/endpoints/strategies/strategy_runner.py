import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict

import ccxt
import numpy as np
import pandas as pd
import redis

from config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StrategyRunner:
    def __init__(self, strategy_name: str):
        self.strategy_name = strategy_name
        self.config = self.load_config()
        self.redis_client = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            decode_responses=True,
        )
        self.exchange = ccxt.binance(
            {
                "apiKey": settings.exchange.binance_us_api_key,
                "secret": settings.exchange.binance_us_secret_key,
                "sandbox": True,  # Use testnet for safety
            }
        )
        self.leaderboard_key = "strategy_leaderboard"
        self.trades = []
        self.current_balance = self.config.get("capital", 1000.0)

    def load_config(self) -> Dict[str, Any]:
        """Load strategy configuration"""
        try:
            config_path = "/app/agent/config.json"
            with open(config_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return {
                "strategy_name": self.strategy_name,
                "capital": 1000.0,
                "timeframe": "1h",
                "symbols": ["BTC/USDT"],
                "risk_per_trade": 0.02,
                "max_positions": 5,
                "stop_loss": 0.05,
                "take_profit": 0.15,
            }

    def fetch_market_data(self, symbol: str, timeframe: str, limit: int = 100) -> pd.DataFrame:
        """Fetch market data from exchange"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(
                ohlcv,
                columns=[
                    "timestamp",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                ],
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            return df
        except Exception as e:
            logger.error(f"Failed to fetch data for {symbol}: {e}")
            return pd.DataFrame()

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators based on strategy type"""
        strategy_type = self.config.get("strategy_type", "momentum")

        if strategy_type == "momentum":
            # RSI
            delta = df["close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df["rsi"] = 100 - (100 / (1 + rs))

            # MACD
            exp1 = df["close"].ewm(span=12).mean()
            exp2 = df["close"].ewm(span=26).mean()
            df["macd"] = exp1 - exp2
            df["signal"] = df["macd"].ewm(span=9).mean()

        elif strategy_type == "mean_reversion":
            # Bollinger Bands
            df["sma"] = df["close"].rolling(window=20).mean()
            df["std"] = df["close"].rolling(window=20).std()
            df["upper_band"] = df["sma"] + (df["std"] * 2)
            df["lower_band"] = df["sma"] - (df["std"] * 2)

        elif strategy_type == "breakout":
            # Support/Resistance levels
            df["high_20"] = df["high"].rolling(window=20).max()
            df["low_20"] = df["low"].rolling(window=20).min()

        elif strategy_type == "volatility":
            # ATR (Average True Range)
            df["tr1"] = df["high"] - df["low"]
            df["tr2"] = abs(df["high"] - df["close"].shift())
            df["tr3"] = abs(df["low"] - df["close"].shift())
            df["tr"] = df[["tr1", "tr2", "tr3"]].max(axis=1)
            df["atr"] = df["tr"].rolling(window=14).mean()

        return df

    def generate_signals(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate trading signals based on strategy type"""
        strategy_type = self.config.get("strategy_type", "momentum")
        signals = {}

        if len(df) < 50:  # Need enough data
            return signals

        current_price = df["close"].iloc[-1]

        if strategy_type == "momentum":
            rsi = df["rsi"].iloc[-1]
            macd = df["macd"].iloc[-1]
            signal = df["signal"].iloc[-1]

            if rsi < 30 and macd > signal:
                signals["action"] = "BUY"
                signals["reason"] = f"RSI oversold ({rsi:.2f}), MACD bullish"
            elif rsi > 70 and macd < signal:
                signals["action"] = "SELL"
                signals["reason"] = f"RSI overbought ({rsi:.2f}), MACD bearish"

        elif strategy_type == "mean_reversion":
            upper_band = df["upper_band"].iloc[-1]
            lower_band = df["lower_band"].iloc[-1]

            if current_price <= lower_band:
                signals["action"] = "BUY"
                signals["reason"] = f"Price at lower Bollinger Band ({current_price:.2f})"
            elif current_price >= upper_band:
                signals["action"] = "SELL"
                signals["reason"] = f"Price at upper Bollinger Band ({current_price:.2f})"

        elif strategy_type == "breakout":
            high_20 = df["high_20"].iloc[-1]
            low_20 = df["low_20"].iloc[-1]

            if current_price > high_20:
                signals["action"] = "BUY"
                signals["reason"] = (
                    f"Breakout above resistance ({current_price:.2f} > {high_20:.2f})"
                )
            elif current_price < low_20:
                signals["action"] = "SELL"
                signals["reason"] = f"Breakout below support ({current_price:.2f} < {low_20:.2f})"

        elif strategy_type == "volatility":
            atr = df["atr"].iloc[-1]
            atr_percent = (atr / current_price) * 100

            if atr_percent > 5:  # High volatility
                signals["action"] = "HOLD"
                signals["reason"] = f"High volatility detected ({atr_percent:.2f}%)"
            else:
                signals["action"] = "BUY"
                signals["reason"] = f"Low volatility, good entry ({atr_percent:.2f}%)"

        return signals

    def execute_trade(self, symbol: str, action: str, reason: str) -> bool:
        """Execute trade on exchange"""
        try:
            # Calculate position size
            risk_amount = self.current_balance * self.config.get("risk_per_trade", 0.02)

            # Get current price
            ticker = self.exchange.fetch_ticker(symbol)
            current_price = ticker["last"]

            # Calculate quantity
            quantity = risk_amount / current_price

            # Execute order
            if action == "BUY":
                order = self.exchange.create_market_buy_order(symbol, quantity)
            else:  # SELL
                order = self.exchange.create_market_sell_order(symbol, quantity)

            # Record trade
            trade = {
                "timestamp": datetime.now().isoformat(),
                "symbol": symbol,
                "action": action,
                "quantity": quantity,
                "price": current_price,
                "reason": reason,
                "order_id": order["id"],
            }

            self.trades.append(trade)
            logger.info(
                f"\ud83d\udcca Executed {action} order: {quantity} {symbol} @ {current_price}"
            )

            return True

        except Exception as e:
            logger.error(f"❌ Trade execution failed: {e}")
            return False

    def update_leaderboard(self):
        """Update strategy performance in leaderboard"""
        try:
            # Calculate performance metrics
            total_trades = len(self.trades)
            if total_trades == 0:
                return

            # Calculate profit/loss (simplified)
            initial_balance = self.config.get("capital", 1000.0)
            profit = self.current_balance - initial_balance

            # Calculate win rate
            winning_trades = sum(1 for trade in self.trades if trade.get("profit", 0) > 0)
            win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0

            # Calculate Sharpe ratio (simplified)
            returns = [trade.get("profit", 0) for trade in self.trades]
            if returns:
                sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
            else:
                sharpe_ratio = 0

            # Update Redis
            performance_data = {
                "profit": profit,
                "trades": total_trades,
                "win_rate": win_rate,
                "sharpe_ratio": sharpe_ratio,
                "last_update": time.time(),
            }

            self.redis_client.hset(
                self.leaderboard_key,
                self.strategy_name,
                json.dumps(performance_data),
            )

            logger.info(
                f"\ud83d\udcca Updated leaderboard: Profit=${profit:.2f}, Win Rate={win_rate:.1f}%"
            )

        except Exception as e:
            logger.error(f"Failed to update leaderboard: {e}")

    def run_strategy(self):
        """Main strategy execution loop"""
        logger.info(f"\ud83d\ude80 Starting strategy: {self.strategy_name}")

        while True:
            try:
                for symbol in self.config.get("symbols", ["BTC/USDT"]):
                    # Fetch market data
                    df = self.fetch_market_data(symbol, self.config.get("timeframe", "1h"))
                    if df.empty:
                        continue

                    # Calculate indicators
                    df = self.calculate_indicators(df)

                    # Generate signals
                    signals = self.generate_signals(df)

                    if signals:
                        action = signals.get("action")
                        reason = signals.get("reason")

                        if action in ["BUY", "SELL"]:
                            # Check if we have enough balance
                            if self.current_balance > 10:  # Minimum balance check
                                success = self.execute_trade(symbol, action, reason)
                                if success:
                                    # Update performance
                                    self.update_leaderboard()

                # Sleep for timeframe duration
                timeframe = self.config.get("timeframe", "1h")
                if timeframe == "1m":
                    sleep_time = 60
                elif timeframe == "5m":
                    sleep_time = 300
                elif timeframe == "15m":
                    sleep_time = 900
                elif timeframe == "1h":
                    sleep_time = 3600
                elif timeframe == "4h":
                    sleep_time = 14400
                else:
                    sleep_time = 3600

                time.sleep(sleep_time)

            except KeyboardInterrupt:
                logger.info(f"\ud83d\uded1 Strategy {self.strategy_name} stopped")
                break
            except Exception as e:
                logger.error(f"Strategy error: {e}")
                time.sleep(60)  # Wait before retrying


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        logger.error("Usage: python strategy_runner.py <strategy_name>")
        sys.exit(1)

    strategy_name = sys.argv[1]
    runner = StrategyRunner(strategy_name)
    runner.run_strategy()


if __name__ == "__main__":
    main()
