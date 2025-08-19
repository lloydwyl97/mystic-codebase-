"""
Real-time Data Processing Service
Handles live trading data processing and streaming
"""

import asyncio
import json
import os
from datetime import datetime
from typing import Any

import aiohttp
import redis
from dotenv import load_dotenv

# Load environment variables
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))


class RealTimeProcessor:
    def __init__(self):
        """Initialize real-time processor"""
        self.redis_client = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            db=int(os.getenv("REDIS_DB", 0)),
            decode_responses=True,
        )
        self.session = None
        self.running = False

    async def start(self):
        """Start the real-time processor"""
        print("ðŸš€ Starting Real-time Data Processor...")
        self.running = True
        self.session = aiohttp.ClientSession()

        # Start multiple processing tasks
        tasks = [
            self.process_market_data(),
            self.process_trade_signals(),
            self.process_portfolio_updates(),
            self.process_risk_alerts(),
        ]

        await asyncio.gather(*tasks)

    async def process_market_data(self):
        """Process live market data"""
        print("ðŸ“Š Starting market data processing...")

        while self.running:
            try:
                # Fetch live market data from multiple sources
                market_data = await self.fetch_live_market_data()

                # Process and store in Redis
                await self.store_market_data(market_data)

                # Publish to Redis channels for real-time updates
                await self.publish_market_updates(market_data)

                await asyncio.sleep(5)  # Update every 5 seconds

            except Exception as e:
                print(f"âŒ Error in market data processing: {e}")
                await asyncio.sleep(10)

    async def process_trade_signals(self):
        """Process live trade signals"""
        print("ðŸŽ¯ Starting trade signal processing...")

        while self.running:
            try:
                # Generate or fetch trade signals
                signals = await self.generate_trade_signals()

                # Store signals in Redis
                await self.store_trade_signals(signals)

                # Publish signals for real-time consumption
                await self.publish_trade_signals(signals)

                await asyncio.sleep(10)  # Update every 10 seconds

            except Exception as e:
                print(f"âŒ Error in trade signal processing: {e}")
                await asyncio.sleep(15)

    async def process_portfolio_updates(self):
        """Process portfolio updates"""
        print("ðŸ’¼ Starting portfolio update processing...")

        while self.running:
            try:
                # Calculate portfolio metrics
                portfolio_data = await self.calculate_portfolio_metrics()

                # Store portfolio data
                await self.store_portfolio_data(portfolio_data)

                # Publish portfolio updates
                await self.publish_portfolio_updates(portfolio_data)

                await asyncio.sleep(30)  # Update every 30 seconds

            except Exception as e:
                print(f"âŒ Error in portfolio processing: {e}")
                await asyncio.sleep(60)

    async def process_risk_alerts(self):
        """Process risk management alerts"""
        print("âš ï¸ Starting risk alert processing...")

        while self.running:
            try:
                # Calculate risk metrics
                risk_data = await self.calculate_risk_metrics()

                # Check for risk alerts
                alerts = await self.check_risk_alerts(risk_data)

                # Store and publish alerts
                if alerts:
                    await self.store_risk_alerts(alerts)
                    await self.publish_risk_alerts(alerts)

                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                print(f"âŒ Error in risk alert processing: {e}")
                await asyncio.sleep(120)

    async def fetch_live_market_data(self) -> dict[str, Any]:
        """Fetch live market data from multiple sources"""
        try:
            market_data = {}

            # Fetch from Binance US
            binance_data = await self.fetch_binance_data()
            if binance_data:
                market_data["binance"] = binance_data

            # Fetch from Coinbase
            coinbase_data = await self.fetch_coinbase_data()
            if coinbase_data:
                market_data["coinbase"] = coinbase_data

            # Add timestamp
            market_data["timestamp"] = datetime.now().isoformat()

            return market_data

        except Exception as e:
            print(f"Error fetching market data: {e}")
            return {}

    async def fetch_binance_data(self) -> dict[str, Any]:
        """Fetch data from Binance US"""
        try:
            symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "SOLUSDT", "DOTUSDT"]
            data = {}

            for symbol in symbols:
                url = f"https://api.binance.us/api/v3/ticker/24hr?symbol={symbol}"
                async with self.session.get(url, timeout=5) as response:
                    if response.status == 200:
                        ticker_data = await response.json()
                        data[symbol] = {
                            "price": float(ticker_data["lastPrice"]),
                            "volume": float(ticker_data["volume"]),
                            "change_24h": float(ticker_data["priceChangePercent"]),
                            "high_24h": float(ticker_data["highPrice"]),
                            "low_24h": float(ticker_data["lowPrice"]),
                        }

            return data

        except Exception as e:
            print(f"Error fetching Binance data: {e}")
            return {}

    async def fetch_coinbase_data(self) -> dict[str, Any]:
        """Fetch data from Coinbase"""
        try:
            symbols = ["BTC-USD", "ETH-USD", "ADA-USD", "SOL-USD", "DOT-USD"]
            data = {}

            for symbol in symbols:
                url = f"https://api.exchange.coinbase.com/products/{symbol}/ticker"
                async with self.session.get(url, timeout=5) as response:
                    if response.status == 200:
                        ticker_data = await response.json()
                        data[symbol] = {
                            "price": float(ticker_data["price"]),
                            "volume": float(ticker_data["volume"]),
                            "bid": float(ticker_data.get("bid", 0)),
                            "ask": float(ticker_data.get("ask", 0)),
                        }

            return data

        except Exception as e:
            print(f"Error fetching Coinbase data: {e}")
            return {}

    async def generate_trade_signals(self) -> list[dict[str, Any]]:
        """Generate trade signals based on market data"""
        try:
            signals = []

            # Get latest market data from Redis
            market_data = self.redis_client.get("market_data")
            if market_data:
                data = json.loads(market_data)

                # Generate signals for each symbol
                for symbol, ticker in data.get("binance", {}).items():
                    # Simple signal generation logic
                    price = ticker["price"]
                    change_24h = ticker["change_24h"]

                    if change_24h > 5:  # Strong buy signal
                        signals.append(
                            {
                                "symbol": symbol,
                                "type": "BUY",
                                "strength": "STRONG",
                                "price": price,
                                "reason": f"24h change: {change_24h:.2f}%",
                                "timestamp": datetime.now().isoformat(),
                            }
                        )
                    elif change_24h < -5:  # Strong sell signal
                        signals.append(
                            {
                                "symbol": symbol,
                                "type": "SELL",
                                "strength": "STRONG",
                                "price": price,
                                "reason": f"24h change: {change_24h:.2f}%",
                                "timestamp": datetime.now().isoformat(),
                            }
                        )

            return signals

        except Exception as e:
            print(f"Error generating trade signals: {e}")
            return []

    async def calculate_portfolio_metrics(self) -> dict[str, Any]:
        """Calculate portfolio metrics"""
        try:
            # Simulate portfolio calculation
            portfolio_data = {
                "total_value": 125000.00,
                "daily_change": 2.5,
                "positions": 4,
                "unrealized_pnl": 1250.00,
                "realized_pnl": 850.00,
                "timestamp": datetime.now().isoformat(),
            }

            return portfolio_data

        except Exception as e:
            print(f"Error calculating portfolio metrics: {e}")
            return {}

    async def calculate_risk_metrics(self) -> dict[str, Any]:
        """Calculate risk metrics"""
        try:
            risk_data = {
                "var_95": -0.0234,
                "cvar_95": -0.0345,
                "volatility": 0.156,
                "beta": 1.12,
                "max_drawdown": -0.089,
                "timestamp": datetime.now().isoformat(),
            }

            return risk_data

        except Exception as e:
            print(f"Error calculating risk metrics: {e}")
            return {}

    async def check_risk_alerts(self, risk_data: dict[str, Any]) -> list[dict[str, Any]]:
        """Check for risk alerts"""
        try:
            alerts = []

            # Check VaR threshold
            if risk_data.get("var_95", 0) < -0.05:
                alerts.append(
                    {
                        "type": "WARNING",
                        "message": "VaR threshold exceeded",
                        "severity": "HIGH",
                        "value": risk_data["var_95"],
                        "timestamp": datetime.now().isoformat(),
                    }
                )

            # Check volatility threshold
            if risk_data.get("volatility", 0) > 0.20:
                alerts.append(
                    {
                        "type": "WARNING",
                        "message": "High volatility detected",
                        "severity": "MEDIUM",
                        "value": risk_data["volatility"],
                        "timestamp": datetime.now().isoformat(),
                    }
                )

            return alerts

        except Exception as e:
            print(f"Error checking risk alerts: {e}")
            return []

    async def store_market_data(self, data: dict[str, Any]):
        """Store market data in Redis"""
        try:
            self.redis_client.set("market_data", json.dumps(data), ex=300)  # 5 minutes TTL
        except Exception as e:
            print(f"Error storing market data: {e}")

    async def store_trade_signals(self, signals: list[dict[str, Any]]):
        """Store trade signals in Redis"""
        try:
            self.redis_client.set("trade_signals", json.dumps(signals), ex=600)  # 10 minutes TTL
        except Exception as e:
            print(f"Error storing trade signals: {e}")

    async def store_portfolio_data(self, data: dict[str, Any]):
        """Store portfolio data in Redis"""
        try:
            self.redis_client.set("portfolio_data", json.dumps(data), ex=1800)  # 30 minutes TTL
        except Exception as e:
            print(f"Error storing portfolio data: {e}")

    async def store_risk_alerts(self, alerts: list[dict[str, Any]]):
        """Store risk alerts in Redis"""
        try:
            self.redis_client.set("risk_alerts", json.dumps(alerts), ex=3600)  # 1 hour TTL
        except Exception as e:
            print(f"Error storing risk alerts: {e}")

    async def publish_market_updates(self, data: dict[str, Any]):
        """Publish market updates to Redis channels"""
        try:
            self.redis_client.publish("market_updates", json.dumps(data))
        except Exception as e:
            print(f"Error publishing market updates: {e}")

    async def publish_trade_signals(self, signals: list[dict[str, Any]]):
        """Publish trade signals to Redis channels"""
        try:
            self.redis_client.publish("trade_signals", json.dumps(signals))
        except Exception as e:
            print(f"Error publishing trade signals: {e}")

    async def publish_portfolio_updates(self, data: dict[str, Any]):
        """Publish portfolio updates to Redis channels"""
        try:
            self.redis_client.publish("portfolio_updates", json.dumps(data))
        except Exception as e:
            print(f"Error publishing portfolio updates: {e}")

    async def publish_risk_alerts(self, alerts: list[dict[str, Any]]):
        """Publish risk alerts to Redis channels"""
        try:
            self.redis_client.publish("risk_alerts", json.dumps(alerts))
        except Exception as e:
            print(f"Error publishing risk alerts: {e}")

    async def stop(self):
        """Stop the real-time processor"""
        print("ðŸ›‘ Stopping Real-time Data Processor...")
        self.running = False
        if self.session:
            await self.session.close()


async def main():
    """Main function"""
    processor = RealTimeProcessor()

    try:
        await processor.start()
    except KeyboardInterrupt:
        print("ðŸ›‘ Received interrupt signal")
    except Exception as e:
        print(f"âŒ Error in main: {e}")
    finally:
        await processor.stop()


if __name__ == "__main__":
    asyncio.run(main())


