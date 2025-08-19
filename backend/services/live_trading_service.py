"""
Live Trading Service
Connects to real trading APIs for live trading operations
"""

import asyncio
import logging
import os
from datetime import datetime
from typing import Any

import ccxt
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LiveTradingService:
    """Service for live trading operations with real APIs"""

    def __init__(self):
        # Prefer Binance US keys; fall back to generic if not set
        self.binance_api_key = os.getenv("BINANCE_US_API_KEY") or os.getenv("BINANCE_API_KEY")
        self.binance_secret = os.getenv("BINANCE_US_SECRET_KEY") or os.getenv("BINANCE_SECRET")

        # Coinbase Advanced Trade API
        self.coinbase_api_key = os.getenv("COINBASE_API_KEY")
        # Support both COINBASE_API_SECRET and legacy COINBASE_SECRET
        self.coinbase_secret = os.getenv("COINBASE_API_SECRET") or os.getenv("COINBASE_SECRET")
        # Passphrase is required by Coinbase
        self.coinbase_passphrase = os.getenv("COINBASE_PASSPHRASE")

        # Initialize exchange connections
        # Use the correct Binance exchange (US if keys provided)
        self.binance = None
        if self.binance_api_key and self.binance_secret:
            try:
                if os.getenv("BINANCE_US_API_KEY") or os.getenv("BINANCE_US_SECRET_KEY"):
                    self.binance = ccxt.binanceus(
                        {
                            "apiKey": self.binance_api_key,
                            "secret": self.binance_secret,
                            "enableRateLimit": True,
                        }
                    )
                else:
                    self.binance = ccxt.binance(
                        {
                            "apiKey": self.binance_api_key,
                            "secret": self.binance_secret,
                            "enableRateLimit": True,
                        }
                    )
            except Exception as e:
                logger.error(f"Error initializing Binance client: {e}")

        self.coinbase = None
        if self.coinbase_api_key and self.coinbase_secret:
            try:
                coinbase_options = {
                    "apiKey": self.coinbase_api_key,
                    "secret": self.coinbase_secret,
                    # CCXT expects 'password' for Coinbase passphrase
                    "password": self.coinbase_passphrase,
                    "enableRateLimit": True,
                }
                self.coinbase = ccxt.coinbase(coinbase_options)
            except Exception as e:
                logger.error(f"Error initializing Coinbase client: {e}")

        # Cache for orders and positions
        self.orders_cache = {}
        self.positions_cache = {}
        self.cache_timeout = 30  # seconds

    async def get_account_balance(self) -> dict[str, Any]:
        """Get account balance from connected exchanges"""
        try:
            balances = {}

            # Get Binance balance
            if self.binance:
                try:
                    binance_balance = await asyncio.to_thread(self.binance.fetch_balance)
                    balances["binance"] = {
                        "total": binance_balance.get("total", {}),
                        "free": binance_balance.get("free", {}),
                        "used": binance_balance.get("used", {}),
                        "timestamp": datetime.now().isoformat(),
                    }
                except Exception as e:
                    logger.error(f"Error fetching Binance balance: {e}")
                    balances["binance"] = {"error": str(e)}

            # Get Coinbase balance
            if self.coinbase:
                try:
                    coinbase_balance = await asyncio.to_thread(self.coinbase.fetch_balance)
                    balances["coinbase"] = {
                        "total": coinbase_balance.get("total", {}),
                        "free": coinbase_balance.get("free", {}),
                        "used": coinbase_balance.get("used", {}),
                        "timestamp": datetime.now().isoformat(),
                    }
                except Exception as e:
                    logger.error(f"Error fetching Coinbase balance: {e}")
                    balances["coinbase"] = {"error": str(e)}

            return {
                "status": "success",
                "balances": balances,
                "timestamp": datetime.now().isoformat(),
                "source": "live_trading_apis",
            }

        except Exception as e:
            logger.error(f"Error fetching account balance: {e}")
            return {
                "status": "error",
                "message": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    async def get_open_orders(self) -> dict[str, Any]:
        """Get open orders from connected exchanges"""
        try:
            orders = {}

            # Get Binance orders
            if self.binance:
                try:
                    binance_orders = await asyncio.to_thread(self.binance.fetch_open_orders)
                    orders["binance"] = [
                        {
                            "id": order["id"],
                            "symbol": order["symbol"],
                            "type": order["type"],
                            "side": order["side"],
                            "amount": order["amount"],
                            "price": order["price"],
                            "status": order["status"],
                            "timestamp": order["timestamp"],
                        }
                        for order in binance_orders
                    ]
                except Exception as e:
                    logger.error(f"Error fetching Binance orders: {e}")
                    orders["binance"] = []

            # Get Coinbase orders
            if self.coinbase:
                try:
                    coinbase_orders = await asyncio.to_thread(self.coinbase.fetch_open_orders)
                    orders["coinbase"] = [
                        {
                            "id": order["id"],
                            "symbol": order["symbol"],
                            "type": order["type"],
                            "side": order["side"],
                            "amount": order["amount"],
                            "price": order["price"],
                            "status": order["status"],
                            "timestamp": order["timestamp"],
                        }
                        for order in coinbase_orders
                    ]
                except Exception as e:
                    logger.error(f"Error fetching Coinbase orders: {e}")
                    orders["coinbase"] = []

            return {
                "status": "success",
                "orders": orders,
                "timestamp": datetime.now().isoformat(),
                "source": "live_trading_apis",
            }

        except Exception as e:
            logger.error(f"Error fetching open orders: {e}")
            return {
                "status": "error",
                "message": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    async def get_trade_history(self, symbol: str = None, limit: int = 100) -> dict[str, Any]:
        """Get trade history from connected exchanges"""
        try:
            trades = {}

            # Get Binance trades
            if self.binance:
                try:
                    binance_trades = await asyncio.to_thread(
                        self.binance.fetch_my_trades, symbol, limit=limit
                    )
                    trades["binance"] = [
                        {
                            "id": trade["id"],
                            "symbol": trade["symbol"],
                            "side": trade["side"],
                            "amount": trade["amount"],
                            "price": trade["price"],
                            "cost": trade["cost"],
                            "fee": trade["fee"],
                            "timestamp": trade["timestamp"],
                        }
                        for trade in binance_trades
                    ]
                except Exception as e:
                    logger.error(f"Error fetching Binance trades: {e}")
                    trades["binance"] = []

            # Get Coinbase trades
            if self.coinbase:
                try:
                    coinbase_trades = await asyncio.to_thread(
                        self.coinbase.fetch_my_trades, symbol, limit=limit
                    )
                    trades["coinbase"] = [
                        {
                            "id": trade["id"],
                            "symbol": trade["symbol"],
                            "side": trade["side"],
                            "amount": trade["amount"],
                            "price": trade["price"],
                            "cost": trade["cost"],
                            "fee": trade["fee"],
                            "timestamp": trade["timestamp"],
                        }
                        for trade in coinbase_trades
                    ]
                except Exception as e:
                    logger.error(f"Error fetching Coinbase trades: {e}")
                    trades["coinbase"] = []

            return {
                "status": "success",
                "trades": trades,
                "timestamp": datetime.now().isoformat(),
                "source": "live_trading_apis",
            }

        except Exception as e:
            logger.error(f"Error fetching trade history: {e}")
            return {
                "status": "error",
                "message": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    async def place_order(
        self,
        exchange: str,
        symbol: str,
        order_type: str,
        side: str,
        amount: float,
        price: float = None,
    ) -> dict[str, Any]:
        """Place a new order on the specified exchange"""
        try:
            if exchange.lower() == "binance" and self.binance:
                order = await asyncio.to_thread(
                    self.binance.create_order,
                    symbol=symbol,
                    type=order_type,
                    side=side,
                    amount=amount,
                    price=price,
                )

                return {
                    "status": "success",
                    "order": {
                        "id": order["id"],
                        "symbol": order["symbol"],
                        "type": order["type"],
                        "side": order["side"],
                        "amount": order["amount"],
                        "price": order["price"],
                        "status": order["status"],
                        "timestamp": order["timestamp"],
                    },
                    "exchange": "binance",
                    "timestamp": datetime.now().isoformat(),
                }

            elif exchange.lower() == "coinbase" and self.coinbase:
                order = await asyncio.to_thread(
                    self.coinbase.create_order,
                    symbol=symbol,
                    type=order_type,
                    side=side,
                    amount=amount,
                    price=price,
                )

                return {
                    "status": "success",
                    "order": {
                        "id": order["id"],
                        "symbol": order["symbol"],
                        "type": order["type"],
                        "side": order["side"],
                        "amount": order["amount"],
                        "price": order["price"],
                        "status": order["status"],
                        "timestamp": order["timestamp"],
                    },
                    "exchange": "coinbase",
                    "timestamp": datetime.now().isoformat(),
                }

            else:
                return {
                    "status": "error",
                    "message": (f"Exchange {exchange} not available or not configured"),
                    "timestamp": datetime.now().isoformat(),
                }

        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return {
                "status": "error",
                "message": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    async def cancel_order(self, exchange: str, order_id: str, symbol: str) -> dict[str, Any]:
        """Cancel an existing order"""
        try:
            if exchange.lower() == "binance" and self.binance:
                await asyncio.to_thread(
                    self.binance.cancel_order, id=order_id, symbol=symbol
                )

                return {
                    "status": "success",
                    "message": f"Order {order_id} cancelled successfully",
                    "exchange": "binance",
                    "timestamp": datetime.now().isoformat(),
                }

            elif exchange.lower() == "coinbase" and self.coinbase:
                await asyncio.to_thread(
                    self.coinbase.cancel_order, id=order_id, symbol=symbol
                )

                return {
                    "status": "success",
                    "message": f"Order {order_id} cancelled successfully",
                    "exchange": "coinbase",
                    "timestamp": datetime.now().isoformat(),
                }

            else:
                return {
                    "status": "error",
                    "message": (f"Exchange {exchange} not available or not configured"),
                    "timestamp": datetime.now().isoformat(),
                }

        except Exception as e:
            logger.error(f"Error cancelling order: {e}")
            return {
                "status": "error",
                "message": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    async def get_positions(self) -> dict[str, Any]:
        """Get current positions from connected exchanges"""
        try:
            positions = {}

            # Get Binance positions (futures)
            if self.binance:
                try:
                    binance_positions = await asyncio.to_thread(self.binance.fetch_positions)
                    positions["binance"] = [
                        {
                            "symbol": pos["symbol"],
                            "side": pos["side"],
                            "size": pos["size"],
                            "notional": pos["notional"],
                            "unrealized_pnl": pos["unrealizedPnl"],
                            "entry_price": pos["entryPrice"],
                            "mark_price": pos["markPrice"],
                            "timestamp": pos["timestamp"],
                        }
                        for pos in binance_positions
                        if float(pos["size"]) > 0
                    ]
                except Exception as e:
                    logger.error(f"Error fetching Binance positions: {e}")
                    positions["binance"] = []

            return {
                "status": "success",
                "positions": positions,
                "timestamp": datetime.now().isoformat(),
                "source": "live_trading_apis",
            }

        except Exception as e:
            logger.error(f"Error fetching positions: {e}")
            return {
                "status": "error",
                "message": str(e),
                "timestamp": datetime.now().isoformat(),
            }


# Global instance
trading_service = LiveTradingService()


