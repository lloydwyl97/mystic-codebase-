"""
Binance Trading Service

Provides real trading capabilities on Binance exchange.
Handles orders, account management, and real-time data.
"""

import logging
import time
from decimal import Decimal
from typing import Any, Dict, Optional

from binance.client import Client
from binance.enums import (
    ORDER_TYPE_MARKET,
    ORDER_TYPE_LIMIT,
    SIDE_BUY,
    SIDE_SELL,
    TIME_IN_FORCE_GTC,
)
from binance.exceptions import BinanceAPIException, BinanceOrderException

logger = logging.getLogger(__name__)


class BinanceTradingService:
    """Binance trading service with real trading capabilities"""

    def __init__(self, api_key: str = "", secret_key: str = "", testnet: bool = True):
        self.api_key = api_key
        self.secret_key = secret_key
        self.testnet = testnet

        # Initialize Binance US client
        self.client = Client(api_key, secret_key, testnet=testnet, tld="us")

        # Account info cache
        self.account_info_cache = {}
        self.cache_timestamp = 0
        self.cache_duration = 30  # seconds

        # Order tracking
        self.active_orders = {}

        logger.info(f"Binance US trading service initialized (testnet: {testnet})")

    async def get_account_info(self) -> Dict[str, Any]:
        """Get account information and balances"""
        now = time.time()

        # Check cache
        if (now - self.cache_timestamp) < self.cache_duration:
            return self.account_info_cache

        try:
            account = self.client.get_account()

            # Process balances
            balances = []
            for balance in account["balances"]:
                if float(balance["free"]) > 0 or float(balance["locked"]) > 0:
                    balances.append(
                        {
                            "asset": balance["asset"],
                            "free": float(balance["free"]),
                            "locked": float(balance["locked"]),
                            "total": (float(balance["free"]) + float(balance["locked"])),
                        }
                    )

            result = {
                "account_type": account["accountType"],
                "permissions": account["permissions"],
                "balances": balances,
                "total_balance_usdt": 0,
                "timestamp": now,
            }

            # Calculate total USDT value
            for balance in balances:
                if balance["asset"] == "USDT":
                    result["total_balance_usdt"] += balance["total"]
                else:
                    # Get current price and calculate USDT value
                    try:
                        ticker = self.client.get_symbol_ticker(symbol=f"{balance['asset']}USDT")
                        price = float(ticker["price"])
                        result["total_balance_usdt"] += balance["total"] * price
                    except Exception as e:
                        logger.warning(f"Could not get price for {balance['asset']}USDT: {e}")
                        pass  # Skip if price not available

            self.account_info_cache = result
            self.cache_timestamp = now
            return result

        except BinanceAPIException as e:
            logger.error(f"Binance API error getting account info: {str(e)}")
            return {"error": str(e), "timestamp": now}
        except Exception as e:
            logger.error(f"Error getting account info: {str(e)}")
            return {"error": str(e), "timestamp": now}

    async def get_market_price(self, symbol: str) -> Dict[str, Any]:
        """Get current market price for a symbol"""
        try:
            ticker = self.client.get_symbol_ticker(symbol=symbol)

            # Get 24hr stats
            stats = self.client.get_ticker(symbol=symbol)

            return {
                "symbol": symbol,
                "price": float(ticker["price"]),
                "price_change": float(stats["priceChange"]),
                "price_change_percent": float(stats["priceChangePercent"]),
                "high_24h": float(stats["highPrice"]),
                "low_24h": float(stats["lowPrice"]),
                "volume": float(stats["volume"]),
                "quote_volume": float(stats["quoteVolume"]),
                "timestamp": time.time(),
            }

        except BinanceAPIException as e:
            logger.error(f"Binance API error getting price for {symbol}: {str(e)}")
            return {
                "symbol": symbol,
                "error": str(e),
                "timestamp": time.time(),
            }

    async def place_market_order(self, symbol: str, side: str, quantity: float) -> Dict[str, Any]:
        """Place a market order"""
        try:
            # Validate side
            if side.upper() not in [SIDE_BUY, SIDE_SELL]:
                raise ValueError("Side must be 'BUY' or 'SELL'")

            # Place order
            order = self.client.order_market(
                symbol=symbol,
                side=side.upper(),
                quantity=quantity,
                type=ORDER_TYPE_MARKET,
            )

            result = {
                "order_id": order["orderId"],
                "symbol": order["symbol"],
                "side": order["side"],
                "type": order["type"],
                "quantity": float(order["origQty"]),
                "status": order["status"],
                "fills": order.get("fills", []),
                "timestamp": time.time(),
            }

            # Calculate average price from fills
            if order.get("fills"):
                total_quote = sum(
                    float(fill["price"]) * float(fill["qty"]) for fill in order["fills"]
                )
                total_qty = sum(float(fill["qty"]) for fill in order["fills"])
                result["average_price"] = total_quote / total_qty if total_qty > 0 else 0

            logger.info(f"Market order placed: {result}")
            return result

        except BinanceOrderException as e:
            logger.error(f"Binance order error: {str(e)}")
            return {
                "error": str(e),
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "timestamp": time.time(),
            }
        except Exception as e:
            logger.error(f"Error placing market order: {str(e)}")
            return {
                "error": str(e),
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "timestamp": time.time(),
            }

    async def place_limit_order(
        self, symbol: str, side: str, quantity: float, price: float
    ) -> Dict[str, Any]:
        """Place a limit order"""
        try:
            # Validate side
            if side.upper() not in [SIDE_BUY, SIDE_SELL]:
                raise ValueError("Side must be 'BUY' or 'SELL'")

            # Use Decimal for precise price formatting
            price_decimal = Decimal(str(price)).quantize(Decimal("0.00000001"))
            quantity_decimal = Decimal(str(quantity)).quantize(Decimal("0.00000001"))

            # Place order
            order = self.client.order_limit(
                symbol=symbol,
                side=side.upper(),
                quantity=str(quantity_decimal),
                price=str(price_decimal),
                type=ORDER_TYPE_LIMIT,
                timeInForce=TIME_IN_FORCE_GTC,
            )

            result = {
                "order_id": order["orderId"],
                "symbol": order["symbol"],
                "side": order["side"],
                "type": order["type"],
                "quantity": float(order["origQty"]),
                "price": float(order["price"]),
                "status": order["status"],
                "timestamp": time.time(),
            }

            logger.info(f"Limit order placed: {result}")
            return result

        except BinanceOrderException as e:
            logger.error(f"Binance order error: {str(e)}")
            return {
                "error": str(e),
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "price": price,
                "timestamp": time.time(),
            }
        except Exception as e:
            logger.error(f"Error placing limit order: {str(e)}")
            return {
                "error": str(e),
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "price": price,
                "timestamp": time.time(),
            }

    async def cancel_order(self, symbol: str, order_id: int) -> Dict[str, Any]:
        """Cancel an order"""
        try:
            result = self.client.cancel_order(symbol=symbol, orderId=order_id)

            return {
                "order_id": result["orderId"],
                "symbol": result["symbol"],
                "status": result["status"],
                "timestamp": time.time(),
            }

        except BinanceAPIException as e:
            logger.error(f"Binance API error canceling order: {str(e)}")
            return {
                "error": str(e),
                "symbol": symbol,
                "order_id": order_id,
                "timestamp": time.time(),
            }

    async def get_open_orders(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Get open orders"""
        try:
            orders = self.client.get_open_orders(symbol=symbol)

            result = {
                "orders": [],
                "count": len(orders),
                "timestamp": time.time(),
            }

            for order in orders:
                order_data = {
                    "order_id": order["orderId"],
                    "symbol": order["symbol"],
                    "side": order["side"],
                    "type": order["type"],
                    "quantity": float(order["origQty"]),
                    "executed_qty": float(order["executedQty"]),
                    "price": (float(order["price"]) if order["price"] != "0" else None),
                    "status": order["status"],
                    "time": order["time"],
                    "update_time": order["updateTime"],
                }
                result["orders"].append(order_data)

            return result

        except BinanceAPIException as e:
            logger.error(f"Binance API error getting open orders: {str(e)}")
            return {
                "orders": [],
                "count": 0,
                "error": str(e),
                "timestamp": time.time(),
            }

    async def get_order_history(
        self, symbol: Optional[str] = None, limit: int = 100
    ) -> Dict[str, Any]:
        """Get order history"""
        try:
            orders = self.client.get_all_orders(symbol=symbol, limit=limit)

            result = {
                "orders": [],
                "count": len(orders),
                "timestamp": time.time(),
            }

            for order in orders:
                order_data = {
                    "order_id": order["orderId"],
                    "symbol": order["symbol"],
                    "side": order["side"],
                    "type": order["type"],
                    "quantity": float(order["origQty"]),
                    "executed_qty": float(order["executedQty"]),
                    "price": (float(order["price"]) if order["price"] != "0" else None),
                    "status": order["status"],
                    "time": order["time"],
                    "update_time": order["updateTime"],
                }

                # Add fills if available
                if "fills" in order:
                    order_data["fills"] = order["fills"]

                result["orders"].append(order_data)

            return result

        except BinanceAPIException as e:
            logger.error(f"Binance API error getting order history: {str(e)}")
            return {
                "orders": [],
                "count": 0,
                "error": str(e),
                "timestamp": time.time(),
            }

    async def get_trade_history(
        self, symbol: Optional[str] = None, limit: int = 100
    ) -> Dict[str, Any]:
        """Get trade history"""
        try:
            trades = self.client.get_my_trades(symbol=symbol, limit=limit)

            result = {
                "trades": [],
                "count": len(trades),
                "timestamp": time.time(),
            }

            for trade in trades:
                trade_data = {
                    "trade_id": trade["id"],
                    "order_id": trade["orderId"],
                    "symbol": trade["symbol"],
                    "side": trade["side"],
                    "quantity": float(trade["qty"]),
                    "price": float(trade["price"]),
                    "quote_qty": float(trade["quoteQty"]),
                    "commission": float(trade["commission"]),
                    "commission_asset": trade["commissionAsset"],
                    "time": trade["time"],
                }
                result["trades"].append(trade_data)

            return result

        except BinanceAPIException as e:
            logger.error(f"Binance API error getting trade history: {str(e)}")
            return {
                "trades": [],
                "count": 0,
                "error": str(e),
                "timestamp": time.time(),
            }

    async def get_symbol_info(self, symbol: str) -> Dict[str, Any]:
        """Get symbol information"""
        try:
            info = self.client.get_symbol_info(symbol)

            if info:
                return {
                    "symbol": info["symbol"],
                    "base_asset": info["baseAsset"],
                    "quote_asset": info["quoteAsset"],
                    "status": info["status"],
                    "filters": info["filters"],
                    "timestamp": time.time(),
                }
            else:
                return {
                    "error": f"Symbol {symbol} not found",
                    "timestamp": time.time(),
                }

        except BinanceAPIException as e:
            logger.error(f"Binance API error getting symbol info: {str(e)}")
            return {
                "error": str(e),
                "symbol": symbol,
                "timestamp": time.time(),
            }

    async def get_exchange_info(self) -> Dict[str, Any]:
        """Get exchange information"""
        try:
            info = self.client.get_exchange_info()

            return {
                "timezone": info["timezone"],
                "server_time": info["serverTime"],
                "symbols": len(info["symbols"]),
                "rate_limits": info["rateLimits"],
                "timestamp": time.time(),
            }

        except BinanceAPIException as e:
            logger.error(f"Binance API error getting exchange info: {str(e)}")
            return {"error": str(e), "timestamp": time.time()}

    async def test_connection(self) -> Dict[str, Any]:
        """Test connection to Binance API"""
        try:
            # Test server time
            server_time = self.client.get_server_time()

            # Test account info (if API keys provided)
            account_test = None
            if self.api_key and self.secret_key:
                try:
                    self.client.get_account()
                    account_test = "success"
                except Exception as e:
                    logger.warning(f"Account access test failed: {e}")
                    account_test = "failed"

            return {
                "connection": "success",
                "server_time": server_time["serverTime"],
                "account_access": account_test,
                "testnet": self.testnet,
                "timestamp": time.time(),
            }

        except Exception as e:
            logger.error(f"Binance connection test failed: {str(e)}")
            return {
                "connection": "failed",
                "error": str(e),
                "testnet": self.testnet,
                "timestamp": time.time(),
            }


# Global instance (will be initialized with API keys)
binance_trading_service = None


def get_binance_trading_service(
    api_key: str = "", secret_key: str = "", testnet: bool = True
) -> BinanceTradingService:
    """Get or create Binance trading service instance"""
    global binance_trading_service
    if binance_trading_service is None:
        binance_trading_service = BinanceTradingService(api_key, secret_key, testnet)
    return binance_trading_service


