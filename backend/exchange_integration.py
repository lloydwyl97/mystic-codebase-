"""
Exchange Integration Module

Provides real trading capabilities through multiple exchange APIs.
"""

import hashlib
import hmac
import logging
import time
import urllib.parse
from dataclasses import dataclass
from datetime import timezone, datetime
from typing import Any, Dict, List, Optional

import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class OrderRequest:
    symbol: str
    side: str  # 'buy' or 'sell'
    order_type: str  # 'market', 'limit', 'stop'
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "GTC"


@dataclass
class OrderResponse:
    order_id: str
    symbol: str
    side: str
    order_type: str
    quantity: float
    price: float
    status: str
    timestamp: datetime
    fills: List[Dict[str, Any]] = None


@dataclass
class Position:
    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    margin_type: str
    timestamp: datetime


class BinanceAPI:
    """Binance US API integration for real trading."""

    def __init__(self, api_key: str = None, api_secret: str = None, testnet: bool = True):
        self.api_key = api_key or "test_key"
        self.api_secret = api_secret or "test_secret"
        self.testnet = testnet
        self.base_url = "https://testnet.binance.vision" if testnet else "https://api.binance.us"
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    def _generate_signature(self, params: Dict[str, Any]) -> str:
        """Generate HMAC signature for authenticated requests."""
        query_string = urllib.parse.urlencode(params)
        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return signature

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Dict[str, Any] = None,
        authenticated: bool = False,
    ) -> Dict[str, Any]:
        """Make HTTP request to Binance API."""
        if params is None:
            params = {}

        if authenticated:
            params["timestamp"] = int(time.time() * 1000)
            params["signature"] = self._generate_signature(params)

        url = f"{self.base_url}{endpoint}"
        headers = {"X-MBX-APIKEY": self.api_key} if authenticated else {}

        try:
            async with self.session.request(
                method, url, params=params, headers=headers
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    logger.error(f"API request failed: {response.status} - {error_text}")
                    return {"error": error_text}
        except Exception as e:
            logger.error(f"Request error: {str(e)}")
            return {"error": str(e)}

    async def get_account_info(self) -> Dict[str, Any]:
        """Get account information."""
        return await self._make_request("GET", "/api/v3/account", authenticated=True)

    async def get_open_orders(self, symbol: str = None) -> List[Dict[str, Any]]:
        """Get open orders."""
        params = {}
        if symbol:
            params["symbol"] = symbol
        return await self._make_request("GET", "/api/v3/openOrders", params, authenticated=True)

    async def get_order_history(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get order history."""
        params = {"symbol": symbol, "limit": limit}
        return await self._make_request("GET", "/api/v3/allOrders", params, authenticated=True)

    async def place_order(self, order: OrderRequest) -> OrderResponse:
        """Place a new order."""
        params = {
            "symbol": order.symbol,
            "side": order.side.upper(),
            "type": order.order_type.upper(),
            "quantity": order.quantity,
        }

        if order.price:
            params["price"] = order.price
        if order.stop_price:
            params["stopPrice"] = order.stop_price
        if order.time_in_force:
            params["timeInForce"] = order.time_in_force

        result = await self._make_request("POST", "/api/v3/order", params, authenticated=True)

        if "error" not in result:
            return OrderResponse(
                order_id=result["orderId"],
                symbol=result["symbol"],
                side=result["side"],
                order_type=result["type"],
                quantity=float(result["origQty"]),
                price=float(result["price"]) if result["price"] != "0" else 0,
                status=result["status"],
                timestamp=datetime.fromtimestamp(result["time"] / 1000, timezone.utc),
            )
        else:
            raise Exception(f"Order placement failed: {result['error']}")

    async def cancel_order(self, symbol: str, order_id: str) -> Dict[str, Any]:
        """Cancel an order."""
        params = {"symbol": symbol, "orderId": order_id}
        return await self._make_request("DELETE", "/api/v3/order", params, authenticated=True)

    async def get_ticker_price(self, symbol: str) -> Dict[str, Any]:
        """Get current price for a symbol."""
        params = {"symbol": symbol}
        return await self._make_request("GET", "/api/v3/ticker/price", params)

    async def get_klines(
        self, symbol: str, interval: str = "1h", limit: int = 100
    ) -> List[List[Any]]:
        """Get candlestick data."""
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        return await self._make_request("GET", "/api/v3/klines", params)


class CoinbaseAPI:
    """Coinbase Pro API integration."""

    def __init__(
        self,
        api_key: str = None,
        api_secret: str = None,
        passphrase: str = None,
    ):
        self.api_key = api_key or "test_key"
        self.api_secret = api_secret or "test_secret"
        self.passphrase = passphrase or "test_passphrase"
        self.base_url = "https://api-public.sandbox.exchange.coinbase.us"  # Sandbox for testing
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    def _generate_signature(
        self, timestamp: str, method: str, request_path: str, body: str = ""
    ) -> str:
        """Generate Coinbase Pro signature."""
        message = timestamp + method + request_path + body
        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            message.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return signature

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Dict[str, Any] = None,
        authenticated: bool = False,
    ) -> Dict[str, Any]:
        """Make HTTP request to Coinbase Pro API."""
        url = f"{self.base_url}{endpoint}"
        headers = {}

        if authenticated:
            timestamp = str(int(time.time()))
            body = ""
            if data:
                body = str(data)

            signature = self._generate_signature(timestamp, method, endpoint, body)
            headers.update(
                {
                    "CB-ACCESS-KEY": self.api_key,
                    "CB-ACCESS-SIGN": signature,
                    "CB-ACCESS-TIMESTAMP": timestamp,
                    "CB-ACCESS-PASSPHRASE": self.passphrase,
                    "Content-Type": "application/json",
                }
            )

        try:
            async with self.session.request(method, url, json=data, headers=headers) as response:
                if response.status in [200, 201]:
                    return await response.json()
                else:
                    error_text = await response.text()
                    logger.error(f"Coinbase API request failed: {response.status} - {error_text}")
                    return {"error": error_text}
        except Exception as e:
            logger.error(f"Coinbase request error: {str(e)}")
            return {"error": str(e)}

    async def get_accounts(self) -> List[Dict[str, Any]]:
        """Get account information."""
        return await self._make_request("GET", "/accounts", authenticated=True)

    async def place_order(self, order: OrderRequest) -> OrderResponse:
        """Place a new order."""
        data = {
            "product_id": order.symbol,
            "side": order.side,
            "type": order.order_type,
            "size": str(order.quantity),
        }

        if order.price:
            data["price"] = str(order.price)

        result = await self._make_request("POST", "/orders", data, authenticated=True)

        if "error" not in result:
            return OrderResponse(
                order_id=result["id"],
                symbol=result["product_id"],
                side=result["side"],
                order_type=result["type"],
                quantity=float(result["size"]),
                price=float(result["price"]) if result.get("price") else 0,
                status=result["status"],
                timestamp=datetime.fromisoformat(result["created_at"].replace("Z", "+00:00")),
            )
        else:
            raise Exception(f"Order placement failed: {result['error']}")


class ExchangeManager:
    """Manages multiple exchange integrations."""

    def __init__(self):
        self.exchanges = {}
        self.active_exchanges = []

    def add_exchange(self, name: str, exchange_instance):
        """Add an exchange to the manager."""
        self.exchanges[name] = exchange_instance
        self.active_exchanges.append(name)
        logger.info(f"Added exchange: {name}")

    async def get_all_positions(self) -> Dict[str, List[Position]]:
        """Get positions from all active exchanges."""
        all_positions = {}

        for exchange_name in self.active_exchanges:
            try:
                exchange = self.exchanges[exchange_name]
                if hasattr(exchange, "get_accounts"):
                    accounts = await exchange.get_accounts()
                    positions = []

                    for account in accounts:
                        if float(account.get("balance", 0)) > 0:
                            position = Position(
                                symbol=account.get("currency", ""),
                                quantity=float(account.get("balance", 0)),
                                entry_price=0,  # Would need to track this separately
                                current_price=0,  # Would need to fetch current price
                                unrealized_pnl=0,
                                realized_pnl=0,
                                margin_type="cash",
                                timestamp=datetime.now(timezone.utc),
                            )
                            positions.append(position)

                    all_positions[exchange_name] = positions
            except Exception as e:
                logger.error(f"Error getting positions from {exchange_name}: {str(e)}")
                all_positions[exchange_name] = []

        return all_positions

    async def place_order_on_all(self, order: OrderRequest) -> Dict[str, OrderResponse]:
        """Place order on all active exchanges."""
        results = {}

        for exchange_name in self.active_exchanges:
            try:
                exchange = self.exchanges[exchange_name]
                if hasattr(exchange, "place_order"):
                    result = await exchange.place_order(order)
                    results[exchange_name] = result
            except Exception as e:
                logger.error(f"Error placing order on {exchange_name}: {str(e)}")
                results[exchange_name] = {"error": str(e)}

        return results

    async def get_market_data(self, symbol: str) -> Dict[str, Dict[str, Any]]:
        """Get market data from all exchanges."""
        market_data = {}

        for exchange_name in self.active_exchanges:
            try:
                exchange = self.exchanges[exchange_name]
                if hasattr(exchange, "get_ticker_price"):
                    data = await exchange.get_ticker_price(symbol)
                    market_data[exchange_name] = data
            except Exception as e:
                logger.error(f"Error getting market data from {exchange_name}: {str(e)}")
                market_data[exchange_name] = {"error": str(e)}

        return market_data


# Global exchange manager instance
exchange_manager = ExchangeManager()


async def initialize_exchanges():
    """Initialize exchange connections."""
    try:
        # Initialize Binance (testnet)
        binance = BinanceAPI(testnet=True)
        exchange_manager.add_exchange("binance", binance)

        # Initialize Coinbase Pro (sandbox)
        coinbase = CoinbaseAPI()
        exchange_manager.add_exchange("coinbase", coinbase)

        logger.info("Exchange integrations initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing exchanges: {str(e)}")


# Remove the problematic asyncio.create_task call
# The exchanges will be initialized when the service manager starts up
