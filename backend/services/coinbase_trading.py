"""
Coinbase Trading Service

Provides real trading capabilities on Coinbase API.
Handles orders, account management, and real-time data.
"""

import base64
import hashlib
import hmac
import json
import logging
import time
from decimal import Decimal
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger(__name__)


class CoinbaseTradingService:
    """Coinbase trading service with real trading capabilities"""

    def __init__(self, api_key: str = "", secret_key: str = "", sandbox: bool = True):
        self.api_key = api_key
        self.secret_key = secret_key
        self.sandbox = sandbox

        # Set base URL based on sandbox mode
        if sandbox:
            self.base_url = "https://api-public.sandbox.exchange.coinbase.us"
        else:
            self.base_url = "https://api.coinbase.com"

        # Account info cache
        self.account_info_cache = {}
        self.cache_timestamp = 0
        self.cache_duration = 30  # seconds

        # Order tracking
        self.active_orders = {}

        logger.info(f"Coinbase trading service initialized (sandbox: {sandbox})")

    def _sign_request(self, method: str, path: str, body: str = "") -> Dict[str, str]:
        """Sign request for Coinbase API authentication"""
        timestamp = str(int(time.time()))
        message = timestamp + method + path + body

        # Decode the private key
        try:
            # Remove header/footer and decode
            key_content = (
                self.secret_key.replace("-----BEGIN EC PRIVATE KEY-----", "")
                .replace("-----END EC PRIVATE KEY-----", "")
                .replace("\n", "")
            )
            private_key = base64.b64decode(key_content)

            # Create signature
            signature = hmac.new(private_key, message.encode("utf-8"), hashlib.sha256)
            signature_b64 = base64.b64encode(signature.digest()).decode("utf-8")

            return {
                "CB-ACCESS-KEY": self.api_key,
                "CB-ACCESS-SIGN": signature_b64,
                "CB-ACCESS-TIMESTAMP": timestamp,
                "CB-ACCESS-PASSPHRASE": "lloyd",  # From env file
                "Content-Type": "application/json",
            }
        except Exception as e:
            logger.error(f"Error signing request: {e}")
            return {}

    async def get_account_info(self) -> Dict[str, Any]:
        """Get account information and balances"""
        now = time.time()

        # Check cache
        if (now - self.cache_timestamp) < self.cache_duration:
            return self.account_info_cache

        try:
            if not self.api_key or not self.secret_key:
                logger.warning("Coinbase API credentials not configured")
                return {
                    "error": "API credentials not configured",
                    "timestamp": now,
                }

            # Make real API call
            path = "/accounts"
            headers = self._sign_request("GET", path)

            if not headers:
                return {"error": "Failed to sign request", "timestamp": now}

            url = f"{self.base_url}{path}"
            response = requests.get(url, headers=headers, timeout=10)

            if response.status_code == 200:
                accounts_data = response.json()

                # Calculate total balance
                total_balance_usd = 0.0
                for account in accounts_data:
                    if account.get("currency") == "USD":
                        total_balance_usd += float(account.get("available", 0))
                    elif account.get("currency") in [
                        "BTC",
                        "ETH",
                        "SOL",
                        "AVAX",
                    ]:
                        # Get current price to calculate USD value
                        try:
                            price_response = requests.get(
                                f"{self.base_url}/products/{account['currency']}-USD/ticker",
                                timeout=5,
                            )
                            if price_response.status_code == 200:
                                price_data = price_response.json()
                                price = float(price_data.get("price", 0))
                                total_balance_usd += float(account.get("available", 0)) * price
                        except Exception as price_error:
                            logger.warning(
                                f"Could not get price for {account.get('currency', 'unknown')}: {price_error}"
                            )
                            pass

                result = {
                    "accounts": accounts_data,
                    "total_balance_usd": total_balance_usd,
                    "timestamp": now,
                    "source": "coinbase_api",
                }

                self.account_info_cache = result
                self.cache_timestamp = now
                return result
            else:
                logger.error(f"Coinbase API error: {response.status_code} - {response.text}")
                return {
                    "error": f"API error: {response.status_code}",
                    "timestamp": now,
                }

        except Exception as e:
            logger.error(f"Error getting account info: {str(e)}")
            return {"error": str(e), "timestamp": now}

    async def get_market_price(self, product_id: str) -> Dict[str, Any]:
        """Get current market price for a product"""
        try:
            # Use public API endpoint for price data
            url = f"{self.base_url}/products/{product_id}/ticker"
            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                data = response.json()
                return {
                    "product_id": product_id,
                    "price": float(data.get("price", 0)),
                    "volume_24h": float(data.get("volume", 0)),
                    "timestamp": time.time(),
                    "source": "coinbase_api",
                }
            else:
                logger.error(f"Coinbase price API error: {response.status_code}")
                return {
                    "product_id": product_id,
                    "error": f"API error: {response.status_code}",
                    "timestamp": time.time(),
                }

        except Exception as e:
            logger.error(f"Error getting price for {product_id}: {str(e)}")
            return {
                "product_id": product_id,
                "error": str(e),
                "timestamp": time.time(),
            }

    async def place_market_order(self, product_id: str, side: str, size: float) -> Dict[str, Any]:
        """Place a market order"""
        try:
            if not self.api_key or not self.secret_key:
                return {
                    "error": "API credentials not configured",
                    "timestamp": time.time(),
                }

            # Use Decimal for precise size formatting
            size_decimal = Decimal(str(size)).quantize(Decimal("0.00000001"))

            # Prepare order data
            order_data = {
                "type": "market",
                "side": side.lower(),
                "product_id": product_id,
                "size": str(size_decimal),
            }

            body = json.dumps(order_data)
            path = "/orders"
            headers = self._sign_request("POST", path, body)

            if not headers:
                return {
                    "error": "Failed to sign request",
                    "timestamp": time.time(),
                }

            url = f"{self.base_url}{path}"
            response = requests.post(url, headers=headers, data=body, timeout=30)

            if response.status_code == 200:
                result = response.json()
                result["source"] = "coinbase_api"
                logger.info(f"Market order placed: {result}")
                return result
            else:
                logger.error(f"Coinbase order API error: {response.status_code} - {response.text}")
                return {
                    "error": f"API error: {response.status_code}",
                    "timestamp": time.time(),
                }

        except Exception as e:
            logger.error(f"Error placing market order: {str(e)}")
            return {
                "error": str(e),
                "product_id": product_id,
                "side": side,
                "size": size,
                "timestamp": time.time(),
            }

    async def place_limit_order(
        self, product_id: str, side: str, size: float, price: float
    ) -> Dict[str, Any]:
        """Place a limit order"""
        try:
            if not self.api_key or not self.secret_key:
                return {
                    "error": "API credentials not configured",
                    "timestamp": time.time(),
                }

            # Prepare order data
            order_data = {
                "type": "limit",
                "side": side.lower(),
                "product_id": product_id,
                "size": str(size),
                "price": str(price),
            }

            body = json.dumps(order_data)
            path = "/orders"
            headers = self._sign_request("POST", path, body)

            if not headers:
                return {
                    "error": "Failed to sign request",
                    "timestamp": time.time(),
                }

            url = f"{self.base_url}{path}"
            response = requests.post(url, headers=headers, data=body, timeout=30)

            if response.status_code == 200:
                result = response.json()
                result["source"] = "coinbase_api"
                logger.info(f"Limit order placed: {result}")
                return result
            else:
                logger.error(f"Coinbase order API error: {response.status_code} - {response.text}")
                return {
                    "error": f"API error: {response.status_code}",
                    "timestamp": time.time(),
                }

        except Exception as e:
            logger.error(f"Error placing limit order: {str(e)}")
            return {
                "error": str(e),
                "product_id": product_id,
                "side": side,
                "size": size,
                "price": price,
                "timestamp": time.time(),
            }

    async def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel an order"""
        try:
            if not self.api_key or not self.secret_key:
                return {
                    "error": "API credentials not configured",
                    "timestamp": time.time(),
                }

            path = f"/orders/{order_id}"
            headers = self._sign_request("DELETE", path)

            if not headers:
                return {
                    "error": "Failed to sign request",
                    "timestamp": time.time(),
                }

            url = f"{self.base_url}{path}"
            response = requests.delete(url, headers=headers, timeout=10)

            if response.status_code == 200:
                result = response.json()
                result["source"] = "coinbase_api"
                return result
            else:
                logger.error(f"Coinbase cancel API error: {response.status_code}")
                return {
                    "error": f"API error: {response.status_code}",
                    "timestamp": time.time(),
                }

        except Exception as e:
            logger.error(f"Error canceling order: {str(e)}")
            return {
                "error": str(e),
                "order_id": order_id,
                "timestamp": time.time(),
            }

    async def get_open_orders(self, product_id: Optional[str] = None) -> Dict[str, Any]:
        """Get open orders"""
        try:
            if not self.api_key or not self.secret_key:
                return {
                    "error": "API credentials not configured",
                    "timestamp": time.time(),
                }

            path = "/orders"
            if product_id:
                path += f"?product_id={product_id}"

            headers = self._sign_request("GET", path)

            if not headers:
                return {
                    "error": "Failed to sign request",
                    "timestamp": time.time(),
                }

            url = f"{self.base_url}{path}"
            response = requests.get(url, headers=headers, timeout=10)

            if response.status_code == 200:
                result = response.json()
                return {
                    "orders": result,
                    "source": "coinbase_api",
                    "timestamp": time.time(),
                }
            else:
                logger.error(f"Coinbase orders API error: {response.status_code}")
                return {
                    "error": f"API error: {response.status_code}",
                    "timestamp": time.time(),
                }

        except Exception as e:
            logger.error(f"Error getting open orders: {str(e)}")
            return {"error": str(e), "timestamp": time.time()}

    async def get_order_history(
        self, product_id: Optional[str] = None, limit: int = 100
    ) -> Dict[str, Any]:
        """Get order history"""
        try:
            if not self.api_key or not self.secret_key:
                return {
                    "error": "API credentials not configured",
                    "timestamp": time.time(),
                }

            path = "/orders"
            if product_id:
                path += f"?product_id={product_id}"
            if limit:
                path += f"{'&' if '?' in path else '?'}limit={limit}"

            headers = self._sign_request("GET", path)

            if not headers:
                return {
                    "error": "Failed to sign request",
                    "timestamp": time.time(),
                }

            url = f"{self.base_url}{path}"
            response = requests.get(url, headers=headers, timeout=10)

            if response.status_code == 200:
                result = response.json()
                return {
                    "orders": result,
                    "count": len(result),
                    "timestamp": time.time(),
                    "source": "coinbase_api",
                }
            else:
                logger.error(f"Coinbase order history API error: {response.status_code}")
                return {
                    "error": f"API error: {response.status_code}",
                    "timestamp": time.time(),
                }

        except Exception as e:
            logger.error(f"Error getting order history: {str(e)}")
            return {
                "orders": [],
                "count": 0,
                "error": str(e),
                "timestamp": time.time(),
            }

    async def get_products(self) -> Dict[str, Any]:
        """Get available products"""
        try:
            # Use public API endpoint for products
            url = f"{self.base_url}/products"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            products_data = response.json()

            result = {"products": [], "count": 0, "timestamp": time.time()}

            for product in products_data:
                if product.get("status") == "online":
                    product_data = {
                        "product_id": product["id"],
                        "base_currency": product["base_currency"],
                        "quote_currency": product["quote_currency"],
                        "status": product["status"],
                        "display_name": product.get("display_name", product["id"]),
                    }
                    result["products"].append(product_data)

            result["count"] = len(result["products"])
            return result

        except Exception as e:
            logger.error(f"Error getting products: {str(e)}")
            return {
                "products": [],
                "count": 0,
                "error": str(e),
                "timestamp": time.time(),
            }

    async def test_connection(self) -> Dict[str, Any]:
        """Test connection to Coinbase API"""
        try:
            # Test by getting products
            products = await self.get_products()

            return {
                "connection": "success",
                "products_count": products.get("count", 0),
                "sandbox": self.sandbox,
                "timestamp": time.time(),
            }

        except Exception as e:
            logger.error(f"Coinbase connection test failed: {str(e)}")
            return {
                "connection": "failed",
                "error": str(e),
                "sandbox": self.sandbox,
                "timestamp": time.time(),
            }


# Global instance (will be initialized with API keys)
coinbase_trading_service = None


def get_coinbase_trading_service(
    api_key: str = "", secret_key: str = "", sandbox: bool = True
) -> CoinbaseTradingService:
    """Get or create Coinbase trading service instance"""
    global coinbase_trading_service
    if coinbase_trading_service is None:
        coinbase_trading_service = CoinbaseTradingService(api_key, secret_key, sandbox)
    return coinbase_trading_service
