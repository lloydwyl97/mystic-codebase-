"""
Auto Execution Service for Mystic AI Trading Platform
Handles live trading execution via CCXT library for multiple exchanges.
"""

import os
import logging
import uuid
from typing import Dict, Any
from datetime import datetime, timezone
import sys

# Add backend to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from backend.modules.ai.persistent_cache import PersistentCache

logger = logging.getLogger(__name__)

try:
    import ccxt
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False
    logger.warning("CCXT library not available. Install with: pip install ccxt")


class AutoExecutionService:
    def __init__(self, test_mode: bool = True):
        """Initialize auto execution service with exchange connections"""
        self.cache = PersistentCache()
        self.test_mode = test_mode

        # Exchange configurations
        self.exchanges = {}
        self.exchange_configs = {
            'binance': {
                'api_key': os.getenv('BINANCE_API_KEY'),
                'secret': os.getenv('BINANCE_SECRET'),
                'sandbox': test_mode,
                'options': {
                    'defaultType': 'spot',
                    'adjustForTimeDifference': True
                }
            },
            'coinbase': {
                'api_key': os.getenv('COINBASE_API_KEY'),
                'secret': os.getenv('COINBASE_SECRET'),
                'sandbox': test_mode,
                'options': {
                    'defaultType': 'spot'
                }
            }
        }

        # Initialize exchanges
        self._init_exchanges()

        logger.info(f"âœ… AutoExecutionService initialized (test_mode: {test_mode})")

    def _init_exchanges(self):
        """Initialize exchange connections"""
        if not CCXT_AVAILABLE:
            logger.error("âŒ CCXT library not available")
            return

        for exchange_id, config in self.exchange_configs.items():
            try:
                if exchange_id == 'binance':
                    exchange = ccxt.binanceus(config)
                elif exchange_id == 'coinbase':
                    exchange = ccxt.coinbase(config)
                else:
                    continue

                # Test connection
                if self.test_mode:
                    logger.info(f"ðŸ”§ {exchange_id} initialized in test mode")
                else:
                    # Load markets to test connection
                    exchange.load_markets()
                    logger.info(f"âœ… {exchange_id} connected successfully")

                self.exchanges[exchange_id] = exchange

            except Exception as e:
                logger.error(f"âŒ Failed to initialize {exchange_id}: {e}")

    def get_balance(self, exchange: str) -> Dict[str, Any]:
        """Get account balance for specified exchange"""
        try:
            if exchange not in self.exchanges:
                return {"success": False, "error": f"Exchange {exchange} not available"}

            ex = self.exchanges[exchange]

            if self.test_mode:
                # Return mock balance for test mode
                return {
                    "success": True,
                    "exchange": exchange,
                    "test_mode": True,
                    "balances": {
                        "USDT": {"free": 10000.0, "used": 0.0, "total": 10000.0},
                        "BTC": {"free": 0.5, "used": 0.0, "total": 0.5},
                        "ETH": {"free": 5.0, "used": 0.0, "total": 5.0}
                    }
                }

            # Get real balance
            balance = ex.fetch_balance()

            return {
                "success": True,
                "exchange": exchange,
                "test_mode": False,
                "balances": balance
            }

        except Exception as e:
            logger.error(f"âŒ Failed to get balance for {exchange}: {e}")
            return {"success": False, "error": str(e)}

    def place_buy_order(self, exchange: str, symbol: str, amount_in_usd: float) -> Dict[str, Any]:
        """Place a buy order for specified amount in USD"""
        try:
            if exchange not in self.exchanges:
                return {"success": False, "error": f"Exchange {exchange} not available"}

            ex = self.exchanges[exchange]

            if self.test_mode:
                # Return mock order for test mode
                order_id = f"test_buy_{uuid.uuid4().hex[:8]}"
                return {
                    "success": True,
                    "order_id": order_id,
                    "symbol": symbol,
                    "side": "buy",
                    "amount_usd": amount_in_usd,
                    "status": "filled",
                    "test_mode": True,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }

            # Get current price for the symbol
            ticker = ex.fetch_ticker(symbol)
            price = ticker['last']

            # Calculate amount in base currency
            amount = amount_in_usd / price

            # Place real order
            order = ex.create_market_buy_order(symbol, amount)

            return {
                "success": True,
                "order_id": order['id'],
                "symbol": symbol,
                "side": "buy",
                "amount_usd": amount_in_usd,
                "amount_base": amount,
                "price": price,
                "status": order['status'],
                "test_mode": False,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "order": order
            }

        except Exception as e:
            logger.error(f"âŒ Failed to place buy order for {symbol}: {e}")
            return {"success": False, "error": str(e)}

    def place_sell_order(self, exchange: str, symbol: str, amount_in_usd: float) -> Dict[str, Any]:
        """Place a sell order for specified amount in USD"""
        try:
            if exchange not in self.exchanges:
                return {"success": False, "error": f"Exchange {exchange} not available"}

            ex = self.exchanges[exchange]

            if self.test_mode:
                # Return mock order for test mode
                order_id = f"test_sell_{uuid.uuid4().hex[:8]}"
                return {
                    "success": True,
                    "order_id": order_id,
                    "symbol": symbol,
                    "side": "sell",
                    "amount_usd": amount_in_usd,
                    "status": "filled",
                    "test_mode": True,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }

            # Get current price for the symbol
            ticker = ex.fetch_ticker(symbol)
            price = ticker['last']

            # Calculate amount in base currency
            amount = amount_in_usd / price

            # Place real order
            order = ex.create_market_sell_order(symbol, amount)

            return {
                "success": True,
                "order_id": order['id'],
                "symbol": symbol,
                "side": "sell",
                "amount_usd": amount_in_usd,
                "amount_base": amount,
                "price": price,
                "status": order['status'],
                "test_mode": False,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "order": order
            }

        except Exception as e:
            logger.error(f"âŒ Failed to place sell order for {symbol}: {e}")
            return {"success": False, "error": str(e)}

    def get_order_status(self, exchange: str, order_id: str) -> Dict[str, Any]:
        """Get status of an order"""
        try:
            if exchange not in self.exchanges:
                return {"success": False, "error": f"Exchange {exchange} not available"}

            ex = self.exchanges[exchange]

            if self.test_mode:
                # Return mock order status for test mode
                return {
                    "success": True,
                    "order_id": order_id,
                    "status": "filled",
                    "test_mode": True,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }

            # Get real order status
            order = ex.fetch_order(order_id)

            return {
                "success": True,
                "order_id": order_id,
                "status": order['status'],
                "test_mode": False,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "order": order
            }

        except Exception as e:
            logger.error(f"âŒ Failed to get order status for {order_id}: {e}")
            return {"success": False, "error": str(e)}

    def cancel_order(self, exchange: str, order_id: str) -> Dict[str, Any]:
        """Cancel an order"""
        try:
            if exchange not in self.exchanges:
                return {"success": False, "error": f"Exchange {exchange} not available"}

            if self.test_mode:
                return {
                    "success": True,
                    "order_id": order_id,
                    "status": "canceled",
                    "test_mode": True
                }

            ex = self.exchanges[exchange]
            result = ex.cancel_order(order_id)

            return {
                "success": True,
                "order_id": order_id,
                "status": "canceled",
                "test_mode": False,
                "result": result
            }

        except Exception as e:
            logger.error(f"âŒ Failed to cancel order {order_id}: {e}")
            return {"success": False, "error": str(e)}

    def get_service_status(self) -> Dict[str, Any]:
        """Get current service status and exchange information"""
        try:
            exchange_status = {}
            for exchange_id, exchange in self.exchanges.items():
                try:
                    if self.test_mode:
                        exchange_status[exchange_id] = {
                            "connected": True,
                            "test_mode": True,
                            "markets_loaded": False
                        }
                    else:
                        markets = exchange.load_markets()
                        exchange_status[exchange_id] = {
                            "connected": True,
                            "test_mode": False,
                            "markets_loaded": len(markets) > 0,
                            "markets_count": len(markets)
                        }
                except Exception as e:
                    exchange_status[exchange_id] = {
                        "connected": False,
                        "error": str(e)
                    }

            return {
                "service": "AutoExecutionService",
                "status": "active",
                "test_mode": self.test_mode,
                "ccxt_available": CCXT_AVAILABLE,
                "exchanges": exchange_status,
                "configuration": {
                    "binance_api_key": "***" if os.getenv('BINANCE_API_KEY') else "NOT_SET",
                    "binance_secret": "***" if os.getenv('BINANCE_SECRET') else "NOT_SET",
                    "coinbase_api_key": "***" if os.getenv('COINBASE_API_KEY') else "NOT_SET",
                    "coinbase_secret": "***" if os.getenv('COINBASE_SECRET') else "NOT_SET"
                }
            }

        except Exception as e:
            logger.error(f"âŒ Failed to get service status: {e}")
            return {"success": False, "error": str(e)}

    def set_test_mode(self, enabled: bool):
        """Enable or disable test mode"""
        self.test_mode = enabled
        logger.info(f"Test mode: {'ENABLED' if enabled else 'DISABLED'}")

        # Reinitialize exchanges with new test mode
        self._init_exchanges()


# Global service instance
auto_execution_service = AutoExecutionService()


def get_auto_execution_service() -> AutoExecutionService:
    """Get the global auto execution service instance"""
    return auto_execution_service


if __name__ == "__main__":
    # Test the service
    service = AutoExecutionService()
    print(f"âœ… AutoExecutionService initialized: {service}")

    # Test balance
    balance = service.get_balance('binance')
    print(f"Balance result: {balance}")

    # Test status
    status = service.get_service_status()
    print(f"Service status: {status['status']}")
    print(f"Test mode: {status['test_mode']}")
    print(f"CCXT available: {status['ccxt_available']}")


