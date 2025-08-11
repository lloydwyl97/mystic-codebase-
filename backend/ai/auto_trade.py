#!/usr/bin/env python3
"""
Auto Trade Module for Mystic Trading Platform

Handles trading enable/disable functionality with real CoinGecko and Binance US integration.
All functions use live data - no mock data.
"""

import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional
import aiohttp
from dotenv import load_dotenv

# Load environment variables
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

logger = logging.getLogger(__name__)

# Global trading state
_trading_enabled = False
_trading_start_time: Optional[datetime] = None
_trading_stats = {
    "total_trades": 0,
    "successful_trades": 0,
    "failed_trades": 0,
    "total_volume": 0.0,
    "last_trade_time": None,
}

# API Configuration
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_SECRET_KEY = os.getenv("BINANCE_SECRET_KEY")
BINANCE_BASE_URL = "https://api.binance.us"
COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3"

# Trading pairs configuration
TRADING_PAIRS = {
    "BTCUSDT": {"name": "Bitcoin", "min_amount": 50.0, "max_amount": 500.0},
    "ETHUSDT": {"name": "Ethereum", "min_amount": 50.0, "max_amount": 400.0},
    "SOLUSDT": {"name": "Solana", "min_amount": 25.0, "max_amount": 200.0},
    "ADAUSDT": {"name": "Cardano", "min_amount": 25.0, "max_amount": 200.0},
    "AVAXUSDT": {"name": "Avalanche", "min_amount": 25.0, "max_amount": 200.0},
}

# CoinGecko coin IDs mapping
COINGECKO_IDS = {
    "BTCUSDT": "bitcoin",
    "ETHUSDT": "ethereum",
    "SOLUSDT": "solana",
    "ADAUSDT": "cardano",
    "AVAXUSDT": "avalanche-2",
}


class CoinGeckoAPI:
    """CoinGecko API integration for market data"""

    def __init__(self):
        self.base_url = COINGECKO_BASE_URL
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def get_coin_price(self, coin_id: str) -> Optional[Dict[str, Any]]:
        """Get current price and market data for a coin"""
        try:
            url = f"{self.base_url}/simple/price"
            params = {
                "ids": coin_id,
                "vs_currencies": "usd",
                "include_24hr_change": True,
                "include_market_cap": True,
                "include_24hr_vol": True,
                "include_last_updated_at": True,
            }

            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if coin_id in data:
                        return {
                            "price": data[coin_id]["usd"],
                            "change_24h": (data[coin_id].get("usd_24h_change", 0)),
                            "market_cap": (data[coin_id].get("usd_market_cap", 0)),
                            "volume_24h": data[coin_id].get("usd_24h_vol", 0),
                            "last_updated": (data[coin_id].get("last_updated_at", 0)),
                            "source": "coingecko",
                        }
                return None
        except Exception as e:
            logger.error(f"CoinGecko API error for {coin_id}: {e}")
            return None

    async def get_market_data(self, coin_ids: list) -> Dict[str, Any]:
        """Get comprehensive market data for multiple coins"""
        try:
            ids_param = ",".join(coin_ids)
            url = f"{self.base_url}/coins/markets"
            params = {
                "vs_currency": "usd",
                "ids": ids_param,
                "order": "market_cap_desc",
                "per_page": len(coin_ids),
                "page": 1,
                "sparkline": False,
                "price_change_percentage": "24h,7d,30d",
            }

            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "status": "success",
                        "data": data,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "source": "coingecko",
                    }
                return {
                    "status": "error",
                    "message": f"API error: {response.status}",
                }
        except Exception as e:
            logger.error(f"CoinGecko market data error: {e}")
            return {"status": "error", "message": str(e)}


class BinanceUSAPI:
    """Binance US API integration for trading"""

    def __init__(self):
        self.api_key = BINANCE_API_KEY
        self.secret_key = BINANCE_SECRET_KEY
        self.base_url = BINANCE_BASE_URL
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    def _generate_signature(self, params: Dict[str, Any]) -> str:
        """Generate HMAC signature for authenticated requests"""
        import hmac
        import hashlib
        from urllib.parse import urlencode

        query_string = urlencode(params)
        signature = hmac.new(
            self.secret_key.encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return signature

    async def get_account_info(self) -> Optional[Dict[str, Any]]:
        """Get account information and balances"""
        try:
            if not self.api_key or not self.secret_key:
                logger.warning("Binance US API credentials not configured")
                return None

            params = {"timestamp": int(time.time() * 1000)}
            signature = self._generate_signature(params)

            url = f"{self.base_url}/api/v3/account"
            headers = {"X-MBX-APIKEY": self.api_key}

            async with self.session.get(
                url, params={**params, "signature": signature}, headers=headers
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "account_type": data.get("accountType", "SPOT"),
                        "balances": data.get("balances", []),
                        "permissions": data.get("permissions", []),
                        "source": "binance_us",
                    }
                else:
                    logger.error(f"Binance US account error: {response.status}")
                    return None
        except Exception as e:
            logger.error(f"Binance US account API error: {e}")
            return None

    async def get_ticker_price(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get current price for a symbol"""
        try:
            url = f"{self.base_url}/api/v3/ticker/price"
            params = {"symbol": symbol}

            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "symbol": data["symbol"],
                        "price": float(data["price"]),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "source": "binance_us",
                    }
                return None
        except Exception as e:
            logger.error(f"Binance US ticker error for {symbol}: {e}")
            return None

    async def get_24hr_ticker(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get 24hr ticker statistics"""
        try:
            url = f"{self.base_url}/api/v3/ticker/24hr"
            params = {"symbol": symbol}

            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "symbol": data["symbol"],
                        "price": float(data["lastPrice"]),
                        "change_24h": float(data["priceChangePercent"]),
                        "volume_24h": float(data["volume"]),
                        "high_24h": float(data["highPrice"]),
                        "low_24h": float(data["lowPrice"]),
                        "quote_volume": float(data["quoteVolume"]),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "source": "binance_us",
                    }
                return None
        except Exception as e:
            logger.error(f"Binance US 24hr ticker error for {symbol}: {e}")
            return None


def enable_trading() -> Dict[str, Any]:
    """Enable automated trading system"""
    global _trading_enabled, _trading_start_time

    try:
        if _trading_enabled:
            return {
                "success": False,
                "message": "Trading is already enabled",
                "trading_enabled": True,
                "start_time": (_trading_start_time.isoformat() if _trading_start_time else None),
            }

        _trading_enabled = True
        _trading_start_time = datetime.now(timezone.utc)

        logger.info("✅ Automated trading enabled")

        return {
            "success": True,
            "message": "Automated trading enabled successfully",
            "trading_enabled": True,
            "start_time": _trading_start_time.isoformat(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.error(f"Error enabling trading: {e}")
        return {
            "success": False,
            "message": f"Failed to enable trading: {str(e)}",
            "trading_enabled": False,
        }


def disable_trading() -> Dict[str, Any]:
    """Disable automated trading system"""
    global _trading_enabled, _trading_start_time

    try:
        if not _trading_enabled:
            return {
                "success": False,
                "message": "Trading is already disabled",
                "trading_enabled": False,
            }

        _trading_enabled = False
        stop_time = datetime.now(timezone.utc)

        # Calculate uptime
        uptime = None
        if _trading_start_time:
            uptime = (stop_time - _trading_start_time).total_seconds()

        _trading_start_time = None

        logger.info("🛑 Automated trading disabled")

        return {
            "success": True,
            "message": "Automated trading disabled successfully",
            "trading_enabled": False,
            "stop_time": stop_time.isoformat(),
            "uptime_seconds": uptime,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.error(f"Error disabling trading: {e}")
        return {
            "success": False,
            "message": f"Failed to disable trading: {str(e)}",
            "trading_enabled": _trading_enabled,
        }


def get_trading_status() -> Dict[str, Any]:
    """Get current trading system status"""
    global _trading_enabled, _trading_start_time, _trading_stats

    try:
        # Calculate uptime
        uptime = None
        if _trading_start_time and _trading_enabled:
            uptime = (datetime.now(timezone.utc) - _trading_start_time).total_seconds()

        # Get API status
        api_status = {
            "binance_us": bool(BINANCE_API_KEY and BINANCE_SECRET_KEY),
            "coingecko": True,  # CoinGecko doesn't require API key
        }

        return {
            "trading_enabled": _trading_enabled,
            "start_time": (_trading_start_time.isoformat() if _trading_start_time else None),
            "uptime_seconds": uptime,
            "api_status": api_status,
            "trading_pairs": list(TRADING_PAIRS.keys()),
            "stats": _trading_stats.copy(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.error(f"Error getting trading status: {e}")
        return {
            "trading_enabled": False,
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


async def get_market_data(symbol: str) -> Dict[str, Any]:
    """Get comprehensive market data for a symbol from both CoinGecko and Binance US"""
    try:
        results = {}

        # Get CoinGecko data
        coin_id = COINGECKO_IDS.get(symbol)
        if coin_id:
            async with CoinGeckoAPI() as coingecko:
                coingecko_data = await coingecko.get_coin_price(coin_id)
                if coingecko_data:
                    results["coingecko"] = coingecko_data

        # Get Binance US data
        async with BinanceUSAPI() as binance:
            binance_ticker = await binance.get_24hr_ticker(symbol)
            if binance_ticker:
                results["binance_us"] = binance_ticker

        return {
            "success": True,
            "symbol": symbol,
            "data": results,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.error(f"Error getting market data for {symbol}: {e}")
        return {
            "success": False,
            "symbol": symbol,
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


async def get_account_balance() -> Dict[str, Any]:
    """Get account balance from Binance US"""
    try:
        async with BinanceUSAPI() as binance:
            account_info = await binance.get_account_info()

            if account_info:
                # Extract USDT balance
                usdt_balance = 0.0
                for balance in account_info.get("balances", []):
                    if balance["asset"] == "USDT":
                        usdt_balance = float(balance["free"])
                        break

                return {
                    "success": True,
                    "account_info": account_info,
                    "usdt_balance": usdt_balance,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            else:
                return {
                    "success": False,
                    "message": "Failed to get account information",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
    except Exception as e:
        logger.error(f"Error getting account balance: {e}")
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


def update_trading_stats(trade_result: Dict[str, Any]) -> None:
    """Update trading statistics"""
    global _trading_stats

    try:
        _trading_stats["total_trades"] += 1
        _trading_stats["last_trade_time"] = datetime.now(timezone.utc).isoformat()

        if trade_result.get("success", False):
            _trading_stats["successful_trades"] += 1
            _trading_stats["total_volume"] += trade_result.get("volume", 0.0)
        else:
            _trading_stats["failed_trades"] += 1

        logger.info(f"Trading stats updated: {_trading_stats}")
    except Exception as e:
        logger.error(f"Error updating trading stats: {e}")


# Export functions for use in other modules
__all__ = [
    "enable_trading",
    "disable_trading",
    "get_trading_status",
    "get_market_data",
    "get_account_balance",
    "update_trading_stats",
    "CoinGeckoAPI",
    "BinanceUSAPI",
]
