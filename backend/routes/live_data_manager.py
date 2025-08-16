"""
Live Data Manager

Handles business logic for live data operations including data processing,
validation, and response formatting.
"""

import logging
import time
from typing import Any, Dict, List, Optional

from backend.services.market_data_sources import (
    SUPPORTED_COINS,
    get_coin_support_summary,
)

logger = logging.getLogger(__name__)


class LiveDataManager:
    """Manages live data business logic and operations."""

    def __init__(self, market_data_service: Any):
        self.market_data_service = market_data_service

    def get_supported_symbols_data(self) -> Dict[str, Any]:
        """Get formatted data for supported symbols."""
        try:
            return {
                "symbols": SUPPORTED_COINS,
                "count": len(SUPPORTED_COINS),
                "timestamp": int(time.time()),
            }
        except Exception as e:
            logger.error(f"Error getting supported symbols: {str(e)}")
            raise

    def get_enhanced_live_data(self) -> Dict[str, Any]:
        """Get enhanced live data for all supported coins."""
        try:
            data = self.market_data_service.get_all_latest_enhanced()
            return {
                "data": data,
                "count": len(data),
                "timestamp": int(time.time()),
            }
        except Exception as e:
            logger.error(f"Error getting enhanced live data: {str(e)}")
            raise

    def get_all_live_data(self) -> Dict[str, Any]:
        """Get all live data for supported coins."""
        try:
            # Get all supported symbols
            all_coins: List[str] = []
            for _, coins in self.market_data_service.coin_config.items():
                all_coins.extend(coins)
            supported_symbols = sorted(set(all_coins))

            results: Dict[str, Any] = {
                "symbols": {},
                "total_symbols": len(supported_symbols),
                "timestamp": int(time.time()),
            }

            # Get data for each symbol
            for symbol in supported_symbols:
                try:
                    cached_data = self.market_data_service.enhanced_cache["data"].get(symbol)

                    if cached_data:
                        results["symbols"][symbol] = {
                            "price": cached_data.get("price"),
                            "change_24h": cached_data.get("change_24h"),
                            "volume": cached_data.get("volume"),
                            "api_source": cached_data.get("api_source"),
                            "timestamp": cached_data.get("timestamp"),
                            "last_update": cached_data.get("last_update"),
                        }
                    else:
                        results["symbols"][symbol] = {
                            "error": "No data available",
                            "timestamp": int(time.time()),
                        }
                except Exception as e:
                    logger.error(f"Error processing symbol {symbol}: {str(e)}")
                    results["symbols"][symbol] = {
                        "error": f"Processing error: {str(e)}",
                        "timestamp": int(time.time()),
                    }

            return results

        except Exception as e:
            logger.error(f"Error getting all live data: {str(e)}")
            raise

    def get_live_data_for_symbol(self, symbol: str) -> Dict[str, Any]:
        """Get live data for a specific symbol."""
        try:
            # Normalize symbol
            symbol = symbol.upper()

            # Check if symbol is supported
            if not any(symbol in coins for coins in self.market_data_service.coin_config.values()):
                raise ValueError(f"Symbol {symbol} not supported")

            # Get cached data
            cached_data = self.market_data_service.enhanced_cache["data"].get(symbol)

            if not cached_data:
                # Try to fetch fresh data using public methods
                try:
                    # Use the public fetch method instead of protected method
                    live_data = self.market_data_service.fetch_market_price(f"{symbol}-USD")
                    if live_data:
                        cached_data = {
                            "price": live_data.price,
                            "change_24h": live_data.change_24h,
                            "volume": getattr(live_data, "volume", None),
                            "api_source": self.market_data_service.current_api,
                            "timestamp": live_data.timestamp,
                            "last_update": int(time.time()),
                        }
                    else:
                        raise ValueError(f"No data available for {symbol}")
                except Exception as fetch_error:
                    logger.error(f"Error fetching data for {symbol}: {str(fetch_error)}")
                    raise ValueError(f"Failed to fetch data for {symbol}")

            return {
                "symbol": symbol,
                "price": cached_data.get("price"),
                "change_24h": cached_data.get("change_24h"),
                "volume": cached_data.get("volume"),
                "api_source": cached_data.get("api_source"),
                "timestamp": cached_data.get("timestamp"),
                "last_update": cached_data.get("last_update"),
            }

        except Exception as e:
            logger.error(f"Error getting live data for {symbol}: {str(e)}")
            raise

    def get_supported_coins_data(self) -> Dict[str, Any]:
        """Get formatted data for supported coins."""
        try:
            return {
                "coins": SUPPORTED_COINS,
                "count": len(SUPPORTED_COINS),
                "timestamp": int(time.time()),
            }
        except Exception as e:
            logger.error(f"Error getting supported coins: {str(e)}")
            raise

    def get_coin_support_info(self) -> Dict[str, Any]:
        """Get detailed coin support information."""
        try:
            support_info = get_coin_support_summary()
            return {
                "support_info": support_info,
                "timestamp": int(time.time()),
            }
        except Exception as e:
            logger.error(f"Error getting coin support info: {str(e)}")
            raise

    def get_coin_summary(self) -> Dict[str, Any]:
        """Get summary of coin data."""
        try:
            # Get all supported symbols
            all_coins: List[str] = []
            for _, coins in self.market_data_service.coin_config.items():
                all_coins.extend(coins)
            supported_symbols = sorted(set(all_coins))

            summary = {
                "total_coins": len(supported_symbols),
                "apis": list(self.market_data_service.coin_config.keys()),
                "cache_stats": {
                    "cached_symbols": len(self.market_data_service.enhanced_cache["data"]),
                    "cache_hits": dict(self.market_data_service.enhanced_cache["cache_hits"]),
                    "cache_misses": dict(self.market_data_service.enhanced_cache["cache_misses"]),
                },
                "timestamp": int(time.time()),
            }

            return summary

        except Exception as e:
            logger.error(f"Error getting coin summary: {str(e)}")
            raise

    def validate_symbol(self, symbol: str) -> bool:
        """Validate if a symbol is supported."""
        symbol = symbol.upper()
        return any(symbol in coins for coins in self.market_data_service.coin_config.values())

    def format_error_response(self, error: str, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Format error response for API endpoints."""
        response = {"error": error, "timestamp": int(time.time())}
        if symbol:
            response["symbol"] = symbol
        return response


# Global instance (will be initialized with market_data_service)
live_data_manager = None


def get_live_data_manager(market_data_service: Any) -> LiveDataManager:
    """Get or create live data manager instance."""
    global live_data_manager
    if live_data_manager is None:
        live_data_manager = LiveDataManager(market_data_service)
    return live_data_manager


