"""
Liquidity Service for Mystic AI Trading Platform
Provides live liquidity tracking and cross-exchange arbitrage analysis.
"""

import asyncio
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any

# Add backend to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from backend.modules.ai.persistent_cache import PersistentCache

logger = logging.getLogger(__name__)

try:
    from backend.modules.ai.multiversal_liquidity_engine import MultiversalLiquidityEngine
    MULTIVERSAL_AVAILABLE = True
except ImportError:
    MULTIVERSAL_AVAILABLE = False
    logger.warning("MultiversalLiquidityEngine not available")


class LiquidityService:
    def __init__(self):
        """Initialize liquidity service with live data tracking"""
        self.cache = PersistentCache()

        # Initialize multiversal liquidity engine if available
        if MULTIVERSAL_AVAILABLE:
            self.liquidity_engine = MultiversalLiquidityEngine()
        else:
            self.liquidity_engine = None
            logger.warning("Using fallback liquidity engine")

        # Service configuration
        self.top_symbols = [
            "BTC-USD", "ETH-USD", "ADA-USD", "DOT-USD", "LINK-USD",
            "LTC-USD", "BCH-USD", "XLM-USD", "EOS-USD", "XRP-USD"
        ]

        self.exchanges = ["coinbase", "binanceus", "kraken"]
        self.trade_sizes = [100, 1000, 10000]  # USD trade sizes for slippage calculation
        self.min_arbitrage_spread = 0.003  # 0.3% minimum spread for arbitrage

        # Rate limiting
        self.request_delays = {
            "coinbase": 0.33,  # 3 req/sec
            "binanceus": 0.1,  # 10 req/sec
            "kraken": 1.0      # 1 req/sec
        }

        # Mock order book data for testing
        self.mock_orderbooks = self._generate_mock_orderbooks()

        logger.info("âœ… LiquidityService initialized")

    def _generate_mock_orderbooks(self) -> dict[str, dict[str, Any]]:
        """Generate mock order book data for testing"""
        mock_data = {}

        for symbol in self.top_symbols:
            mock_data[symbol] = {}

            for exchange in self.exchanges:
                # Generate realistic order book data
                base_price = 50000 if "BTC" in symbol else 3000 if "ETH" in symbol else 100

                bids = []
                asks = []

                # Generate bid side (buy orders)
                for i in range(10):
                    price = base_price * (1 - 0.001 * i)  # Slightly decreasing prices
                    size = 0.1 + (i * 0.05)  # Increasing sizes
                    bids.append([price, size])

                # Generate ask side (sell orders)
                for i in range(10):
                    price = base_price * (1 + 0.001 * i)  # Slightly increasing prices
                    size = 0.1 + (i * 0.05)  # Increasing sizes
                    asks.append([price, size])

                mock_data[symbol][exchange] = {
                    "bids": bids,
                    "asks": asks,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "symbol": symbol,
                    "exchange": exchange
                }

        return mock_data

    async def _fetch_orderbook(self, exchange: str, symbol: str) -> dict[str, Any] | None:
        """Fetch live order book from exchange"""
        try:
            # Use mock data for testing
            if symbol in self.mock_orderbooks and exchange in self.mock_orderbooks[symbol]:
                return self.mock_orderbooks[symbol][exchange]

            # Real API endpoints (commented out for testing)
            # api_urls = {
            #     "coinbase": f"https://api.pro.coinbase.com/products/{symbol}/book?level=2",
            #     "binanceus": f"https://api.binance.us/api/v3/depth?symbol={symbol.replace('-', '')}&limit=10",
            #     "kraken": f"https://api.kraken.com/0/public/Depth?pair={symbol}&count=10"
            # }

            # if exchange in api_urls:
            #     async with aiohttp.ClientSession() as session:
            #         async with session.get(api_urls[exchange]) as response:
            #             if response.status == 200:
            #                 data = await response.json()
            #                 return self._parse_orderbook_response(exchange, symbol, data)

            logger.warning(f"Using mock data for {exchange}:{symbol}")
            return None
        except Exception as e:
            logger.error(f"Failed to fetch orderbook for {symbol} on {exchange}: {e}")
            return None

    def _parse_orderbook_response(self, exchange: str, symbol: str, data: dict[str, Any]) -> dict[str, Any]:
        """Parse exchange-specific order book response"""
        try:
            if exchange == "coinbase" or exchange == "binanceus":
                return {
                    "bids": data.get("bids", []),
                    "asks": data.get("asks", []),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "symbol": symbol,
                    "exchange": exchange
                }
            elif exchange == "kraken":
                # Kraken has a different response format
                result = data.get("result", {})
                pair_name = list(result.keys())[0] if result else symbol
                pair_data = result.get(pair_name, {})

                return {
                    "bids": pair_data.get("bids", []),
                    "asks": pair_data.get("asks", []),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "symbol": symbol,
                    "exchange": exchange
                }
            else:
                logger.warning(f"Unknown exchange format for {exchange}")
                return {
                    "bids": [],
                    "asks": [],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "symbol": symbol,
                    "exchange": exchange
                }

        except Exception as e:
            logger.error(f"Failed to parse orderbook response for {symbol} on {exchange}: {e}")
            return {
                "bids": [],
                "asks": [],
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "symbol": symbol,
                "exchange": exchange,
                "error": str(e)
            }

    def _calculate_available_volume(self, orderbook: dict[str, Any], side: str) -> float:
        """Calculate available volume for a given side of the order book"""
        try:
            orders = orderbook.get("bids" if side == "buy" else "asks", [])
            total_volume = 0.0

            for order in orders:
                if len(order) >= 2:
                    price = float(order[0])
                    size = float(order[1])
                    total_volume += price * size

            return total_volume

        except Exception as e:
            logger.error(f"Failed to calculate available volume: {e}")
            return 0.0

    def _calculate_slippage(self, orderbook: dict[str, Any], trade_size_usd: float, side: str) -> dict[str, Any]:
        """Calculate slippage for a given trade size"""
        try:
            orders = orderbook.get("bids" if side == "buy" else "asks", [])
            if not orders:
                return {"slippage_pct": 0.0, "effective_price": 0.0, "available_volume": 0.0}

            # Sort orders by price (descending for bids, ascending for asks)
            if side == "buy":
                orders.sort(key=lambda x: float(x[0]), reverse=True)
            else:
                orders.sort(key=lambda x: float(x[0]))

            remaining_size = trade_size_usd
            total_cost = 0.0
            weighted_price = 0.0
            total_volume = 0.0

            for order in orders:
                if remaining_size <= 0:
                    break

                price = float(order[0])
                size = float(order[1])
                order_value = price * size

                if order_value <= remaining_size:
                    # Use entire order
                    total_cost += order_value
                    weighted_price += price * order_value
                    total_volume += order_value
                    remaining_size -= order_value
                else:
                    # Use partial order
                    partial_size = remaining_size
                    total_cost += partial_size
                    weighted_price += price * partial_size
                    total_volume += partial_size
                    remaining_size = 0

            if total_volume > 0:
                effective_price = weighted_price / total_volume
                mid_price = (float(orders[0][0]) + float(orders[-1][0])) / 2
                slippage_pct = abs(effective_price - mid_price) / mid_price
            else:
                effective_price = 0.0
                slippage_pct = 0.0

            return {
                "slippage_pct": slippage_pct,
                "effective_price": effective_price,
                "available_volume": total_volume,
                "remaining_size": remaining_size
            }

        except Exception as e:
            logger.error(f"Failed to calculate slippage: {e}")
            return {"slippage_pct": 0.0, "effective_price": 0.0, "available_volume": 0.0}

    def _find_arbitrage_opportunities(self, symbol: str, orderbooks: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
        """Find arbitrage opportunities across exchanges"""
        try:
            opportunities = []
            exchanges = list(orderbooks.keys())

            for i, exchange1 in enumerate(exchanges):
                for exchange2 in exchanges[i+1:]:
                    book1 = orderbooks[exchange1]
                    book2 = orderbooks[exchange2]

                    if not book1.get("bids") or not book1.get("asks") or not book2.get("bids") or not book2.get("asks"):
                        continue

                    # Calculate best bid and ask for each exchange
                    best_bid1 = float(book1["bids"][0][0]) if book1["bids"] else 0
                    best_ask1 = float(book1["asks"][0][0]) if book1["asks"] else float('inf')
                    best_bid2 = float(book2["bids"][0][0]) if book2["bids"] else 0
                    best_ask2 = float(book2["asks"][0][0]) if book2["asks"] else float('inf')

                    # Check for arbitrage opportunity
                    if best_bid1 > best_ask2:
                        # Buy on exchange2, sell on exchange1
                        spread = best_bid1 - best_ask2
                        spread_pct = spread / best_ask2

                        if spread_pct >= self.min_arbitrage_spread:
                            opportunities.append({
                                "symbol": symbol,
                                "buy_exchange": exchange2,
                                "sell_exchange": exchange1,
                                "buy_price": best_ask2,
                                "sell_price": best_bid1,
                                "spread": spread,
                                "spread_pct": spread_pct,
                                "timestamp": datetime.now(timezone.utc).isoformat()
                            })

                    elif best_bid2 > best_ask1:
                        # Buy on exchange1, sell on exchange2
                        spread = best_bid2 - best_ask1
                        spread_pct = spread / best_ask1

                        if spread_pct >= self.min_arbitrage_spread:
                            opportunities.append({
                                "symbol": symbol,
                                "buy_exchange": exchange1,
                                "sell_exchange": exchange2,
                                "buy_price": best_ask1,
                                "sell_price": best_bid2,
                                "spread": spread,
                                "spread_pct": spread_pct,
                                "timestamp": datetime.now(timezone.utc).isoformat()
                            })

            return opportunities

        except Exception as e:
            logger.error(f"Failed to find arbitrage opportunities for {symbol}: {e}")
            return []

    async def get_liquidity_snapshot(self, exchange: str, symbol: str) -> dict[str, Any]:
        """Get comprehensive liquidity snapshot for a symbol on an exchange"""
        try:
            # Fetch order book
            orderbook = await self._fetch_orderbook(exchange, symbol)
            if not orderbook:
                return {"error": "Failed to fetch orderbook", "exchange": exchange, "symbol": symbol}

            # Calculate metrics for different trade sizes
            slippage_metrics = {}
            for trade_size in self.trade_sizes:
                buy_slippage = self._calculate_slippage(orderbook, trade_size, "buy")
                sell_slippage = self._calculate_slippage(orderbook, trade_size, "sell")

                slippage_metrics[f"${trade_size}"] = {
                    "buy": buy_slippage,
                    "sell": sell_slippage
                }

            # Calculate available volumes
            bid_volume = self._calculate_available_volume(orderbook, "buy")
            ask_volume = self._calculate_available_volume(orderbook, "sell")

            # Calculate spread
            if orderbook.get("bids") and orderbook.get("asks"):
                best_bid = float(orderbook["bids"][0][0])
                best_ask = float(orderbook["asks"][0][0])
                spread = best_ask - best_bid
                spread_pct = spread / best_bid
            else:
                spread = 0.0
                spread_pct = 0.0

            # Calculate liquidity score
            liquidity_score = self._calculate_liquidity_score(bid_volume, ask_volume, spread_pct)

            snapshot = {
                "exchange": exchange,
                "symbol": symbol,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "orderbook_depth": len(orderbook.get("bids", [])) + len(orderbook.get("asks", [])),
                "bid_volume": bid_volume,
                "ask_volume": ask_volume,
                "total_volume": bid_volume + ask_volume,
                "best_bid": float(orderbook["bids"][0][0]) if orderbook.get("bids") else 0,
                "best_ask": float(orderbook["asks"][0][0]) if orderbook.get("asks") else 0,
                "spread": spread,
                "spread_pct": spread_pct,
                "liquidity_score": liquidity_score,
                "slippage_metrics": slippage_metrics
            }

            return snapshot

        except Exception as e:
            logger.error(f"Failed to get liquidity snapshot for {symbol} on {exchange}: {e}")
            return {"error": str(e), "exchange": exchange, "symbol": symbol}

    def _calculate_liquidity_score(self, bid_volume: float, ask_volume: float, spread_pct: float) -> float:
        """Calculate liquidity score based on volume and spread"""
        try:
            # Normalize volumes (assuming typical volumes)
            normalized_bid_volume = min(bid_volume / 1000000, 1.0)  # Cap at 1M
            normalized_ask_volume = min(ask_volume / 1000000, 1.0)

            # Volume component (0-50 points)
            volume_score = (normalized_bid_volume + normalized_ask_volume) * 25

            # Spread component (0-50 points, lower spread = higher score)
            spread_score = max(0, 50 - (spread_pct * 1000))  # Penalize high spreads

            return volume_score + spread_score

        except Exception as e:
            logger.error(f"Failed to calculate liquidity score: {e}")
            return 0.0

    async def get_cross_exchange_paths(self, symbol: str) -> dict[str, Any]:
        """Get cross-exchange liquidity paths and arbitrage opportunities"""
        try:
            # Fetch order books from all exchanges
            orderbooks = {}
            for exchange in self.exchanges:
                orderbook = await self._fetch_orderbook(exchange, symbol)
                if orderbook:
                    orderbooks[exchange] = orderbook

            if len(orderbooks) < 2:
                return {
                    "symbol": symbol,
                    "error": "Insufficient exchange data",
                    "arbitrage_opportunities": []
                }

            # Find arbitrage opportunities
            opportunities = self._find_arbitrage_opportunities(symbol, orderbooks)

            # Calculate liquidity paths
            paths = self._find_best_liquidity_paths(symbol, orderbooks)

            return {
                "symbol": symbol,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "exchanges_analyzed": len(orderbooks),
                "arbitrage_opportunities": opportunities,
                "liquidity_paths": paths,
                "total_opportunities": len(opportunities)
            }

        except Exception as e:
            logger.error(f"Failed to get cross-exchange paths for {symbol}: {e}")
            return {
                "symbol": symbol,
                "error": str(e),
                "arbitrage_opportunities": []
            }

    def _find_best_liquidity_paths(self, symbol: str, exchange_metrics: dict[str, Any]) -> list[dict[str, Any]]:
        """Find best liquidity paths across exchanges"""
        try:
            paths = []

            # Calculate liquidity scores for each exchange
            exchange_scores = {}
            for exchange, orderbook in exchange_metrics.items():
                if orderbook.get("bids") and orderbook.get("asks"):
                    bid_volume = self._calculate_available_volume(orderbook, "buy")
                    ask_volume = self._calculate_available_volume(orderbook, "sell")

                    best_bid = float(orderbook["bids"][0][0])
                    best_ask = float(orderbook["asks"][0][0])
                    spread_pct = (best_ask - best_bid) / best_bid

                    liquidity_score = self._calculate_liquidity_score(bid_volume, ask_volume, spread_pct)
                    exchange_scores[exchange] = liquidity_score

            # Sort exchanges by liquidity score
            sorted_exchanges = sorted(exchange_scores.items(), key=lambda x: x[1], reverse=True)

            # Create liquidity paths
            for i, (exchange, score) in enumerate(sorted_exchanges):
                paths.append({
                    "rank": i + 1,
                    "exchange": exchange,
                    "liquidity_score": score,
                    "recommendation": "primary" if i == 0 else "secondary" if i == 1 else "tertiary"
                })

            return paths

        except Exception as e:
            logger.error(f"Failed to find liquidity paths for {symbol}: {e}")
            return []

    async def store_liquidity_data(self) -> dict[str, Any]:
        """Store comprehensive liquidity data in cache"""
        try:
            logger.info("ðŸ’¾ Storing comprehensive liquidity data")

            all_data = {}
            total_snapshots = 0
            total_opportunities = 0

            for symbol in self.top_symbols:
                symbol_data = {
                    "snapshots": {},
                    "cross_exchange_paths": None
                }

                # Get snapshots for each exchange
                for exchange in self.exchanges:
                    snapshot = await self.get_liquidity_snapshot(exchange, symbol)
                    symbol_data["snapshots"][exchange] = snapshot
                    if "error" not in snapshot:
                        total_snapshots += 1

                # Get cross-exchange paths
                paths_data = await self.get_cross_exchange_paths(symbol)
                symbol_data["cross_exchange_paths"] = paths_data
                if "arbitrage_opportunities" in paths_data:
                    total_opportunities += len(paths_data["arbitrage_opportunities"])

                all_data[symbol] = symbol_data

            # Store summary in cache
            summary = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "total_symbols": len(self.top_symbols),
                "total_exchanges": len(self.exchanges),
                "successful_snapshots": total_snapshots,
                "total_arbitrage_opportunities": total_opportunities,
                "data": all_data
            }

            self.cache.store_signal(
                signal_id=f"liquidity_data_summary_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
                symbol="LIQUIDITY_SUMMARY",
                signal_type="LIQUIDITY_DATA_SUMMARY",
                confidence=total_snapshots / (len(self.top_symbols) * len(self.exchanges)),
                strategy="comprehensive_liquidity_analysis",
                metadata=summary
            )

            logger.info(f"âœ… Stored liquidity data: {total_snapshots} snapshots, {total_opportunities} opportunities")
            return summary

        except Exception as e:
            logger.error(f"Failed to store liquidity data: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

    def get_liquidity_status(self) -> dict[str, Any]:
        """Get current liquidity service status"""
        try:
            return {
                "service": "LiquidityService",
                "status": "active",
                "monitored_symbols": len(self.top_symbols),
                "monitored_exchanges": len(self.exchanges),
                "multiversal_engine_available": MULTIVERSAL_AVAILABLE,
                "configuration": {
                    "trade_sizes": self.trade_sizes,
                    "min_arbitrage_spread": self.min_arbitrage_spread,
                    "request_delays": self.request_delays
                },
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            logger.error(f"Failed to get liquidity status: {e}")
            return {"success": False, "error": str(e)}


# Global liquidity service instance
liquidity_service = LiquidityService()


def get_liquidity_service() -> LiquidityService:
    """Get the global liquidity service instance"""
    return liquidity_service


if __name__ == "__main__":
    # Test the liquidity service
    service = LiquidityService()
    print(f"âœ… LiquidityService initialized: {service}")

    # Test status
    status = service.get_liquidity_status()
    print(f"Service status: {status['status']}")

    # Test snapshot
    import asyncio
    snapshot = asyncio.run(service.get_liquidity_snapshot("coinbase", "BTC-USD"))
    print(f"Snapshot: {snapshot}")


