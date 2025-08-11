"""
Multiversal Liquidity Engine for Mystic AI Trading Platform
Analyzes cross-exchange liquidity and identifies arbitrage opportunities.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone
import sys
import os

# Add backend to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from modules.ai.persistent_cache import PersistentCache

logger = logging.getLogger(__name__)

# Conditional CCXT import
try:
    import ccxt
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False
    logger.warning("CCXT not available. Install with: pip install ccxt")


class MultiversalLiquidityEngine:
    def __init__(self):
        """Initialize multiversal liquidity engine with exchange configurations"""
        self.cache = PersistentCache()

        # Exchange configurations
        self.exchanges = {
            'binanceus': {
                'name': 'Binance US',
                'api_key': os.getenv('BINANCE_API_KEY'),
                'secret': os.getenv('BINANCE_SECRET'),
                'sandbox': True
            },
            'coinbase': {
                'name': 'Coinbase US',
                'api_key': os.getenv('COINBASE_API_KEY'),
                'secret': os.getenv('COINBASE_SECRET'),
                'sandbox': True
            },
            'kraken': {
                'name': 'Kraken US',
                'api_key': os.getenv('KRAKEN_API_KEY'),
                'secret': os.getenv('KRAKEN_SECRET'),
                'sandbox': True
            }
        }

        # Trading parameters
        self.min_arbitrage_spread = 0.003  # 0.3%
        self.max_slippage = 0.001  # 0.1%
        self.min_volume_usd = 100.0  # Minimum trade size
        self.max_volume_usd = 10000.0  # Maximum trade size

        # Top symbols to monitor
        self.top_symbols = [
            'BTC-USD', 'ETH-USD', 'ADA-USD', 'SOL-USD', 'DOT-USD',
            'LINK-USD', 'MATIC-USD', 'AVAX-USD', 'UNI-USD', 'ATOM-USD'
        ]

        # Exchange instances
        self.exchange_instances = {}
        self._init_exchanges()

        logger.info("✅ MultiversalLiquidityEngine initialized")

    def _init_exchanges(self):
        """Initialize exchange instances"""
        try:
            if not CCXT_AVAILABLE:
                logger.warning("CCXT not available - using mock data")
                return

            for exchange_id, config in self.exchanges.items():
                try:
                    # Create exchange instance
                    exchange_class = getattr(ccxt, exchange_id)
                    exchange = exchange_class({
                        'apiKey': config['api_key'],
                        'secret': config['secret'],
                        'sandbox': config['sandbox'],
                        'enableRateLimit': True
                    })

                    self.exchange_instances[exchange_id] = exchange
                    logger.info(f"✅ Initialized {config['name']}")

                except Exception as e:
                    logger.error(f"Failed to initialize {config['name']}: {e}")

        except Exception as e:
            logger.error(f"Failed to initialize exchanges: {e}")

    def _get_mock_orderbook(self, symbol: str, exchange: str) -> Dict[str, Any]:
        """Generate mock orderbook data for testing"""
        try:
            import random
            import time

            # Base price for different symbols
            base_prices = {
                'BTC-USD': 45000,
                'ETH-USD': 3000,
                'ADA-USD': 0.5,
                'SOL-USD': 100,
                'DOT-USD': 7,
                'LINK-USD': 15,
                'MATIC-USD': 0.8,
                'AVAX-USD': 25,
                'UNI-USD': 6,
                'ATOM-USD': 10
            }

            base_price = base_prices.get(symbol, 100)
            timestamp = int(time.time() * 1000)

            # Add some randomness and spread
            spread = random.uniform(0.001, 0.005)
            mid_price = base_price * (1 + random.uniform(-0.1, 0.1))
            bid_price = mid_price * (1 - spread / 2)
            ask_price = mid_price * (1 + spread / 2)

            # Generate order book levels
            bids = []
            asks = []

            for i in range(10):
                bid_level = {
                    'price': bid_price * (1 - i * 0.001),
                    'amount': random.uniform(0.1, 10.0)
                }
                bids.append(bid_level)

                ask_level = {
                    'price': ask_price * (1 + i * 0.001),
                    'amount': random.uniform(0.1, 10.0)
                }
                asks.append(ask_level)

            return {
                'symbol': symbol,
                'bids': bids,
                'asks': asks,
                'timestamp': timestamp,
                'exchange': exchange
            }

        except Exception as e:
            logger.error(f"Failed to generate mock orderbook: {e}")
            return {}

    async def _fetch_orderbook(self, exchange_id: str, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch order book from exchange"""
        try:
            if not CCXT_AVAILABLE:
                # Return mock data
                return self._get_mock_orderbook(symbol, exchange_id)

            exchange = self.exchange_instances.get(exchange_id)
            if not exchange:
                logger.warning(f"Exchange {exchange_id} not initialized")
                return None

            # Fetch order book
            orderbook = await exchange.fetch_order_book(symbol)
            return orderbook

        except Exception as e:
            logger.error(f"Failed to fetch orderbook for {symbol} on {exchange_id}: {e}")
            return None

    def _calculate_best_prices(self, orderbook: Dict[str, Any]) -> Tuple[float, float]:
        """Calculate best bid and ask prices from orderbook"""
        try:
            bids = orderbook.get('bids', [])
            asks = orderbook.get('asks', [])

            if not bids or not asks:
                return 0.0, 0.0

            # Best bid is highest price willing to buy
            best_bid = float(bids[0][0]) if bids else 0.0

            # Best ask is lowest price willing to sell
            best_ask = float(asks[0][0]) if asks else 0.0

            return best_bid, best_ask

        except Exception as e:
            logger.error(f"Failed to calculate best prices: {e}")
            return 0.0, 0.0

    def _calculate_volume_capacity(self, orderbook: Dict[str, Any], target_volume_usd: float) -> Dict[str, float]:
        """Calculate maximum volume that can be traded at current prices"""
        try:
            bids = orderbook.get('bids', [])
            asks = orderbook.get('asks', [])

            if not bids or not asks:
                return {"capacity": 0.0, "bid_price": 0.0, "ask_price": 0.0}

            # Calculate volume capacity for bids (buying)
            bid_volume = 0.0
            bid_price = 0.0
            total_bid_value = 0.0
            for bid in bids:
                price = float(bid[0])
                volume = float(bid[1])
                volume_usd = price * volume

                if bid_volume + volume_usd <= target_volume_usd:
                    bid_volume += volume_usd
                    total_bid_value += price * volume
                    bid_price = total_bid_value / bid_volume if bid_volume > 0 else price
                else:
                    remaining_volume = target_volume_usd - bid_volume
                    bid_volume += remaining_volume
                    total_bid_value += price * remaining_volume
                    bid_price = total_bid_value / bid_volume if bid_volume > 0 else price
                    break

            # Calculate volume capacity for asks (selling)
            ask_volume = 0.0
            ask_price = 0.0
            total_ask_value = 0.0
            for ask in asks:
                price = float(ask[0])
                volume = float(ask[1])
                volume_usd = price * volume

                if ask_volume + volume_usd <= target_volume_usd:
                    ask_volume += volume_usd
                    total_ask_value += price * volume
                    ask_price = total_ask_value / ask_volume if ask_volume > 0 else price
                else:
                    remaining_volume = target_volume_usd - ask_volume
                    ask_volume += remaining_volume
                    total_ask_value += price * remaining_volume
                    ask_price = total_ask_value / ask_volume if ask_volume > 0 else price
                    break

            # Calculate capacity and use the prices
            capacity = min(bid_volume, ask_volume)
            
            # Use the calculated prices in the return value
            return {
                "capacity": capacity,
                "bid_price": bid_price,
                "ask_price": ask_price
            }

        except Exception as e:
            logger.error(f"Failed to calculate volume capacity: {e}")
            return {"capacity": 0.0, "bid_price": 0.0, "ask_price": 0.0}

    def _detect_arbitrage_opportunities(self, orderbooks: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect arbitrage opportunities across exchanges"""
        try:
            opportunities = []

            # Group orderbooks by symbol
            symbols = set()
            for exchange_id, exchange_orderbooks in orderbooks.items():
                symbols.update(exchange_orderbooks.keys())

            for symbol in symbols:
                # Collect orderbooks for this symbol across exchanges
                symbol_orderbooks = {}
                for exchange_id, exchange_orderbooks in orderbooks.items():
                    if symbol in exchange_orderbooks:
                        symbol_orderbooks[exchange_id] = exchange_orderbooks[symbol]

                if len(symbol_orderbooks) < 2:
                    continue

                # Calculate best prices for each exchange
                exchange_prices = {}
                for exchange_id, orderbook in symbol_orderbooks.items():
                    best_bid, best_ask = self._calculate_best_prices(orderbook)
                    if best_bid > 0 and best_ask > 0:
                        exchange_prices[exchange_id] = {
                            'bid': best_bid,
                            'ask': best_ask,
                            'orderbook': orderbook
                        }

                # Find arbitrage opportunities
                exchanges = list(exchange_prices.keys())
                for i, buy_exchange in enumerate(exchanges):
                    for sell_exchange in exchanges[i+1:]:
                        buy_prices = exchange_prices[buy_exchange]
                        sell_prices = exchange_prices[sell_exchange]

                        # Check buy low, sell high opportunity
                        if buy_prices['ask'] < sell_prices['bid']:
                            spread = (sell_prices['bid'] - buy_prices['ask']) / buy_prices['ask']

                            if spread >= self.min_arbitrage_spread:
                                # Calculate volume capacity
                                buy_capacity_data = self._calculate_volume_capacity(
                                    buy_prices['orderbook'], self.max_volume_usd
                                )
                                sell_capacity_data = self._calculate_volume_capacity(
                                    sell_prices['orderbook'], self.max_volume_usd
                                )

                                max_volume = min(buy_capacity_data["capacity"], sell_capacity_data["capacity"])

                                if max_volume >= self.min_volume_usd:
                                    opportunities.append({
                                        'symbol': symbol,
                                        'buy_exchange': self.exchanges[buy_exchange]['name'],
                                        'sell_exchange': self.exchanges[sell_exchange]['name'],
                                        'buy_price': buy_prices['ask'],
                                        'sell_price': sell_prices['bid'],
                                        'profit_pct': spread * 100,
                                        'max_volume_usd': max_volume,
                                        'timestamp': datetime.now(timezone.utc).isoformat()
                                    })

                        # Check reverse opportunity (sell low, buy high)
                        if sell_prices['ask'] < buy_prices['bid']:
                            spread = (buy_prices['bid'] - sell_prices['ask']) / sell_prices['ask']

                            if spread >= self.min_arbitrage_spread:
                                # Calculate volume capacity
                                buy_capacity_data = self._calculate_volume_capacity(
                                    buy_prices['orderbook'], self.max_volume_usd
                                )
                                sell_capacity_data = self._calculate_volume_capacity(
                                    sell_prices['orderbook'], self.max_volume_usd
                                )

                                max_volume = min(buy_capacity_data["capacity"], sell_capacity_data["capacity"])

                                if max_volume >= self.min_volume_usd:
                                    opportunities.append({
                                        'symbol': symbol,
                                        'buy_exchange': self.exchanges[sell_exchange]['name'],
                                        'sell_exchange': self.exchanges[buy_exchange]['name'],
                                        'buy_price': sell_prices['ask'],
                                        'sell_price': buy_prices['bid'],
                                        'profit_pct': spread * 100,
                                        'max_volume_usd': max_volume,
                                        'timestamp': datetime.now(timezone.utc).isoformat()
                                    })

            return opportunities

        except Exception as e:
            logger.error(f"Failed to detect arbitrage opportunities: {e}")
            return []

    async def _fetch_all_orderbooks(self) -> Dict[str, Dict[str, Any]]:
        """Fetch order books from all exchanges for all symbols"""
        try:
            all_orderbooks = {}

            for exchange_id in self.exchanges.keys():
                exchange_orderbooks = {}

                for symbol in self.top_symbols:
                    try:
                        orderbook = await self._fetch_orderbook(exchange_id, symbol)
                        if orderbook:
                            exchange_orderbooks[symbol] = orderbook

                        # Rate limiting
                        await asyncio.sleep(0.1)

                    except Exception as e:
                        logger.error(f"Failed to fetch {symbol} from {exchange_id}: {e}")

                if exchange_orderbooks:
                    all_orderbooks[exchange_id] = exchange_orderbooks

            return all_orderbooks

        except Exception as e:
            logger.error(f"Failed to fetch all orderbooks: {e}")
            return {}

    async def find_arbitrage_opportunities(self) -> List[Dict[str, Any]]:
        """Find arbitrage opportunities across all exchanges"""
        try:
            logger.info("🔍 Searching for arbitrage opportunities...")

            # Fetch all orderbooks
            orderbooks = await self._fetch_all_orderbooks()

            if not orderbooks:
                logger.warning("No orderbooks available")
                return []

            # Detect opportunities
            opportunities = self._detect_arbitrage_opportunities(orderbooks)

            # Store opportunities in cache
            if opportunities:
                for opportunity in opportunities:
                    signal_id = f"arbitrage_{opportunity['symbol']}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
                    self.cache.store_signal(
                        signal_id=signal_id,
                        symbol=opportunity['symbol'],
                        signal_type="ARBITRAGE_OPPORTUNITY",
                        confidence=min(opportunity['profit_pct'] / 10.0, 1.0),
                        strategy="multiversal_liquidity",
                        metadata=opportunity
                    )

            logger.info(f"✅ Found {len(opportunities)} arbitrage opportunities")
            return opportunities

        except Exception as e:
            logger.error(f"Failed to find arbitrage opportunities: {e}")
            return []

    def best_liquidity_path(self, symbol: str) -> Dict[str, Any]:
        """Find the best liquidity path for a specific symbol"""
        try:
            # Get recent arbitrage signals for this symbol
            signals = self.cache.get_signals_by_type("ARBITRAGE_OPPORTUNITY", limit=20)
            symbol_opportunities = [
                signal for signal in signals
                if signal.get("metadata", {}).get("symbol") == symbol
            ]

            if not symbol_opportunities:
                return {
                    "symbol": symbol,
                    "success": False,
                    "reason": "No recent arbitrage data available"
                }

            # Find the best opportunity
            best_opportunity = max(
                symbol_opportunities,
                key=lambda x: x.get("metadata", {}).get("profit_pct", 0)
            )

            metadata = best_opportunity.get("metadata", {})

            return {
                "symbol": symbol,
                "success": True,
                "best_path": {
                    "buy_exchange": metadata.get("buy_exchange"),
                    "sell_exchange": metadata.get("sell_exchange"),
                    "buy_price": metadata.get("buy_price"),
                    "sell_price": metadata.get("sell_price"),
                    "profit_pct": metadata.get("profit_pct"),
                    "max_volume_usd": metadata.get("max_volume_usd"),
                    "timestamp": metadata.get("timestamp")
                },
                "confidence": best_opportunity.get("confidence", 0.0)
            }

        except Exception as e:
            logger.error(f"Failed to find best liquidity path for {symbol}: {e}")
            return {
                "symbol": symbol,
                "success": False,
                "error": str(e)
            }

    def get_routing_map(self) -> Dict[str, Any]:
        """Get current routing map for liquidity optimization"""
        try:
            # Get recent arbitrage opportunities from cache
            recent_opportunities = self.cache.get_signals_by_type("ARBITRAGE_OPPORTUNITY", limit=50)
            
            # Build routing map
            routing_map = {
                "exchanges": {},
                "symbols": {},
                "best_routes": {},
                "liquidity_scores": {}
            }
            
            # Exchange routing info
            for exchange_id, config in self.exchanges.items():
                routing_map["exchanges"][exchange_id] = {
                    "name": config["name"],
                    "connected": exchange_id in self.exchange_instances,
                    "liquidity_score": 0.0,
                    "active_symbols": 0
                }
            
            # Symbol routing info
            for symbol in self.top_symbols:
                symbol_opportunities = [
                    opp for opp in recent_opportunities
                    if opp.get("symbol") == symbol
                ]
                
                if symbol_opportunities:
                    best_opportunity = max(
                        symbol_opportunities,
                        key=lambda x: x.get("metadata", {}).get("profit_pct", 0)
                    )
                    
                    metadata = best_opportunity.get("metadata", {})
                    routing_map["symbols"][symbol] = {
                        "best_buy_exchange": metadata.get("buy_exchange"),
                        "best_sell_exchange": metadata.get("sell_exchange"),
                        "profit_pct": metadata.get("profit_pct", 0),
                        "volume_capacity": metadata.get("max_volume_usd", 0),
                        "last_updated": metadata.get("timestamp")
                    }
                    
                    # Update exchange scores
                    buy_exchange = metadata.get("buy_exchange")
                    sell_exchange = metadata.get("sell_exchange")
                    
                    if buy_exchange and buy_exchange in routing_map["exchanges"]:
                        routing_map["exchanges"][buy_exchange]["liquidity_score"] += 1
                        routing_map["exchanges"][buy_exchange]["active_symbols"] += 1
                    
                    if sell_exchange and sell_exchange in routing_map["exchanges"]:
                        routing_map["exchanges"][sell_exchange]["liquidity_score"] += 1
                        routing_map["exchanges"][sell_exchange]["active_symbols"] += 1
            
            return routing_map
            
        except Exception as e:
            logger.error(f"Failed to get routing map: {e}")
            return {
                "exchanges": {},
                "symbols": {},
                "best_routes": {},
                "liquidity_scores": {}
            }

    def get_liquidity_status(self) -> Dict[str, Any]:
        """Get current liquidity engine status"""
        try:
            return {
                "service": "MultiversalLiquidityEngine",
                "status": "active",
                "exchanges_connected": len(self.exchange_instances),
                "ccxt_available": CCXT_AVAILABLE,
                "monitored_symbols": self.top_symbols,
                "parameters": {
                    "min_arbitrage_spread": self.min_arbitrage_spread,
                    "max_slippage": self.max_slippage,
                    "min_volume_usd": self.min_volume_usd,
                    "max_volume_usd": self.max_volume_usd
                },
                "exchanges": {
                    exchange_id: {
                        "name": config["name"],
                        "connected": exchange_id in self.exchange_instances
                    }
                    for exchange_id, config in self.exchanges.items()
                }
            }

        except Exception as e:
            logger.error(f"Failed to get liquidity status: {e}")
            return {"success": False, "error": str(e)}


# Global multiversal liquidity engine instance
multiversal_liquidity_engine = MultiversalLiquidityEngine()


def get_multiversal_liquidity_engine() -> MultiversalLiquidityEngine:
    """Get the global multiversal liquidity engine instance"""
    return multiversal_liquidity_engine


async def main():
    """Test the multiversal liquidity engine"""
    engine = MultiversalLiquidityEngine()
    print(f"✅ MultiversalLiquidityEngine initialized: {engine}")

    # Test arbitrage opportunities
    opportunities = await engine.find_arbitrage_opportunities()
    print(f"Arbitrage opportunities: {opportunities}")

    # Test best liquidity path
    if opportunities:
        best_path = engine.best_liquidity_path(opportunities[0]['symbol'])
        print(f"Best liquidity path: {best_path}")

    # Test status
    status = engine.get_liquidity_status()
    print(f"Liquidity status: {status['status']}")
    print(f"CCXT available: {status['ccxt_available']}")


if __name__ == "__main__":
    # Run async test
    asyncio.run(main())
