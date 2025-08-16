"""
Initialize Persistent Data Cache
Populates the persistent cache with sample data for testing endpoints
"""

import asyncio
import logging

from backend.ai.persistent_cache import get_persistent_cache

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def init_persistent_sample_data():
    """Initialize persistent cache with sample market data"""
    try:
        cache = get_persistent_cache()

        # Sample Binance data
        binance_data = {
            "BTCUSDT": 45000.0,
            "ETHUSDT": 3200.0,
            "ADAUSDT": 0.45,
            "SOLUSDT": 150.0,
        }

        # Sample Coinbase data
        coinbase_data = {
            "BTC-USD": 45000.0,
            "ETH-USD": 3200.0,
            "ADA-USD": 0.45,
            "SOL-USD": 150.0,
        }

        # Sample CoinGecko data with more detailed information
        coingecko_data = {
            "bitcoin": {
                "symbol": "BTC",
                "rank": 1,
                "price": 45000.0,
                "market_cap": 850000000000,
                "volume_24h": 25000000000,
                "price_change_24h": 2.5,
            },
            "ethereum": {
                "symbol": "ETH",
                "rank": 2,
                "price": 3200.0,
                "market_cap": 380000000000,
                "volume_24h": 15000000000,
                "price_change_24h": 1.8,
            },
            "cardano": {
                "symbol": "ADA",
                "rank": 8,
                "price": 0.45,
                "market_cap": 15000000000,
                "volume_24h": 800000000,
                "price_change_24h": 5.2,
            },
            "solana": {
                "symbol": "SOL",
                "rank": 9,
                "price": 150.0,
                "market_cap": 12000000000,
                "volume_24h": 1200000000,
                "price_change_24h": 8.3,
            },
            "polkadot": {
                "symbol": "DOT",
                "rank": 12,
                "price": 25.0,
                "market_cap": 8000000000,
                "volume_24h": 500000000,
                "price_change_24h": -0.5,
            },
            "chainlink": {
                "symbol": "LINK",
                "rank": 15,
                "price": 18.0,
                "market_cap": 6000000000,
                "volume_24h": 400000000,
                "price_change_24h": 3.2,
            },
            "polygon": {
                "symbol": "MATIC",
                "rank": 14,
                "price": 0.85,
                "market_cap": 7000000000,
                "volume_24h": 600000000,
                "price_change_24h": 4.1,
            },
            "avalanche": {
                "symbol": "AVAX",
                "rank": 20,
                "price": 35.0,
                "market_cap": 4000000000,
                "volume_24h": 300000000,
                "price_change_24h": 6.7,
            },
            "cosmos": {
                "symbol": "ATOM",
                "rank": 25,
                "price": 12.0,
                "market_cap": 3000000000,
                "volume_24h": 200000000,
                "price_change_24h": 2.1,
            },
            "uniswap": {
                "symbol": "UNI",
                "rank": 22,
                "price": 8.5,
                "market_cap": 3500000000,
                "volume_24h": 250000000,
                "price_change_24h": 1.5,
            },
        }

        # Update persistent cache
        cache.update_binance(binance_data)
        cache.update_coinbase(coinbase_data)
        cache.update_coingecko(coingecko_data)

        logger.info("âœ… Persistent sample data initialized:")
        logger.info(f"   Binance: {len(binance_data)} symbols")
        logger.info(f"   Coinbase: {len(coinbase_data)} symbols")
        logger.info(f"   CoinGecko: {len(coingecko_data)} coins")
        logger.info(f"   Cache file: {cache.cache_file}")

        return True

    except Exception as e:
        logger.error(f"âŒ Error initializing persistent sample data: {e}")
        return False


if __name__ == "__main__":
    # Initialize persistent sample data
    asyncio.run(init_persistent_sample_data())
    print("Persistent sample data initialized successfully!")


