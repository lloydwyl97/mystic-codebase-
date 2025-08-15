"""
Mystic AI Trading Platform - App Factory
Creates and configures the FastAPI application with all routes and middleware.
"""

import time
import asyncio
import logging
import os
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Add modules directory to Python path
modules_path = os.path.join(os.path.dirname(__file__), '..', 'modules')
if modules_path not in sys.path:
    sys.path.insert(0, modules_path)

# Add ai directory to Python path
ai_path = os.path.join(os.path.dirname(__file__), '..', 'ai')
if ai_path not in sys.path:
    sys.path.insert(0, ai_path)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global data fetcher manager
data_fetcher_manager = None


def create_app() -> FastAPI:
    ...

"""
Mystic AI Trading Platform - App Factory
Creates and configures the FastAPI application with all routes and middleware.
"""

import time
import asyncio
import logging
import os
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Add modules directory to Python path
modules_path = os.path.join(os.path.dirname(__file__), '..', 'modules')
if modules_path not in sys.path:
    sys.path.insert(0, modules_path)

# Add ai directory to Python path
ai_path = os.path.join(os.path.dirname(__file__), '..', 'ai')
if ai_path not in sys.path:
    sys.path.insert(0, ai_path)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global data fetcher manager
data_fetcher_manager = None


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    logger.info("Starting app creation...")
    app = FastAPI(
        title="Mystic AI Trading Platform",
        description="Advanced AI-powered cryptocurrency trading platform",
        version="1.0.0",
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # In production, specify exact origins
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    logger.info("✅ FastAPI app created")

    # Initialize comprehensive data fetchers for live market data
    async def start_data_fetchers():
        global data_fetcher_manager
        try:
            logger.info("🔄 Initializing comprehensive market data fetchers...")

            # Initialize the proper DataFetcherManager
            try:
                from data_fetchers import DataFetcherManager
                from shared_cache import SharedCache

                # Create shared cache instance
                shared_cache = SharedCache()

                # Initialize the data fetcher manager
                data_fetcher_manager = DataFetcherManager(shared_cache)

                # Start all data fetchers
                await data_fetcher_manager.start_all()
                logger.info("✅ DataFetcherManager initialized and started")

            except Exception as e:
                logger.error(f"❌ Error initializing DataFetcherManager: {e}")
                data_fetcher_manager = None

            # Start a comprehensive background task to fetch market data
            async def fetch_comprehensive_market_data():
                import asyncio
                import aiohttp
                import time
                try:
                    import sys
                    import os
                    # Add ai directory to Python path
                    ai_path = os.path.join(os.path.dirname(__file__), '..', 'ai')
                    if ai_path not in sys.path:
                        sys.path.insert(0, ai_path)
                    from ai.persistent_cache import get_persistent_cache
                    cache = get_persistent_cache()
                except ImportError as e:
                    logger.warning(f"persistent_cache not available: {e}, skipping comprehensive market data")
                    return

                session = None

                try:
                    session = aiohttp.ClientSession()

                    while True:
                        try:
                            # Fetch comprehensive Binance US data
                            binance_data = {}
                            binance_symbols = [
                                "BTCUSDT",
                                "ETHUSDT",
                            ]

                            for symbol in binance_symbols:
                                try:
                                    # Get 24hr ticker data (includes price, volume, change, high, low)
                                    url = f"https://api.binance.us/api/v3/ticker/24hr?symbol={symbol}"

                                    async with session.get(url, timeout=10) as response:
                                        if response.status == 200:
                                            data = await response.json()
                                            price = float(data["lastPrice"])
                                            volume = float(data["volume"])
                                            change_24h = float(data["priceChangePercent"])
                                            high_24h = float(data["highPrice"])
                                            low_24h = float(data["lowPrice"])

                                            binance_data[symbol] = {
                                                "price": price,
                                                "volume": volume,
                                                "change_24h": change_24h,
                                                "high_24h": high_24h,
                                                "low_24h": low_24h,
                                                "bid": float(data.get("bidPrice", 0)),
                                                "ask": float(data.get("askPrice", 0)),
                                                "timestamp": time.time(),
                                            }
                                        else:
                                            logger.error(
                                                f"Error fetching Binance US data for {symbol}: HTTP {response.status}"
                                            )
                                except Exception as e:
                                    logger.error(f"Error fetching Binance US data for {symbol}: {e}")

                                # Rate limiting: 1 second delay for Binance US
                                await asyncio.sleep(1)

                            # Fetch comprehensive Coinbase data using public endpoints
                            coinbase_data = {}
                            coinbase_symbols = [
                                "BTC-USD",
                                "ETH-USD",
                            ]

                            for symbol in coinbase_symbols:
                                try:
                                    # Get data from Coinbase Exchange API (public endpoints)
                                    ticker_url = f"https://api.exchange.coinbase.com/products/{symbol}/ticker"

                                    async with session.get(ticker_url, timeout=10) as response:
                                        if response.status == 200:
                                            ticker_data = await response.json()

                                            price = float(ticker_data.get("price", 0))
                                            volume = float(ticker_data.get("size", 0))
                                            bid = float(ticker_data.get("bid", 0))
                                            ask = float(ticker_data.get("ask", 0))

                                            coinbase_data[symbol] = {
                                                "price": price,
                                                "volume": volume,
                                                "bid": bid,
                                                "ask": ask,
                                                "product_id": symbol,
                                                "base_currency": symbol.split("-")[0],
                                                "quote_currency": symbol.split("-")[1],
                                                "status": "online",
                                                "trading_enabled": True,
                                                "timestamp": time.time(),
                                            }
                                        else:
                                            logger.error(
                                                f"❌ Coinbase ticker API error for {symbol}: HTTP {response.status}"
                                            )
                                except Exception as e:
                                    logger.error(f"❌ Error fetching Coinbase data for {symbol}: {e}")

                                # Rate limiting: 3 second delay for Coinbase (respectful to 3 req/sec limit)
                                await asyncio.sleep(3)

                            # Fetch comprehensive CoinGecko data (includes market cap, rank, etc.)
                            coingecko_data = {}
                            coingecko_ids = [
                                "bitcoin",
                                "ethereum",
                                "solana",
                                "cardano",
                                "dogecoin",
                            ]

                            try:
                                # Use simple/price endpoint for better rate limiting
                                ids_param = ",".join(coingecko_ids)
                                url = f"https://api.coingecko.com/api/v3/simple/price?ids={ids_param}&vs_currencies=usd&include_market_cap=true&include_24hr_vol=true&include_24hr_change=true"

                                async with session.get(url, timeout=10) as response:
                                    if response.status == 200:
                                        data = await response.json()

                                        for coin_id, coin_data in data.items():
                                            if coin_id in coingecko_ids:
                                                coingecko_data[coin_id] = {
                                                    "symbol": coin_id.upper(),
                                                    "name": coin_id.title(),
                                                    "rank": 0,  # Not available in simple endpoint
                                                    "price": coin_data.get("usd", 0),
                                                    "market_cap": coin_data.get("usd_market_cap", 0),
                                                    "volume_24h": coin_data.get("usd_24h_vol", 0),
                                                    "price_change_24h": coin_data.get("usd_24h_change", 0),
                                                    "high_24h": 0,  # Not available in simple endpoint
                                                    "low_24h": 0,  # Not available in simple endpoint
                                                    "timestamp": time.time(),
                                                }
                            except Exception as e:
                                logger.error(f"Error fetching CoinGecko data: {e}")

                            # Rate limiting: 5 second delay for CoinGecko (respectful to 50 calls/min limit)
                            await asyncio.sleep(5)

                            # Update persistent cache with comprehensive data
                            if binance_data:
                                cache.update_binance(binance_data)
                                logger.info(
                                    f"✅ Updated Binance US data: {len(binance_data)} symbols with comprehensive data"
                                )

                            if coinbase_data:
                                cache.update_coinbase(coinbase_data)
                                logger.info(
                                    f"✅ Updated Coinbase data: {len(coinbase_data)} symbols with comprehensive data"
                                )

                            if coingecko_data:
                                cache.update_coingecko(coingecko_data)
                                logger.info(
                                    f"✅ Updated CoinGecko data: {len(coingecko_data)} coins with market cap, rank, and comprehensive data"
                                )

                            # Wait 120 seconds before next fetch (very respectful to APIs)
                            await asyncio.sleep(120)

                        except Exception as e:
                            logger.error(f"Error in comprehensive market data fetch loop: {e}")
                            await asyncio.sleep(60)  # Wait longer on error

                except Exception as e:
                    logger.error(f"Error starting comprehensive market data fetcher: {e}")
                finally:
                    if session:
                        await session.close()

            # Start the comprehensive market data fetcher as a background task
            asyncio.create_task(fetch_comprehensive_market_data())
            logger.info("✅ Comprehensive market data fetcher started")

        except Exception as e:
            logger.error(f"❌ Error starting comprehensive data fetchers: {e}")

    # Start data fetchers on app startup
    @app.on_event("startup")
    async def startup_event():
        await start_data_fetchers()

    # Include consolidated router - replaces all individual router loading
    try:
        logger.info("🔄 Loading consolidated endpoints...")
        from endpoints.consolidated_router import router as consolidated_router

        app.include_router(consolidated_router, prefix="/api")
        logger.info(
            f"✅ Included consolidated router with {len(consolidated_router.routes)} routes"
        )

        # Also include Crypto Autoengine router directly to expose single-prefix /api routes
        try:
            from crypto_autoengine_api import router as crypto_router

            app.include_router(crypto_router)
            logger.info("✅ Included Crypto Autoengine router (single /api prefix)")
        except Exception as e2:
            logger.error(f"❌ Error loading Crypto Autoengine router: {e2}")

        # Add UI routes to main router
        try:
            from routes.ui import router as ui_router

            app.include_router(ui_router)
            logger.info("✅ Included UI router")
        except Exception as e2:
            logger.error(f"❌ Error loading UI router: {e2}")

        # Register AI explain router
        try:
            from endpoints.ai_explain import router as ai_explain_router

            app.include_router(ai_explain_router)
            logger.info("✅ Included AI explain router")
        except Exception as e2:
            logger.error(f"❌ Error loading AI explain router: {e2}")

        try:
            from endpoints.dashboard_missing.dashboard_missing_endpoints import router as dashboard_missing_router

            app.include_router(dashboard_missing_router)
            logger.info("✅ Included dashboard missing endpoints router")
        except Exception as e2:
            logger.error(f"❌ Error loading dashboard missing endpoints router: {e2}")

        try:
            from routes.ai_dashboard import router as ai_dashboard_router

            app.include_router(ai_dashboard_router)
            logger.info("✅ Included AI dashboard router")
        except Exception as e2:
            logger.error(f"❌ Error loading AI dashboard router: {e2}")

        # Register AI explain router
        try:
            from endpoints.ai_explain import router as ai_explain_router

            app.include_router(ai_explain_router)
            logger.info("✅ Included AI explain router (fallback)")
        except Exception as e2:
            logger.error(f"❌ Error loading AI explain router (fallback): {e2}")

        try:
            from routes.websocket import router as websocket_router

            app.include_router(websocket_router)
            logger.info("✅ Included WebSocket router")
        except Exception as e2:
            logger.error(f"❌ Error loading WebSocket router: {e2}")

    except Exception as e:
        logger.error(f"❌ Error loading consolidated endpoints: {e}")

        # Fallback to individual routers if consolidated router fails
        logger.info("🔄 Falling back to individual routers...")
        try:
            from routes.dashboard import router as dashboard_router

            app.include_router(dashboard_router, prefix="/api")
            logger.info("✅ Included dashboard router (fallback)")
        except Exception as e2:
            logger.error(f"❌ Error loading dashboard router (fallback): {e2}")

        try:
            from api_endpoints import router as main_api_router

            app.include_router(main_api_router, prefix="/api")
            logger.info("✅ Included main API router (fallback)")
        except Exception as e2:
            logger.error(f"❌ Error loading main API router (fallback): {e2}")

        # Add missing UI routes
        try:
            from routes.ui import router as ui_router

            app.include_router(ui_router)
            logger.info("✅ Included UI router")
        except Exception as e2:
            logger.error(f"❌ Error loading UI router: {e2}")

        try:
            from endpoints.dashboard_missing.dashboard_missing_endpoints import router as dashboard_missing_router

            app.include_router(dashboard_missing_router)
            logger.info("✅ Included dashboard missing endpoints router")
        except Exception as e2:
            logger.error(f"❌ Error loading dashboard missing endpoints router: {e2}")

        try:
            from routes.ai_dashboard import router as ai_dashboard_router

            app.include_router(ai_dashboard_router)
            logger.info("✅ Included AI dashboard router")
        except Exception as e2:
            logger.error(f"❌ Error loading AI dashboard router: {e2}")

    # Health check endpoint
    @app.get("/health")
    async def health_check():
        """Health check endpoint"""
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "service": "mystic-backend",
            "version": "1.0.0",
        }

    # Root endpoint
    @app.get("/")
    async def root():
        """Root endpoint"""
        return {
            "message": "Mystic AI Trading Platform API",
            "version": "1.0.0",
            "status": "running",
            "timestamp": time.time(),
        }

    logger.info("✅ App factory completed successfully")
    return app
