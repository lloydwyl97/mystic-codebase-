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

# Max symbols per exchange for comprehensive market data (configurable via env)
MAX_PER_EXCHANGE = int(os.getenv("COMPREHENSIVE_MAX_PER_EXCHANGE", "10"))
MAX_BINANCEUS = int(os.getenv("COMPREHENSIVE_MAX_BINANCEUS", MAX_PER_EXCHANGE))
MAX_COINBASE = int(os.getenv("COMPREHENSIVE_MAX_COINBASE", MAX_PER_EXCHANGE))

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

 

 

 
    # Initialize comprehensive data fetchers for live market data
    async def start_data_fetchers():
        global data_fetcher_manager
        try:
            logger.info("ðŸ”„ Initializing comprehensive market data fetchers...")

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
                logger.error(f"âŒ Error initializing DataFetcherManager: {e}")
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
                    from backend.modules.ai.persistent_cache import get_persistent_cache
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

                            for symbol in binance_symbols[:MAX_BINANCEUS]:
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

                            for symbol in coinbase_symbols[:MAX_COINBASE]:
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
                                                f"âŒ Coinbase ticker API error for {symbol}: HTTP {response.status}"
                                            )
                                except Exception as e:
                                    logger.error(f"âŒ Error fetching Coinbase data for {symbol}: {e}")

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
                                    f"âœ… Updated Binance US data: {len(binance_data)} symbols with comprehensive data"
                                )

                            if coinbase_data:
                                cache.update_coinbase(coinbase_data)
                                logger.info(
                                    f"âœ… Updated Coinbase data: {len(coinbase_data)} symbols with comprehensive data"
                                )

                            if coingecko_data:
                                cache.update_coingecko(coingecko_data)
                                logger.info(
                                    f"âœ… Updated CoinGecko data: {len(coingecko_data)} coins with market cap, rank, and comprehensive data"
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
            logger.error(f"âŒ Error starting comprehensive data fetchers: {e}")

    # Start data fetchers on app startup
    @app.on_event("startup")
    async def startup_event():
        await start_data_fetchers()

        # Expand symbol seeding/caching to match UI top 10 (non-blocking)
        try:
            import aiohttp
            seed_symbols = [
                "BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT","ADAUSDT","XRPUSDT",
                "DOGEUSDT","AVAXUSDT","DOTUSDT","LTCUSDT",
            ]
            async with aiohttp.ClientSession() as session:
                for sym in seed_symbols:
                    try:
                        url = f"https://api.binance.us/api/v3/ticker/24hr?symbol={sym}"
                        async with session.get(url, timeout=10) as resp:
                            _ = await resp.text()
                    except Exception as e:
                        logger.warning(f"Symbol seed fetch failed for {sym}: {e}")
        except Exception as e:
            logger.warning(f"Symbol seed loop skipped: {e}")

    # Include consolidated router - replaces all individual router loading
    try:
        logger.info("🔧 Loading consolidated endpoints...")
        from backend.endpoints.consolidated_router import router as consolidated_router

        app.include_router(consolidated_router, prefix="/api")
        logger.info(
            f"✅ Included consolidated router with {len(consolidated_router.routes)} routes"
        )

        # Also include Crypto Autoengine router under /api to keep UI paths consistent
        try:
            from crypto_autoengine_api import router as crypto_router

            app.include_router(crypto_router, prefix="/api")
            logger.info("✅ Included Crypto Autoengine router under /api")
        except Exception as e2:
            logger.error(f"âŒ Error loading Crypto Autoengine router: {e2}")

        # Include minimal compat router under /api for UI-only paths
        try:
            from backend.endpoints.compat_router import router as compat_router

            app.include_router(compat_router, prefix="/api")
            logger.info("✅ Included compatibility router under /api")
        except Exception as e2:
            logger.error(f"âŒ Error loading compatibility router: {e2}")

        # Add UI routes to main router
        try:
            from backend.routes.ui import router as ui_router

            app.include_router(ui_router, prefix="/api")
            logger.info("✅ Included UI router")
        except Exception as e2:
            logger.error(f"âŒ Error loading UI router: {e2}")

        # Register AI explain router
        try:
            from backend.endpoints.ai_explain import router as ai_explain_router

            app.include_router(ai_explain_router)
            logger.info("✅ Included AI explain router")
        except Exception as e2:
            logger.error(f"âŒ Error loading AI explain router: {e2}")

        try:
            from backend.endpoints.dashboard_missing.dashboard_missing_endpoints import router as dashboard_missing_router

            app.include_router(dashboard_missing_router)
            logger.info("✅ Included dashboard missing endpoints router")
        except Exception as e2:
            logger.error(f"âŒ Error loading dashboard missing endpoints router: {e2}")

        try:
            from backend.routes.ai_dashboard import router as ai_dashboard_router

            app.include_router(ai_dashboard_router)
            logger.info("✅ Included AI dashboard router")
        except Exception as e2:
            logger.error(f"âŒ Error loading AI dashboard router: {e2}")

        # Also include UI compatibility routers in fallback mode
        try:
            from backend.endpoints.market.candles_endpoints import router as candles_router
            app.include_router(candles_router)
            logger.info("✅ Included market candles router (fallback)")
        except Exception as e2:
            logger.error(f"âŒ Error loading candles router (fallback): {e2}")

        try:
            from backend.endpoints.portfolio.transactions_endpoints import router as transactions_router
            app.include_router(transactions_router)
            logger.info("✅ Included portfolio transactions router (fallback)")
        except Exception as e2:
            logger.error(f"âŒ Error loading transactions router (fallback): {e2}")

        

        try:
            from backend.endpoints.ai.ai_leaderboard_endpoints import router as ai_leaderboard_router
            app.include_router(ai_leaderboard_router)
            logger.info("✅ Included AI leaderboard router (fallback)")
        except Exception as e2:
            logger.error(f"âŒ Error loading AI leaderboard router (fallback): {e2}")

        try:
            from backend.endpoints.ai.ai_analytics_endpoints import router as ai_analytics_router
            app.include_router(ai_analytics_router)
            logger.info("✅ Included AI analytics router (fallback)")
        except Exception as e2:
            logger.error(f"âŒ Error loading AI analytics router (fallback): {e2}")

        try:
            from backend.endpoints.signals.whale_alerts_endpoints import router as whale_alerts_router
            app.include_router(whale_alerts_router)
            logger.info("✅ Included whale alerts router (fallback)")
        except Exception as e2:
            logger.error(f"âŒ Error loading whale alerts router (fallback): {e2}")

        try:
            from backend.endpoints.health.ai_health_endpoints import router as ai_health_router
            app.include_router(ai_health_router)
            logger.info("✅ Included AI health router (fallback)")
        except Exception as e2:
            logger.error(f"âŒ Error loading AI health router (fallback): {e2}")

        # Register AI explain router
        try:
            from backend.endpoints.ai_explain import router as ai_explain_router

            app.include_router(ai_explain_router)
            logger.info("✅ Included AI explain router (fallback)")
        except Exception as e2:
            logger.error(f"âŒ Error loading AI explain router (fallback): {e2}")

        try:
            from backend.routes.websocket import router as websocket_router

            # Note: websocket router defines an HTTP status at /api/websocket/status already
            # so we include it without extra prefix to avoid double-API duplication.
            app.include_router(websocket_router)
            logger.info("✅ Included WebSocket router")
        except Exception as e2:
            logger.error(f"âŒ Error loading WebSocket router: {e2}")

        # Register thin alias/new routers for UI compatibility
        try:
            from backend.endpoints.market.candles_endpoints import router as candles_router
            app.include_router(candles_router)
            logger.info("✅ Included market candles router")
        except Exception as e2:
            logger.error(f"âŒ Error loading candles router: {e2}")

        # legacy alias router removed

        try:
            from backend.endpoints.portfolio.transactions_endpoints import router as transactions_router
            app.include_router(transactions_router)
            logger.info("✅ Included portfolio transactions router")
        except Exception as e2:
            logger.error(f"âŒ Error loading transactions router: {e2}")

        # legacy live trading alias router removed

        try:
            from backend.endpoints.ai.ai_leaderboard_endpoints import router as ai_leaderboard_router
            app.include_router(ai_leaderboard_router)
            logger.info("✅ Included AI leaderboard router")
        except Exception as e2:
            logger.error(f"âŒ Error loading AI leaderboard router: {e2}")

        try:
            from backend.endpoints.ai.ai_analytics_endpoints import router as ai_analytics_router
            app.include_router(ai_analytics_router)
            logger.info("✅ Included AI analytics router")
        except Exception as e2:
            logger.error(f"âŒ Error loading AI analytics router: {e2}")

        try:
            from backend.endpoints.signals.whale_alerts_endpoints import router as whale_alerts_router
            app.include_router(whale_alerts_router)
            logger.info("✅ Included whale alerts router")
        except Exception as e2:
            logger.error(f"âŒ Error loading whale alerts router: {e2}")

        try:
            from backend.endpoints.health.ai_health_endpoints import router as ai_health_router
            app.include_router(ai_health_router)
            logger.info("✅ Included AI health router")
        except Exception as e2:
            logger.error(f"âŒ Error loading AI health router: {e2}")

        # legacy live notifications alias router removed

        logger.info("Routers mounted OK: market candles and consolidated bundle (if present)")

    except Exception as e:
        logger.error(f"âŒ Error loading consolidated endpoints: {e}")

        # Fallback to individual routers if consolidated router fails
        logger.info("🔧 Falling back to individual routers...")
        try:
            from backend.routes.dashboard import router as dashboard_router

            app.include_router(dashboard_router, prefix="/api")
            logger.info("✅ Included dashboard router (fallback)")
        except Exception as e2:
            logger.error(f"âŒ Error loading dashboard router (fallback): {e2}")

        try:
            from api_endpoints import router as main_api_router

            app.include_router(main_api_router, prefix="/api")
            logger.info("✅ Included main API router (fallback)")
        except Exception as e2:
            logger.error(f"âŒ Error loading main API router (fallback): {e2}")

        # Add missing UI routes
        try:
            from backend.routes.ui import router as ui_router

            app.include_router(ui_router)
            logger.info("✅ Included UI router")
        except Exception as e2:
            logger.error(f"âŒ Error loading UI router: {e2}")

        try:
            from backend.endpoints.dashboard_missing.dashboard_missing_endpoints import router as dashboard_missing_router

            app.include_router(dashboard_missing_router)
            logger.info("✅ Included dashboard missing endpoints router")
        except Exception as e2:
            logger.error(f"âŒ Error loading dashboard missing endpoints router: {e2}")

        try:
            from backend.routes.ai_dashboard import router as ai_dashboard_router

            app.include_router(ai_dashboard_router)
            logger.info("✅ Included AI dashboard router")
        except Exception as e2:
            logger.error(f"âŒ Error loading AI dashboard router: {e2}")

    

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


