"""
Live Trading Endpoints

Provides live trading endpoints using real market data and exchange APIs.
Integrates CoinGecko, Binance, and Coinbase services.
"""

import logging
import time
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Query

# Import configuration
from config import settings
from services.binance_trading import get_binance_trading_service
from services.coinbase_trading import get_coinbase_trading_service

# Import live services
from services.live_market_data import live_market_data_service

router = APIRouter()
logger = logging.getLogger(__name__)

# ============================================================================
# LIVE MARKET DATA ENDPOINTS (CoinGecko)
# ============================================================================


@router.get("/live/market-data")
async def get_live_market_data(
    currency: str = Query("usd", description="Currency for prices"),
    per_page: int = Query(100, description="Number of coins to return"),
) -> Dict[str, Any]:
    """Get live market data for top cryptocurrencies"""
    try:
        data = await live_market_data_service.get_market_data(currency, per_page)
        return {"status": "success", "data": data, "source": "coingecko"}
    except Exception as e:
        logger.error(f"Error getting live market data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting market data: {str(e)}")


@router.get("/live/market-data/{symbol}")
async def get_live_market_data_by_symbol(
    symbol: str, currency: str = Query("usd", description="Currency for price")
) -> Dict[str, Any]:
    """Get live market data for a specific symbol"""
    try:
        # Get price data for the symbol
        price_data = await live_market_data_service.get_coin_price(symbol, currency)

        # Get detailed data
        details = await live_market_data_service.get_coin_details(symbol)

        return {
            "status": "success",
            "data": {
                "symbol": symbol.upper(),
                "price": price_data,
                "details": details,
            },
            "source": "coingecko",
        }
    except Exception as e:
        logger.error(f"Error getting live market data for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting market data: {str(e)}")


@router.get("/live/coin/{coin_id}")
async def get_live_coin_data(
    coin_id: str,
    currency: str = Query("usd", description="Currency for price"),
) -> Dict[str, Any]:
    """Get live data for a specific coin"""
    try:
        # Get price data
        price_data = await live_market_data_service.get_coin_price(coin_id, currency)

        # Get detailed data
        details = await live_market_data_service.get_coin_details(coin_id)

        return {
            "status": "success",
            "data": {"price": price_data, "details": details},
            "source": "coingecko",
        }
    except Exception as e:
        logger.error(f"Error getting live coin data for {coin_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting coin data: {str(e)}")


@router.get("/live/trending")
async def get_trending_coins() -> Dict[str, Any]:
    """Get trending coins in the last 24 hours"""
    try:
        data = await live_market_data_service.get_trending_coins()
        return {"status": "success", "data": data, "source": "coingecko"}
    except Exception as e:
        logger.error(f"Error getting trending coins: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting trending coins: {str(e)}")


@router.get("/live/global")
async def get_global_market_data() -> Dict[str, Any]:
    """Get global cryptocurrency market data"""
    try:
        data = await live_market_data_service.get_global_data()
        return {"status": "success", "data": data, "source": "coingecko"}
    except Exception as e:
        logger.error(f"Error getting global data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting global data: {str(e)}")


@router.get("/live/search")
async def search_coins(
    query: str = Query(..., description="Search query for coins")
) -> Dict[str, Any]:
    """Search for coins by name or symbol"""
    try:
        data = await live_market_data_service.search_coins(query)
        return {"status": "success", "data": data, "source": "coingecko"}
    except Exception as e:
        logger.error(f"Error searching coins: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error searching coins: {str(e)}")


@router.get("/live/historical/{coin_id}")
async def get_historical_data(
    coin_id: str,
    days: int = Query(30, description="Number of days of historical data"),
    currency: str = Query("usd", description="Currency for prices"),
) -> Dict[str, Any]:
    """Get historical price data for a coin"""
    try:
        data = await live_market_data_service.get_historical_data(coin_id, days, currency)
        return {"status": "success", "data": data, "source": "coingecko"}
    except Exception as e:
        logger.error(f"Error getting historical data for {coin_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting historical data: {str(e)}")


# ============================================================================
# BINANCE TRADING ENDPOINTS
# ============================================================================


@router.get("/binance/account")
async def get_binance_account() -> Dict[str, Any]:
    """Get Binance account information and balances"""
    try:
        # Get service instance
        binance_service = get_binance_trading_service(
            api_key=settings.exchange.binance_us_api_key or "",
            secret_key=settings.exchange.binance_us_secret_key or "",
            testnet=settings.exchange.testnet,
        )

        data = await binance_service.get_account_info()
        return {"status": "success", "data": data, "exchange": "binance"}
    except Exception as e:
        logger.error(f"Error getting Binance account: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting Binance account: {str(e)}")


@router.get("/binance/price/{symbol}")
async def get_binance_price(symbol: str) -> Dict[str, Any]:
    """Get current price for a symbol on Binance"""
    try:
        binance_service = get_binance_trading_service(
            api_key=settings.exchange.binance_us_api_key or "",
            secret_key=settings.exchange.binance_us_secret_key or "",
            testnet=settings.exchange.testnet,
        )

        data = await binance_service.get_market_price(symbol)
        return {"status": "success", "data": data, "exchange": "binance"}
    except Exception as e:
        logger.error(f"Error getting Binance price for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting Binance price: {str(e)}")


@router.post("/binance/order/market")
async def place_binance_market_order(
    symbol: str = Query(..., description="Trading symbol"),
    side: str = Query(..., description="Order side (BUY/SELL)"),
    quantity: float = Query(..., description="Order quantity"),
) -> Dict[str, Any]:
    """Place a market order on Binance"""
    try:
        binance_service = get_binance_trading_service(
            api_key=settings.exchange.binance_us_api_key or "",
            secret_key=settings.exchange.binance_us_secret_key or "",
            testnet=settings.exchange.testnet,
        )

        data = await binance_service.place_market_order(symbol, side, quantity)
        return {"status": "success", "data": data, "exchange": "binance"}
    except Exception as e:
        logger.error(f"Error placing Binance market order: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error placing Binance order: {str(e)}")


@router.post("/binance/order/limit")
async def place_binance_limit_order(
    symbol: str = Query(..., description="Trading symbol"),
    side: str = Query(..., description="Order side (BUY/SELL)"),
    quantity: float = Query(..., description="Order quantity"),
    price: float = Query(..., description="Limit price"),
) -> Dict[str, Any]:
    """Place a limit order on Binance"""
    try:
        binance_service = get_binance_trading_service(
            api_key=settings.exchange.binance_us_api_key or "",
            secret_key=settings.exchange.binance_us_secret_key or "",
            testnet=settings.exchange.testnet,
        )

        data = await binance_service.place_limit_order(symbol, side, quantity, price)
        return {"status": "success", "data": data, "exchange": "binance"}
    except Exception as e:
        logger.error(f"Error placing Binance limit order: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error placing Binance order: {str(e)}")


@router.get("/binance/orders")
async def get_binance_orders(
    symbol: Optional[str] = Query(None, description="Filter by symbol")
) -> Dict[str, Any]:
    """Get open orders on Binance"""
    try:
        binance_service = get_binance_trading_service(
            api_key=settings.exchange.binance_us_api_key or "",
            secret_key=settings.exchange.binance_us_secret_key or "",
            testnet=settings.exchange.testnet,
        )

        data = await binance_service.get_open_orders(symbol)
        return {"status": "success", "data": data, "exchange": "binance"}
    except Exception as e:
        logger.error(f"Error getting Binance orders: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting Binance orders: {str(e)}")


@router.get("/binance/history")
async def get_binance_history(
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    limit: int = Query(100, description="Number of orders to return"),
) -> Dict[str, Any]:
    """Get order history on Binance"""
    try:
        binance_service = get_binance_trading_service(
            api_key=settings.exchange.binance_us_api_key or "",
            secret_key=settings.exchange.binance_us_secret_key or "",
            testnet=settings.exchange.testnet,
        )

        data = await binance_service.get_order_history(symbol, limit)
        return {"status": "success", "data": data, "exchange": "binance"}
    except Exception as e:
        logger.error(f"Error getting Binance history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting Binance history: {str(e)}")


@router.get("/binance/trades")
async def get_binance_trades(
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    limit: int = Query(100, description="Number of trades to return"),
) -> Dict[str, Any]:
    """Get trade history on Binance"""
    try:
        binance_service = get_binance_trading_service(
            api_key=settings.exchange.binance_us_api_key or "",
            secret_key=settings.exchange.binance_us_secret_key or "",
            testnet=settings.exchange.testnet,
        )

        data = await binance_service.get_trade_history(symbol, limit)
        return {"status": "success", "data": data, "exchange": "binance"}
    except Exception as e:
        logger.error(f"Error getting Binance trades: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting Binance trades: {str(e)}")


# ============================================================================
# COINBASE TRADING ENDPOINTS
# ============================================================================


@router.get("/coinbase/account")
async def get_coinbase_account() -> Dict[str, Any]:
    """Get Coinbase account information and balances"""
    try:
        coinbase_service = get_coinbase_trading_service(
            api_key=settings.exchange.coinbase_api_key or "",
            secret_key=settings.exchange.coinbase_secret_key or "",
            sandbox=settings.exchange.testnet,
        )

        data = await coinbase_service.get_account_info()
        return {"status": "success", "data": data, "exchange": "coinbase"}
    except Exception as e:
        logger.error(f"Error getting Coinbase account: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting Coinbase account: {str(e)}")


@router.get("/coinbase/price/{product_id}")
async def get_coinbase_price(product_id: str) -> Dict[str, Any]:
    """Get current price for a product on Coinbase"""
    try:
        coinbase_service = get_coinbase_trading_service(
            api_key=settings.exchange.coinbase_api_key or "",
            secret_key=settings.exchange.coinbase_secret_key or "",
            sandbox=settings.exchange.testnet,
        )

        data = await coinbase_service.get_market_price(product_id)
        return {"status": "success", "data": data, "exchange": "coinbase"}
    except Exception as e:
        logger.error(f"Error getting Coinbase price for {product_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting Coinbase price: {str(e)}")


@router.post("/coinbase/order/market")
async def place_coinbase_market_order(
    product_id: str = Query(..., description="Product ID"),
    side: str = Query(..., description="Order side (BUY/SELL)"),
    size: float = Query(..., description="Order size"),
) -> Dict[str, Any]:
    """Place a market order on Coinbase"""
    try:
        coinbase_service = get_coinbase_trading_service(
            api_key=settings.exchange.coinbase_api_key or "",
            secret_key=settings.exchange.coinbase_secret_key or "",
            sandbox=settings.exchange.testnet,
        )

        data = await coinbase_service.place_market_order(product_id, side, size)
        return {"status": "success", "data": data, "exchange": "coinbase"}
    except Exception as e:
        logger.error(f"Error placing Coinbase market order: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error placing Coinbase order: {str(e)}")


@router.post("/coinbase/order/limit")
async def place_coinbase_limit_order(
    product_id: str = Query(..., description="Product ID"),
    side: str = Query(..., description="Order side (BUY/SELL)"),
    size: float = Query(..., description="Order size"),
    price: float = Query(..., description="Limit price"),
) -> Dict[str, Any]:
    """Place a limit order on Coinbase"""
    try:
        coinbase_service = get_coinbase_trading_service(
            api_key=settings.exchange.coinbase_api_key or "",
            secret_key=settings.exchange.coinbase_secret_key or "",
            sandbox=settings.exchange.testnet,
        )

        data = await coinbase_service.place_limit_order(product_id, side, size, price)
        return {"status": "success", "data": data, "exchange": "coinbase"}
    except Exception as e:
        logger.error(f"Error placing Coinbase limit order: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error placing Coinbase order: {str(e)}")


@router.get("/coinbase/orders")
async def get_coinbase_orders(
    product_id: Optional[str] = Query(None, description="Filter by product ID")
) -> Dict[str, Any]:
    """Get open orders on Coinbase"""
    try:
        coinbase_service = get_coinbase_trading_service(
            api_key=settings.exchange.coinbase_api_key or "",
            secret_key=settings.exchange.coinbase_secret_key or "",
            sandbox=settings.exchange.testnet,
        )

        data = await coinbase_service.get_open_orders(product_id)
        return {"status": "success", "data": data, "exchange": "coinbase"}
    except Exception as e:
        logger.error(f"Error getting Coinbase orders: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting Coinbase orders: {str(e)}")


@router.get("/coinbase/products")
async def get_coinbase_products() -> Dict[str, Any]:
    """Get available products on Coinbase"""
    try:
        coinbase_service = get_coinbase_trading_service(
            api_key=settings.exchange.coinbase_api_key or "",
            secret_key=settings.exchange.coinbase_secret_key or "",
            sandbox=settings.exchange.testnet,
        )

        data = await coinbase_service.get_products()
        return {"status": "success", "data": data, "exchange": "coinbase"}
    except Exception as e:
        logger.error(f"Error getting Coinbase products: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting Coinbase products: {str(e)}",
        )


# ============================================================================
# MULTI-EXCHANGE ENDPOINTS
# ============================================================================


@router.get("/live/price-comparison/{symbol}")
async def get_price_comparison(symbol: str) -> Dict[str, Any]:
    """Get price comparison across multiple exchanges"""
    try:
        result: Dict[str, Any] = {
            "symbol": symbol,
            "prices": {},
            "timestamp": time.time(),
        }

        # Get CoinGecko price
        try:
            coingecko_price = await live_market_data_service.get_coin_price(symbol.lower(), "usd")
            result["prices"]["coingecko"] = coingecko_price
        except Exception as e:
            logger.warning(f"Failed to get CoinGecko price for {symbol}: {str(e)}")
            result["prices"]["coingecko"] = {"error": "Price not available"}

        # Get Binance price if API keys are configured
        if settings.exchange.binance_us_api_key:
            try:
                binance_service = get_binance_trading_service(
                    api_key=settings.exchange.binance_us_api_key,
                    secret_key=settings.exchange.binance_us_secret_key,
                    testnet=settings.exchange.testnet,
                )
                binance_price = await binance_service.get_market_price(f"{symbol.upper()}USDT")
                result["prices"]["binance"] = binance_price
            except Exception as e:
                logger.warning(f"Failed to get Binance price for {symbol}: {str(e)}")
                result["prices"]["binance"] = {"error": "Price not available"}

        # Get Coinbase price if API keys are configured
        if settings.exchange.coinbase_api_key:
            try:
                coinbase_service = get_coinbase_trading_service(
                    api_key=settings.exchange.coinbase_api_key,
                    secret_key=settings.exchange.coinbase_secret_key,
                    sandbox=settings.exchange.testnet,
                )
                coinbase_price = await coinbase_service.get_market_price(f"{symbol.upper()}-USD")
                result["prices"]["coinbase"] = coinbase_price
            except Exception as e:
                logger.warning(f"Failed to get Coinbase price for {symbol}: {str(e)}")
                result["prices"]["coinbase"] = {"error": "Price not available"}

        return {"status": "success", "data": result}

    except Exception as e:
        logger.error(f"Error getting price comparison for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting price comparison: {str(e)}")


@router.get("/live/exchange-status")
async def get_exchange_status() -> Dict[str, Any]:
    """Get status of all connected exchanges"""
    try:
        result: Dict[str, Any] = {"exchanges": {}, "timestamp": time.time()}

        # Test CoinGecko
        try:
            global_data = await live_market_data_service.get_global_data()
            result["exchanges"]["coingecko"] = {
                "status": "connected",
                "data": global_data,
            }
        except Exception as e:
            logger.warning(f"CoinGecko connection failed: {str(e)}")
            result["exchanges"]["coingecko"] = {"status": "disconnected"}

        # Test Binance
        if settings.exchange.binance_us_api_key:
            try:
                binance_service = get_binance_trading_service(
                    api_key=settings.exchange.binance_us_api_key,
                    secret_key=settings.exchange.binance_us_secret_key,
                    testnet=settings.exchange.testnet,
                )
                binance_status = await binance_service.test_connection()
                result["exchanges"]["binance"] = binance_status
            except Exception as e:
                logger.warning(f"Binance connection failed: {str(e)}")
                result["exchanges"]["binance"] = {"status": "disconnected"}
        else:
            result["exchanges"]["binance"] = {"status": "not_configured"}

        # Test Coinbase
        if settings.exchange.coinbase_api_key:
            try:
                coinbase_service = get_coinbase_trading_service(
                    api_key=settings.exchange.coinbase_api_key,
                    secret_key=settings.exchange.coinbase_secret_key,
                    sandbox=settings.exchange.testnet,
                )
                coinbase_status = await coinbase_service.test_connection()
                result["exchanges"]["coinbase"] = coinbase_status
            except Exception as e:
                logger.warning(f"Coinbase connection failed: {str(e)}")
                result["exchanges"]["coinbase"] = {"status": "disconnected"}
        else:
            result["exchanges"]["coinbase"] = {"status": "not_configured"}

        return {"status": "success", "data": result}

    except Exception as e:
        logger.error(f"Error getting exchange status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting exchange status: {str(e)}")


logger.info("✅ Live trading endpoints loaded successfully")
