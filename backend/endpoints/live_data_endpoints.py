"""
Live Data API Endpoints
Real API endpoints using live market data and trading services
"""

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

# Import live services
from backend.services.live_market_data import live_market_data_service
from backend.services.live_trading_service import trading_service

router = APIRouter(prefix="/live", tags=["Live Data"])


@router.get("/market/prices")
async def get_live_prices(symbols: str = Query(..., description="Comma-separated list of symbols")):
    """Get live prices for multiple symbols"""
    try:
        symbol_list = [s.strip() for s in symbols.split(",")]
        result = await live_market_data_service.get_live_prices(symbol_list)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching live prices: {str(e)}")


@router.get("/market/overview")
async def get_market_overview():
    """Get market overview with major indices and crypto"""
    try:
        result = await live_market_data_service.get_market_overview()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching market overview: {str(e)}")


@router.get("/market/historical/{symbol}")
async def get_historical_data(
    symbol: str,
    timeframe: str = Query("1d", description="Timeframe: 1m, 5m, 15m, 30m, 1h, 4h, 1d"),
    limit: int = Query(100, description="Number of candles to fetch"),
):
    """Get historical data for charting"""
    try:
        result = await live_market_data_service.get_historical_data(symbol, timeframe, limit)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching historical data: {str(e)}")


@router.get("/trading/balance")
async def get_account_balance():
    """Get account balance from connected exchanges"""
    try:
        result = await trading_service.get_account_balance()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching account balance: {str(e)}")


@router.get("/trading/orders")
async def get_open_orders():
    """Get open orders from connected exchanges"""
    try:
        result = await trading_service.get_open_orders()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching open orders: {str(e)}")


@router.get("/trading/trades")
async def get_trade_history(
    symbol: Optional[str] = Query(None, description="Symbol to filter trades"),
    limit: int = Query(100, description="Number of trades to fetch"),
):
    """Get trade history from connected exchanges"""
    try:
        result = await trading_service.get_trade_history(symbol, limit)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching trade history: {str(e)}")


@router.get("/trading/positions")
async def get_positions():
    """Get current positions from connected exchanges"""
    try:
        result = await trading_service.get_positions()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching positions: {str(e)}")


@router.post("/trading/order")
async def place_order(
    exchange: str = Query(..., description="Exchange: binance, coinbase"),
    symbol: str = Query(..., description="Trading symbol"),
    order_type: str = Query(..., description="Order type: market, limit"),
    side: str = Query(..., description="Order side: buy, sell"),
    amount: float = Query(..., description="Order amount"),
    price: Optional[float] = Query(None, description="Order price (required for limit orders)"),
):
    """Place a new order on the specified exchange"""
    try:
        if order_type == "limit" and price is None:
            raise HTTPException(status_code=400, detail="Price is required for limit orders")

        result = await trading_service.place_order(
            exchange, symbol, order_type, side, amount, price
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error placing order: {str(e)}")


@router.delete("/trading/order/{order_id}")
async def cancel_order(
    order_id: str,
    exchange: str = Query(..., description="Exchange: binance, coinbase"),
    symbol: str = Query(..., description="Trading symbol"),
):
    """Cancel an existing order"""
    try:
        result = await trading_service.cancel_order(exchange, order_id, symbol)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error cancelling order: {str(e)}")


@router.get("/status")
async def get_live_data_status():
    """Get status of live data connections"""
    try:
        status = {
            "market_data": {
                "binance": live_market_data_service.binance is not None,
                "coinbase": live_market_data_service.coinbase is not None,
                "yahoo_finance": True,  # Always available
            },
            "trading": {
                "binance": trading_service.binance is not None,
                "coinbase": trading_service.coinbase is not None,
            },
            "timestamp": datetime.now().isoformat(),
        }
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting status: {str(e)}")



