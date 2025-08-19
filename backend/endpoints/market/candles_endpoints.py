"""
Market Candles Endpoint
Provides OHLCV data in a normalized format for UI consumption.
"""

import logging
from typing import Any, TypedDict, cast

import aiohttp
from fastapi import APIRouter, Response

router = APIRouter(prefix="/api/market", tags=["market"])

logger = logging.getLogger(__name__)


async def _fetch_binanceus_klines(symbol: str, interval: str, limit: int) -> list[list[Any]]:
    url = "https://api.binance.us/api/v3/klines"
    params = {"symbol": symbol.upper(), "interval": interval, "limit": min(max(limit, 1), 1000)}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=10) as resp:
                if resp.status == 200:
                    data: Any = await resp.json()
                    if isinstance(data, list):
                        return cast(list[list[Any]], data)  # Raw klines arrays
                else:
                    body = await resp.text()
                    logger.warning(f"BinanceUS klines HTTP {resp.status}: {body}")
    except Exception as e:
        logger.warning(f"Error fetching BinanceUS klines for {symbol}: {e}")
    return []


class Candle(TypedDict):
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float


def _map_klines_to_ohlcv(items: list[list[Any]]) -> list[Candle]:
    out: list[Candle] = []
    for k in items:
        try:
            # Binance klines schema: [ openTime, open, high, low, close, volume, closeTime, ... ]
            out.append(
                {
                    "timestamp": int(k[0]),
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[5]),
                }
            )
        except Exception:
            # Skip malformed row
            continue
    return out


@router.get("/candles")
async def get_candles(symbol: str, response: Response, interval: str = "1h", limit: int = 200) -> list[Candle]:
    """Return normalized OHLCV list for a symbol.

    Always returns 200 with a list (possibly empty) to avoid UI errors.
    """
    try:
        # Defensive: coerce interval to allowed set to prevent cache poisoning
        allowed = {"1m", "5m", "15m", "1h", "4h", "1d"}
        if interval not in allowed:
            interval = "1h"
        # Set short cache header to reduce load and smooth charts
        response.headers["Cache-Control"] = "public, max-age=15"

        klines = await _fetch_binanceus_klines(symbol=symbol, interval=interval, limit=limit)
        if not klines:
            return []
        return _map_klines_to_ohlcv(klines)
    except Exception as e:
        logger.warning(f"/api/market/candles error for {symbol}: {e}")
        return []


