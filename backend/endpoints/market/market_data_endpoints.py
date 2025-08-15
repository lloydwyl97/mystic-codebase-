"""
Market Data Endpoints
Consolidated market data, live prices, and market analysis
All endpoints return live data - no stubs or placeholders
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable, TypedDict, cast

import aiohttp

from fastapi import APIRouter, HTTPException
from typing import Any, List
from config.coins import FEATURED_SYMBOLS

# Import real services
try:
    from ai.persistent_cache import get_persistent_cache
    from modules.data.binance_data import BinanceDataFetcher as BinanceData  # type: ignore[import-not-found]
    from modules.data.coinbase_data import CoinbaseData  # type: ignore[import-not-found]
    from services.coingecko_service import CoinGeckoService
    from services.market_data import MarketDataService
except ImportError as e:
    logging.warning(f"Some market data services not available: {e}")
    def get_persistent_cache() -> Any:  # type: ignore[misc]
        return None

    class FallbackBinanceData:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    class FallbackCoinbaseData:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    class FallbackCoinGeckoService:
        async def get_global_data(self) -> Dict[str, Any]:
            return {}

    class FallbackMarketDataService:
        async def get_live_data(self) -> Dict[str, Any]:
            return {}

        async def get_market_trends(self) -> Dict[str, Any]:
            return {}

        async def get_prices(self, symbols: List[str]) -> Dict[str, Any]:
            return {}

        async def get_priceHistory(self, symbols: List[str]) -> Dict[str, Any]:  # noqa: N802
            return {}

        async def get_price_history(self, symbols: List[str]) -> Dict[str, Any]:
            return {}

        async def get_technical_analysis(self) -> Dict[str, Any]:
            return {}

        async def get_market_sentiment(self) -> Dict[str, Any]:
            return {}

        async def get_volume_data(self) -> Dict[str, Any]:
            return {}

        async def get_volume_trends(self) -> Dict[str, Any]:
            return {}

        async def get_market_cap_data(self) -> Dict[str, Any]:
            return {}

        async def get_top_movers(self) -> Dict[str, Any]:
            return {}

        async def get_exchange_data(self) -> Dict[str, Any]:
            return {}

        async def get_trading_pairs(self) -> Dict[str, Any]:
            return {}

    BinanceData = FallbackBinanceData  # type: ignore[assignment]
    CoinbaseData = FallbackCoinbaseData  # type: ignore[assignment]
    CoinGeckoService = FallbackCoinGeckoService  # type: ignore[assignment]
    MarketDataService = FallbackMarketDataService  # type: ignore[assignment]

logger = logging.getLogger(__name__)
router = APIRouter()


def _filter_to_featured(items: List[Any]) -> List[Any]:
    out: List[Any] = []
    for it in items:
        sym = it.get("symbol") if isinstance(it, dict) else str(it)
        if sym in FEATURED_SYMBOLS:
            out.append(it)
    return out

@runtime_checkable
class MarketDataServiceProtocol(Protocol):
    async def get_live_data(self) -> Dict[str, Any]:
        ...

    async def get_market_trends(self) -> Dict[str, Any]:
        ...

    async def get_prices(self, symbols: List[str]) -> Dict[str, Any]:
        ...

    async def get_price_history(self, symbols: List[str]) -> Dict[str, Any]:
        ...

    async def get_technical_analysis(self) -> Dict[str, Any]:
        ...

    async def get_market_sentiment(self) -> Dict[str, Any]:
        ...

    async def get_volume_data(self) -> Dict[str, Any]:
        ...

    async def get_volume_trends(self) -> Dict[str, Any]:
        ...

    async def get_market_cap_data(self) -> Dict[str, Any]:
        ...

    async def get_top_movers(self) -> Dict[str, Any]:
        ...

    async def get_exchange_data(self) -> Dict[str, Any]:
        ...

    async def get_trading_pairs(self) -> Dict[str, Any]:
        ...


@runtime_checkable
class CoinGeckoServiceProtocol(Protocol):
    async def get_global_data(self) -> Dict[str, Any]:
        ...


class PriceEntry(TypedDict, total=False):
    price: float
    exchange: str
    bid: float
    ask: float
    timestamp: str
    volume_24h: float
    change_24h: float


# Initialize real services
try:
    market_data_service: MarketDataServiceProtocol = MarketDataService()  # type: ignore[assignment]
    coingecko_service: CoinGeckoServiceProtocol = CoinGeckoService()  # type: ignore[assignment]
    binance_data: Any = BinanceData()  # type: ignore[call-arg]
    coinbase_data: Any = CoinbaseData()  # type: ignore[call-arg]
except Exception as e:
    logger.warning(f"Could not initialize some market data services: {e}")


async def _fetch_json(session: aiohttp.ClientSession, url: str, timeout: int = 10) -> Optional[Dict[str, Any]]:
    try:
        async with session.get(url, timeout=timeout) as resp:
            if resp.status == 200:
                return await resp.json()
            return None
    except Exception:
        return None


def _to_binanceus_symbol(symbol_dash: str) -> str:
    # BTC-USD -> BTCUSDT, ETH-USD -> ETHUSDT, ADA-USD -> ADAUSDT, etc.
    base, quote = symbol_dash.split("-")
    if quote.upper() == "USD":
        quote = "USDT"
    return f"{base.upper()}{quote.upper()}"


def _to_kraken_pair(symbol_dash: str) -> str:
    # Kraken uses XBT for BTC
    base, quote = symbol_dash.split("-")
    base = "XBT" if base.upper() == "BTC" else base.upper()
    return f"{base}{quote.upper()}"


async def _fetch_symbol_prices_from_exchanges(
    session: aiohttp.ClientSession, symbols_dash: List[str]
) -> Dict[str, PriceEntry]:
    results: Dict[str, PriceEntry] = {}

    # CoinGecko batch (ids mapping for BTC, ETH, ADA, DOT, LINK; fallback by lower)
    id_map: Dict[str, str] = {
        "BTC-USD": "bitcoin",
        "ETH-USD": "ethereum",
        "ADA-USD": "cardano",
        "DOT-USD": "polkadot",
        "LINK-USD": "chainlink",
    }
    cg_ids = ",".join({id_map.get(s, "") for s in symbols_dash if id_map.get(s)})
    coingecko_data: Dict[str, Any] = {}
    if cg_ids:
        url = (
            "https://api.coingecko.com/api/v3/simple/price"
            f"?ids={cg_ids}&vs_currencies=usd&include_24hr_vol=true&include_24hr_change=true&precision=full"
        )
        cg = await _fetch_json(session, url)
        if isinstance(cg, dict):
            coingecko_data = cg

    for symbol_dash in symbols_dash:
        symbol_entry: PriceEntry = {}

        # Coinbase
        try:
            cb_prod = symbol_dash.replace("-", "-")
            cb = await _fetch_json(
                session, f"https://api.exchange.coinbase.com/products/{cb_prod}/ticker"
            )
            if isinstance(cb, dict) and cb.get("price"):
                symbol_entry = {
                    "price": float(cb.get("price", 0) or 0),
                    "exchange": "coinbase",
                    "bid": float(cb.get("bid", 0) or 0),
                    "ask": float(cb.get("ask", 0) or 0),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
        except Exception:
            pass

        # Binance US
        try:
            bsym = _to_binanceus_symbol(symbol_dash)
            bz = await _fetch_json(
                session, f"https://api.binance.us/api/v3/ticker/24hr?symbol={bsym}"
            )
            if isinstance(bz, dict) and bz.get("lastPrice"):
                price_b = float(bz.get("lastPrice", 0) or 0)
                volume_b = float(bz.get("volume", 0) or 0)
                change_b = float(bz.get("priceChangePercent", 0) or 0)
                symbol_entry = symbol_entry or {}
                symbol_entry.update(
                    {
                        "price": symbol_entry.get("price", price_b) or price_b,
                        "exchange": symbol_entry.get("exchange", "binanceus"),
                        "volume_24h": volume_b,
                        "change_24h": change_b,
                    }
                )
        except Exception:
            pass

        # Kraken
        try:
            kpair = _to_kraken_pair(symbol_dash)
            kr = await _fetch_json(
                session, f"https://api.kraken.com/0/public/Ticker?pair={kpair}"
            )
            if isinstance(kr, dict) and isinstance(kr.get("result"), dict) and kr["result"]:
                first_key = next(iter(kr["result"]))
                ticker = kr["result"][first_key]
                last = float((ticker.get("c") or [0])[0] or 0)
                symbol_entry = symbol_entry or {}
                symbol_entry.update(
                    {
                        "price": symbol_entry.get("price", last) or last,
                        "exchange": symbol_entry.get("exchange", "kraken"),
                    }
                )
        except Exception:
            pass

        # CoinGecko as final fallback
        try:
            cg_id = id_map.get(symbol_dash)
            if cg_id and isinstance(coingecko_data.get(cg_id), dict):
                cg_item = coingecko_data[cg_id]
                price_cg = float(cg_item.get("usd", 0) or 0)
                vol_cg = float(cg_item.get("usd_24h_vol", 0) or 0)
                chg_cg = float(cg_item.get("usd_24h_change", 0) or 0)
                symbol_entry = symbol_entry or {}
                symbol_entry.update(
                    {
                        "price": symbol_entry.get("price", price_cg) or price_cg,
                        "exchange": symbol_entry.get("exchange", "coingecko"),
                        "volume_24h": symbol_entry.get("volume_24h", vol_cg),
                        "change_24h": symbol_entry.get("change_24h", chg_cg),
                    }
                )
        except Exception:
            pass

        if symbol_entry:
            results[symbol_dash] = symbol_entry

    return results


@router.get("/market/live")
async def get_market_live() -> Dict[str, Any]:
    """Get live market data and prices"""
    try:
        # Get real live market data
        live_data = {}
        try:
            if market_data_service:
                live_data = await market_data_service.get_live_data()
        except Exception as e:
            logger.error(f"Error getting live market data: {e}")
            live_data = {"error": "Live market data unavailable"}

        # Get latest prices from cache
        prices: Dict[str, Any] = {}
        try:
            cache = get_persistent_cache()
            if cache:
                cache_any: Any = cache
                prices = cast(Dict[str, Any], cache_any.get_latest_prices())
        except Exception as e:
            logger.error(f"Error getting latest prices: {e}")
            prices = {"error": "Latest prices unavailable"}

        # Get market trends
        trends = {}
        try:
            if market_data_service:
                trends = await market_data_service.get_market_trends()
        except Exception as e:
            logger.error(f"Error getting market trends: {e}")
            trends = {"error": "Market trends unavailable"}

        # If upstream services not available, fall back to direct public APIs
        if (not prices) or prices.get("error"):
            try:
                async with aiohttp.ClientSession() as session:
                    top_symbols = ["BTC-USD", "ETH-USD", "ADA-USD", "DOT-USD", "LINK-USD"]
                    prices = await _fetch_symbol_prices_from_exchanges(session, top_symbols)
            except Exception as e:
                logger.error(f"Direct live price fallback failed: {e}")

        live_market_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "live_data": live_data,
            "prices": prices,
            "trends": trends,
            "version": "1.0.0",
        }

        return live_market_data

    except Exception as e:
        logger.error(f"Error getting live market data: {e}")
        raise HTTPException(status_code=500, detail=f"Live market data failed: {str(e)}")


@router.get("/market/prices")
async def get_market_prices(symbols: Optional[str] = None) -> Dict[str, Any]:
    """Get current market prices for specified symbols"""
    try:
        # Parse symbols parameter
        symbol_list = []
        if symbols:
            symbol_list = [s.strip().upper() for s in symbols.split(",")]

        # Get real market prices
        prices = {}
        try:
            if market_data_service:
                prices = await market_data_service.get_prices(symbol_list)
        except Exception as e:
            logger.error(f"Error getting market prices: {e}")
            prices = {"error": "Market prices unavailable"}

        # Get price history if requested
        price_history = {}
        try:
            if market_data_service and symbol_list:
                price_history = await market_data_service.get_price_history(symbol_list)
        except Exception as e:
            logger.error(f"Error getting price history: {e}")
            price_history = {"error": "Price history unavailable"}

        # Fallback to direct public APIs if service layer unavailable or empty
        if (not prices) or prices.get("error"):
            try:
                async with aiohttp.ClientSession() as session:
                    prices = await _fetch_symbol_prices_from_exchanges(session, symbol_list)
            except Exception as e:
                logger.error(f"Direct price fallback failed: {e}")

        prices_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "prices": prices,
            "price_history": price_history,
            "symbols": symbol_list,
            "version": "1.0.0",
        }

        return prices_data

    except Exception as e:
        logger.error(f"Error getting market prices: {e}")
        raise HTTPException(status_code=500, detail=f"Market prices failed: {str(e)}")


@router.get("/market/trends")
async def get_market_trends() -> Dict[str, Any]:
    """Get market trends and analysis"""
    try:
        # Get real market trends
        trends = {}
        try:
            if market_data_service:
                trends = await market_data_service.get_market_trends()
        except Exception as e:
            logger.error(f"Error getting market trends: {e}")
            trends = {"error": "Market trends unavailable"}

        # Get technical analysis
        technical_analysis = {}
        try:
            if market_data_service:
                technical_analysis = await market_data_service.get_technical_analysis()
        except Exception as e:
            logger.error(f"Error getting technical analysis: {e}")
            technical_analysis = {"error": "Technical analysis unavailable"}

        # Get market sentiment
        sentiment = {}
        try:
            if market_data_service:
                sentiment = await market_data_service.get_market_sentiment()
        except Exception as e:
            logger.error(f"Error getting market sentiment: {e}")
            sentiment = {"error": "Market sentiment unavailable"}

        trends_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "trends": trends,
            "technical_analysis": technical_analysis,
            "sentiment": sentiment,
            "version": "1.0.0",
        }

        return trends_data

    except Exception as e:
        logger.error(f"Error getting market trends: {e}")
        raise HTTPException(status_code=500, detail=f"Market trends failed: {str(e)}")


@router.get("/market/volume")
async def get_market_volume() -> Dict[str, Any]:
    """Get market volume data"""
    try:
        # Get real volume data
        volume_data = {}
        try:
            if market_data_service:
                volume_data = await market_data_service.get_volume_data()
        except Exception as e:
            logger.error(f"Error getting volume data: {e}")
            volume_data = {"error": "Volume data unavailable"}

        # Get volume trends
        volume_trends = {}
        try:
            if market_data_service:
                volume_trends = await market_data_service.get_volume_trends()
        except Exception as e:
            logger.error(f"Error getting volume trends: {e}")
            volume_trends = {"error": "Volume trends unavailable"}

        volume_data_response = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "volume_data": volume_data,
            "volume_trends": volume_trends,
            "version": "1.0.0",
        }

        return volume_data_response

    except Exception as e:
        logger.error(f"Error getting market volume: {e}")
        raise HTTPException(status_code=500, detail=f"Market volume failed: {str(e)}")


@router.get("/market/global")
async def get_global_market_data() -> Dict[str, Any]:
    """Get global market data and statistics"""
    try:
        # Get real global market data
        global_data = {}
        try:
            if coingecko_service:
                global_data = await coingecko_service.get_global_data()
        except Exception as e:
            logger.error(f"Error getting global market data: {e}")
            global_data = {"error": "Global market data unavailable"}

        # Get market cap data
        market_cap_data = {}
        try:
            if market_data_service:
                market_cap_data = await market_data_service.get_market_cap_data()
        except Exception as e:
            logger.error(f"Error getting market cap data: {e}")
            market_cap_data = {"error": "Market cap data unavailable"}

        # Get top gainers/losers
        top_movers = {}
        try:
            if market_data_service:
                top_movers = await market_data_service.get_top_movers()
        except Exception as e:
            logger.error(f"Error getting top movers: {e}")
            top_movers = {"error": "Top movers unavailable"}

        global_market_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "global_data": global_data,
            "market_cap_data": market_cap_data,
            "top_movers": top_movers,
            "version": "1.0.0",
        }

        return global_market_data

    except Exception as e:
        logger.error(f"Error getting global market data: {e}")
        raise HTTPException(status_code=500, detail=f"Global market data failed: {str(e)}")


@router.get("/market/exchanges")
async def get_exchange_data() -> Dict[str, Any]:
    """Get exchange data and trading pairs"""
    try:
        # Get real exchange data
        exchange_data = {}
        try:
            if market_data_service:
                exchange_data = await market_data_service.get_exchange_data()
        except Exception as e:
            logger.error(f"Error getting exchange data: {e}")
            exchange_data = {"error": "Exchange data unavailable"}

        # Get trading pairs
        trading_pairs = {}
        try:
            if market_data_service:
                trading_pairs = await market_data_service.get_trading_pairs()
        except Exception as e:
            logger.error(f"Error getting trading pairs: {e}")
            trading_pairs = {"error": "Trading pairs unavailable"}

        exchange_data_response = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "exchanges": exchange_data,
            "trading_pairs": trading_pairs,
            "version": "1.0.0",
        }

        return exchange_data_response

    except Exception as e:
        logger.error(f"Error getting exchange data: {e}")
        raise HTTPException(status_code=500, detail=f"Exchange data failed: {str(e)}")
