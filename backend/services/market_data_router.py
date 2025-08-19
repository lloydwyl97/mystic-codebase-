from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable
from typing import Any, TypeVar

T = TypeVar("T")

from backend.exchanges.base_adapter import BaseExchangeAdapter  # type: ignore[import-not-found]
from backend.exchanges.binanceus_adapter import BinanceUSAdapter  # type: ignore[import-not-found]
from backend.exchanges.coinbase_adapter import CoinbaseAdapter  # type: ignore[import-not-found]
from backend.exchanges.coingecko_client import CoinGeckoClient  # type: ignore[import-not-found]
from backend.exchanges.kraken_adapter import KrakenAdapter  # type: ignore[import-not-found]
from backend.models.market_types import OHLCV, Ticker  # type: ignore[import-not-found]
from backend.utils.symbols import is_top4, normalize_symbol_to_dash  # type: ignore[import-not-found]

logger = logging.getLogger(__name__)


class _InMemoryCache:
    def __init__(self) -> None:
        self._store: dict[str, tuple[float, Any]] = {}

    def get(self, key: str) -> Any | None:
        item = self._store.get(key)
        if not item:
            return None
        exp, val = item
        if exp < time.time():
            self._store.pop(key, None)
            return None
        return val

    def set(self, key: str, val: Any, ttl: int) -> None:
        self._store[key] = (time.time() + ttl, val)


class MarketDataRouter:
    def __init__(self) -> None:
        self.adapters: dict[str, BaseExchangeAdapter] = {
            "coinbase": CoinbaseAdapter(),
            "binanceus": BinanceUSAdapter(),
            "kraken": KrakenAdapter(),
        }
        self.coingecko = CoinGeckoClient()
        self.cache = _InMemoryCache()
        self.ttl_short = 3
        self.rate_limits_per_minute: dict[str, int] = {
            "coinbase": 60,
            "binanceus": 120,
            "kraken": 60,
        }
        self._last_call_ts: dict[str, float] = {}
        self._backoff_base = 0.25
        self._backoff_max = 2.0

    async def _throttle(self, exchange: str) -> None:
        limit = max(self.rate_limits_per_minute.get(exchange, 60), 1)
        min_interval = 60.0 / float(limit)
        last = self._last_call_ts.get(exchange, 0.0)
        now = time.time()
        if last > 0 and (now - last) < min_interval:
            wait_s = min_interval - (now - last)
            await asyncio.sleep(wait_s)
        self._last_call_ts[exchange] = time.time()

    async def _with_backoff(self, coro: Awaitable[T], exchange: str, symbol: str, kind: str) -> T:
        attempt = 0
        delay = self._backoff_base
        while True:
            try:
                return await coro
            except Exception as exc:  # noqa: BLE001
                attempt += 1
                if attempt >= 3:
                    logger.warning(
                        "router_error",
                        extra={
                            "exchange": exchange,
                            "symbol": normalize_symbol_to_dash(symbol),
                            "kind": kind,
                            "attempts": attempt,
                            "error": str(exc),
                        },
                    )
                    raise
                await asyncio.sleep(min(delay, self._backoff_max))
                delay *= 2

    async def get_enabled_adapters(self) -> list[str]:
        return list(self.adapters.keys())

    def _cache_key(self, exchange: str, symbol: str, kind: str) -> str:
        s = normalize_symbol_to_dash(symbol)
        return f"{exchange}:{s}:{kind}"

    async def get_ticker(self, exchange: str, symbol: str) -> Ticker:
        if not is_top4(exchange, normalize_symbol_to_dash(symbol)):
            raise ValueError("Symbol not allowed; top-4 per exchange enforced")
        key = self._cache_key(exchange, symbol, "ticker")
        cached = self.cache.get(key)
        if cached:
            return cached
        adapter = self.adapters[exchange]
        await self._throttle(exchange)
        logger.debug(
            "router_get_ticker",
            extra={"exchange": exchange, "symbol": normalize_symbol_to_dash(symbol)},
        )
        result = await self._with_backoff(
            adapter.get_ticker(symbol), exchange, symbol, "ticker"
        )
        self.cache.set(key, result, self.ttl_short)
        return result

    async def get_ohlcv(self, exchange: str, symbol: str, interval: str, limit: int = 200) -> list[OHLCV]:
        if not is_top4(exchange, normalize_symbol_to_dash(symbol)):
            raise ValueError("Symbol not allowed; top-4 per exchange enforced")
        key = self._cache_key(exchange, symbol, f"ohlcv:{interval}:{limit}")
        cached = self.cache.get(key)
        if cached:
            return cached
        adapter = self.adapters[exchange]
        await self._throttle(exchange)
        logger.debug(
            "router_get_ohlcv",
            extra={
                "exchange": exchange,
                "symbol": normalize_symbol_to_dash(symbol),
                "interval": interval,
                "limit": limit,
            },
        )
        result = await self._with_backoff(
            adapter.get_ohlcv(symbol, interval, limit), exchange, symbol, "ohlcv"
        )
        self.cache.set(key, result, self.ttl_short)
        return result

    async def fanout_tickers(self, symbol: str) -> dict[str, Ticker]:
        tasks = {}
        for name in await self.get_enabled_adapters():
            if is_top4(name, normalize_symbol_to_dash(symbol)):
                tasks[name] = asyncio.create_task(self.get_ticker(name, symbol))
        results: dict[str, Ticker] = {}
        for name, task in list(tasks.items()):  # type: ignore[assignment]
            try:
                results[name] = await task
            except Exception:
                continue
        return results




