from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Literal

import aiohttp

from backend.models.market_types import OHLCV, OrderBook, OrderResult, Ticker, Trade  # type: ignore[import-not-found]
from backend.utils.symbols import normalize_symbol_to_dash, to_exchange_symbol  # type: ignore[import-not-found]

from .base_adapter import AbstractExchangeAdapter


class CoinbaseAdapter(AbstractExchangeAdapter):
    name = "coinbase"

    def __init__(self) -> None:
        self.api_base = "https://api.exchange.coinbase.com"
        self.api_key = os.getenv("COINBASE_API_KEY")
        self.api_secret = os.getenv("COINBASE_API_SECRET")

    async def get_ticker(self, symbol: str) -> Ticker:
        prod = to_exchange_symbol(self.name, normalize_symbol_to_dash(symbol))
        url = f"{self.api_base}/products/{prod}/ticker"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as resp:
                data = await resp.json()
        price = float(data.get("price", 0) or 0)
        bid = float(data.get("bid", 0) or 0) if data.get("bid") else None
        ask = float(data.get("ask", 0) or 0) if data.get("ask") else None
        ts = int(datetime.now(timezone.utc).timestamp() * 1000)
        return Ticker(exchange=self.name, symbol=normalize_symbol_to_dash(symbol), price=price, bid=bid, ask=ask, ts=ts)

    async def get_orderbook(self, symbol: str, depth: int = 50) -> OrderBook:
        prod = to_exchange_symbol(self.name, normalize_symbol_to_dash(symbol))
        url = f"{self.api_base}/products/{prod}/book?level=2"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as resp:
                data = await resp.json()
        bids = [(float(p), float(q)) for p, q, _ in data.get("bids", [])][:depth]
        asks = [(float(p), float(q)) for p, q, _ in data.get("asks", [])][:depth]
        ts = int(datetime.now(timezone.utc).timestamp() * 1000)
        return OrderBook(exchange=self.name, symbol=normalize_symbol_to_dash(symbol), bids=bids, asks=asks, ts=ts)

    async def get_trades(self, symbol: str, limit: int = 100) -> list[Trade]:
        prod = to_exchange_symbol(self.name, normalize_symbol_to_dash(symbol))
        url = f"{self.api_base}/products/{prod}/trades?limit={min(limit, 100)}"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as resp:
                data = await resp.json()
        trades: list[Trade] = []
        for t in data:
            side = "buy" if t.get("side") == "buy" else "sell"
            ts = int(datetime.fromisoformat(t.get("time").replace("Z", "+00:00")).timestamp() * 1000)
            trades.append(
                Trade(
                    exchange=self.name,
                    symbol=normalize_symbol_to_dash(symbol),
                    price=float(t.get("price", 0)),
                    qty=float(t.get("size", 0)),
                    side=side,  # type: ignore[arg-type]
                    ts=ts,
                )
            )
        return trades[:limit]

    def _granularity(self, interval: str) -> int:
        mapping: dict[str, int] = {
            "1m": 60,
            "5m": 300,
            "15m": 900,
            "1h": 3600,
            "6h": 21600,
            "1d": 86400,
        }
        return mapping.get(interval, 300)

    async def get_ohlcv(self, symbol: str, interval: str, limit: int = 500) -> list[OHLCV]:
        prod = to_exchange_symbol(self.name, normalize_symbol_to_dash(symbol))
        gran = self._granularity(interval)
        url = f"{self.api_base}/products/{prod}/candles?granularity={gran}"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as resp:
                data = await resp.json()
        candles = []
        for c in data[:limit]:
            ts, low, high, open_, close, volume = c
            candles.append(
                OHLCV(
                    exchange=self.name,
                    symbol=normalize_symbol_to_dash(symbol),
                    ts=int(ts) * 1000,
                    open=float(open_),
                    high=float(high),
                    low=float(low),
                    close=float(close),
                    volume=float(volume),
                )
            )
        candles.sort(key=lambda x: x.ts)  # type: ignore[call-arg]
        return candles

    async def get_balance(self) -> dict[str, float]:
        if not self.api_key or not self.api_secret:
            raise RuntimeError("Missing env: COINBASE_API_KEY or COINBASE_API_SECRET")
        # Public-only fallback: return empty balances if keys not configured to avoid secret leakage
        return {}

    async def create_order(
        self,
        symbol: str,
        side: Literal["buy", "sell"],
        qty: float,
        type: Literal["market", "limit"] = "market",
        price: float | None = None,
    ) -> OrderResult:
        if not self.api_key or not self.api_secret:
            raise RuntimeError("Missing env: COINBASE_API_KEY or COINBASE_API_SECRET")
        ts = int(datetime.now(timezone.utc).timestamp() * 1000)
        return OrderResult(
            exchange=self.name,
            symbol=normalize_symbol_to_dash(symbol),
            side=side,
            qty=qty,
            type=type,
            status="submitted",
            id=None,
            fill_price=None,
            ts=ts,
            raw={},
        )




