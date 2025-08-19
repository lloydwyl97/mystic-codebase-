from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal, cast

import aiohttp

from backend.models.market_types import OHLCV, OrderBook, OrderResult, Ticker, Trade  # type: ignore[import-not-found]
from backend.utils.symbols import normalize_symbol_to_dash, to_exchange_symbol  # type: ignore[import-not-found]

from .base_adapter import AbstractExchangeAdapter


class KrakenAdapter(AbstractExchangeAdapter):
    name = "kraken"

    def __init__(self) -> None:
        self.api_base = "https://api.kraken.com"

    async def get_ticker(self, symbol: str) -> Ticker:
        pair = to_exchange_symbol(self.name, normalize_symbol_to_dash(symbol))
        url = f"{self.api_base}/0/public/Ticker?pair={pair}"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as resp:
                data = await resp.json()
        result = cast(dict[str, Any], data.get("result", {}))
        first_key = next(iter(result)) if result else None
        t: dict[str, Any] = cast(dict[str, Any], result.get(first_key, {})) if first_key else {}
        last = float((t.get("c", [0])[0] if isinstance(t.get("c"), list) else 0) or 0)
        bid = float((t.get("b", [0])[0] if isinstance(t.get("b"), list) else 0) or 0) if t.get("b") is not None else None
        ask = float((t.get("a", [0])[0] if isinstance(t.get("a"), list) else 0) or 0) if t.get("a") is not None else None
        ts = int(datetime.now(timezone.utc).timestamp() * 1000)
        return Ticker(exchange=self.name, symbol=normalize_symbol_to_dash(symbol), price=last, bid=bid, ask=ask, ts=ts)

    async def get_orderbook(self, symbol: str, depth: int = 50) -> OrderBook:
        pair = to_exchange_symbol(self.name, normalize_symbol_to_dash(symbol))
        url = f"{self.api_base}/0/public/Depth?pair={pair}&count={min(max(depth, 5), 100)}"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as resp:
                data = await resp.json()
        result = cast(dict[str, Any], data.get("result", {}))
        first_key = next(iter(result)) if result else None
        ob: dict[str, Any] = cast(dict[str, Any], result.get(first_key, {})) if first_key else {}
        bids = [(float(p), float(q)) for p, q, _ in cast(list[list[Any]], ob.get("bids", []))][:depth]
        asks = [(float(p), float(q)) for p, q, _ in cast(list[list[Any]], ob.get("asks", []))][:depth]
        ts = int(datetime.now(timezone.utc).timestamp() * 1000)
        return OrderBook(exchange=self.name, symbol=normalize_symbol_to_dash(symbol), bids=bids, asks=asks, ts=ts)

    async def get_trades(self, symbol: str, limit: int = 100) -> list[Trade]:
        pair = to_exchange_symbol(self.name, normalize_symbol_to_dash(symbol))
        url = f"{self.api_base}/0/public/Trades?pair={pair}"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as resp:
                data = await resp.json()
        result = cast(dict[str, Any], data.get("result", {}))
        first_key = next(iter(result)) if result else None
        arr = cast(list[list[Any]], result.get(first_key, [])) if first_key else []
        out: list[Trade] = []
        for t in arr[:limit]:
            price = float(t[0])
            qty = float(t[1])
            ts = int(float(t[2]) * 1000)
            side = "buy" if t[3] == "b" else "sell"
            out.append(
                Trade(exchange=self.name, symbol=normalize_symbol_to_dash(symbol), price=price, qty=qty, side=side, ts=ts)
            )
        return out

    async def get_ohlcv(self, symbol: str, interval: str, limit: int = 500) -> list[OHLCV]:
        pair = to_exchange_symbol(self.name, normalize_symbol_to_dash(symbol))
        interval_map = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}
        it = interval_map.get(interval, 5)
        url = f"{self.api_base}/0/public/OHLC?pair={pair}&interval={it}"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as resp:
                data = await resp.json()
        result = cast(dict[str, Any], data.get("result", {}))
        first_key = next(iter(result)) if result else None
        arr = cast(list[list[Any]], result.get(first_key, [])) if first_key else []
        ohlc: list[OHLCV] = []
        for c in arr[:limit]:
            ts = int(c[0]) * 1000
            ohlc.append(
                OHLCV(
                    exchange=self.name,
                    symbol=normalize_symbol_to_dash(symbol),
                    ts=ts,
                    open=float(c[1]),
                    high=float(c[2]),
                    low=float(c[3]),
                    close=float(c[4]),
                    volume=float(c[6]),
                )
            )
        return ohlc

    async def get_balance(self) -> dict[str, float]:
        return {}

    async def create_order(
        self,
        symbol: str,
        side: Literal["buy", "sell"],
        qty: float,
        type: Literal["market", "limit"] = "market",
        price: float | None = None,
    ) -> OrderResult:
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




