from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Literal

import aiohttp

from .base_adapter import AbstractExchangeAdapter
from backend.models.market_types import Ticker, OrderBook, Trade, OHLCV, OrderResult  # type: ignore[import-not-found]
from backend.utils.symbols import to_exchange_symbol, normalize_symbol_to_dash  # type: ignore[import-not-found]


class BinanceUSAdapter(AbstractExchangeAdapter):
    name = "binanceus"

    def __init__(self) -> None:
        self.api_base = "https://api.binance.us"

    async def get_ticker(self, symbol: str) -> Ticker:
        pair = to_exchange_symbol(self.name, normalize_symbol_to_dash(symbol))
        url = f"{self.api_base}/api/v3/ticker/24hr?symbol={pair}"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as resp:
                data = await resp.json()
        price = float(data.get("lastPrice", 0) or 0)
        bid = float(data.get("bidPrice", 0) or 0) if data.get("bidPrice") else None
        ask = float(data.get("askPrice", 0) or 0) if data.get("askPrice") else None
        ts = int(datetime.now(timezone.utc).timestamp() * 1000)
        return Ticker(exchange=self.name, symbol=normalize_symbol_to_dash(symbol), price=price, bid=bid, ask=ask, ts=ts)

    async def get_orderbook(self, symbol: str, depth: int = 50) -> OrderBook:
        pair = to_exchange_symbol(self.name, normalize_symbol_to_dash(symbol))
        url = f"{self.api_base}/api/v3/depth?symbol={pair}&limit={min(max(depth, 5), 100)}"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as resp:
                data = await resp.json()
        bids = [(float(p), float(q)) for p, q in data.get("bids", [])][:depth]
        asks = [(float(p), float(q)) for p, q in data.get("asks", [])][:depth]
        ts = int(datetime.now(timezone.utc).timestamp() * 1000)
        return OrderBook(exchange=self.name, symbol=normalize_symbol_to_dash(symbol), bids=bids, asks=asks, ts=ts)

    async def get_trades(self, symbol: str, limit: int = 100) -> List[Trade]:
        pair = to_exchange_symbol(self.name, normalize_symbol_to_dash(symbol))
        url = f"{self.api_base}/api/v3/trades?symbol={pair}&limit={min(limit, 1000)}"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as resp:
                data = await resp.json()
        out: List[Trade] = []
        for t in data:
            ts = int(t.get("time", 0))
            out.append(
                Trade(
                    exchange=self.name,
                    symbol=normalize_symbol_to_dash(symbol),
                    price=float(t.get("price", 0)),
                    qty=float(t.get("qty", 0)),
                    side="buy" if not t.get("isBuyerMaker", False) else "sell",
                    ts=ts,
                )
            )
        return out[:limit]

    async def get_ohlcv(self, symbol: str, interval: str, limit: int = 500) -> List[OHLCV]:
        pair = to_exchange_symbol(self.name, normalize_symbol_to_dash(symbol))
        url = f"{self.api_base}/api/v3/klines?symbol={pair}&interval={interval}&limit={min(limit, 1000)}"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as resp:
                data = await resp.json()
        candles: List[OHLCV] = []
        for k in data:
            open_time = int(k[0])
            candles.append(
                OHLCV(
                    exchange=self.name,
                    symbol=normalize_symbol_to_dash(symbol),
                    ts=open_time,
                    open=float(k[1]),
                    high=float(k[2]),
                    low=float(k[3]),
                    close=float(k[4]),
                    volume=float(k[5]),
                )
            )
        return candles

    async def get_balance(self) -> Dict[str, float]:
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




