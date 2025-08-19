from __future__ import annotations

import abc
from typing import Literal, Protocol, runtime_checkable

from backend.models.market_types import OHLCV, OrderBook, OrderResult, Ticker, Trade  # type: ignore[import-not-found]


@runtime_checkable
class BaseExchangeAdapter(Protocol):
    name: str

    async def get_ticker(self, symbol: str) -> Ticker: ...

    async def get_orderbook(self, symbol: str, depth: int = 50) -> OrderBook: ...

    async def get_trades(self, symbol: str, limit: int = 100) -> list[Trade]: ...

    async def get_ohlcv(self, symbol: str, interval: str, limit: int = 500) -> list[OHLCV]: ...

    async def get_balance(self) -> dict[str, float]: ...

    async def create_order(
        self,
        symbol: str,
        side: Literal["buy", "sell"],
        qty: float,
        type: Literal["market", "limit"] = "market",
        price: float | None = None,
    ) -> OrderResult: ...


class AbstractExchangeAdapter(abc.ABC):
    name: str

    @abc.abstractmethod
    async def get_ticker(self, symbol: str) -> Ticker: ...

    @abc.abstractmethod
    async def get_orderbook(self, symbol: str, depth: int = 50) -> OrderBook: ...

    @abc.abstractmethod
    async def get_trades(self, symbol: str, limit: int = 100) -> list[Trade]: ...

    @abc.abstractmethod
    async def get_ohlcv(self, symbol: str, interval: str, limit: int = 500) -> list[OHLCV]: ...

    @abc.abstractmethod
    async def get_balance(self) -> dict[str, float]: ...

    @abc.abstractmethod
    async def create_order(
        self,
        symbol: str,
        side: Literal["buy", "sell"],
        qty: float,
        type: Literal["market", "limit"] = "market",
        price: float | None = None,
    ) -> OrderResult: ...




