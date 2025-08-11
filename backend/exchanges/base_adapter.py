from __future__ import annotations

import abc
from typing import Dict, List, Literal, Protocol, runtime_checkable

from models.market_types import Ticker, OrderBook, Trade, OHLCV, OrderResult  # type: ignore[import-not-found]


@runtime_checkable
class BaseExchangeAdapter(Protocol):
    name: str

    async def get_ticker(self, symbol: str) -> Ticker: ...

    async def get_orderbook(self, symbol: str, depth: int = 50) -> OrderBook: ...

    async def get_trades(self, symbol: str, limit: int = 100) -> List[Trade]: ...

    async def get_ohlcv(self, symbol: str, interval: str, limit: int = 500) -> List[OHLCV]: ...

    async def get_balance(self) -> Dict[str, float]: ...

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
    async def get_trades(self, symbol: str, limit: int = 100) -> List[Trade]: ...

    @abc.abstractmethod
    async def get_ohlcv(self, symbol: str, interval: str, limit: int = 500) -> List[OHLCV]: ...

    @abc.abstractmethod
    async def get_balance(self) -> Dict[str, float]: ...

    @abc.abstractmethod
    async def create_order(
        self,
        symbol: str,
        side: Literal["buy", "sell"],
        qty: float,
        type: Literal["market", "limit"] = "market",
        price: float | None = None,
    ) -> OrderResult: ...


