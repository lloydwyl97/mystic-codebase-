from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


@dataclass
class Ticker:
    exchange: str
    symbol: str
    price: float
    bid: float | None
    ask: float | None
    ts: int


@dataclass
class OrderBook:
    exchange: str
    symbol: str
    bids: list[tuple[float, float]]
    asks: list[tuple[float, float]]
    ts: int


@dataclass
class Trade:
    exchange: str
    symbol: str
    price: float
    qty: float
    side: Literal["buy", "sell"]
    ts: int


@dataclass
class OHLCV:
    exchange: str
    symbol: str
    ts: int
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class OrderResult:
    exchange: str
    symbol: str
    side: Literal["buy", "sell"]
    qty: float
    type: Literal["market", "limit"]
    status: Literal["submitted", "filled", "rejected", "error"]
    id: str | None
    fill_price: float | None
    ts: int
    raw: dict[str, Any]




