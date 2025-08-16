from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple, Dict, Any


@dataclass
class Ticker:
    exchange: str
    symbol: str
    price: float
    bid: Optional[float]
    ask: Optional[float]
    ts: int


@dataclass
class OrderBook:
    exchange: str
    symbol: str
    bids: List[Tuple[float, float]]
    asks: List[Tuple[float, float]]
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
    id: Optional[str]
    fill_price: Optional[float]
    ts: int
    raw: Dict[str, Any]




