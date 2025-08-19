from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Ticker(BaseModel):
    price: float = Field(..., ge=0)
    exchange: str | None
    bid: float | None
    ask: float | None
    timestamp: str | None
    volume_24h: float | None
    change_24h: float | None


class Candle(BaseModel):
    timestamp: Any
    open: float
    high: float
    low: float
    close: float
    volume: float | None


class OHLCV(BaseModel):
    data: dict[str, list[Any]] | None = None
    candles: list[Candle] | None = None


class OrderBookLevel(BaseModel):
    price: float
    size: float


class OrderBook(BaseModel):
    bids: list[OrderBookLevel]
    asks: list[OrderBookLevel]
    timestamp: str | None


class Trade(BaseModel):
    trade_id: str | None
    price: float
    size: float | None
    side: str | None
    timestamp: str | None


class Balance(BaseModel):
    asset: str
    free: float
    locked: float | None


class AIHeartbeat(BaseModel):
    running: bool
    strategies_active: int = 0
    last_decision_ts: str | None = None
    queue_depth: int | None = None


class AlertItem(BaseModel):
    id: str | None
    title: str | None
    message: str
    level: str | None
    timestamp: str | None


def truncate_snippet(payload: Any, max_len: int = 300) -> str:
    try:
        import json

        s = json.dumps(payload, default=str)[: max_len + 3]
        return s if len(s) <= max_len else s[:max_len] + "..."
    except Exception:
        s = str(payload)
        return s if len(s) <= max_len else s[:max_len] + "..."


