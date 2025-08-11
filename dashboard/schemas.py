from __future__ import annotations

from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field, model_validator


class Ticker(BaseModel):
    price: float = Field(..., ge=0)
    exchange: Optional[str]
    bid: Optional[float]
    ask: Optional[float]
    timestamp: Optional[str]
    volume_24h: Optional[float]
    change_24h: Optional[float]


class Candle(BaseModel):
    timestamp: Any
    open: float
    high: float
    low: float
    close: float
    volume: Optional[float]


class OHLCV(BaseModel):
    data: Optional[Dict[str, List[Any]]] = None
    candles: Optional[List[Candle]] = None


class OrderBookLevel(BaseModel):
    price: float
    size: float


class OrderBook(BaseModel):
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]
    timestamp: Optional[str]


class Trade(BaseModel):
    trade_id: Optional[str]
    price: float
    size: Optional[float]
    side: Optional[str]
    timestamp: Optional[str]


class Balance(BaseModel):
    asset: str
    free: float
    locked: Optional[float]


class AIHeartbeat(BaseModel):
    running: bool
    strategies_active: int = 0
    last_decision_ts: Optional[str] = None
    queue_depth: Optional[int] = None


class AlertItem(BaseModel):
    id: Optional[str]
    title: Optional[str]
    message: str
    level: Optional[str]
    timestamp: Optional[str]


def truncate_snippet(payload: Any, max_len: int = 300) -> str:
    try:
        import json

        s = json.dumps(payload, default=str)[: max_len + 3]
        return s if len(s) <= max_len else s[:max_len] + "..."
    except Exception:
        s = str(payload)
        return s if len(s) <= max_len else s[:max_len] + "..."


