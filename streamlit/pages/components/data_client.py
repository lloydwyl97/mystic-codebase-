from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import streamlit as st

from dashboard import data_client as core
from .state import get_app_state


@dataclass
class FetchResult:
    data: Any
    latency_ms: int
    cached_age_s: Optional[int]


@st.cache_data(show_spinner=False, ttl=2)
def get_prices(symbols: List[str]) -> FetchResult:
    r = core.get_prices(symbols)
    return FetchResult(data=r.data, latency_ms=r.latency_ms, cached_age_s=r.cache_age_s)


@st.cache_data(show_spinner=False, ttl=2)
def get_ohlcv(symbol: str, timeframe: str, limit: int = 300) -> FetchResult:
    state = get_app_state()
    exch = str(state["exchange"])  # type: ignore[index]
    r = core.get_ohlcv(exch, symbol, timeframe)
    return FetchResult(data=r.data, latency_ms=r.latency_ms, cached_age_s=r.cache_age_s)


@st.cache_data(show_spinner=False, ttl=2)
def get_trades(symbol: str, limit: int = 100) -> FetchResult:
    state = get_app_state()
    exch = str(state["exchange"])  # type: ignore[index]
    r = core.get_trades(exch, symbol, limit)
    return FetchResult(data=r.data, latency_ms=r.latency_ms, cached_age_s=r.cache_age_s)


@st.cache_data(show_spinner=False, ttl=2)
def get_orders() -> FetchResult:
    # No direct order endpoint in core client; reuse live trades/orders via backend
    r = core.system_health()
    return FetchResult(data=r.data, latency_ms=r.latency_ms, cached_age_s=r.cache_age_s)


@st.cache_data(show_spinner=False, ttl=2)
def get_positions() -> FetchResult:
    r = core.system_health()
    return FetchResult(data=r.data, latency_ms=r.latency_ms, cached_age_s=r.cache_age_s)


@st.cache_data(show_spinner=False, ttl=2)
def get_balance() -> FetchResult:
    state = get_app_state()
    exch = str(state["exchange"])  # type: ignore[index]
    r = core.get_balances(exch)
    return FetchResult(data=r.data, latency_ms=r.latency_ms, cached_age_s=r.cache_age_s)


@st.cache_data(show_spinner=False, ttl=2)
def get_system_health() -> FetchResult:
    r = core.system_health()
    return FetchResult(data=r.data, latency_ms=r.latency_ms, cached_age_s=r.cache_age_s)


@st.cache_data(show_spinner=False, ttl=2)
def get_health_check() -> FetchResult:
    r = core.get_health_check()
    return FetchResult(data=r.data, latency_ms=r.latency_ms, cached_age_s=r.cache_age_s)


def clear_cache() -> bool:
    # Use system clear-cache if exposed in backend via core (not implemented)
    return False


@st.cache_data(show_spinner=False, ttl=2)
def get_autobuy_status() -> FetchResult:
    r = core.get_autobuy_status()
    return FetchResult(data=r.data, latency_ms=r.latency_ms, cached_age_s=r.cache_age_s)


@st.cache_data(show_spinner=False, ttl=2)
def get_autobuy_signals(limit: int = 50) -> FetchResult:
    r = core.get_autobuy_signals(limit)
    return FetchResult(data=r.data, latency_ms=r.latency_ms, cached_age_s=r.cache_age_s)


def start_autobuy() -> bool:
    return core.start_autobuy()


def stop_autobuy() -> bool:
    return core.stop_autobuy()


def compute_spread_from_price_entry(entry: Dict[str, Any]) -> Optional[float]:
    return core.compute_spread_from_price_entry(entry)


def recent_update_ts(payload: Any) -> Optional[str]:
    try:
        if isinstance(payload, dict) and "timestamp" in payload:
            v = payload.get("timestamp")
            return str(v) if v is not None else None
    except Exception:
        return None
    return None


