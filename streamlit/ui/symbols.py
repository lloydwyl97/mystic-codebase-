from __future__ import annotations

import os
from typing import Any, List, cast

import streamlit as st

from streamlit.ui.data_adapter import fetch_candles


# Preferred ordered list for BinanceUS (top-10 target)
BINANCEUS_ORDERED: List[str] = [
    "BTCUSDT",
    "ETHUSDT",
    "ADAUSDT",
    "BNBUSDT",
    "SOLUSDT",
    "DOGEUSDT",
    "AVAXUSDT",
    "MATICUSDT",
    "LTCUSDT",
    "ATOMUSDT",
]


_st = cast(Any, st)


@_st.cache_data(show_spinner=False, ttl=60)
def get_working_symbols(
    preferred: List[str] | None = None,
    exchange: str = "binanceus",
    interval: str = "1h",
    target_count: int = 10,
) -> List[str]:
    symbols = list(preferred or BINANCEUS_ORDERED)
    working: List[str] = []
    for sym in symbols:
        try:
            res = fetch_candles(exchange=exchange, symbol=sym, interval=interval)
            candles = res.get("candles")
            if candles:
                working.append(sym)
                if len(working) >= target_count:
                    break
        except Exception:
            # Skip on any probe error
            continue
    return working


def ensure_state_defaults() -> None:
    s = _st.session_state
    # Lock to BinanceUS and 1h interval on app start
    s.setdefault("exchange", os.getenv("DISPLAY_EXCHANGE", "binanceus"))
    s["exchange"] = "binanceus"
    s.setdefault("interval", "1h")
    s["interval"] = "1h"
    # Maintain legacy key for compatibility
    s.setdefault("timeframe", "1h")
    s["timeframe"] = "1h"


def render_symbol_strip() -> List[str]:
    ensure_state_defaults()
    with _st.spinner("Loading BinanceUS symbols..."):
        symbols = get_working_symbols()

    # Default current symbol if missing or invalid
    if symbols:
        cur = _st.session_state.get("symbol")
        if not cur or cur not in symbols:
            _st.session_state["symbol"] = symbols[0]

    cols = _st.columns(max(1, len(symbols)))
    for i, sym in enumerate(symbols):
        label = sym
        is_selected = _st.session_state.get("symbol") == sym
        btn_kwargs = {"type": "primary"} if is_selected else {}
        with cols[i]:
            if _st.button(label, key=f"symbtn_{sym}", use_container_width=True, **btn_kwargs):
                _st.session_state["exchange"] = "binanceus"
                _st.session_state["symbol"] = sym
                _st.session_state["interval"] = "1h"
                _st.session_state["timeframe"] = "1h"
                _st.rerun()

    return symbols


