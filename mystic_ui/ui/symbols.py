from __future__ import annotations

import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple, cast

import streamlit as st

from streamlit.ui.data_adapter import fetch_candles
from streamlit.top10_resolver import resolve_top10


# Legacy fallback list retained but unused by default; kept to satisfy "never remove unless it's upgraded"
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

# ----------------- module-level smart cache (stale-while-revalidate) -----------------
_CACHE_TTL_S: int = 300  # 5 minutes
_symbols_cache: Dict[str, Any] = {"data": None, "ts": 0.0}
_refresh_lock = threading.Lock()
_refreshing: bool = False


def _parse_force_symbols_env() -> List[str]:
    raw = os.getenv("FORCE_SYMBOLS", "").strip()
    if not raw:
        return []
    items = [s.strip() for s in raw.split(",") if s.strip()]
    # Normalize to upper with dash USD if omitted quote is a USD variant handled by adapter
    return [s.upper() for s in items]


def _compute_working_symbols(
    preferred: Optional[List[str]] = None,
    exchange: str = "binanceus",
    interval: str = "1h",
    target_count: int = 10,
) -> Tuple[List[str], List[str]]:
    """Compute validated working symbols and return (working, dropped_forced).

    - Honors FORCE_SYMBOLS exclusively when set; validates via real candles.
    - Otherwise resolves Top 10 dynamically with fallback to preferred/legacy list.
    """
    force_list = _parse_force_symbols_env()
    dropped_forced: List[str] = []

    # Source candidate list
    if force_list:
        candidates = force_list
    else:
        candidates = resolve_top10(exchange=exchange, timeframe="1m") or list(preferred or BINANCEUS_ORDERED)

    working: List[str] = []
    for sym in candidates:
        try:
            res = fetch_candles(exchange=exchange, symbol=sym, interval="1m")
            candles = res.get("candles")
            if candles:
                working.append(sym)
                if not force_list and len(working) >= target_count:
                    break
            else:
                if force_list:
                    dropped_forced.append(sym)
        except Exception:
            if force_list:
                dropped_forced.append(sym)
            continue

    # When FORCE_SYMBOLS is set, do not auto-trim to target_count; keep all validated
    if not force_list and len(working) > target_count:
        working = working[:target_count]

    return working, dropped_forced


def _refresh_cache_async(
    preferred: Optional[List[str]], exchange: str, interval: str, target_count: int
) -> None:
    global _refreshing
    with _refresh_lock:
        if _refreshing:
            return
        _refreshing = True
    try:
        data, dropped = _compute_working_symbols(preferred, exchange, interval, target_count)
        _symbols_cache["data"] = data
        _symbols_cache["ts"] = time.time()
        # Store dropped list for optional user warning on next render
        _symbols_cache["dropped_forced"] = dropped
    finally:
        _refreshing = False


def get_working_symbols(
    preferred: List[str] | None = None,
    exchange: str = "binanceus",
    interval: str = "1h",
    target_count: int = 10,
) -> List[str]:
    now = time.time()
    cached = _symbols_cache.get("data")
    ts = float(_symbols_cache.get("ts") or 0.0)
    age = now - ts if ts else None

    # Fresh cache
    if cached and age is not None and age < _CACHE_TTL_S:
        return cast(List[str], cached)

    # Stale cache: trigger background refresh and return stale immediately
    if cached:
        threading.Thread(
            target=_refresh_cache_async,
            args=(preferred, exchange, interval, target_count),
            daemon=True,
        ).start()
        return cast(List[str], cached)

    # No cache yet: compute synchronously
    data, dropped = _compute_working_symbols(preferred, exchange, interval, target_count)
    _symbols_cache["data"] = data
    _symbols_cache["ts"] = now
    _symbols_cache["dropped_forced"] = dropped
    return data


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
    forced = _parse_force_symbols_env()
    spinner_msg = (
        "Loading forced symbols (validated via candles)..."
        if forced
        else "Loading BinanceUS Top 10 (BinanceUS)..."
    )
    with _st.spinner(spinner_msg):
        symbols = get_working_symbols()

    # If FORCE_SYMBOLS was provided, warn about any dropped items
    if forced:
        dropped: List[str] = cast(List[str], _symbols_cache.get("dropped_forced") or [])
        if dropped:
            _st.warning(
                f"Dropped forced symbols with no working candles: {', '.join(dropped)}",
                icon="⚠️",
            )

    # Default current symbol if missing or invalid
    if symbols:
        cur = _st.session_state.get("symbol")
        if not cur or cur not in symbols:
            _st.session_state["symbol"] = symbols[0]

    left, right = _st.columns([3, 1])
    with left:
        cur_sym = _st.session_state.get("symbol")
        idx = symbols.index(cur_sym) if symbols and cur_sym in symbols else 0
        selected = _st.selectbox(
            "Symbol",
            symbols,
            index=idx,
            key="symbol_select",
            help="Top 10 auto-resolved from BinanceUS by 24h volume",
        ) if symbols else None
        if selected and _st.session_state.get("symbol") != selected:
            _st.session_state["exchange"] = "binanceus"
            _st.session_state["symbol"] = selected
            _st.session_state["interval"] = "1h"
            _st.session_state["timeframe"] = "1h"
            _st.rerun()
    with right:
        if _st.button("↻ Refresh Top 10", key="refresh_top10", use_container_width=True):
            try:
                resolve_top10(exchange="binanceus", timeframe="1m", force_refresh=True)
            finally:
                # Clear our local cache and re-run; background refresh will repopulate
                _symbols_cache["data"] = None
                _symbols_cache["ts"] = 0.0
                _st.rerun()

    # Debug expander: shows accepted Top 10 with volumes and rejected with reasons
    with _st.expander("Debug / Top 10 Discovery", expanded=False):
        try:
            dbg_any = resolve_top10(exchange="binanceus", timeframe="1m", debug=True)
            debug: Dict[str, Any] = cast(Dict[str, Any], dbg_any)
            accepted = cast(List[Dict[str, Any]], debug.get("accepted", []))
            rejected = cast(List[Dict[str, Any]], debug.get("rejected", []))

            _st.subheader("Accepted (Top 10)")
            if accepted:
                acc_rows = [
                    {
                        "Rank": a.get("rank"),
                        "Symbol": a.get("symbol"),
                        "24h Quote Vol": round(float(a.get("quote_volume", 0.0)), 2),
                        "24h Base Vol": round(float(a.get("base_volume", 0.0)), 2),
                    }
                    for a in accepted
                ]
                _st.table(acc_rows)
            else:
                _st.caption("No accepted symbols.")

            _st.subheader("Rejected")
            if rejected:
                rej_rows = [
                    {
                        "Symbol": r.get("symbol"),
                        "Reason": r.get("reason"),
                    }
                    for r in rejected
                ]
                _st.table(rej_rows)
            else:
                _st.caption("No rejections recorded.")

            if err := debug.get("http_error"):
                _st.error(str(err))
        except Exception as e:
            _st.exception(e)

    return symbols


