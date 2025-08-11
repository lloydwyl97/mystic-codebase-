from __future__ import annotations

from typing import Dict, List, Any, cast
import streamlit as st


# Attempt to use backend mapping for top-4 symbols per exchange
def _load_exchange_top4() -> Dict[str, List[str]]:
    try:
        import sys
        import os
        backend_root = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "backend")
        backend_root = os.path.abspath(backend_root)
        if backend_root not in sys.path:
            sys.path.append(backend_root)
        from utils.symbols import EXCHANGE_TOP4 as BACKEND_TOP4  # type: ignore[assignment]
        # ensure concrete typing using explicit loop to satisfy type checkers
        out: Dict[str, List[str]] = {}
        items_fn: Any = getattr(BACKEND_TOP4, "items", None)  # type: ignore[no-untyped-call]
        if callable(items_fn):
            for k_any, v_any in items_fn():  # type: ignore[misc]
                key_str = str(k_any)
                try:
                    vals_iter = list(v_any)
                except Exception:
                    vals_iter = []
                out[key_str] = [str(x) for x in vals_iter]
        return out
    except Exception:
        return {
            "coinbase": ["BTC-USD", "ETH-USD", "ADA-USD", "SOL-USD"],
            "binanceus": ["BTC-USD", "ETH-USD", "ADA-USD", "SOL-USD"],
            "kraken": ["BTC-USD", "ETH-USD", "ADA-USD", "SOL-USD"],
        }


EXCHANGES: List[str] = [
    "coinbase",
    "binanceus",
    "kraken",
]

EXCHANGE_TOP4: Dict[str, List[str]] = _load_exchange_top4()


def _normalize_symbol(symbol: str) -> str:
    s = symbol.upper().replace("/", "-")
    if s.endswith("-USDT"):
        s = s.replace("-USDT", "-USD")
    return s


def get_app_state() -> Dict[str, object]:
    if "exchange" not in st.session_state:
        st.session_state.exchange = "coinbase"
    if "symbol" not in st.session_state:
        st.session_state.symbol = EXCHANGE_TOP4.get(st.session_state.exchange, ["BTC-USD"])[0]
    if "timeframe" not in st.session_state:
        st.session_state.timeframe = "1h"
    if "refresh_sec" not in st.session_state:
        st.session_state.refresh_sec = 3
    if "live_mode" not in st.session_state:
        st.session_state.live_mode = True
    return {
        "exchange": st.session_state.exchange,
        "symbol": st.session_state.symbol,
        "timeframe": st.session_state.timeframe,
        "refresh_sec": int(st.session_state.refresh_sec),
        "live_mode": bool(st.session_state.live_mode),
    }


def set_exchange(value: str) -> None:
    value = value.lower()  # type: ignore[assignment]
    if value not in EXCHANGES:  # type: ignore[operator]
        return
    if st.session_state.get("exchange") != value:
        st.session_state.exchange = value
        # Reset symbol to first top-4 for this exchange
        st.session_state.symbol = EXCHANGE_TOP4.get(value, ["BTC-USD"])[0]
        st.rerun()


def set_symbol(value: str) -> None:
    sym = _normalize_symbol(value)
    allowed = EXCHANGE_TOP4.get(st.session_state.get("exchange", "coinbase"), [])
    if sym in allowed and st.session_state.get("symbol") != sym:
        st.session_state.symbol = sym
        st.rerun()


def set_timeframe(value: str) -> None:
    allowed: List[str] = ["1m", "5m", "15m", "1h", "4h", "1d"]
    val = value if value in allowed else st.session_state.get("timeframe", "1h")
    if st.session_state.get("timeframe") != val:
        st.session_state.timeframe = val
        st.rerun()


def set_refresh_sec(value: int) -> None:
    v = max(1, min(int(value), 10))
    st.session_state.refresh_sec = v


def set_live_mode(value: bool) -> None:
    st.session_state.live_mode = bool(value)


def render_sidebar_controls() -> None:
    """Render unified sidebar controls bound to session state.
    Uses `EXCHANGE_TOP4` mapping; triggers safe rerun on changes.
    """
    state = get_app_state()
    st.sidebar.markdown("**Global Controls**")

    # Exchange selector
    exch_list = list(EXCHANGES)
    cur_ex = str(state["exchange"])
    try:
        cur_idx = exch_list.index(cur_ex)
    except ValueError:
        cur_idx = 0
    sel_ex = st.sidebar.selectbox("Exchange", exch_list, index=cur_idx)
    if sel_ex != cur_ex:
        set_exchange(sel_ex)  # safe rerun inside setter
        return

    # Symbol selector from mapping
    symbols = EXCHANGE_TOP4.get(sel_ex, ["BTC-USD"]) or ["BTC-USD"]
    cur_sym = str(state["symbol"])
    try:
        sym_idx = symbols.index(cur_sym)
    except ValueError:
        sym_idx = 0
    sel_sym = st.sidebar.selectbox("Symbol", symbols, index=sym_idx)
    if sel_sym != cur_sym:
        set_symbol(sel_sym)  # safe rerun inside setter
        return

    # Optional icon preview
    try:
        # Local import to avoid hard dependency during module import
        try:
            from .icons import get_coin_icon, render_text_badge  # type: ignore
        except Exception:
            from streamlit.icons import get_coin_icon, render_text_badge  # type: ignore
        icon_path = get_coin_icon(sel_sym)
        if icon_path:
            st.sidebar.image(icon_path, width=28)
        else:
            render_text_badge(sel_sym, size=32)
    except Exception:
        pass

    # Timeframe and refresh controls
    tfs: List[str] = ["1m", "5m", "15m", "1h", "4h", "1d"]
    cur_tf = str(state["timeframe"])
    try:
        tf_idx = tfs.index(cur_tf)
    except ValueError:
        tf_idx = 3
    sel_tf = st.sidebar.selectbox("Timeframe", tfs, index=tf_idx)
    if sel_tf != cur_tf:
        set_timeframe(sel_tf)
        return

    live_t = st.sidebar.toggle("Live Refresh", value=bool(state["live_mode"]))
    if bool(live_t) != bool(state["live_mode"]):
        set_live_mode(bool(live_t))

    cur_ref = int(cast(int, state["refresh_sec"]))
    # Slider returns int; guard for type checkers
    value_raw = st.sidebar.slider("Refresh interval (s)", min_value=1, max_value=10, value=int(cur_ref), step=1)
    new_ref_int: int = int(value_raw)
    if new_ref_int != cur_ref:
        set_refresh_sec(new_ref_int)

