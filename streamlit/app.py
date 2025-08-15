import os, sys, streamlit as st
from typing import Any, cast

# Self-bootstrap sys.path so all pages can import project modules
_ST   = os.path.abspath(os.path.dirname(__file__))      # .../streamlit
ROOT  = os.path.abspath(os.path.join(_ST, ".."))        # repo root
if ROOT not in sys.path: sys.path.insert(0, ROOT)
if _ST  not in sys.path: sys.path.insert(0, _ST)

from streamlit.ui.theme import inject_global_theme  # inject Coinbase-dark theme first
inject_global_theme()

# Use a lax-typed handle to avoid stub limitations in linter
_st = cast(Any, st)

# Force collapsed sidebar and wide layout
_st.set_page_config(page_title="Mystic Super Dashboard — BinanceUS", layout="wide", initial_sidebar_state="collapsed")  # type: ignore[attr-defined]

# Hide Streamlit's sidebar navigation and sidebar toggle UI
_st.markdown(
    """
    <style>
    /* Hide the pages navigation and the entire sidebar area */
    [data-testid="stSidebar"], [data-testid="stSidebarNav"], [data-testid="stSidebarHeader"], [data-testid="collapsedControl"] { display: none !important; }
    /* Pull main container full width since sidebar is gone */
    .block-container { padding-top: 10px !important; }
    /* Top bar styling */
    .topbar { display: flex; align-items: center; justify-content: space-between; gap: 16px; padding: 10px 12px; background: var(--bg-soft); border: 1px solid rgba(255,255,255,0.06); border-radius: 12px; }
    .topbar-left { display: flex; align-items: center; gap: 10px; }
    .topbar-center { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; justify-content: center; }
    .topbar-right { display: flex; align-items: center; gap: 12px; }
    .app-title { font-size: 18px; font-weight: 700; letter-spacing: 0.2px; }
    .pill { display:inline-flex; align-items:center; gap:6px; padding:4px 10px; border-radius:999px; background: rgba(22,82,240,0.12); color:#CFE0FF; border:1px solid rgba(22,82,240,0.35); font-weight:700; font-size:12.5px; }
    .dot { width:10px; height:10px; border-radius:999px; display:inline-block; }
    .dot.ok { background:#16C784; box-shadow:0 0 0 3px rgba(22,199,132,0.18); }
    .dot.warn { background:#F59E0B; box-shadow:0 0 0 3px rgba(245,158,11,0.18); }
    .dot.bad { background:#FF5A5F; box-shadow:0 0 0 3px rgba(255,90,95,0.18); }
    .label { color: var(--text-dim); font-weight: 600; font-size: 12.5px; }
    .spacer { width: 1px; height: 22px; background: rgba(255,255,255,0.08); }
    .symbol-btn > button { padding: 6px 10px !important; font-size: 13px !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Lock display exchange for all pages
_st.session_state["exchange"] = os.getenv("DISPLAY_EXCHANGE", "binanceus")  # type: ignore[attr-defined]

# Initialize symbol/timeframe defaults
from backend.config.coins import FEATURED_SYMBOLS, FEATURED_EXCHANGE  # type: ignore[attr-defined]

if "symbol" not in _st.session_state:
    _st.session_state["symbol"] = os.getenv("DISPLAY_SYMBOL", (FEATURED_SYMBOLS[0] if FEATURED_SYMBOLS else "BTCUSDT"))  # type: ignore[attr-defined]
if "timeframe" not in _st.session_state:
    _st.session_state["timeframe"] = os.getenv("DISPLAY_TIMEFRAME", "1h")  # type: ignore[attr-defined]

from streamlit.api_client import dc_health, dc_autobuy_status, dc_ai_heartbeat, dc_get_prices  # type: ignore[attr-defined]

health = dc_ai_heartbeat()  # AI heartbeat for AI status
autobuy = dc_autobuy_status()
sys_health = dc_health()

from typing import Mapping, Sequence

def _status_class(val: object) -> str:
    try:
        if isinstance(val, Mapping):
            d = cast(Mapping[str, Any], val)
            ok_flag = bool(d.get("ok"))
            healthy_flag = bool(d.get("healthy"))
            running_flag = bool(d.get("running"))
            status_flag = d.get("status")
            status_ok = False
            if isinstance(status_flag, str):
                status_ok = status_flag.lower() in ("ok", "healthy", "up")
            elif isinstance(status_flag, bool):
                status_ok = status_flag is True
            ok = ok_flag or healthy_flag or status_ok or running_flag
            return "ok" if ok else "bad"
        if isinstance(val, Sequence) and not isinstance(val, (str, bytes, bytearray)):
            seq = cast(Sequence[Any], val)
            return "ok" if len(seq) > 0 else "warn"
        return "ok" if bool(val) else "warn"
    except Exception:
        return "warn"

"""Top header toolbar (main body)"""
container = _st.container()
with container:
    left, center, right = _st.columns([3, 6, 3])

    with left:
        _st.markdown(
            """
            <div class="topbar-left">
              <div class="app-title">Mystic Super Dashboard</div>
              <span class="pill">BinanceUS</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with center:
        # Top-10 symbol strip (buttons)
        top_symbols = [s for s in FEATURED_SYMBOLS][:10] if FEATURED_SYMBOLS else ["BTCUSDT", "ETHUSDT"]
        cols = _st.columns(len(top_symbols)) if top_symbols else _st.columns(1)
        for i, sym in enumerate(top_symbols):
            with cols[i]:
                if _st.button(sym, key=f"sym_{sym}"):
                    _st.session_state["symbol"] = sym

    with right:
        # Timeframe selector with label "1h (server)"
        _st.caption("1h (server)")
        tf_options = ["1m", "5m", "15m", "1h", "4h", "1d"]
        try:
            current_idx = tf_options.index(str(_st.session_state.get("timeframe", "1h")))
        except ValueError:
            current_idx = 3
        tf = _st.selectbox("Timeframe", tf_options, index=current_idx, key="timeframe_select")
        _st.session_state["timeframe"] = tf

        # AI / Autobuy status dots
        h_cls = _status_class(health.get("data"))  # AI heartbeat
        a_cls = _status_class(autobuy.get("data"))
        _st.markdown(
            f"<span class='label'>AI</span> <span class='dot {h_cls}'></span>  &nbsp;&nbsp;"
            f"<span class='label'>Autobuy</span> <span class='dot {a_cls}'></span>",
            unsafe_allow_html=True,
        )

        # Small "Server Info" dropdown
        with _st.expander("Server Info", expanded=False):
            from os import environ as _env
            api_url = _env.get("MYSTIC_BACKEND", "http://127.0.0.1:9000")
            # minimal, useful info including last success/error timestamps if present
            ai_data = cast(Any, health).get("data")
            ab_data = cast(Any, autobuy).get("data")
            sh_data = cast(Any, sys_health).get("data")
            info: dict[str, Any] = {
                "backend": api_url,
                "last_success": (
                    cast(Any, ai_data).get("timestamp") if isinstance(ai_data, dict) and cast(Any, ai_data).get("timestamp") else None
                ),
                "last_error": (
                    cast(Any, sh_data).get("error") if isinstance(sh_data, dict) and cast(Any, sh_data).get("error") else None
                ),
                "ai": ai_data,
                "autobuy": ab_data,
                "exchange": _st.session_state.get("exchange"),
                "timeframe": _st.session_state.get("timeframe"),
            }
            _st.code(info)

# Optional: price strip under toolbar for selected symbol
try:
    _prices = dc_get_prices([_st.session_state.get("symbol", "BTCUSDT")], exchange=_st.session_state.get("exchange"))
    if _prices.get("data"):
        _st.caption(f"{_st.session_state.get('symbol')} price: {_prices['data']}")
except Exception:
    pass

