from __future__ import annotations

import os
import sys
import shutil
from pathlib import Path
import pandas as pd
from datetime import datetime, timezone
import plotly.graph_objects as go  # type: ignore[import-not-found]
import streamlit as st

# Ensure project root is importable so `dashboard` resolves
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.append(_ROOT)

# Canonical components root: components.*
from streamlit.data_client import (
    get_prices,
    get_ohlcv,
    get_trades,
    compute_spread_from_price_entry,
    get_health_check,
    get_ai_heartbeat,
)
from streamlit.state import get_app_state, render_sidebar_controls
from streamlit.icons import get_coin_icon, render_text_badge  # noqa: F401
from streamlit.ui_guard import display_guard

# Backup
_p = Path(__file__)
_bak = _p.with_suffix(_p.suffix + ".bak_cursor")
try:
    if not _bak.exists():
        shutil.copyfile(_p, _bak)
except Exception:
    pass


def main() -> None:
    st.set_page_config(page_title="Markets", layout="wide")
    # Unified sidebar controls bound to session state
    render_sidebar_controls()
    state = get_app_state()
    sym = str(state["symbol"])  # normalized dash
    # Icon badge for selected symbol
    with display_guard("Symbol Icon"):
        icon_path = get_coin_icon(sym)
        ic_col, _ = st.columns([1, 5])
        with ic_col:
            if icon_path:
                st.image(icon_path, width=32)
            else:
                render_text_badge(sym, size=32)
    # Top banner status pills
    try:
        hc = get_health_check()
        from typing import Any, Dict, List, cast

        hdata: Dict[str, Any] = cast(Dict[str, Any], hc.data) if isinstance(hc.data, dict) else {}
        adapters: List[str] = [str(x) for x in cast(List[Any], hdata.get("adapters", []))] if isinstance(hdata.get("adapters"), list) else []
        autobuy_state: str = str(hdata.get("autobuy", ""))
        sys_state: str = str(hdata.get("status", ""))
        status_map = {
            "CB": "✅" if "coinbase" in adapters else "⚠️",
            "BUS": "✅" if "binanceus" in adapters else "⚠️",
            "KRA": "✅" if "kraken" in adapters else "⚠️",
            "CGK": "✅" if "coingecko" in adapters else "⚠️",
            "AI": "✅" if autobuy_state == "ready" else ("⚠️" if autobuy_state else "❌"),
            "SYS": "✅" if sys_state == "ok" else ("⚠️" if sys_state else "❌"),
        }
        pills = " ".join([f"<span style='padding:4px 8px;border-radius:12px;background:#222;color:#eee;margin-right:6px'>{k} {v}</span>" for k, v in status_map.items()])
        st.markdown(pills, unsafe_allow_html=True)
        st.caption(f"Last refresh: {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}")
    except Exception:
        pass

    # AI status chip
    try:
        ai = get_ai_heartbeat()
        running = bool(ai.data.get("running")) if isinstance(ai.data, dict) else False
        last_ts = ai.data.get("last_decision_ts") if isinstance(ai.data, dict) else None
        st.markdown(
            f"<div style='margin:6px 0;padding:6px 10px;display:inline-block;border-radius:12px;background:{'#154' if running else '#441'};color:#eee'>AI: {'Running' if running else 'Idle'} • {last_ts or '—'}</div>",
            unsafe_allow_html=True,
        )
    except Exception:
        pass

    with display_guard("Prices & KPIs"):
        pr = get_prices([sym])
    entry: Dict[str, Any] = {}
    if isinstance(pr.data, dict):
        from typing import Any, Dict, cast
        prices_obj: Dict[str, Any] = cast(Dict[str, Any], pr.data.get("prices", pr.data))
        if isinstance(prices_obj, dict):
            entry = cast(Dict[str, Any], prices_obj.get(sym, {}))

    price_val = float(entry.get("price", 0) or 0)
    vol_val = float(entry.get("volume_24h", 0) or 0)
    chg_val = float(entry.get("change_24h", 0) or 0)
    spread_val = compute_spread_from_price_entry(entry) or 0.0
    last_ts = entry.get("timestamp") or (pr.data.get("timestamp") if isinstance(pr.data, dict) else None)

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.metric("Price", f"${price_val:,.2f}")
    with k2:
        st.metric("24h Change %", f"{chg_val:.2f}%")
    with k3:
        st.metric("Spread", f"${spread_val:,.2f}")
    with k4:
        st.metric("24h Volume", f"{vol_val:,.0f}")
    with k5:
        st.metric("Last Update", str(last_ts) if last_ts else "—")

    with display_guard("OHLCV Chart"):
        candles = get_ohlcv(str(state["exchange"]), sym, str(state["timeframe"]))
    cdata = candles.data or {}
    try:
        ts = cdata.get("data", {}).get("timestamps", []) if isinstance(cdata, dict) else [c.get("timestamp") for c in cdata]
        opens = cdata.get("data", {}).get("opens", []) if isinstance(cdata, dict) else [c.get("open") for c in cdata]
        highs = cdata.get("data", {}).get("highs", []) if isinstance(cdata, dict) else [c.get("high") for c in cdata]
        lows = cdata.get("data", {}).get("lows", []) if isinstance(cdata, dict) else [c.get("low") for c in cdata]
        closes = cdata.get("data", {}).get("closes", []) if isinstance(cdata, dict) else [c.get("close") for c in cdata]
        fig = go.Figure(data=[go.Candlestick(x=ts, open=opens, high=highs, low=lows, close=closes)])
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        st.info("No OHLCV data")

    with display_guard("Recent Trades"):
        tr = get_trades(str(state["exchange"]), sym, 100)
    trades = tr.data or []
    if trades:
        try:
            df = pd.DataFrame(trades)
            st.dataframe(df.tail(100), use_container_width=True)
        except Exception:
            st.json(trades)
    else:
        st.info("Live: no data yet")

    with st.expander("Debug"):
        payload_size = len(str(pr.data)) if pr.data is not None else 0
        st.write({
            "prices_latency_ms": pr.latency_ms,
            "ohlcv_latency_ms": candles.latency_ms,
            "trades_latency_ms": tr.latency_ms,
            "prices_payload_size": payload_size,
            "prices_cache_age_s": pr.cache_age_s,
        })


if __name__ == "__main__":
    main()


