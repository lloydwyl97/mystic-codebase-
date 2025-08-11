from __future__ import annotations

import os
import sys
import shutil
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import streamlit as st

# Ensure project root is importable so `dashboard` resolves
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.append(_ROOT)

# Canonical components root: components.*
from streamlit.data_client import (
    get_autobuy_status,
    get_autobuy_signals,
    start_autobuy,
    stop_autobuy,
    get_health_check,
    get_ai_heartbeat,
    get_autobuy_decision,
)
from streamlit.state import get_app_state, render_sidebar_controls
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
    st.set_page_config(page_title="Signals & Autobuy", layout="wide")
    # Unified sidebar controls bound to session state
    render_sidebar_controls()
    state = get_app_state()
    sym = str(state["symbol"])  # normalized dash
    tf = str(state["timeframe"])  # echo below

    # Context echo to ensure no desync with Markets
    st.caption(f"Context • Symbol: {sym} • Timeframe: {tf}")

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

    with display_guard("Autobuy Status"):
        hb = get_autobuy_status()
        ai_hb = get_ai_heartbeat()
        st.caption("Autobuy status")
        st.json(hb.data)
    try:
        running = bool(ai_hb.data.get("running")) if isinstance(ai_hb.data, dict) else False
        last_dec = ai_hb.data.get("last_decision_ts") if isinstance(ai_hb.data, dict) else None
        st.markdown(f"AI Heartbeat: {'✅ Running' if running else '❌ Idle'} • Last decision: {last_dec or '—'}")
    except Exception:
        pass

    with display_guard("Signals Table"):
        sig = get_autobuy_signals(100)
    from typing import Any, Dict, List, cast
    signals_raw: Any = sig.data
    if isinstance(signals_raw, dict):
        signals_raw = cast(Dict[str, Any], signals_raw).get("signals", signals_raw)
    signals: List[Dict[str, Any]] = cast(List[Dict[str, Any]], signals_raw or [])
    if signals:
        sdf = pd.DataFrame(signals)
        if "symbol" in sdf.columns:
            sdf = sdf[sdf["symbol"].astype(str).str.upper() == sym.upper()]
        st.dataframe(sdf.tail(100), use_container_width=True)
    else:
        st.info("Live: no data yet")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Start Autobuy", use_container_width=True):
            if start_autobuy():
                st.rerun()
    with c2:
        if st.button("Stop Autobuy", use_container_width=True):
            if stop_autobuy():
                st.rerun()

    with st.expander("Decision Trace (fetch → features → decision → execution)"):
        if st.button("Run Decision Trace", key="run_decision_trace"):
            try:
                res = get_autobuy_decision(sym)
                st.write({
                    "latency_ms": res.latency_ms,
                    "cache_age_s": res.cache_age_s,
                    "payload_size": len(str(res.data)) if res.data is not None else 0,
                })
                st.json(res.data)
            except Exception as e:  # noqa: BLE001
                st.warning(f"Decision trace failed: {e}")

    with st.expander("Debug"):
        payload_size = len(str(sig.data)) if sig.data is not None else 0
        st.write({
            "autobuy_status_latency_ms": hb.latency_ms,
            "signals_latency_ms": sig.latency_ms,
            "signals_payload_size": payload_size,
            "signals_cache_age_s": sig.cache_age_s,
        })


if __name__ == "__main__":
    main()


