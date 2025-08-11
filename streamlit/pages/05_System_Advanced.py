from __future__ import annotations

import os
import sys
import shutil
from pathlib import Path
import streamlit as st
from datetime import datetime, timezone

# Ensure project root is importable so `dashboard` package resolves
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.append(_ROOT)

# Canonical components root: components.*
from streamlit.data_client import system_health, advanced_events, advanced_performance, get_health_check
from streamlit.state import render_sidebar_controls
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
    st.set_page_config(page_title="System / Advanced Tech", layout="wide")
    # Unified sidebar controls bound to session state
    render_sidebar_controls()

    # Top banner status pills for consistency
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

    with display_guard("Health"):
        h = system_health()
        st.subheader("Health")
        st.json(h.data)

    with display_guard("Events & Alerts"):
        e = advanced_events()
        st.subheader("Events & Alerts")
        st.json(e.data)

    with display_guard("Performance"):
        p = advanced_performance()
        st.subheader("Performance")
        st.json(p.data)

    with st.expander("Debug"):
        st.write({
            "health_latency_ms": h.latency_ms,
            "events_latency_ms": e.latency_ms,
            "performance_latency_ms": p.latency_ms,
            "health_payload_size": len(str(h.data)) if h.data is not None else 0,
            "events_payload_size": len(str(e.data)) if e.data is not None else 0,
            "performance_payload_size": len(str(p.data)) if p.data is not None else 0,
            "health_cache_age_s": h.cache_age_s,
            "events_cache_age_s": e.cache_age_s,
            "performance_cache_age_s": p.cache_age_s,
        })


if __name__ == "__main__":
    main()


