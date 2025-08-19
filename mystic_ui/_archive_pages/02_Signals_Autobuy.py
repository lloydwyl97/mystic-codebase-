from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any, cast

import pandas as pd
import streamlit as st

from mystic_ui._archive_pages.components.common_utils import (  # public wrapper
    get_app_state,
    inject_global_theme,  # public wrapper
    render_sidebar_controls,
    safe_number_format,  # public wrapper
)
from mystic_ui.api_client import request_json as _req

# Silence type checker for Streamlit's dynamic attributes
_st = cast(Any, st)


def _extract_signals(payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    data: Any = payload.get("signals", payload)
    if isinstance(data, list):
        return cast(list[dict[str, Any]], data)
    return []


def _extract_status_ok(payload: dict[str, Any] | None) -> bool:
    try:
        if not isinstance(payload, dict):
            return False
        ok_flag = bool(payload.get("ok"))
        healthy_flag = bool(payload.get("healthy"))
        status_val: Any = payload.get("status")
        status_ok = False
        if isinstance(status_val, str):
            status_ok = status_val.lower() in ("ok", "healthy", "up", "ready")
        elif isinstance(status_val, bool):
            status_ok = status_val is True
        return ok_flag or healthy_flag or status_ok
    except Exception:
        return False


def main() -> None:
    # set_page_config is centralized in mystic_ui/app.py
    inject_global_theme()

    # Sidebar controls and shared state
    render_sidebar_controls()
    s: MutableMapping[str, Any] = get_app_state()

    # Fetch data
    with _st.spinner("Loading signals and autobuy status…"):
        try:
            signals_payload_any = _req("GET", "/api/autobuy/signals", params={"limit": 100})
        except Exception:
            signals_payload_any = None
        try:
            autobuy_payload_any = _req("GET", "/api/autobuy/status")
        except Exception:
            autobuy_payload_any = None

    signals_payload: dict[str, Any] | None = cast(dict[str, Any] | None, signals_payload_any if isinstance(signals_payload_any, dict) else None)
    autobuy_payload: dict[str, Any] | None = cast(dict[str, Any] | None, autobuy_payload_any if isinstance(autobuy_payload_any, dict) else None)

    if not signals_payload:
        _st.info("Signals unavailable")

    if not autobuy_payload:
        _st.info("Autobuy status unavailable")

    signals: list[dict[str, Any]] = _extract_signals(signals_payload)
    unique_symbols = len({str(x.get("symbol", "")).upper() for x in signals}) if signals else 0
    total_signals = len(signals)
    limit_val = 0
    try:
        if isinstance(signals_payload, dict):
            limit_val = int(signals_payload.get("limit") or 0)
    except Exception:
        limit_val = 0

    # Header & summary metrics (clean cards)
    _st.subheader("Live Signals & Autobuy")
    m1, m2, m3, m4 = _st.columns(4)
    with m1:
        _st.metric("Total Signals", safe_number_format(total_signals, 0))
    with m2:
        _st.metric("Unique Symbols", safe_number_format(unique_symbols, 0))
    with m3:
        _st.metric("Limit", safe_number_format(limit_val, 0))
    with m4:
        ok = _extract_status_ok(autobuy_payload)
        _st.metric("Autobuy Status", "OK" if ok else "Check")

    # Signals list → clean table
    _st.markdown("### Signals")
    if signals:
        # Optional: filter by selected symbol before DataFrame creation to avoid pandas typing issues
        selected_symbol = str(s.get("symbol", "")).upper()
        if selected_symbol:
            filtered_signals: list[dict[str, Any]] = [
                row for row in signals
                if str(row.get("symbol", "")).upper() == selected_symbol
            ]
        else:
            filtered_signals = signals
        df = pd.DataFrame(filtered_signals)
        # Stable column order if common fields exist
        preferred_cols = [
            "timestamp", "symbol", "action", "confidence", "price", "reason", "strategy"
        ]
        cols = [c for c in preferred_cols if c in df.columns] + [
            c for c in df.columns if c not in preferred_cols
        ]
        df = df[cols]
        _st.dataframe(df.tail(200), use_container_width=True)
    else:
        _st.info("No signals yet.")

    # Controls (optional start/stop could live elsewhere; keeping display read-only here)

    # Raw JSON expanders (exact API payloads only here)
    with _st.expander("Raw JSON: Signals", expanded=False):
        _st.json(signals_payload)
    with _st.expander("Raw JSON: Autobuy", expanded=False):
        _st.json(autobuy_payload)


if __name__ == "__main__":
    main()



