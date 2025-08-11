"""Unified sidebar state for Streamlit dashboard.

This module re-exports state helpers from the legacy `streamlit/pages/components/state.py`
to provide a canonical import path `streamlit.state` across all pages.
"""

from __future__ import annotations

from typing import Any, Dict, List

try:
    # Prefer canonical path within this package
    from streamlit.pages.components.state import (  # type: ignore
        get_app_state,
        set_exchange,
        set_symbol,
        set_timeframe,
        set_refresh_sec,
        set_live_mode,
        EXCHANGES,
        EXCHANGE_TOP4,
        render_sidebar_controls,
    )
except Exception:  # pragma: no cover - safety fallback for import edge cases
    # Fallback to relative legacy path resolution
    from .pages.components.state import (  # type: ignore
        get_app_state,
        set_exchange,
        set_symbol,
        set_timeframe,
        set_refresh_sec,
        set_live_mode,
        EXCHANGES,
        EXCHANGE_TOP4,
        render_sidebar_controls,
    )

__all__ = [
    "get_app_state",
    "set_exchange",
    "set_symbol",
    "set_timeframe",
    "set_refresh_sec",
    "set_live_mode",
    "EXCHANGES",
    "EXCHANGE_TOP4",
    "render_sidebar_controls",
]


