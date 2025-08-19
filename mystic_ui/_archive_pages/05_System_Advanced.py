from __future__ import annotations

import os
import sys
from typing import Any, Dict, Optional, List, cast

import streamlit as st

# Ensure project root is importable so `streamlit.*` package resolves consistently
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.append(_ROOT)

from mystic_ui.api_client import request_json as _req
from mystic_ui._archive_pages.components.common_utils import render_sidebar_controls, display_guard

_st = cast(Any, st)


def _safe_len(x: Any) -> int:
    try:
        return len(x)  # type: ignore[arg-type]
    except Exception:
        return 0


def _display_small_stat(label: str, value: Any, help_text: Optional[str] = None) -> None:
    _st.metric(label, value if value is not None else "N/A", help=help_text)


def main() -> None:
    # set_page_config is centralized in mystic_ui/app.py
    render_sidebar_controls()

    # Grid: Feature Flags | AI Status | Autobuy Status | Server Health Pings
    c1, c2, c3, c4 = _st.columns(4)

    # ---------- Feature Flags ----------
    with display_guard("Feature Flags"):
        with c1:
            _st.subheader("Feature Flags")
            features_payload = cast(Optional[Dict[str, Any]], _req("GET", "/api/features"))
            if isinstance(features_payload, dict) and "__meta__" in features_payload:
                meta = cast(Dict[str, Any], features_payload.get("__meta__") or {})
                route = meta.get("route")
                status = meta.get("status")
                err = meta.get("error")
                msg = "Features unavailable"
                if route or status is not None:
                    msg += f" — {route or ''} {f'({status})' if status is not None else ''}"
                if err:
                    msg += f" • {err}"
                _st.info(msg.strip())
            features_list: List[Any] = []
            enabled_count: int = 0
            if features_payload is not None:
                raw_features = features_payload.get("features")
                if isinstance(raw_features, list):
                    features_list = cast(List[Any], raw_features)
                try:
                    cnt = 0
                    for item in features_list:
                        if isinstance(item, dict):
                            d = cast(Dict[str, Any], item)
                            if d.get("enabled") is True:
                                cnt += 1
                    enabled_count = cnt
                except Exception:
                    enabled_count = 0
            _display_small_stat("Enabled", enabled_count)
            _display_small_stat("Total", _safe_len(features_list))
            with _st.expander("Raw JSON", expanded=False):
                _st.json(features_payload or {})

    # ---------- AI Status ----------
    with display_guard("AI Status"):
        with c2:
            _st.subheader("AI Status")
            ai_payload = cast(Optional[Dict[str, Any]], _req("GET", "/api/ai/heartbeat"))
            if isinstance(ai_payload, dict) and "__meta__" in ai_payload:
                meta = cast(Dict[str, Any], ai_payload.get("__meta__") or {})
                route = meta.get("route")
                status = meta.get("status")
                err = meta.get("error")
                msg = "AI heartbeat unavailable"
                if route or status is not None:
                    msg += f" — {route or ''} {f'({status})' if status is not None else ''}"
                if err:
                    msg += f" • {err}"
                _st.info(msg.strip())
            running: Optional[bool] = None
            strategies_active: Optional[int] = None
            if isinstance(ai_payload, dict):
                running = ai_payload.get("running")
                strategies_active = ai_payload.get("strategies_active")
            _display_small_stat("Running", "✅" if running else ("❌" if running is not None else "N/A"))
            _display_small_stat("Strategies", strategies_active)
            with _st.expander("Raw JSON", expanded=False):
                _st.json(ai_payload or {})

    # ---------- Autobuy Status ----------
    with display_guard("Autobuy Status"):
        with c3:
            _st.subheader("Autobuy Status")
            ab_payload = cast(Optional[Dict[str, Any]], _req("GET", "/api/autobuy/status"))
            if isinstance(ab_payload, dict) and "__meta__" in ab_payload:
                meta = cast(Dict[str, Any], ab_payload.get("__meta__") or {})
                route = meta.get("route")
                status = meta.get("status")
                err = meta.get("error")
                msg = "Autobuy status unavailable"
                if route or status is not None:
                    msg += f" — {route or ''} {f'({status})' if status is not None else ''}"
                if err:
                    msg += f" • {err}"
                _st.info(msg.strip())
            status_text: Optional[str] = None
            active_orders: Optional[int] = None
            if ab_payload is not None:
                # Best-effort extraction from common shapes
                status_obj: Dict[str, Any] = {}
                svc_obj: Dict[str, Any] = {}
                s_val = ab_payload.get("status")
                if isinstance(s_val, dict):
                    status_obj = cast(Dict[str, Any], s_val)
                svc_val = ab_payload.get("service_status")
                if isinstance(svc_val, dict):
                    svc_obj = cast(Dict[str, Any], svc_val)
                status_text = cast(Optional[str], status_obj.get("status"))
                ao_val = svc_obj.get("active_orders")
                active_orders = ao_val if isinstance(ao_val, int) else None
            _display_small_stat("State", status_text or "N/A")
            _display_small_stat("Active Orders", active_orders)
            with _st.expander("Raw JSON", expanded=False):
                _st.json(ab_payload or {})

    # ---------- Server Health Pings ----------
    with display_guard("Server Health Pings"):
        with c4:
            _st.subheader("Server Health")
            basic = cast(Optional[Dict[str, Any]], _req("GET", "/api/health"))
            detailed = cast(Optional[Dict[str, Any]], _req("GET", "/health/comprehensive"))
            # Notices if metadata only
            for name, payload in ("basic", basic), ("detailed", detailed):
                if isinstance(payload, dict) and "__meta__" in payload:
                    meta = cast(Dict[str, Any], payload.get("__meta__") or {})
                    route = meta.get("route")
                    status = meta.get("status")
                    err = meta.get("error")
                    msg = f"Server health ({name}) unavailable"
                    if route or status is not None:
                        msg += f" — {route or ''} {f'({status})' if status is not None else ''}"
                    if err:
                        msg += f" • {err}"
                    _st.info(msg.strip())
            # Prefer detailed status if available, else fallback to basic
            status_value = None
            if isinstance(detailed, dict):
                status_value = detailed.get("status")
            if status_value is None and isinstance(basic, dict):
                status_value = basic.get("status")
            _display_small_stat("Status", status_value or "N/A")
            with _st.expander("Raw JSON", expanded=False):
                _st.json({
                    "basic": basic or {},
                    "detailed": detailed or {},
                })


if __name__ == "__main__":
    main()



