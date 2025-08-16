from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, cast

import streamlit as st

# Ensure project root is importable so `streamlit.*` utilities resolve
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.append(_ROOT)

from streamlit.data_client import get_alerts, get_health_check, get_ai_heartbeat
from streamlit.state import render_sidebar_controls
from streamlit.ui_guard import display_guard


SEVERITY_COLOR = {
    "critical": "#ff4d4f",
    "high": "#ff7a45",
    "error": "#ff4d4f",
    "warn": "#faad14",
    "warning": "#faad14",
    "info": "#1890ff",
    "success": "#52c41a",
    "low": "#2db7f5",
}


def _severity_color(sev: Optional[str]) -> str:
    s = (sev or "").strip().lower()
    return SEVERITY_COLOR.get(s, "#444")


def _to_iso(ts: Any) -> str:
    try:
        if isinstance(ts, (int, float)):
            return datetime.fromtimestamp(float(ts), tz=timezone.utc).strftime("%H:%M:%S")
        if isinstance(ts, str):
            # show hh:mm:ss when possible
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                return dt.strftime("%H:%M:%S")
            except Exception:
                return ts
    except Exception:
        pass
    return "—"


def _as_dict(x: Any) -> Dict[str, Any]:
    if isinstance(x, dict):
        return cast(Dict[str, Any], x)
    if hasattr(x, "data") and isinstance(getattr(x, "data"), dict):
        return cast(Dict[str, Any], getattr(x, "data"))
    return {}


def main() -> None:
    _st = cast(Any, st)
    _st.set_page_config(page_title="Alerts", layout="wide")

    # Sidebar controls (exchange/symbol/timeframe)
    render_sidebar_controls()

    # Health/AI summary pills
    try:
        hc = get_health_check()
        hdata: Dict[str, Any] = _as_dict(hc)
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
        pills = " ".join([
            f"<span style='padding:4px 8px;border-radius:12px;background:#222;color:#eee;margin-right:6px'>{k} {v}</span>"
            for k, v in status_map.items()
        ])
        _st.markdown(pills, unsafe_allow_html=True)
        try:
            ai = get_ai_heartbeat()
            ai_data: Dict[str, Any] = _as_dict(ai)
            running = bool(ai_data.get("running"))
            last_ts = ai_data.get("last_decision_ts")
            _st.markdown(
                f"<div style='margin:6px 0;padding:6px 10px;display:inline-block;border-radius:12px;background:{'#154' if running else '#441'};color:#eee'>AI: {'Running' if running else 'Idle'} • {last_ts or '—'}</div>",
                unsafe_allow_html=True,
            )
        except Exception:
            pass
        _st.caption(f"Last refresh: {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}")
    except Exception:
        pass

    # Alerts fetch
    with display_guard("Alerts Feed"):
        res: Any = get_alerts(200)
        if isinstance(res, dict) and "__meta__" in res:
            meta_raw: Any = res["__meta__"] if "__meta__" in res else {}
            meta: Dict[str, Any] = cast(Dict[str, Any], meta_raw if isinstance(meta_raw, dict) else {})
            route = cast(Optional[str], meta.get("route"))
            status = cast(Optional[int], meta.get("status"))
            err = cast(Optional[str], meta.get("error"))
            msg = "Alerts unavailable"
            if route or status:
                msg += f" — {route or ''} {f'({status})' if status is not None else ''}"
            if err:
                msg += f" • {err}"
            _st.info(msg.strip())

    # Normalize alerts list
    alerts: List[Dict[str, Any]] = []
    raw_payload: Any = None
    if res is None:
        alerts = []
    else:
        if isinstance(res, dict) and "__meta__" in res:
            raw_payload = cast(Dict[str, Any], res)
        else:
            res_obj: Any = res
            if hasattr(res_obj, "data"):
                raw_payload = getattr(res_obj, "data")
            else:
                raw_payload = res_obj
        data: Any = raw_payload
        if isinstance(data, dict):
            data_dict: Dict[str, Any] = cast(Dict[str, Any], data)
            if "__meta__" in data_dict:
                alerts = []
            else:
                inner_any: Any = data_dict.get("data")
                if isinstance(inner_any, dict):
                    inner_dict: Dict[str, Any] = cast(Dict[str, Any], inner_any)
                    notif_any: Any = inner_dict.get("notifications")
                    alerts_any: Any = inner_dict.get("alerts")
                    if isinstance(notif_any, list):
                        alerts = cast(List[Dict[str, Any]], notif_any)
                    elif isinstance(alerts_any, list):
                        alerts = cast(List[Dict[str, Any]], alerts_any)
                    else:
                        alerts = []
                elif isinstance(inner_any, list):
                    alerts = cast(List[Dict[str, Any]], inner_any)
                else:
                    # some backends return the list at top-level
                    top_alerts_any: Any = data_dict.get("alerts") or data_dict.get("notifications")
                    alerts = cast(List[Dict[str, Any]], top_alerts_any or [])
        elif isinstance(data, list):
            alerts = cast(List[Dict[str, Any]], data)

    _st.subheader("Recent Alerts")
    if alerts:
        # Compact timeline/list with severity color accents
        for a in alerts[:200]:
            ad: Dict[str, Any] = a
            sev_val: Any = ad["severity"] if "severity" in ad else ad["level"] if "level" in ad else ad["type"] if "type" in ad else "info"
            sev = str(sev_val)
            color = _severity_color(sev)
            ts_val: Any = ad["timestamp"] if "timestamp" in ad else ad["time"] if "time" in ad else ad.get("ts")
            ts = _to_iso(ts_val)
            title_val: Any = ad["title"] if "title" in ad else ad["message"] if "message" in ad else ad.get("text") or "Alert"
            title = str(title_val)
            desc_val: Any = ad["description"] if "description" in ad else ad["details"] if "details" in ad else ad.get("reason") or ""
            desc = str(desc_val)
            badge = f"<span style='display:inline-block;min-width:8px;height:8px;border-radius:50%;background:{color};margin-right:8px;vertical-align:middle'></span>"
            line = f"<div style='display:flex;align-items:center;gap:8px;margin:6px 0'>" \
                   f"{badge}<span style='font-weight:600'>{ts}</span>" \
                   f"<span style='color:{color};font-weight:600;text-transform:uppercase;font-size:12px;background:rgba(0,0,0,0.2);padding:2px 6px;border-radius:10px'>{sev}</span>" \
                   f"<span style='flex:1'></span>" \
                   f"</div>"
            _st.markdown(line, unsafe_allow_html=True)
            _st.markdown(f"<div style='margin:-6px 0 8px 16px'><span style='font-weight:600'>{title}</span>" + (f" — <span style='opacity:0.8'>{desc}</span>" if desc else "") + "</div>", unsafe_allow_html=True)
    else:
        _st.info("No recent alerts.")

    # Raw JSON expander
    with _st.expander("Raw JSON: Alerts"):
        _st.json(raw_payload)


if __name__ == "__main__":
    main()


