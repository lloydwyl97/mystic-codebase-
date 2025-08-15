from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, Tuple, TypedDict, Optional, cast

import streamlit as st
import shutil
from pathlib import Path

# Ensure project root is importable so `dashboard` resolves
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.append(_ROOT)

from streamlit.data_client import (
    get_ticker,
    get_ohlcv,
    get_trades,
    get_balances,
    get_autobuy_heartbeat,
    get_ai_heartbeat,
    system_health,
    get_alerts,
    get_ai_signals,
)
from dashboard.schemas import truncate_snippet, Ticker as Tkr, OHLCV as Ohl, Trade as Trd, Balance as Bal, AIHeartbeat as AIHB, AlertItem as Alrt
from streamlit.ui_guard import display_guard

# Backup copy
_p = Path(__file__)
_bak = _p.with_suffix(_p.suffix + ".bak_cursor")
try:
    if not _bak.exists():
        shutil.copyfile(_p, _bak)
except Exception:
    pass


ROWS = [
    "coinbase",
    "binanceus",
    "kraken",
    "coingecko",
    "autobuy",
    "ai_signals",
    "alerts",
    "system",
]

COLUMNS = [
    "ping",
    "ticker",
    "ohlcv",
    "orderbook",
    "trades",
    "balance",
    "ai_heartbeat",
    "alerts_stream",
]


class OBPreview(TypedDict, total=False):
    bid: Optional[float]
    ask: Optional[float]


def _validate_schema(row: str, col: str, payload: Any) -> Tuple[bool, Optional[str]]:
    try:
        if payload is None:
            return False, "no payload"
        if row in ("coinbase", "binanceus", "kraken", "coingecko"):
            if col == "ticker":
                Tkr(**cast(Dict[str, Any], payload))
                return True, None
            if col == "ohlcv":
                if isinstance(payload, dict):
                    Ohl(**cast(Dict[str, Any], payload))
                    return True, None
                if isinstance(payload, list):
                    # list of candles accepted by our validator wrapper
                    return True, None
            if col == "trades":
                if isinstance(payload, list) and payload:
                    Trd(**cast(Dict[str, Any], payload[0]))
                    return True, None
                if isinstance(payload, dict):
                    Trd(**cast(Dict[str, Any], payload))
                    return True, None
            if col == "balance":
                if isinstance(payload, list) and payload:
                    Bal(**cast(Dict[str, Any], payload[0]))
                    return True, None
                if isinstance(payload, dict):
                    Bal(**cast(Dict[str, Any], payload))
                    return True, None
        if row == "autobuy" and col in ("ping", "ai_heartbeat"):
            if isinstance(payload, dict):
                # try nested service_status shape or simple heartbeat
                pdict: Dict[str, Any] = payload
                svc_raw: Any = pdict.get("service_status", pdict)
                svc: Dict[str, Any] = svc_raw if isinstance(svc_raw, dict) else {}
                running_flag = str(svc.get("status", "")).lower() in ("active", "running", "ready")
                strategies_active_val = int(svc.get("strategies_active", 0) or 0)
                last_decision_val = str(pdict.get("timestamp", "")) if pdict else None
                queue_depth_val = int(svc.get("queue_depth", 0) or 0)
                AIHB(
                    running=bool(running_flag),
                    strategies_active=strategies_active_val,
                    last_decision_ts=last_decision_val,
                    queue_depth=queue_depth_val,
                )
                return True, None
        if row in ("ai_signals", "alerts") and col == "alerts_stream":
            if isinstance(payload, list) and payload:
                first: Any = payload[0]
                if isinstance(first, dict):
                    Alrt(**cast(Dict[str, Any], first))
                    return True, None
        return False, "unvalidated"
    except Exception as e:  # noqa: BLE001
        return False, str(e)


def _probe(row: str, col: str) -> Tuple[str, int, Any]:
    start = time.perf_counter()
    data: Any = None
    try:
        if row in ("coinbase", "binanceus", "kraken", "coingecko"):
            if col == "ping":
                r = system_health()
                data = r.data
            elif col == "ticker":
                r = get_ticker(row if row != "coingecko" else "coinbase", "BTC-USD")
                data = r.data
            elif col == "ohlcv":
                r = get_ohlcv(row if row != "coingecko" else "coinbase", "BTC-USD", "1h")
                data = r.data
            elif col == "orderbook":
                r = get_ticker(row if row != "coingecko" else "coinbase", "BTC-USD")
                dd: Dict[str, Any] = r.data if isinstance(r.data, dict) else {}
                bid_obj: Any = dd.get("bid", None)
                ask_obj: Any = dd.get("ask", None)
                bid_val = float(bid_obj) if isinstance(bid_obj, (int, float)) else None
                ask_val = float(ask_obj) if isinstance(ask_obj, (int, float)) else None
                data = cast(Any, OBPreview(bid=bid_val, ask=ask_val))
            elif col == "trades":
                r = get_trades(row if row != "coingecko" else "coinbase", "BTC-USD", 5)
                # trades can be list or dict; coerce to list when possible for clarity
                payload_tr: Any = r.data
                if isinstance(payload_tr, dict):
                    # some backends wrap trades under a key; prefer any list-like value if present
                    maybe_list_any: Any = payload_tr.get("trades")
                    if isinstance(maybe_list_any, list):
                        data = cast(Any, maybe_list_any)
                    else:
                        data = cast(Dict[str, Any], payload_tr)
                else:
                    data = cast(Any, payload_tr)
            elif col == "balance":
                r = get_balances(row if row != "coingecko" else "coinbase")
                data = r.data
        elif row == "autobuy":
            if col == "ping":
                r = get_autobuy_heartbeat()
                data = r.data
            elif col == "ai_heartbeat":
                r = get_ai_heartbeat()
                data = r.data
        elif row == "ai_signals":
            if col == "ping":
                # minimal availability check via AI signals endpoint
                r = get_ai_signals("BTC-USD", "1h")
                data = r.data
            elif col == "alerts_stream":
                # treat presence of signals as stream indicator
                r = get_ai_signals("BTC-USD", "1h")
                data = r.data
        elif row == "alerts":
            if col == "ping":
                # system health as a quick ping
                r = system_health()
                data = r.data
            elif col == "alerts_stream":
                # call real alerts endpoint
                r = get_alerts(50)
                data = r.data
        elif row == "system":
            if col in ("ping", "ai_heartbeat"):
                r = system_health()
                data = r.data
    except Exception as e:
        data = {"error": str(e)}
    latency_ms = int((time.perf_counter() - start) * 1000)
    if data is None:
        status = "❌"
    elif isinstance(data, dict):
        data_dict: Dict[str, Any] = data
        status = "⚠️" if bool(data_dict.get("error")) else "✅"
    else:
        status = "✅"
    out_data: Any = data
    return status, latency_ms, out_data


def main() -> None:
    st.set_page_config(page_title="Connectivity Matrix", layout="wide")
    with display_guard("Connectivity Header"):
        st.title("Connectivity Matrix")
        st.caption(f"Last refresh: {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}")
    grid: Dict[Tuple[str, str], Tuple[str, int, Any]] = {}

    with display_guard("Connectivity Columns"):
        cols = st.columns(len(COLUMNS) + 1)
        cols[0].markdown("**Service**")
        for j, c in enumerate(COLUMNS, start=1):
            cols[j].markdown(f"**{c}**")

    for row in ROWS:
        with display_guard(f"Row {row}"):
            cells = st.columns(len(COLUMNS) + 1)
            cells[0].markdown(f"`{row}`")
            for j, col in enumerate(COLUMNS, start=1):
                status, latency, payload = _probe(row, col)
                key = f"{row}-{col}"
                label = f"{status} {latency}ms"
                if cells[j].button(label, key=key):
                    with st.expander(f"{row}.{col} details", expanded=True):
                        snippet = truncate_snippet(payload, 300)
                        size = len(str(payload)) if payload is not None else 0
                        ok, err = _validate_schema(row, col, payload)
                        st.write({"payload_size": size, "latency_ms": latency, "schema_valid": ok, "error": err})
                        st.code(snippet)
                grid[(row, col)] = (status, latency, payload)

    with display_guard("Legend"):
        with st.expander("Legend / Diagnostics", expanded=False):
            st.write("✅ ok, ⚠️ degraded, ❌ fail")


if __name__ == "__main__":
    main()


