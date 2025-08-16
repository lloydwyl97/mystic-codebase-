# --- MystIC path bootstrap (put at very top) ---
import os, sys
_ROOT    = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_ST      = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_BACKEND = os.path.join(_ROOT, "backend")
for _p in (_ROOT, _ST, _BACKEND):
    if _p not in sys.path: sys.path.insert(0, _p)
# -----------------------------------------------
# --- path setup for app + backend config ---
ROOT = os.path.abspath(os.path.dirname(__file__) + "/..")   # repo root
if ROOT not in sys.path: sys.path.append(ROOT)
ST   = os.path.abspath(os.path.dirname(__file__))           # streamlit/
if ST not in sys.path: sys.path.append(ST)
# -------------------------------------------
import streamlit as st
import pandas as pd
import requests
from typing import Any, Dict, cast

from backend.config.coins import FEATURED_EXCHANGE, FEATURED_SYMBOLS
from data_client import get_ohlcv, get_trades

API = os.environ.get("MYSTIC_BACKEND", "http://127.0.0.1:9000")
_st = cast(Any, st)
_st.set_page_config(page_title=f"Mystic — {FEATURED_EXCHANGE.upper()}", layout="wide")
_st.title(f"{FEATURED_EXCHANGE.upper()} — Live AI Dashboard")


def chart_one(symbol: str):
    colA, colB = _st.columns([2, 1])

    with colA:
        res_ohlcv = get_ohlcv(FEATURED_EXCHANGE, symbol, "1m")
        # Show compact notice if only metadata returned
        if isinstance(res_ohlcv, dict) and "__meta__" in res_ohlcv:
            meta_any: Any = res_ohlcv["__meta__"] if "__meta__" in res_ohlcv else {}
            meta: Dict[str, Any] = cast(Dict[str, Any], meta_any if isinstance(meta_any, dict) else {})
            route = meta.get("route")
            status = meta.get("status")
            err = meta.get("error")
            msg = "OHLCV unavailable"
            if route or status is not None:
                msg += f" — {route or ''} {f'({status})' if status is not None else ''}"
            if err:
                msg += f" • {err}"
            _st.info(msg.strip())
            candles = None
        else:
            payload: Any = res_ohlcv
            # Try common shapes: top-level dict with "candles", or direct list
            candles = None
            if isinstance(payload, dict):
                p_dict: Dict[str, Any] = cast(Dict[str, Any], payload)
                c_any = p_dict.get("candles")
                if isinstance(c_any, list):
                    candles = cast(list[Any], c_any)
            elif isinstance(payload, list):
                candles = cast(list[Any], payload)
        if candles:
            try:
                df_any: Any = pd.DataFrame(candles)
                df = cast(pd.DataFrame, df_any)
                # Ensure columns exist with safe defaults
                if "ts" in df.columns:
                    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True, errors="coerce")  # type: ignore[arg-type]
                    idx_any: Any = df.set_index("ts")  # type: ignore[call-overload]
                    idx_df = cast(pd.DataFrame, idx_any)
                    cols = [c for c in ("close", "c") if c in idx_df.columns]
                    _st.line_chart(idx_df[cols])
                else:
                    _st.dataframe(df.tail(200), use_container_width=True)
            except Exception:
                _st.json(candles)
        else:
            _st.info("No OHLCV from backend")

    with colB:
        res_tr = get_trades(FEATURED_EXCHANGE, symbol, 50)
        trades: list[Any] = []
        if isinstance(res_tr, dict) and "__meta__" in res_tr:
            meta_any: Any = res_tr["__meta__"] if "__meta__" in res_tr else {}
            meta: Dict[str, Any] = cast(Dict[str, Any], meta_any if isinstance(meta_any, dict) else {})
            route = meta.get("route")
            status = meta.get("status")
            err = meta.get("error")
            msg = "Trades unavailable"
            if route or status is not None:
                msg += f" — {route or ''} {f'({status})' if status is not None else ''}"
            if err:
                msg += f" • {err}"
            _st.info(msg.strip())
        else:
            payload_tr: Any = res_tr
            if isinstance(payload_tr, dict):
                p2: Dict[str, Any] = cast(Dict[str, Any], payload_tr)
                t_any = p2.get("trades") or p2.get("data")
                if isinstance(t_any, list):
                    trades = t_any
            elif isinstance(payload_tr, list):
                trades = payload_tr
        _st.caption(f"{symbol} — recent trades ({len(trades)})")
        if trades:
            _st.dataframe(trades, height=460, use_container_width=True)
        else:
            _st.info("No trades from backend")


def ai_explain(symbol: str):
    _st.subheader(f"AI — What it used for {symbol}")
    try:
        r = requests.get(f"{API}/api/ai/explain/attribution", params={"symbol": symbol}, timeout=6)
        data: Any = r.json()
    except Exception as e:
        _st.error(f"Explain error: {e}")
        return
    if not (isinstance(data, dict) and cast(Dict[str, Any], data).get("ok")):
        _st.info("No AI attribution yet.")
        return
    used_any: Any = cast(Dict[str, Any], data).get("used") or {}
    used: Dict[str, Any] = cast(Dict[str, Any], used_any if isinstance(used_any, dict) else {})
    _st.json(
        {
            "mode": os.environ.get("AI_TRADE_MODE", "off"),
            "inputs": used.get("inputs"),
            "weights": used.get("weights"),
            "reason": used.get("reason"),
            "ts": cast(Dict[str, Any], data).get("ts"),
        }
    )


for sym in FEATURED_SYMBOLS:
    _st.header(sym)
    chart_one(sym)
    ai_explain(sym)
    _st.divider()


