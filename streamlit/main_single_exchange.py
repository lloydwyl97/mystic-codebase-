# --- MystIC path bootstrap (put at very top) ---
import os, sys
_ROOT    = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_ST      = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_BACKEND = os.path.join(_ROOT, "backend")
for _p in (_ROOT, _ST, _BACKEND):
    if _p not in sys.path: sys.path.insert(0, _p)
# -----------------------------------------------
# --- path setup for app + backend config ---
import os, sys
ROOT = os.path.abspath(os.path.dirname(__file__) + "/..")   # repo root
if ROOT not in sys.path: sys.path.append(ROOT)
ST   = os.path.abspath(os.path.dirname(__file__))           # streamlit/
if ST not in sys.path: sys.path.append(ST)
# -------------------------------------------
import streamlit as st
import pandas as pd
import requests

from backend.config.coins import FEATURED_EXCHANGE, FEATURED_SYMBOLS
from data_client import get_ohlcv, get_trades

API = os.environ.get("MYSTIC_BACKEND", "http://127.0.0.1:9000")
st.set_page_config(page_title=f"Mystic — {FEATURED_EXCHANGE.upper()}", layout="wide")
st.title(f"{FEATURED_EXCHANGE.upper()} — Live AI Dashboard")


def chart_one(symbol: str):
    colA, colB = st.columns([2, 1])

    with colA:
        res = get_ohlcv(FEATURED_EXCHANGE, symbol, "1m")
        candles = (res.data or {}).get("candles") if isinstance(res.data, dict) else None
        if candles:
            df = pd.DataFrame(candles, columns=["ts", "open", "high", "low", "close", "volume"])
            df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
            st.line_chart(df.set_index("ts")[ ["close"] ])
        else:
            st.info("No OHLCV from backend")

    with colB:
        res = get_trades(FEATURED_EXCHANGE, symbol, 50)
        trades = res.data or []
        st.caption(f"{symbol} — recent trades ({len(trades)})")
        st.dataframe(trades, height=460, use_container_width=True)


def ai_explain(symbol: str):
    st.subheader(f"AI — What it used for {symbol}")
    try:
        r = requests.get(f"{API}/api/ai/explain/attribution", params={"symbol": symbol}, timeout=6)
        data = r.json()
    except Exception as e:
        st.error(f"Explain error: {e}")
        return
    if not data.get("ok"):
        st.info("No AI attribution yet.")
        return
    used = data.get("used") or {}
    st.json(
        {
            "mode": os.environ.get("AI_TRADE_MODE", "off"),
            "inputs": used.get("inputs"),
            "weights": used.get("weights"),
            "reason": used.get("reason"),
            "ts": data.get("ts"),
        }
    )


for sym in FEATURED_SYMBOLS:
    st.header(sym)
    chart_one(sym)
    ai_explain(sym)
    st.divider()


