# Super Dashboard  BinanceUS only, no sidebar, dark theme
import os
import math
from typing import List
import pandas as pd
import streamlit as st
from datetime import datetime, timezone

from mystic_ui.api_client import get_candles
from mystic_ui.top10_resolver import resolve_top10

# ---------- Page & Theme ----------
st.set_page_config(
    page_title="Mystic Super Dashboard  BinanceUS",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Hide sidebar + tighten layout + Coinbase-dark-ish palette
st.markdown("""
<style>
/* Hide default sidebar & hamburger */
section[data-testid="stSidebar"] { display: none !important; }
button[kind="header"] { visibility: hidden; }

/* Coinbase-dark vibe */
:root {
  --bg: #0b0d12;           /* page background */
  --bg2: #12151c;          /* card background */
  --fg: #e6e8f0;           /* primary text */
  --muted: #a6adbb;        /* secondary text */
  --primary: #1652f0;      /* accent blue */
  --up: #16c784;
  --down: #ea3943;
  --grid: #1c2230;
}

html, body, [data-testid="stAppViewContainer"] {
  background-color: var(--bg) !important;
  color: var(--fg) !important;
}
[data-testid="stToolbar"] { display: none !important; }
.block-container { padding-top: 1.2rem; padding-bottom: 1.2rem; }
.stMetric { background: var(--bg2); border-radius: 12px; padding: 12px; }
hr { border: none; border-top: 1px solid var(--grid); margin: 0.75rem 0; }
.css-zt5igj { color: var(--fg) !important; }
</style>
""", unsafe_allow_html=True)

# ---------- Helpers ----------
def df_from_candles(candles: List[dict]) -> pd.DataFrame:
    if not candles:
        return pd.DataFrame(columns=["time","open","high","low","close","volume"])
    df = pd.DataFrame(candles)
    # backend uses 'timestamp' in ms
    if "timestamp" in df.columns:
        df["time"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert(None)
    elif "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_convert(None)
    else:
        df["time"] = pd.to_datetime("now")
    # standard numeric cols
    for c in ["open","high","low","close","volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            df[c] = math.nan
    return df[["time","open","high","low","close","volume"]].sort_values("time")

@st.cache_data(ttl=300, show_spinner=False)
def load_top10(tf: str) -> List[str]:
    return resolve_top10(timeframe=tf, limit=300)

@st.cache_data(ttl=15, show_spinner=False)
def load_candles(sym: str, tf: str):
    return get_candles(symbol=sym, timeframe=tf, limit=300, exchange="binanceus")

def kpi_row(df: pd.DataFrame):
    if df.empty:
        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("Last", ""); c2.metric("24h Δ", "")
        c3.metric("High 24h",""); c4.metric("Low 24h",""); c5.metric("Vol 24h","")
        return
    last = df["close"].iloc[-1]
    # take 24 rows as "24h" regardless of interval returned; works even if backend returns 1h
    window = df.tail(24) if len(df) >= 24 else df
    first = window["close"].iloc[0]
    delta = last - first
    pct = (delta/first*100) if first else 0.0
    hi = window["high"].max()
    lo = window["low"].min()
    vol = window["volume"].sum()
    c1,c2,c3,c4,c5 = st.columns([1,1,1,1,1])
    c1.metric("Last", f"{last:,.2f}")
    c2.metric("24h Δ", f"{delta:,.2f}  ({pct:+.2f}%)", delta)
    c3.metric("High 24h", f"{hi:,.2f}")
    c4.metric("Low 24h", f"{lo:,.2f}")
    c5.metric("Volume 24h", f"{vol:,.0f}")

# ---------- UI ----------
st.title(" Mystic Super Dashboard  BinanceUS")

# Controls
left, mid, right = st.columns([2,2,1])
with left:
    timeframe = st.selectbox(
        "Timeframe",
        options=["1m","5m","1h","4h","1d"],
        index=2,
        help="Sent as ?timeframe= to your backend (it may still return 1h; UI handles it).",
    )
with mid:
    top10 = load_top10(timeframe)
    symbol = st.selectbox("Symbol (Top 10 by 24h volume)", options=top10, index=0)
with right:
    do_refresh = st.button("Refresh", use_container_width=True)

st.divider()

# Data fetch
res = load_candles(symbol, timeframe) if not do_refresh else get_candles(symbol, timeframe, 300, "binanceus")
candles = res.get("candles") or []
df = df_from_candles(candles)

# KPIs
kpi_row(df)

# Chart
st.subheader("Price (Candles)")
if df.empty:
    st.warning("No candles returned. (Check symbol is hyphen format like BTC-USD and backend is running.)")
else:
    import plotly.graph_objects as go
    fig = go.Figure(data=[
        go.Candlestick(
            x=df["time"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name=symbol
        )
    ])
    fig.update_layout(
        height=520,
        margin=dict(l=10,r=10,t=10,b=10),
        paper_bgcolor="#0b0d12",
        plot_bgcolor="#0b0d12",
        font_color="#e6e8f0",
        xaxis=dict(gridcolor="#1c2230"),
        yaxis=dict(gridcolor="#1c2230"),
    )
    st.plotly_chart(fig, use_container_width=True)

# Raw JSON & debug
with st.expander("Raw JSON (candles)"):
    st.json(res.get("raw", {}))
with st.expander("Request info"):
    st.write({"url": res.get("url"), "params": res.get("params"), "live_data": res.get("live_data")})
