import os
from typing import Any

import pandas as pd
import streamlit as st

# Base env is read by api_client; we keep here for display convenience.
BASE = os.getenv("MYSTIC_BACKEND", "http://127.0.0.1:8000")

def metric_row(items: list[tuple[str, Any, str | None]]):
    cols = st.columns(len(items))
    for col, (label, val, help_text) in zip(cols, items, strict=False):
        col.metric(label, val, help=help_text)

def to_df(candles: dict[str, list[Any]]) -> pd.DataFrame:
    if not candles or not candles.get("timestamps"):
        return pd.DataFrame()
    df = pd.DataFrame({
        "ts": candles["timestamps"],
        "open": candles.get("opens", []),
        "high": candles.get("highs", []),
        "low": candles.get("lows", []),
        "close": candles.get("closes", []),
        "volume": candles.get("volumes", []),
    })
    # ms -> datetime if needed
    if df["ts"].dtype != "datetime64[ns]":
        try:
            df["ts"] = pd.to_datetime(df["ts"], unit="ms")
        except Exception:
            try:
                df["ts"] = pd.to_datetime(df["ts"])
            except Exception:
                pass
    return df

def timeframe_select(default="1h"):
    return st.selectbox("Timeframe", ["5m","1h","4h","1d"], index=["5m","1h","4h","1d"].index(default))

def safe_json(title: str, data: Any):
    with st.expander(title, expanded=False):
        st.json(data)

def titled_table(title: str, data: Any):
    st.subheader(title)
    if isinstance(data, list):
        st.dataframe(data)
    else:
        st.json(data)

