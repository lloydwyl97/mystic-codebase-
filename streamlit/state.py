import os, streamlit as st
from typing import MutableMapping, Any, cast

_st = cast(Any, st)

def get_app_state() -> MutableMapping[str, Any]:
    s = cast(MutableMapping[str, Any], _st.session_state)
    s.setdefault("exchange", os.getenv("DISPLAY_EXCHANGE", "binanceus"))
    s.setdefault("symbol",   os.getenv("DISPLAY_SYMBOL",  "BTCUSDT"))
    s.setdefault("timeframe","1m")
    return s

def render_sidebar_controls() -> None:
    s = get_app_state()
    _st.sidebar.caption("Exchange & Symbol (locked by env unless changed here)")
    s["exchange"] = _st.sidebar.text_input("Exchange", s["exchange"]) 
    s["symbol"]   = _st.sidebar.text_input("Symbol",   s["symbol"]) 
    s["timeframe"]= _st.sidebar.text_input("Timeframe",s["timeframe"]) 
