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
    _st.caption("Exchange & Symbol (locked by env unless changed here)")
    cols = _st.columns([1,1,1])
    with cols[0]:
        s["exchange"] = _st.text_input("Exchange", s["exchange"]) 
    with cols[1]:
        s["symbol"]   = _st.text_input("Symbol",   s["symbol"]) 
    with cols[2]:
        s["timeframe"]= _st.text_input("Timeframe",s["timeframe"]) 
