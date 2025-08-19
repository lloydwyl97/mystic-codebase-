import streamlit as st

from mystic_ui.api_client import request_json
from mystic_ui.display import render_kpis, render_table


def render() -> None:
	st.title(" Live Trading Control")

	col = st.columns(3)
	if col[0].button("Start Trading"):
		st.write(request_json("POST", "/trading/start", json={}))
	if col[1].button("Stop Trading"):
		st.write(request_json("POST", "/trading/stop", json={}))
	if col[2].button("Status"):
		st.write(request_json("GET", "/trading"))

	st.subheader("Balance / Positions / Trades")
	balance = request_json("GET", "/live/trading/balance") or {}
	render_kpis([
		("Total USD", (balance.get("total_usd") or balance.get("total")), None),
	])

	st.markdown("**Positions**")
	render_table(request_json("GET", "/live/trading/positions"))

	st.markdown("**Recent Trades**")
	render_table(request_json("GET", "/live/trading/trades"))


