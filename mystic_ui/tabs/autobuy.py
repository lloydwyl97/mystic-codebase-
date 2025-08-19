import streamlit as st

from mystic_ui.api_client import request_json
from mystic_ui.display import render_kpis, render_table


def render() -> None:
	st.title(" AutoBuy")

	col = st.columns(4)
	if col[0].button("Start AutoBuy"):
		st.write(request_json("POST", "/autobuy/start", json={}))
	if col[1].button("Stop AutoBuy"):
		st.write(request_json("POST", "/autobuy/stop", json={}))
	status_payload = request_json("GET", "/autobuy/status") or {}
	st.subheader("Status")
	render_kpis([
		("Enabled", str(status_payload.get("enabled", status_payload.get("trading_enabled"))), None),
		("Active Orders", status_payload.get("active_orders"), None),
		("Success Rate %", status_payload.get("success_rate"), None),
	])
	if col[3].button("Signals"):
		st.write(request_json("GET", "/autobuy/signals"))

	st.subheader("Config")
	render_table(request_json("GET", "/autobuy/config"))
	st.subheader("Stats")
	render_table(request_json("GET", "/autobuy/stats"))
	st.subheader("Trades")
	render_table(request_json("GET", "/autobuy/trades"))


