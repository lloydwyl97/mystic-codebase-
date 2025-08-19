import streamlit as st

from mystic_ui.api_client import request_json
from mystic_ui.display import render_kpis, render_table


def render() -> None:
	st.title(" Portfolio")

	ov = request_json("GET", "/portfolio/overview") or {}
	positions = request_json("GET", "/portfolio/positions") or []
	render_kpis([
		("Equity", ov.get("equity", ov.get("total_value")), None),
		("PNL (Total)", ov.get("pnl_total", ov.get("total_pnl")), None),
		("Positions", len(positions), None),
	])

	st.subheader("Positions")
	render_table(positions)

	st.subheader("Transactions")
	render_table(request_json("GET", "/portfolio/transactions"))

	st.subheader("Risk Metrics")
	render_table(request_json("GET", "/portfolio/risk-metrics"))


