import streamlit as st

from mystic_ui.api_client import request_json
from mystic_ui.display import render_kpis, render_table


def render() -> None:
	st.title(" AI Strategies")

	st.subheader("Strategies")
	render_table(request_json("GET", "/ai/strategies"))
	st.subheader("Leaderboard")
	render_table(request_json("GET", "/ai/strategies/leaderboard"))
	st.subheader("Status")
	status = request_json("GET", "/ai/status") or {}
	render_kpis([
		("CPU %", status.get("systemMetrics", {}).get("cpuUsage"), None),
		("Mem %", status.get("systemMetrics", {}).get("memoryUsage"), None),
		("API Calls/min", status.get("systemMetrics", {}).get("apiCallsPerMinute"), None),
	])

	st.divider()
	st.subheader("Update Strategy (raw JSON)")
	txt = st.text_area("Payload", "{\n  \"name\": \"example\",\n  \"params\": {\"window\": 24}\n}")
	if st.button("POST /api/ai/update-strategy"):
		import json
		try:
			st.json(request_json("POST", "/ai/update-strategy", json=json.loads(txt)))
		except Exception as e:
			st.error(str(e))


