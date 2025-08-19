import streamlit as st

from mystic_ui.api_client import request_json
from mystic_ui.display import render_kpis, render_table


def render() -> None:
	st.title(" Health")

	sys_health = request_json("GET", "/system/health") or {}
	ai_health = request_json("GET", "/ai/system/health") or {}
	basic = request_json("GET", "/health") or {}

	st.subheader("System Health")
	render_table(sys_health)

	st.subheader("AI Health")
	render_table(ai_health)

	st.subheader("Service")
	render_kpis([
		("Service", basic.get("service"), None),
		("Status", basic.get("status"), None),
	])


