import streamlit as st

from mystic_ui.api_client import request_json
from mystic_ui.display import render_table


def render() -> None:
	st.title(" Analytics")

	st.subheader("System Performance")
	render_table(request_json("GET", "/system/performance"))

	st.subheader("Portfolio Performance")
	render_table(request_json("GET", "/portfolio/performance"))

	st.subheader("AI Performance")
	ai_perf = request_json("GET", "/ai/performance/analytics")
	render_table(ai_perf)


