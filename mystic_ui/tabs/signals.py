import streamlit as st

from mystic_ui.api_client import request_json
from mystic_ui.display import render_table


def render() -> None:
	st.title(" Signals")

	st.subheader("Signals")
	render_table(request_json("GET", "/signals"))
	st.subheader("Whale Alerts")
	render_table(request_json("GET", "/whale/alerts"))


