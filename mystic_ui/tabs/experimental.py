import streamlit as st
from mystic_ui.api_client import request_json as _request_json


def render() -> None:
	st.title(" Experimental")

	col = st.columns(3)
	if col[0].button("Start All"):
		st.write(_request_json("POST", "/experimental/start-all", {}))
	if col[1].button("Stop All"):
		st.write(_request_json("POST", "/experimental/stop-all", {}))
	if col[2].button("Status"):
		st.write(_request_json("GET", "/experimental/status"))

	st.subheader("Services")
	st.json(_request_json("GET", "/experimental/services"))


