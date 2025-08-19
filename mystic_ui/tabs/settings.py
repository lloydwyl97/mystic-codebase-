import streamlit as st
from mystic_ui.api_client import request_json as _request_json


def render() -> None:
	st.title(" Settings")

	st.json(_request_json("GET", "/system/config"))


