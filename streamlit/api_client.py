"""Canonical API client facade for Streamlit dashboard.

Wraps the existing legacy client at `streamlit/pages/components/api_client.py`
to provide import path `from streamlit.api_client import api_client`.
Ensures BASE_URL is centralized via `DASHBOARD_BASE_URL`.
"""

from __future__ import annotations

import os

# Centralize base URL
BASE_URL = os.getenv("DASHBOARD_BASE_URL", "http://127.0.0.1:9000").rstrip("/")
os.environ.setdefault("BACKEND_URL", BASE_URL)
if not globals().get("_API_BASE_URL_LOGGED"):
    print(f"[streamlit.api_client] BASE_URL={BASE_URL}")
    _API_BASE_URL_LOGGED = True  # type: ignore

try:
    from streamlit.pages.components.api_client import api_client  # type: ignore
except Exception:  # pragma: no cover - fallback
    from .pages.components.api_client import api_client  # type: ignore

__all__ = ["api_client"]


