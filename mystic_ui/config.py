import os
from urllib.parse import urljoin


def _ensure_no_double_api(base: str) -> str:
	# Normalize to avoid trailing slashes and double /api
	b = base.rstrip("/")
	if b.endswith("/api"):
		return b
	return f"{b}/api"


MYSTIC_BACKEND = os.getenv("MYSTIC_BACKEND", "http://127.0.0.1:9000").rstrip("/")
API_BASE = _ensure_no_double_api(MYSTIC_BACKEND)


def api_url(path: str) -> str:
	"""
	Join API_BASE with a relative path like '/market/prices'.
	Ensures exactly one '/api' is present and single slashes.
	"""
	# strip leading slashes to avoid urljoin quirks
	p = path.lstrip("/")
	return urljoin(API_BASE + "/", p)


