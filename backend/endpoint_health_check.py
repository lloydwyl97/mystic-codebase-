import os
from typing import Iterable
import requests

BASE = os.environ.get("MYSTIC_BACKEND", "http://127.0.0.1:9000").rstrip("/")

def _url(p: str) -> str:
	return f"{BASE}/{p.lstrip('/')}"

def _expect(path: str, expected: Iterable[int] = (200,), follow_redirects: bool = True) -> requests.Response:
	r = requests.get(_url(path), timeout=10, allow_redirects=follow_redirects)
	assert r.status_code in expected, f"{path} => {r.status_code}, expected {expected}"
	return r

def run():
	# Canonical must be 200
	_expect("/api/market/candles?symbol=BTCUSDT&interval=1h&limit=1", expected=(200,))
	_expect("/api/alerts/recent", expected=(200,))

if __name__ == "__main__":
	run()


