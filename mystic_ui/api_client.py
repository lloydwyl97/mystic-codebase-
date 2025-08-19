import json
import os
import typing as t

import requests


class ApiError(Exception):
	pass


def _request_json(
	method: str,
	path: str,
	params: dict[str, t.Any] | None = None,
	data: dict[str, t.Any] | str | None = None,
	headers: dict[str, str] | None = None,
	json_body: t.Any = None,
) -> t.Any:
	from mystic_ui.config import api_url
	url = api_url(path)
	r = requests.request(
		method,
		url,
		params=params,
		data=data,
		headers=headers,
		json=json_body,
		timeout=20,
	)
	if r.status_code >= 400:
		raise ApiError(f"{method} {url} -> {r.status_code}: {r.text[:300]}")
	ct = r.headers.get("content-type", "")
	if "application/json" in ct:
		return r.json()
	# tolerate text/json anomalies
	try:
		return json.loads(r.text)
	except Exception as e:
		raise ApiError(f"Invalid JSON from {url}: {e}") from e


def _coerce_candles_to_columns(payload: t.Any) -> dict[str, list[t.Any]]:
	"""Normalize candles payload to columnar dict for UI consumption."""
	if isinstance(payload, dict) and all(
		k in payload for k in ("timestamps", "opens", "highs", "lows", "closes", "volumes")
	):
		return t.cast(dict[str, list[t.Any]], payload)

	if isinstance(payload, list) and payload and isinstance(payload[0], dict):
		ts: list[t.Any] = []
		o: list[float] = []
		h: list[float] = []
		low_values: list[float] = []
		c: list[float] = []
		v: list[float] = []
		for row_any in t.cast(list[t.Any], payload):
			row: dict[str, t.Any] = t.cast(dict[str, t.Any], row_any)
			# tolerate different key spellings
			ts.append(row.get("time") or row.get("timestamp") or row.get("t"))
			o.append(float(row.get("open") or row.get("o") or 0))
			h.append(float(row.get("high") or row.get("h") or 0))
			low_values.append(float(row.get("low") or row.get("l") or 0))
			c.append(float(row.get("close") or row.get("c") or 0))
			v.append(float(row.get("volume") or row.get("v") or 0))
		return {"timestamps": ts, "opens": o, "highs": h, "lows": low_values, "closes": c, "volumes": v}

	raise ApiError("Unrecognized candles payload shape")

# === PUBLIC WRAPPERS (use these in UI code) ===
def request_json(
	arg1: str,
	arg2: t.Any | None = None,
	params: dict[str, t.Any] | None = None,
	*,
	data: dict[str, t.Any] | str | None = None,
	headers: dict[str, str] | None = None,
	json: t.Any = None,
) -> t.Any:
	"""
	Flexible HTTP JSON wrapper with passthrough of common kwargs.

	- request_json("GET", "/path", params={...}) → backend GET
	- request_json("POST", "/path", json={...}) → backend POST
	- request_json("https://host/path", params={...}) → absolute GET
	"""
	method_like = (arg1 or "").upper()
	if method_like in {"GET", "POST", "PUT", "DELETE", "PATCH"}:
		return _request_json(
			arg1,
			t.cast(str, arg2),
			params=params,
			data=data,
			headers=headers,
			json_body=json,
		)
	# Treat as absolute URL (GET only)
	url = arg1
	r = requests.get(url, params=t.cast(dict[str, t.Any] | None, arg2), timeout=20)
	if r.status_code >= 400:
		raise ApiError(f"GET {url} -> {r.status_code}: {r.text[:300]}")
	ct = r.headers.get("content-type", "")
	if "application/json" in ct:
		return r.json()
	try:
		return json.loads(r.text)
	except Exception as e:
		raise ApiError(f"Invalid JSON from {url}: {e}") from e

def coerce_candles_to_columns(data: t.Any) -> dict[str, list[t.Any]]:
	return _coerce_candles_to_columns(data)

BACKEND = os.environ.get("MYSTIC_BACKEND", "").rstrip("/")

def _join_url(base: str, path: str) -> str:
	if not base:
		return path
	if path.startswith("http"):
		return path
	return f"{base}/{path.lstrip('/')}"

# Last-known-good cache for candles to avoid UI flicker on transient errors
_LKG_CANDLES: dict[tuple[str, str, str, int], dict[str, list[t.Any]]] = {}

def get_candles(
	symbol: str,
	exchange: str = "binance",
	interval: str = "1h",
	limit: int = 200,
) -> dict[str, list[t.Any]]:
	"""
	Primary: /api/market/candles
	Fallback: /market/candles (for clients that add /api automatically)
	Removed: double-api variant and /api/binance/history (wrong shape)
	"""
	params = {"symbol": symbol, "interval": interval, "limit": limit}
	candidates = [
		"/api/market/candles",  # canonical
		"/market/candles",      # fallback if base already includes /api
	]
	key = (symbol, exchange, interval, limit)
	for path in candidates:
		try:
			url = _join_url(BACKEND, path)
			r = requests.get(url, params=params, timeout=10)
			if r.status_code == 200:
				data = r.json()
				# Ensure timestamps are ms (convert seconds -> ms if needed)
				if isinstance(data, list) and data and isinstance(data[0], dict):
					data_dicts = t.cast(list[dict[str, t.Any]], data)
					ts = data_dicts[0].get("timestamp")
					if isinstance(ts, int | float) and ts < 10_000_000_000:
						for r_ in data_dicts:
							tv = r_.get("timestamp")
							if isinstance(tv, int | float):
								r_["timestamp"] = int(tv * 1000)
				# Expect list[dict] with keys: timestamp, open, high, low, close, volume
				if isinstance(data, list):
					columns = _coerce_candles_to_columns(data)
					if columns.get("timestamps"):
						_LKG_CANDLES[key] = columns
					return columns
		except Exception:
			# try next candidate
			pass
	# fallback to last-known-good if available; else UI-safe empty columns shape
	return _LKG_CANDLES.get(key, {"timestamps": [], "opens": [], "highs": [], "lows": [], "closes": [], "volumes": []})


def get_prices() -> dict[str, t.Any]:
	return _request_json("GET", "/market/prices")


def get_live() -> dict[str, t.Any]:
	return _request_json("GET", "/market/live")


def get_ws_status() -> dict[str, t.Any]:
	return _request_json("GET", "/websocket/status")


def get_alerts_recent() -> list[dict[str, t.Any]]:
	"""
	Canonical alerts. Fallback to /api/alerts only if old router is present.
	Intentionally does NOT call legacy live notifications alias post-alias removal.
	"""
	candidates = [
		"/api/alerts/recent",
		"/api/alerts",	# legacy, optional
	]
	for path in candidates:
		try:
			r = requests.get(_join_url(BACKEND, path), timeout=10)
			if r.status_code == 200:
				data = r.json()
				# Accept list or dict-wrapped list; normalize to list
				if isinstance(data, dict) and "alerts" in data and isinstance(data["alerts"], list):
					return t.cast(list[dict[str, t.Any]], data["alerts"])
				if isinstance(data, list):
					return t.cast(list[dict[str, t.Any]], data)
		except Exception:
			pass
	return []
