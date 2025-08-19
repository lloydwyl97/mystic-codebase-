from __future__ import annotations

from typing import Any

_VOLUME_KEYS = ("quoteVolume", "volume_quote", "volume_24h", "volume", "vol")

def _lower(d: dict[str, Any]) -> dict[str, Any]:
	return {str(k).lower(): v for k, v in d.items()}

def _rows(payload: Any) -> list[dict]:
	if isinstance(payload, list):
		return [r for r in payload if isinstance(r, dict)]
	if isinstance(payload, dict):
		for key in ("data", "rows", "items", "result"):
			v = payload.get(key)
			if isinstance(v, list):
				return [r for r in v if isinstance(r, dict)]
		vals = list(payload.values())
		if vals and all(isinstance(x, dict) for x in vals):
			return vals
	return []

def resolve_top10_binanceus(base_api: str, limit: int = 10) -> list[str]:
	try:
		from mystic_ui.api_client import request_json as _request_json
	except Exception:
		from mystic_ui.api_client import _request_json as _request_json

	symbols: list[str] = []
	try:
		global_payload = _request_json(f"{base_api}/market/global", params=None)
		ranked: list[tuple[str, float]] = []
		for row in _rows(global_payload):
			L = _lower(row)
			sym = row.get("symbol") or L.get("symbol")
			if not sym:
				continue
			exch = (L.get("exchange") or "").strip().lower()
			if exch and exch != "binanceus":
				continue
			if not exch and not (sym.endswith("USDT") or sym.endswith("USD")):
				continue
			vol = 0.0
			for k in _VOLUME_KEYS:
				if k in row:
					try:
						vol = float(row[k]); break
					except Exception:
						pass
				lk = k.lower()
				if lk in L:
					try:
						vol = float(L[lk]); break
					except Exception:
						pass
			ranked.append((sym, vol))
		ranked.sort(key=lambda x: x[1], reverse=True)
		symbols = [s for s, _ in ranked[:limit]]
	except Exception:
		symbols = []

	if len(symbols) < limit:
		fallback = [
			"BTCUSDT","ETHUSDT","SOLUSDT","ADAUSDT","XRPUSDT",
			"DOGEUSDT","LTCUSDT","BCHUSDT","LINKUSDT","TRXUSDT",
		]
		for s in fallback:
			if s not in symbols:
				symbols.append(s)
			if len(symbols) >= limit:
				break
	return symbols[:limit]

