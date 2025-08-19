"""
Compatibility Router
Non-invasive shim that exposes missing UI endpoints and path aliases under /api
without changing business logic. Prefer delegating to existing routers/services.
"""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="", tags=["compat"])  # mounted under /api by app factory


@router.get("/system/performance")
async def system_performance() -> Dict[str, Any]:
	try:
		# Try enhanced API performance if available
		try:
			from backend.endpoints.enhanced_api.enhanced_api_endpoints import performance_monitor  # type: ignore
		except Exception:
			performance_monitor = None  # type: ignore
		if performance_monitor and hasattr(performance_monitor, "get_metrics"):
			return {"performance": performance_monitor.get_metrics()}
		# Fallback minimal payload
		return {"performance": {"available": False}}
	except Exception as e:
		raise HTTPException(status_code=500, detail=f"system/performance error: {e}")


@router.get("/system/config")
async def system_config() -> Dict[str, Any]:
	try:
		cfg: Dict[str, Any] = {}
		try:
			from backend.config import settings  # type: ignore
			cfg = {
				"environment": getattr(settings, "ENV", None),
				"version": getattr(settings, "VERSION", None),
				"exchanges": getattr(settings, "EXCHANGES", None),
			}
		except Exception:
			cfg = {}
		return {"config": cfg}
	except Exception as e:
		raise HTTPException(status_code=500, detail=f"system/config error: {e}")


@router.get("/coins")
async def coins_proxy() -> Dict[str, Any]:
	"""Expose /api/coins by delegating to crypto_autoengine if present."""
	try:
		try:
			from crypto_autoengine_api import get_coins as _get_coins  # type: ignore
		except Exception:
			_get_coins = None  # type: ignore
		if _get_coins:
			return await _get_coins()  # type: ignore[misc]
		return {"all_symbols": [], "enabled_symbols": []}
	except Exception as e:
		raise HTTPException(status_code=500, detail=f"coins proxy error: {e}")


