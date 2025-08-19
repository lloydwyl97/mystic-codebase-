from __future__ import annotations

from typing import Any, Dict, List, Optional, cast

import pandas as pd
import streamlit as st

from mystic_ui.api_client import request_json as _req
from mystic_ui._archive_pages.components.common_utils import get_app_state, render_sidebar_controls  # public wrapper
from mystic_ui._archive_pages.components.common_utils import safe_number_format  # public wrapper
from mystic_ui._archive_pages.components.common_utils import inject_global_theme  # public wrapper

_st = cast(Any, st)


def _extract_positions(payload: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
	if not isinstance(payload, dict):
		return []
	# Accept multiple possible shapes
	positions_obj: Any = (
		payload.get("positions")
		or payload.get("portfolio", {}).get("positions")
		or payload.get("data", {}).get("positions")
		or payload.get("data")
	)
	if isinstance(positions_obj, list):
		return cast(List[Dict[str, Any]], positions_obj)
	return []


def _extract_orders(payload: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
	if not isinstance(payload, dict):
		return []
	orders_obj: Any = payload.get("orders", payload.get("data"))
	if isinstance(orders_obj, list):
		return cast(List[Dict[str, Any]], orders_obj)
	return []


def _extract_pnl_summary(payload_overview: Optional[Dict[str, Any]]) -> Dict[str, Any]:
	# Try common locations for P&L / equity snapshot
	ov = payload_overview if isinstance(payload_overview, dict) else {}
	portfolio: Dict[str, Any] = cast(Dict[str, Any], ov.get("portfolio", {})) if isinstance(ov.get("portfolio"), dict) else {}
	perf: Dict[str, Any] = cast(Dict[str, Any], ov.get("performance", {})) if isinstance(ov.get("performance"), dict) else {}
	market_data: Dict[str, Any] = cast(Dict[str, Any], ov.get("market_data", {})) if isinstance(ov.get("market_data"), dict) else {}

	return {
		"equity": portfolio.get("total_value") or portfolio.get("equity") or perf.get("equity"),
		"unrealized_pnl": portfolio.get("unrealized_pnl") or perf.get("unrealized_pnl"),
		"realized_pnl": portfolio.get("realized_pnl") or perf.get("realized_pnl"),
		"daily_change_pct": portfolio.get("daily_change") or perf.get("daily_change_pct"),
		"market": market_data.get("source"),
	}


def _format_positions_table(rows: List[Dict[str, Any]]) -> pd.DataFrame:
	if not rows:
		return pd.DataFrame()
	df = pd.DataFrame(rows)
	preferred_cols = [
		"symbol", "side", "qty", "entry_price", "mark_price", "unrealized_pnl", "pnl_pct",
	]
	cols = [c for c in preferred_cols if c in df.columns] + [c for c in df.columns if c not in preferred_cols]
	df = df[cols]
	# Number coercion for numeric-like columns
	num_cols: List[str] = [str(c) for c in df.columns if c not in ("symbol", "side", "status")]
	for c in num_cols:
		try:
			df[c] = pd.to_numeric(df[c], errors="coerce")  # type: ignore[call-overload]
		except Exception:
			pass
	return df


def _format_orders_table(rows: List[Dict[str, Any]]) -> pd.DataFrame:
	if not rows:
		return pd.DataFrame()
	df = pd.DataFrame(rows)
	preferred = [
		"timestamp", "symbol", "side", "order_type", "status", "price", "avg_fill_price", "qty", "filled", "remaining",
	]
	cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
	df = df[cols]
	numeric: List[str] = ["price", "avg_fill_price", "qty", "filled", "remaining"]
	for c in numeric:
		if c in df.columns:
			try:
				df[c] = pd.to_numeric(df[c], errors="coerce")  # type: ignore[call-overload]
			except Exception:
				pass
	return df


def main() -> None:
	# set_page_config is centralized in mystic_ui/app.py
	inject_global_theme()
	render_sidebar_controls()
	_ = get_app_state()  # ensure defaults

	# Fetch data with guards
	with _st.spinner("Loading portfolio & orders…"):
		try:
			ov_payload_any = _req("GET", "/api/portfolio/overview")
		except Exception:
			ov_payload_any = None
		try:
			pos_payload_any = _req("GET", "/api/live/trading/positions")
		except Exception:
			pos_payload_any = None
		try:
			ord_payload_any = _req("GET", "/api/live/trading/orders")
		except Exception:
			ord_payload_any = None

	ov_payload: Optional[Dict[str, Any]] = cast(Optional[Dict[str, Any]], ov_payload_any if isinstance(ov_payload_any, dict) else None)
	pos_payload: Optional[Dict[str, Any]] = cast(Optional[Dict[str, Any]], pos_payload_any if isinstance(pos_payload_any, dict) else None)
	ord_payload: Optional[Dict[str, Any]] = cast(Optional[Dict[str, Any]], ord_payload_any if isinstance(ord_payload_any, dict) else None)

	positions = _extract_positions(pos_payload or ov_payload)
	orders = _extract_orders(ord_payload)
	pnl = _extract_pnl_summary(ov_payload)

	# Compact info notices if any payloads are empty/unavailable
	if not ov_payload:
		_st.info("Portfolio overview unavailable")
	if not pos_payload:
		_st.info("Positions unavailable")
	if not ord_payload:
		_st.info("Orders unavailable")

	# Header strip with compact P&L metrics
	m1, m2, m3, m4 = _st.columns(4)
	with m1:
		_st.metric("Equity", safe_number_format(pnl.get("equity"), 2))
	with m2:
		_st.metric("Unrealized P&L", safe_number_format(pnl.get("unrealized_pnl"), 2))
	with m3:
		_st.metric("Realized P&L", safe_number_format(pnl.get("realized_pnl"), 2))
	with m4:
		chg = pnl.get("daily_change_pct")
		delta = f"{safe_number_format(chg, 2)}%" if chg is not None else None
		_st.metric("24h Change", delta or "0.00%")

	# Two-column layout: positions and orders
	left, right = _st.columns([3, 2])
	with left:
		_st.markdown("### Positions")
		if positions:
			df_pos = _format_positions_table(positions)
			_st.dataframe(df_pos, use_container_width=True)
		else:
			_st.info("No positions.")
	with right:
		_st.markdown("### Open Orders")
		if orders:
			df_ord = _format_orders_table(orders)
			_st.dataframe(df_ord, use_container_width=True)
		else:
			_st.info("No open orders.")

	# Raw payload expanders
	with _st.expander("Raw JSON: Portfolio", expanded=False):
		_st.json(ov_payload)
	with _st.expander("Raw JSON: Orders", expanded=False):
		_st.json(ord_payload)


if __name__ == "__main__":
	main()



