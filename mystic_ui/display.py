from __future__ import annotations
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import pandas as pd
import streamlit as st


def render_kpis(items: Sequence[Tuple[str, Any, Optional[str]]]) -> None:
	cols = st.columns(len(items))
	for col, (label, value, help_text) in zip(cols, items):
		try:
			col.metric(label, value if value is not None else "-")
		except Exception:
			col.metric(label, "-")


def to_table_rows(data: Any) -> List[Dict[str, Any]]:
	"""Best-effort: turn various API responses into list[dict] rows.
	- list[dict] -> itself
	- dict with 'data' list -> that list
	- dict with obvious collection keys -> pick first list-like
	"""
	if isinstance(data, list):
		return [x for x in data if isinstance(x, dict)]
	if isinstance(data, dict):
		if isinstance(data.get("data"), list):
			return [x for x in data["data"] if isinstance(x, dict)]
		# heuristic: choose first list value of dict
		for v in data.values():
			if isinstance(v, list) and v and isinstance(v[0], dict):
				return v
	return []


def render_table(data: Any, *, max_rows: int = 200, caption: Optional[str] = None) -> None:
	rows = to_table_rows(data)
	if not rows:
		st.info("No data available")
		return
	df = pd.DataFrame(rows)
	if len(df) > max_rows:
		df = df.head(max_rows)
	st.dataframe(df, use_container_width=True)
	if caption:
		st.caption(caption)


def try_get(d: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
	for k in keys:
		if k in d:
			return d[k]
	return default


def render_line_series(series: Mapping[str, Any], *, x_key: str, y_key: str, title: Optional[str] = None) -> None:
	try:
		df = pd.DataFrame(series)
		if x_key in df.columns and y_key in df.columns:
			st.line_chart(df.set_index(x_key)[y_key])
			if title:
				st.caption(title)
		else:
			st.info("Not enough data for chart")
	except Exception:
		st.info("Chart unavailable")


