from __future__ import annotations

from typing import Optional
from pathlib import Path
import streamlit as st
import plotly.graph_objects as go


def _asset_paths(base: str) -> list[Path]:
    here = Path(__file__).resolve()
    # Streamlit repo root -> streamlit/assets/icons
    icons_root = here.parent.parent.parent / "assets" / "icons"
    return [icons_root / f"{base}.png", icons_root / f"{base}.svg"]


def get_coin_icon(base_or_symbol: str) -> Optional[str]:
    base = base_or_symbol.upper().split("-")[0].replace("/", "-")
    for p in _asset_paths(base):
        try:
            if p.exists():
                return str(p)
        except Exception:
            continue
    return None


def render_text_badge(base_or_symbol: str, size: int = 36):
    base = base_or_symbol.upper().split("-")[0]
    fig = go.Figure()
    fig.add_shape(type="circle", x0=0, y0=0, x1=1, y1=1, line=dict(color="#888"))
    fig.add_annotation(x=0.5, y=0.5, text=base, showarrow=False, font=dict(size=14, color="#EEE"))
    fig.update_layout(
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        width=size, height=size, margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig, use_container_width=False)


