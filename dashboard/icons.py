from __future__ import annotations

import streamlit as st
from pathlib import Path
from typing import Optional


def _asset_paths(base: str) -> list[Path]:
    here = Path(__file__).resolve()
    icons_root = here.parent.parent / "streamlit" / "assets" / "icons"
    return [icons_root / f"{base}.png", icons_root / f"{base}.svg"]


def get_coin_icon(base_or_symbol: str) -> Optional[str]:
    """Get coin icon path if exists, return None if not found"""
    base = base_or_symbol.upper().split("-")[0].split("/")[0]
    for p in _asset_paths(base):
        try:
            if p.exists():
                return str(p)
        except Exception:
            continue
    return None


def render_text_badge(symbol: str, size: int = 32) -> None:
    """Render a text badge for symbols without icons"""
    base = symbol.upper().split("-")[0].split("/")[0]
    # Create a circular text badge
    st.markdown(
        f"""
        <div style="
            width: {size}px; 
            height: {size}px; 
            border-radius: 50%; 
            background: linear-gradient(135deg, #00e6e6, #0099cc);
            display: flex; 
            align-items: center; 
            justify-content: center; 
            color: white; 
            font-weight: bold; 
            font-size: {max(8, size//4)}px;
            margin: 4px 0;
        ">
            {base[:3]}
        </div>
        """,
        unsafe_allow_html=True
    )


