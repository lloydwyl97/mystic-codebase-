from __future__ import annotations

import os
import typing as t


_ICON_DIR = os.path.join(os.path.dirname(__file__), "assets", "icons")


def coin_svg_path(symbol: str) -> str | None:
    """Return local SVG path for a coin symbol if available."""
    sym = (symbol or "").upper()
    filename = f"{sym}.svg"
    path = os.path.join(_ICON_DIR, filename)
    return path if os.path.exists(path) else None


def get_coin_icon(symbol: str) -> dict[str, t.Any]:
    """Return a descriptor for UI to render an icon.

    Descriptor:
      {"type":"svg","path": ".../BTC.svg"} or {"type":"none"}
    """
    p = coin_svg_path(symbol)
    if p:
        return {"type": "svg", "path": p}
    return {"type": "none"}


def render_text_badge(text: str) -> dict[str, str]:
    """Return a style descriptor for a text badge; UI can decide rendering."""
    return {"type": "text", "text": str(text)}


