from typing import List

from backend.config.coins import FEATURED_SYMBOLS


def filter_featured_symbols(symbols: List[str]) -> List[str]:
    return [s for s in symbols if s in FEATURED_SYMBOLS]




