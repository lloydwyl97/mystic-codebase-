from fastapi import APIRouter

from backend.config.coins import FEATURED_SYMBOLS


router = APIRouter(prefix="/api/markets", tags=["markets"])


def _filter_to_featured(items):
    out = []
    for it in items:
        sym = it.get("symbol") if isinstance(it, dict) else str(it)
        if sym in FEATURED_SYMBOLS:
            out.append(it)
    return out

# Note: This module intentionally does not declare routes in this codebase variant.
# The helper above is provided for use by any instruments-returning handlers that import it.





