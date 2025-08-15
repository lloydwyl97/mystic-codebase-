import os
import sys

from typing import Any, Dict
from fastapi import APIRouter, Query

# Allow importing from backend services without package prefix
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from services.ai_attribution import load_attribution  # noqa: E402


router = APIRouter(prefix="/api/ai/explain", tags=["ai"])


@router.get("/attribution")
def attribution(symbol: str = Query(...)) -> Dict[str, Any]:
    data: Dict[str, Any] = load_attribution(symbol)
    return {"ok": True, "symbol": symbol, **data}


