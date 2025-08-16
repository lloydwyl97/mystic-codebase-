"""
Missing Risk Management Endpoints

Provides missing risk management endpoints that return live data:
- Risk Status
"""

import logging
from typing import Any, Dict

from fastapi import APIRouter, HTTPException


logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/api/risk/status")
async def get_risk_status() -> Dict[str, Any]:
    """
    Get risk management status with live data

    Returns comprehensive risk management information including:
    - Current risk levels
    - Risk metrics
    - Position limits
    - Risk alerts
    """
    try:
        from backend.services.risk_service import RiskService

        risk_service = RiskService()
        risk_status = await risk_service.get_status()
        return risk_status
    except Exception as e:
        logger.error(f"Error getting risk status: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting risk status: {str(e)}")



