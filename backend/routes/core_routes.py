"""
Core Routes
Basic API endpoints for the application
"""

import logging
from typing import Any

from fastapi import APIRouter

logger = logging.getLogger("core_routes")

router = APIRouter(tags=["core"])


@router.get("/")
async def root() -> dict[str, str]:
    """Root endpoint"""
    return {"message": "Mystic AI Trading Platform API"}


@router.get("/test")
async def test_endpoint() -> dict[str, Any]:
    """Test endpoint for basic functionality"""
    return {
        "status": "success",
        "message": "API is working correctly",
        "timestamp": "2024-01-15T14:30:00Z",
    }


@router.get("/version")
async def get_version() -> dict[str, Any]:
    """Get application version information"""
    return {
        "version": "1.0.0",
        "name": "Mystic AI Trading Platform",
        "description": ("Advanced AI-powered trading platform with cosmic analysis"),
        "features": [
            "AI Trading Engine",
            "Multi-Wallet Management",
            "DeFi Yield Optimization",
            "Cold Storage Automation",
            "Real-time Dashboard",
        ],
    }


