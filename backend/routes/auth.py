"""
Auth Router - Authentication and Authorization

Contains login, logout, refresh token, and authentication endpoints.
"""

import logging
from typing import Dict

from fastapi import APIRouter, HTTPException

router = APIRouter()
logger = logging.getLogger(__name__)

# ============================================================================
# AUTHENTICATION ENDPOINTS
# ============================================================================


@router.post("/api/auth/login")
async def login(credentials: Dict[str, str]):
    """User login endpoint"""
    try:
        # Real authentication using auth service
        from services.auth_service import get_auth_service

        auth_service = get_auth_service()
        result = await auth_service.login(credentials)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Login failed: {str(e)}")


@router.post("/api/auth/logout")
async def logout():
    """User logout endpoint"""
    try:
        # Real logout using auth service
        from services.auth_service import get_auth_service

        auth_service = get_auth_service()
        result = await auth_service.logout()
        return result
    except Exception as e:
        logger.error(f"Logout error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Logout failed: {str(e)}")


@router.post("/api/auth/refresh")
async def refresh_token():
    """Refresh access token endpoint"""
    try:
        # Real token refresh using auth service
        from services.auth_service import get_auth_service

        auth_service = get_auth_service()
        result = await auth_service.refresh_token()
        return result
    except Exception as e:
        logger.error(f"Token refresh error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Token refresh failed: {str(e)}")
