"""
Simplified API Endpoints for Mystic Trading

Contains core API endpoint definitions that are not covered by modular endpoint files.
Uses shared endpoints to eliminate duplication with api_endpoints.py.
"""

import logging

from fastapi import APIRouter

# Import shared endpoints
from shared_endpoints import register_shared_endpoints

router = APIRouter()
logger = logging.getLogger(__name__)

# Register all shared endpoints with no prefix (for simplified version)
register_shared_endpoints(router, prefix="")

# Add any simplified-specific endpoints here if needed
# (These would be endpoints that only exist in the simplified version)

logger.info("✅ Simplified API endpoints loaded with shared endpoint consolidation")
