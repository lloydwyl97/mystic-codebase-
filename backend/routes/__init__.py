"""
DEPRECATED: Routes Module
This module is deprecated and replaced by endpoints/consolidated_router.py
All route loading is now handled by the consolidated router system
"""

import logging

from fastapi import FastAPI

logger = logging.getLogger(__name__)


def include_all_routes(app: FastAPI) -> None:
    """DEPRECATED: Include all route modules in the FastAPI app"""

    logger.warning(
        "âš ï¸ routes/__init__.py is deprecated - use endpoints/consolidated_router.py instead"
    )
    logger.info("âœ… Routes loading skipped - using consolidated router system")

    # No route loading here - all handled by consolidated_router.py
    # This function is kept for backward compatibility but does nothing


