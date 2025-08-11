"""
DEPRECATED: Router Setup
This file is deprecated and replaced by endpoints/consolidated_router.py
All router loading is now handled by the consolidated router system
"""

import logging

logger = logging.getLogger("main")

# This file is deprecated - all router loading is now handled by consolidated_router.py
logger.warning("⚠️ router_setup.py is deprecated - use endpoints/consolidated_router.py instead")

# Import app from main instead of app_factory
try:
    from main import app
except ImportError:
    # Fallback: create app directly
    from fastapi import FastAPI

    app = FastAPI(title="Mystic AI Trading Platform")

# No router loading here - all handled by consolidated_router.py
logger.info("✅ Router setup skipped - using consolidated router system")
