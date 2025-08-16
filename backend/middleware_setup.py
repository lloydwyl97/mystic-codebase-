import logging

from fastapi.middleware.cors import CORSMiddleware

from backend.middleware.manager import get_middleware_manager

from .app_factory import app

logger = logging.getLogger("main")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:3000",
        "http://localhost:3000",
        "http://127.0.0.1:3001",
        "http://localhost:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register middleware if available
try:
    middleware_manager = get_middleware_manager()
    middleware_manager.register_all(app)
    logger.info("âœ… Middleware registered successfully")
except Exception as e:
    logger.warning(f"âš ï¸ Middleware registration failed: {e}")


