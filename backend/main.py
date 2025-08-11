"""
Mystic AI Trading Platform - Main Application Entry Point

Clean, simple FastAPI application entry point.
All complex logic is handled by the app factory.
"""

import uvicorn
from app_factory import create_app

# Create the FastAPI application using the factory
app = create_app()

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=9000, reload=True, log_level="info")
