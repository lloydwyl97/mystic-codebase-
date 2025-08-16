#!/usr/bin/env python3
"""
Docker runner script for Mystic Trading Platform
Sets up proper Python path and runs the FastAPI app
"""

import os
import sys

# Add the current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from main import app
except ImportError:
    # Fallback if the path modification didn't work
    sys.path.insert(0, current_dir)
    from main import app

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


