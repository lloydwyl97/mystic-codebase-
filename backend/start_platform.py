#!/usr/bin/env python3
"""
Mystic AI Trading Platform Startup Script
Handles initialization and starts the platform
"""

import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("platform.log"),
        ],
    )
    return logging.getLogger("startup")


def check_dependencies():
    """Check if all required dependencies are installed"""
    logger = logging.getLogger("startup")

    required_packages = [
        "fastapi",
        "uvicorn",
        "sqlalchemy",
        "redis",
        "requests",
        "pandas",
        "numpy",
    ]

    missing_packages: List[str] = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        logger.error(f"Missing packages: {missing_packages}")
        logger.info("Run: pip install -r requirements.txt")
        return False

    logger.info("âœ… All dependencies are installed")
    return True


def initialize_database():
    """Initialize database with live data connections"""
    logger = logging.getLogger("startup")

    try:
        from setup_wallet_system import main as setup_main

        if setup_main():
            logger.info("âœ… Database initialized successfully")
            return True
        else:
            logger.error("âŒ Database initialization failed")
            return False
    except Exception as e:
        logger.error(f"âŒ Database initialization error: {e}")
        return False


def start_backend():
    """Start the backend server"""
    logger = logging.getLogger("startup")

    try:
        logger.info("ðŸš€ Starting backend server...")

        # Start uvicorn server
        process = subprocess.Popen(
            [
                "python",
                "-m",
                "uvicorn",
                "main:app",
                "--host",
                "127.0.0.1",
                "--port",
                "9000",
                "--reload",
            ]
        )

        # Wait a moment for server to start
        time.sleep(3)

        if process.poll() is None:
            logger.info("âœ… Backend server started successfully")
            logger.info("ðŸ“Š API Documentation: http://localhost:9000/docs")
            logger.info("ðŸ¥ Health Check: http://localhost:9000/api/health")
            return process
        else:
            logger.error("âŒ Backend server failed to start")
            return None

    except Exception as e:
        logger.error(f"âŒ Backend startup error: {e}")
        return None


def start_frontend():
    """Start the frontend development server"""
    logger = logging.getLogger("startup")

    frontend_dir = Path("../frontend")
    if not frontend_dir.exists():
        logger.warning("âš ï¸ Frontend directory not found, skipping frontend startup")
        return None

    try:
        logger.info("ðŸš€ Starting frontend server...")

        # Change to frontend directory
        os.chdir(frontend_dir)

        # Start npm development server
        process = subprocess.Popen(["npm", "start"])

        # Wait a moment for server to start
        time.sleep(5)

        if process.poll() is None:
            logger.info("âœ… Frontend server started successfully")
            logger.info("ðŸ“± Dashboard: http://localhost:3000")
            return process
        else:
            logger.error("âŒ Frontend server failed to start")
            return None

    except Exception as e:
        logger.error(f"âŒ Frontend startup error: {e}")
        return None


def main():
    """Main startup function"""
    logger = setup_logging()

    logger.info("ðŸŽ¯ Mystic AI Trading Platform Startup")
    logger.info("=" * 50)

    # Check dependencies
    if not check_dependencies():
        return False

    # Initialize database
    if not initialize_database():
        return False

    # Start backend
    backend_process = start_backend()
    if not backend_process:
        return False

    # Start frontend (optional)
    frontend_process = start_frontend()

    logger.info("=" * 50)
    logger.info("ðŸŽ‰ Platform startup completed!")
    logger.info("")
    logger.info("ðŸ“Š Available Services:")
    logger.info("   Backend API: http://localhost:9000")
    logger.info("   API Docs: http://localhost:9000/docs")
    logger.info("   Health Check: http://localhost:9000/api/health")
    if frontend_process:
        logger.info("   Dashboard: http://localhost:3000")
    logger.info("")
    logger.info("ðŸ›‘ Press Ctrl+C to stop all services")

    try:
        # Keep the processes running
        while True:
            time.sleep(1)

            # Check if processes are still running
            if backend_process and backend_process.poll() is not None:
                logger.error("âŒ Backend process stopped unexpectedly")
                break

            if frontend_process and frontend_process.poll() is not None:
                logger.warning("âš ï¸ Frontend process stopped")
                frontend_process = None

    except KeyboardInterrupt:
        logger.info("ðŸ›‘ Shutting down platform...")

        # Stop processes
        if backend_process:
            backend_process.terminate()
            backend_process.wait()
            logger.info("âœ… Backend stopped")

        if frontend_process:
            frontend_process.terminate()
            frontend_process.wait()
            logger.info("âœ… Frontend stopped")

        logger.info("ðŸ‘‹ Platform shutdown complete")
        return True


if __name__ == "__main__":
    main()


