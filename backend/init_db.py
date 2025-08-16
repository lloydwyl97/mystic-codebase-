#!/usr/bin/env python3
"""
Database Initialization Script for Mystic Trading Platform
Creates and initializes the database with required tables and data
"""

import logging
import os
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def init_database():
    """Initialize the database with required tables"""
    logger.info("Initializing database...")

    try:
        # Import database modules
        from config import DATABASE_URL
        from database import create_tables, init_db

        logger.info(f"Database URL: {DATABASE_URL}")

        # Initialize database
        init_db()
        logger.info("âœ“ Database initialized")

        # Create tables
        create_tables()
        logger.info("âœ“ Tables created")

        return True

    except ImportError as e:
        logger.error(f"Database modules not available: {e}")
        return False
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        return False


def create_initial_data():
    """Create initial data for production"""
    logger.info("Creating initial production data...")

    try:
        # Import data initialization functions
        from init_data import init_sample_data
        import asyncio

        # Initialize with live data sources
        asyncio.run(init_sample_data())
        logger.info("âœ“ Initial data created")
        return True

    except ImportError as e:
        logger.warning(f"Data initialization modules not available: {e}")
        return True  # Not critical
    except Exception as e:
        logger.error(f"Data initialization failed: {e}")
        return False


def main():
    """Main database initialization procedure"""
    logger.info("ðŸ—„ï¸  Starting Database Initialization")

    # Change to backend directory if needed
    if not Path("main.py").exists():
        if Path("backend/main.py").exists():
            os.chdir("backend")
            logger.info("Changed to backend directory")

    # Initialize database
    if not init_database():
        logger.error("Database initialization failed")
        return 1

    # Create initial data (optional)
    create_initial_data()

    logger.info("ðŸŽ‰ Database initialization completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())


