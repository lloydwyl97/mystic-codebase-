"""
Database utilities for Mystic Trading Platform

Provides database connection and management with standardized error handling.
"""

import logging
import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

try:
    import redis
except ImportError:
    redis = None

from backend.utils.exceptions import (
    DatabaseConnectionException,
    DatabaseException,
    handle_exception,
)

logger = logging.getLogger(__name__)

# Database configuration
DATABASE_PATH = "mystic_trading.db"

# Redis configuration
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0
REDIS_PASSWORD = None


# MockRedis class removed - no longer needed in production


# Global Redis client instance
_redis_client: Any | None = None


class DatabaseManager:
    """Database manager for SQLite operations with connection pooling and error handling"""

    def __init__(self, db_path: str = DATABASE_PATH):
        self.db_path = db_path
        self.connection: sqlite3.Connection | None = None
        self.is_connected = False

    def connect(self) -> None:
        """Establish database connection"""
        if self.is_connected and self.connection:
            return

        try:
            self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
            self.connection.row_factory = sqlite3.Row
            self.is_connected = True
            logger.info(f"Connected to database: {self.db_path}")
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise DatabaseConnectionException(f"Failed to connect to database: {e}")

    def disconnect(self) -> None:
        """Close database connection"""
        if self.connection:
            self.connection.close()
            self.connection = None
            self.is_connected = False
            logger.info("Database connection closed")

    def execute_query(self, query: str, parameters: tuple[Any, ...] = ()) -> list[Any]:
        """Execute a SQL query and return results"""
        if not self.is_connected or not self.connection:
            raise DatabaseConnectionException("Database not connected")

        try:
            cursor = self.connection.cursor()
            upper_query = query.strip().upper()
            # Skip BEGIN if already in a transaction
            if upper_query.startswith("BEGIN") and self.connection.in_transaction:
                return []
            cursor.execute(query, parameters)

            if upper_query.startswith(("SELECT", "PRAGMA")):
                return cursor.fetchall()
            elif upper_query.startswith(("BEGIN", "COMMIT", "ROLLBACK")):
                # Don't auto-commit transaction control statements
                return []
            else:
                if not self.connection.in_transaction:
                    self.connection.commit()
                return []

        except sqlite3.Error as e:
            logger.error(f"Query execution failed: {e}")
            raise DatabaseException(f"Query execution failed: {e}")

    def get_connection(self) -> sqlite3.Connection:
        """Get the current database connection"""
        if not self.is_connected or not self.connection:
            raise DatabaseConnectionException("Database not connected")
        return self.connection

    def __enter__(self) -> "DatabaseManager":
        """Context manager entry"""
        self.connect()
        return self

    def __exit__(
        self,
        exc_type: type | None,
        exc_val: BaseException | None,
        exc_tb: Any | None,
    ) -> None:
        """Context manager exit"""
        self.disconnect()


# Global database manager instance
_db_manager: DatabaseManager | None = None


def get_db() -> DatabaseManager:
    """Get singleton database manager instance"""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager


def init_database() -> None:
    """Initialize database with required tables"""
    db_manager = get_db()
    try:
        db_manager.connect()

        # Create tables
        db_manager.execute_query(
            """
            CREATE TABLE IF NOT EXISTS market_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                price REAL NOT NULL,
                timestamp TEXT NOT NULL,
                volume REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        db_manager.execute_query(
            """
            CREATE TABLE IF NOT EXISTS user_settings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                setting_name TEXT NOT NULL,
                setting_value TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        db_manager.execute_query(
            """
            CREATE TABLE IF NOT EXISTS trading_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                signal_type TEXT NOT NULL,
                price REAL,
                timestamp TEXT NOT NULL,
                confidence REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        db_manager.execute_query(
            """
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                trade_type TEXT NOT NULL,
                quantity REAL NOT NULL,
                price REAL NOT NULL,
                pnl REAL,
                strategy TEXT,
                status TEXT DEFAULT 'completed',
                entry_reason TEXT,
                exit_reason TEXT,
                risk_level TEXT,
                tags TEXT,
                timestamp TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        logger.info("Database initialized successfully")

    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise DatabaseException(f"Database initialization failed: {e}")
    finally:
        db_manager.disconnect()


@handle_exception("Failed to get Redis client", DatabaseException)
def get_redis_client() -> Any:
    """Get Redis client instance with connection pooling."""
    global _redis_client

    if _redis_client is None:
        try:
            _redis_client = create_redis_client()
            logger.info("Redis client initialized successfully")
        except Exception as e:
            logger.error(f"Redis connection error: {str(e)}")
            raise DatabaseException(f"Failed to initialize Redis client: {str(e)}")

    return _redis_client


@handle_exception("Failed to create Redis client", DatabaseException)
def create_redis_client() -> Any:
    """Create a Redis client with proper error handling."""
    try:
        if redis is None:
            raise DatabaseException("Redis module not available")

        client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            password=REDIS_PASSWORD,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
            retry_on_timeout=True,
            health_check_interval=30,
        )
        client.ping()
        return client
    except Exception as e:
        raise DatabaseException(f"Failed to create Redis client: {str(e)}")


@handle_exception("Failed to get database connection", DatabaseException)
def get_database_connection() -> sqlite3.Connection:
    """Get a SQLite database connection."""
    try:
        conn: sqlite3.Connection = sqlite3.connect(DATABASE_PATH, check_same_thread=False)
        conn.row_factory = sqlite3.Row  # Enable row factory for named access
        return conn
    except Exception as e:
        logger.error(f"Database connection error: {str(e)}")
        # Return in-memory database as fallback
        conn = sqlite3.connect(":memory:", check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn


def get_db_connection() -> sqlite3.Connection:
    """Get a database connection (alias for compatibility)"""
    return get_database_connection()


@handle_exception("Failed to initialize database", DatabaseException)
def initialize_database() -> None:
    """Initialize the database with basic tables."""
    conn: sqlite3.Connection = get_database_connection()
    try:
        cursor: sqlite3.Cursor = conn.cursor()

        # Create basic tables
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS market_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                price REAL NOT NULL,
                timestamp TEXT NOT NULL,
                volume REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS user_settings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                setting_name TEXT NOT NULL,
                setting_value TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS trading_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                signal_type TEXT NOT NULL,
                price REAL,
                timestamp TEXT NOT NULL,
                confidence REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                trade_type TEXT NOT NULL,
                quantity REAL NOT NULL,
                price REAL NOT NULL,
                pnl REAL,
                strategy TEXT,
                status TEXT DEFAULT 'completed',
                entry_reason TEXT,
                exit_reason TEXT,
                risk_level TEXT,
                tags TEXT,
                timestamp TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timeframe TEXT NOT NULL,
                total_trades INTEGER,
                winning_trades INTEGER,
                losing_trades INTEGER,
                win_rate REAL,
                total_pnl REAL,
                max_drawdown REAL,
                sharpe_ratio REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        conn.commit()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization error: {str(e)}")
        raise DatabaseException(f"Database initialization failed: {e}")
    finally:
        conn.close()


@contextmanager
def get_db_context() -> Generator[sqlite3.Connection, None, None]:
    """Context manager for database connections."""
    conn: sqlite3.Connection = get_database_connection()
    try:
        yield conn
    finally:
        conn.close()


# Initialize database on module import
initialize_database()


