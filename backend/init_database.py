#!/usr/bin/env python3
"""
Database Initialization Script for Mystic Trading Platform
Fixes database file access issues and ensures proper setup
"""

import os
import sys
import sqlite3
import json
from pathlib import Path
from datetime import datetime


def ensure_database_directory():
    """Ensure database directory exists with proper permissions"""
    try:
        # Get the current working directory
        current_dir = Path.cwd()
        db_dir = current_dir / "data"

        # Create data directory if it doesn't exist
        if not db_dir.exists():
            db_dir.mkdir(parents=True, exist_ok=True)
            print(f"Created database directory: {db_dir}")

        # Set proper permissions
        os.chmod(db_dir, 0o755)

        return db_dir
    except Exception as e:
        print(f"Error creating database directory: {e}")
        return None


def create_database_file(db_path):
    """Create SQLite database file with proper permissions"""
    try:
        # Create database file if it doesn't exist
        if not db_path.exists():
            # Create empty database file
            conn = sqlite3.connect(db_path)
            conn.close()

            # Set proper permissions
            os.chmod(db_path, 0o644)
            print(f"Created database file: {db_path}")

        return True
    except Exception as e:
        print(f"Error creating database file: {e}")
        return False


def initialize_database_schema(db_path):
    """Initialize database with basic schema"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Create tables
        tables = [
            """
            CREATE TABLE IF NOT EXISTS market_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                price REAL NOT NULL,
                volume REAL NOT NULL,
                change_24h REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                exchange TEXT DEFAULT 'binance'
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS portfolios (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                portfolio_id TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                total_value REAL NOT NULL,
                cash REAL NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                portfolio_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                amount REAL NOT NULL,
                average_price REAL NOT NULL,
                current_value REAL NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS trade_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                amount REAL NOT NULL,
                price REAL NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                strategy TEXT,
                portfolio_id TEXT,
                status TEXT DEFAULT 'executed'
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS strategies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                description TEXT,
                parameters TEXT,
                performance_metrics TEXT,
                is_active BOOLEAN DEFAULT 1,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS system_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                level TEXT NOT NULL,
                component TEXT NOT NULL,
                message TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT
            )
            """,
        ]

        for table_sql in tables:
            cursor.execute(table_sql)

        # Create indexes
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_market_data_symbol ON market_data(symbol)",
            "CREATE INDEX IF NOT EXISTS idx_market_data_timestamp ON market_data(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_market_data_symbol_timestamp ON market_data(symbol, timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_portfolios_id ON portfolios(portfolio_id)",
            "CREATE INDEX IF NOT EXISTS idx_positions_portfolio ON positions(portfolio_id)",
            "CREATE INDEX IF NOT EXISTS idx_positions_symbol ON positions(symbol)",
            "CREATE INDEX IF NOT EXISTS idx_positions_portfolio_symbol ON positions(portfolio_id, symbol)",
            "CREATE INDEX IF NOT EXISTS idx_trade_logs_symbol ON trade_logs(symbol)",
            "CREATE INDEX IF NOT EXISTS idx_trade_logs_timestamp ON trade_logs(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_trade_logs_symbol_timestamp ON trade_logs(symbol, timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_strategies_name ON strategies(name)",
            "CREATE INDEX IF NOT EXISTS idx_strategies_active ON strategies(is_active)",
            "CREATE INDEX IF NOT EXISTS idx_system_logs_timestamp ON system_logs(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_system_logs_level ON system_logs(level)",
            "CREATE INDEX IF NOT EXISTS idx_system_logs_component ON system_logs(component)",
        ]

        for index_sql in indexes:
            cursor.execute(index_sql)

        conn.commit()
        conn.close()

        print("Database schema initialized successfully")
        return True

    except Exception as e:
        print(f"Error initializing database schema: {e}")
        return False


def insert_sample_data(db_path):
    """Insert sample data for testing"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Check if sample data already exists
        cursor.execute("SELECT COUNT(*) FROM portfolios")
        if cursor.fetchone()[0] > 0:
            print("Sample data already exists, skipping...")
            return True

        # Insert sample portfolio
        cursor.execute(
            """
            INSERT INTO portfolios (portfolio_id, name, total_value, cash)
            VALUES (?, ?, ?, ?)
        """,
            ("default", "Default Trading Portfolio", 100000.0, 15000.0),
        )

        # Insert sample positions
        positions = [
            ("default", "BTC/USDT", 0.5, 50000.0, 25000.0),
            ("default", "ETH/USDT", 5.0, 3000.0, 15000.0),
            ("default", "BNB/USDT", 50.0, 300.0, 15000.0),
            ("default", "ADA/USDT", 10000.0, 1.5, 15000.0),
            ("default", "SOL/USDT", 100.0, 150.0, 15000.0),
        ]

        cursor.executemany(
            """
            INSERT INTO positions (portfolio_id, symbol, amount, average_price, current_value)
            VALUES (?, ?, ?, ?, ?)
        """,
            positions,
        )

        # Insert sample strategies
        strategies = [
            (
                "Momentum_AI_v1",
                "AI-powered momentum trading strategy",
                json.dumps({"lookback_period": 14, "threshold": 0.02}),
                json.dumps({"sharpe_ratio": 1.85, "total_return": 0.23}),
            ),
            (
                "Mean_Reversion_AI_v2",
                "AI-powered mean reversion strategy",
                json.dumps({"window_size": 20, "std_dev_threshold": 2.0}),
                json.dumps({"sharpe_ratio": 1.42, "total_return": 0.18}),
            ),
        ]

        cursor.executemany(
            """
            INSERT INTO strategies (name, description, parameters, performance_metrics)
            VALUES (?, ?, ?, ?)
        """,
            strategies,
        )

        # Insert sample market data
        market_data = [
            ("BTC/USDT", 50000.0, 10000.0, 2.5),
            ("ETH/USDT", 3000.0, 8000.0, 1.8),
            ("BNB/USDT", 300.0, 5000.0, 3.2),
            ("ADA/USDT", 1.5, 3000.0, -1.2),
            ("SOL/USDT", 150.0, 4000.0, 4.1),
        ]

        cursor.executemany(
            """
            INSERT INTO market_data (symbol, price, volume, change_24h)
            VALUES (?, ?, ?, ?)
        """,
            market_data,
        )

        conn.commit()
        conn.close()

        print("Sample data inserted successfully")
        return True

    except Exception as e:
        print(f"Error inserting sample data: {e}")
        return False


def test_database_connection(db_path):
    """Test database connection and basic operations"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Test basic queries
        cursor.execute("SELECT COUNT(*) FROM portfolios")
        portfolio_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM strategies")
        strategy_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM market_data")
        market_data_count = cursor.fetchone()[0]

        conn.close()

        print("Database connection test successful:")
        print(f"  - Portfolios: {portfolio_count}")
        print(f"  - Strategies: {strategy_count}")
        print(f"  - Market data records: {market_data_count}")

        return True

    except Exception as e:
        print(f"Database connection test failed: {e}")
        return False


def main():
    """Main initialization function"""
    print("=== Mystic Trading Database Initialization ===")
    print(f"Timestamp: {datetime.now()}")

    # Ensure database directory exists
    db_dir = ensure_database_directory()
    if not db_dir:
        print("Failed to create database directory")
        sys.exit(1)

    # Create database file
    db_path = db_dir / "mystic_trading.db"
    if not create_database_file(db_path):
        print("Failed to create database file")
        sys.exit(1)

    # Initialize schema
    if not initialize_database_schema(db_path):
        print("Failed to initialize database schema")
        sys.exit(1)

    # Insert sample data
    if not insert_sample_data(db_path):
        print("Failed to insert sample data")
        sys.exit(1)

    # Test connection
    if not test_database_connection(db_path):
        print("Database connection test failed")
        sys.exit(1)

    print("\n=== Database Initialization Complete ===")
    print(f"Database location: {db_path}")
    print(f"Database size: {db_path.stat().st_size / 1024:.2f} KB")

    # Update environment variable for Docker
    print("\nSet DATABASE_URL environment variable to:")
    print(f"sqlite:///{db_path}")


if __name__ == "__main__":
    main()
