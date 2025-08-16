#!/usr/bin/env python3
"""
Setup script for Mystic AI Wallet System
Initializes database tables and sample data for the real-time wallet panel
"""

import json
import sqlite3
from datetime import datetime


def setup_database():
    """Initialize database with required tables"""
    db_path = "simulation_trades.db"

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create enhanced trades table
    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS simulated_trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        symbol TEXT NOT NULL,
        action TEXT NOT NULL,
        price REAL NOT NULL,
        confidence REAL,
        simulated_profit REAL,
        strategy TEXT,
        mystic_signals TEXT,
        wallet_source TEXT DEFAULT 'Main AI Trading'
    )
    """
    )

    # Create wallet allocations table
    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS wallet_allocations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        wallet_name TEXT NOT NULL,
        allocation_percent REAL NOT NULL,
        current_balance REAL DEFAULT 0,
        last_updated TEXT,
        status TEXT DEFAULT 'active'
    )
    """
    )

    # Create yield tracking table
    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS yield_positions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        provider TEXT NOT NULL,
        protocol TEXT NOT NULL,
        amount_deployed REAL NOT NULL,
        apy REAL NOT NULL,
        start_date TEXT,
        status TEXT DEFAULT 'active'
    )
    """
    )

    # Create cold wallet syncs table
    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS cold_wallet_syncs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        amount REAL NOT NULL,
        timestamp TEXT NOT NULL,
        threshold_triggered REAL,
        status TEXT DEFAULT 'completed'
    )
    """
    )

    conn.commit()
    conn.close()
    print("âœ… Database tables created successfully")


def insert_live_data():
    """Initialize live data connections and empty tables for real-time data"""
    db_path = "simulation_trades.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Initialize empty wallet allocations - will be populated by live trading system
    cursor.execute("DELETE FROM wallet_allocations")
    cursor.execute("DELETE FROM yield_positions")
    cursor.execute("DELETE FROM cold_wallet_syncs")
    cursor.execute("DELETE FROM simulated_trades")

    # Create initial wallet structure for live data
    live_wallets = [
        ("Main AI Trading", 50.0, 0.0),  # Will be populated by live balance
        ("Backup AI Trading", 30.0, 0.0),  # Will be populated by live balance
        ("Cold Storage Vault", 20.0, 0.0),  # Will be populated by live balance
    ]

    cursor.executemany(
        """
    INSERT OR REPLACE INTO wallet_allocations (wallet_name, allocation_percent, current_balance, last_updated, status)
    VALUES (?, ?, ?, ?, 'waiting_for_live_data')
    """,
        [(w[0], w[1], w[2], datetime.timezone.utcnow().isoformat()) for w in live_wallets],
    )

    # Initialize empty yield positions - will be populated by live DeFi APIs
    cursor.execute(
        """
    INSERT OR REPLACE INTO yield_positions (provider, protocol, amount_deployed, apy, start_date, status)
    VALUES ('Initializing', 'Live Data', 0.0, 0.0, ?, 'waiting_for_live_data')
    """,
        (datetime.timezone.utcnow().isoformat(),),
    )

    # Initialize empty cold wallet syncs - will be populated by live wallet monitoring
    cursor.execute(
        """
    INSERT OR REPLACE INTO cold_wallet_syncs (amount, timestamp, threshold_triggered, status)
    VALUES (0.0, ?, 0.0, 'waiting_for_live_data')
    """,
        (datetime.timezone.utcnow().isoformat(),),
    )

    conn.commit()
    conn.close()
    print("âœ… Live data tables initialized - waiting for real-time data connections")
    print("   Note: Data will be populated by live trading system and API connections")


def create_ai_model_state():
    """Create initial AI model state file"""
    model_state = {
        "version": 1,
        "mode": "training",
        "confidence_threshold": 0.75,
        "avg_profit_threshold": 0.5,
        "adjustment_count": 0,
        "last_update": datetime.timezone.utcnow().isoformat(),
        "performance_metrics": {
            "total_trades": 0,
            "win_rate": 0.0,
            "avg_profit": 0.0,
            "total_profit": 0.0,
        },
    }

    with open("ai_model_state.json", "w") as f:
        json.dump(model_state, f, indent=2)

    print("âœ… AI model state file created")


def create_config_files():
    """Create configuration files for the system"""

    # Environment variables template
    env_template = """
# Mystic AI Wallet System Configuration
SIM_DB_PATH=simulation_trades.db
MODEL_STATE_PATH=ai_model_state.json

# Discord/Telegram Notifications (optional)
DISCORD_WEBHOOK=your_discord_webhook_url
TELEGRAM_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_chat_id

# Cold Wallet Configuration
COLD_WALLET_THRESHOLD=250.00
COLD_WALLET_ADDRESS=your_cold_wallet_address

# Yield Rotation Settings
YIELD_ROTATION_THRESHOLD=0.005
MAX_YIELD_ALLOCATION=0.4
    """.strip()

    with open("env.example", "w") as f:
        f.write(env_template)

    print("âœ… Configuration files created")


def main():
    """Main setup function"""
    print("ðŸš€ Setting up Mystic AI Wallet System...")

    try:
        setup_database()
        insert_live_data()
        create_ai_model_state()
        create_config_files()

        print("\nðŸŽ‰ Setup completed successfully!")
        print("\nðŸ“‹ Next steps:")
        print("1. Copy env.example to .env and configure your settings")
        print("2. Start the backend: uvicorn main:app --reload")
        print("3. Start the frontend: npm start")
        print("4. Access the dashboard at: http://localhost:3000")
        print("5. View API docs at: http://localhost:8000/docs")

    except Exception as e:
        print(f"âŒ Setup failed: {str(e)}")
        return False

    return True


if __name__ == "__main__":
    main()


