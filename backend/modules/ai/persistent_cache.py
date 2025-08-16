"""
Persistent Cache Module for AI Services
Provides SQLite-based persistent caching functionality for AI services and data.
"""

import sqlite3
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


class PersistentCache:
    def __init__(self, db_path: str = "cache.db"):
        """Initialize persistent cache with SQLite database"""
        self.db_path = db_path
        self.connection = None
        self._init_database()

    def _init_database(self):
        """Initialize database and create tables if they don't exist"""
        try:
            self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
            self.connection.row_factory = sqlite3.Row

            # Create tables
            self._create_tables()
            logger.info(f"âœ… Persistent cache initialized with database: {self.db_path}")
        except Exception as e:
            logger.error(f"âŒ Failed to initialize persistent cache: {e}")
            raise

    def _create_tables(self):
        """Create database tables if they don't exist"""
        cursor = self.connection.cursor()

        # Market data table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS market_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                exchange TEXT NOT NULL,
                symbol TEXT NOT NULL,
                price REAL NOT NULL,
                volume REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Trade journal table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trade_journal (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id TEXT UNIQUE NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                quantity REAL NOT NULL,
                price REAL NOT NULL,
                total_value REAL NOT NULL,
                exchange TEXT NOT NULL,
                status TEXT DEFAULT 'pending',
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # AI signals table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ai_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_id TEXT UNIQUE NOT NULL,
                symbol TEXT NOT NULL,
                signal_type TEXT NOT NULL,
                confidence REAL NOT NULL,
                price_target REAL,
                stop_loss REAL,
                take_profit REAL,
                strategy TEXT,
                metadata TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create indexes for better performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_market_data_symbol ON market_data(symbol)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_market_data_timestamp ON market_data(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_trade_journal_symbol ON trade_journal(symbol)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_trade_journal_timestamp ON trade_journal(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_ai_signals_symbol ON ai_signals(symbol)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_ai_signals_timestamp ON ai_signals(timestamp)")

        self.connection.commit()

    def set_price(self, exchange: str, symbol: str, price: float, volume: Optional[float] = None) -> bool:
        """Store price data in the cache"""
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                INSERT INTO market_data (exchange, symbol, price, volume)
                VALUES (?, ?, ?, ?)
            """, (exchange, symbol, price, volume))
            self.connection.commit()
            return True
        except Exception as e:
            logger.error(f"âŒ Failed to set price for {symbol}: {e}")
            return False

    def get_latest_price(self, exchange: str, symbol: str) -> Optional[float]:
        """Get the latest price for a symbol from a specific exchange"""
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                SELECT price FROM market_data
                WHERE exchange = ? AND symbol = ?
                ORDER BY timestamp DESC
                LIMIT 1
            """, (exchange, symbol))
            result = cursor.fetchone()
            return result['price'] if result else None
        except Exception as e:
            logger.error(f"âŒ Failed to get latest price for {symbol}: {e}")
            return None

    def get_price_history(self, exchange: str, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get price history for a symbol"""
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                SELECT price, volume, timestamp
                FROM market_data
                WHERE exchange = ? AND symbol = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (exchange, symbol, limit))
            return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"âŒ Failed to get price history for {symbol}: {e}")
            return []

    def log_trade(self, trade_id: str, symbol: str, side: str, quantity: float,
                  price: float, exchange: str, status: str = "pending") -> bool:
        """Log a trade in the journal"""
        try:
            total_value = quantity * price
            cursor = self.connection.cursor()
            cursor.execute("""
                INSERT INTO trade_journal (trade_id, symbol, side, quantity, price, total_value, exchange, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (trade_id, symbol, side, quantity, price, total_value, exchange, status))
            self.connection.commit()
            return True
        except Exception as e:
            logger.error(f"âŒ Failed to log trade {trade_id}: {e}")
            return False

    def get_trades(self, symbol: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get trade history"""
        try:
            cursor = self.connection.cursor()
            if symbol:
                cursor.execute("""
                    SELECT * FROM trade_journal
                    WHERE symbol = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (symbol, limit))
            else:
                cursor.execute("""
                    SELECT * FROM trade_journal
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (limit,))
            return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"âŒ Failed to get trades: {e}")
            return []

    def store_signal(self, signal_id: str, symbol: str, signal_type: str, confidence: float,
                    price_target: Optional[float] = None, stop_loss: Optional[float] = None,
                    take_profit: Optional[float] = None, strategy: Optional[str] = None,
                    metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Store an AI signal"""
        try:
            metadata_json = json.dumps(metadata) if metadata else None
            cursor = self.connection.cursor()
            cursor.execute("""
                INSERT INTO ai_signals (signal_id, symbol, signal_type, confidence, price_target,
                                      stop_loss, take_profit, strategy, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (signal_id, symbol, signal_type, confidence, price_target, stop_loss,
                  take_profit, strategy, metadata_json))
            self.connection.commit()
            return True
        except Exception as e:
            logger.error(f"âŒ Failed to store signal {signal_id}: {e}")
            return False

    def get_signals(self, symbol: Optional[str] = None, signal_type: Optional[str] = None,
                   limit: int = 100) -> List[Dict[str, Any]]:
        """Get AI signals"""
        try:
            cursor = self.connection.cursor()
            if symbol and signal_type:
                cursor.execute("""
                    SELECT * FROM ai_signals
                    WHERE symbol = ? AND signal_type = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (symbol, signal_type, limit))
            elif symbol:
                cursor.execute("""
                    SELECT * FROM ai_signals
                    WHERE symbol = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (symbol, limit))
            elif signal_type:
                cursor.execute("""
                    SELECT * FROM ai_signals
                    WHERE signal_type = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (signal_type, limit))
            else:
                cursor.execute("""
                    SELECT * FROM ai_signals
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (limit,))

            results = []
            for row in cursor.fetchall():
                result = dict(row)
                if result.get('metadata'):
                    try:
                        result['metadata'] = json.loads(result['metadata'])
                    except (json.JSONDecodeError, TypeError):
                        pass
                results.append(result)
            return results
        except Exception as e:
            logger.error(f"âŒ Failed to get signals: {e}")
            return []

    def get_signals_by_type(self, signal_type: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get signals by type (alias for get_signals with signal_type parameter)"""
        return self.get_signals(signal_type=signal_type, limit=limit)

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            cursor = self.connection.cursor()

            # Market data stats
            cursor.execute("SELECT COUNT(*) as count FROM market_data")
            market_data_count = cursor.fetchone()['count']

            # Trade journal stats
            cursor.execute("SELECT COUNT(*) as count FROM trade_journal")
            trade_count = cursor.fetchone()['count']

            # AI signals stats
            cursor.execute("SELECT COUNT(*) as count FROM ai_signals")
            signal_count = cursor.fetchone()['count']

            # Latest timestamps
            cursor.execute("SELECT MAX(timestamp) as latest FROM market_data")
            latest_market = cursor.fetchone()['latest']

            cursor.execute("SELECT MAX(timestamp) as latest FROM trade_journal")
            latest_trade = cursor.fetchone()['latest']

            cursor.execute("SELECT MAX(timestamp) as latest FROM ai_signals")
            latest_signal = cursor.fetchone()['latest']

            return {
                "market_data_count": market_data_count,
                "trade_count": trade_count,
                "signal_count": signal_count,
                "latest_market_update": latest_market,
                "latest_trade": latest_trade,
                "latest_signal": latest_signal,
                "database_size": Path(self.db_path).stat().st_size if Path(self.db_path).exists() else 0
            }
        except Exception as e:
            logger.error(f"âŒ Failed to get cache stats: {e}")
            return {}

    def clear_old_data(self, days: int = 30) -> bool:
        """Clear old data from cache"""
        try:
            cursor = self.connection.cursor()
            cutoff_date = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
            cutoff_date = cutoff_date.replace(day=cutoff_date.day - days)

            # Clear old market data
            cursor.execute("DELETE FROM market_data WHERE timestamp < ?", (cutoff_date,))
            market_deleted = cursor.rowcount

            # Clear old trade journal entries (keep successful trades longer)
            cursor.execute("DELETE FROM trade_journal WHERE timestamp < ? AND status != 'completed'", (cutoff_date,))
            trade_deleted = cursor.rowcount

            # Clear old AI signals
            cursor.execute("DELETE FROM ai_signals WHERE timestamp < ?", (cutoff_date,))
            signal_deleted = cursor.rowcount

            self.connection.commit()
            logger.info(f"âœ… Cleared old data: {market_deleted} market records, {trade_deleted} trades, {signal_deleted} signals")
            return True
        except Exception as e:
            logger.error(f"âŒ Failed to clear old data: {e}")
            return False

    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()

    def __del__(self):
        """Destructor to ensure connection is closed"""
        self.close()


# Global cache instance
persistent_cache = PersistentCache()


def get_persistent_cache() -> PersistentCache:
    """Get the global persistent cache instance"""
    return persistent_cache


# --- Compatibility shims (legacy update_* methods) ---
# Ensure PersistentCache has update_binance / update_coinbase / update_coingecko
try:
    _PC = PersistentCache  # class defined above in this module
    def _mk_update(ns_key):
        def _update(self, data=None, **kwargs):
            payload = data if data is not None else kwargs
            setter = getattr(self, "set", None)
            if callable(setter):
                return setter(f"market:{ns_key}", payload)
            # minimal in-memory fallback (should rarely be used)
            if not hasattr(self, "_store"):
                self._store = {}
            self._store[f"market:{ns_key}"] = payload
            return True
        return _update

    if not hasattr(_PC, "update_binance"):
        _PC.update_binance = _mk_update("binanceus")
    if not hasattr(_PC, "update_coinbase"):
        _PC.update_coinbase = _mk_update("coinbase")
    if not hasattr(_PC, "update_coingecko"):
        _PC.update_coingecko = _mk_update("coingecko")
except Exception:
    # Dont break module import if anything unexpected happens
    pass
