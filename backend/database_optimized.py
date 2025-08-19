"""
Optimized Database Manager for Mystic Trading Platform

Provides high-performance database operations with:
- Connection pooling
- Query caching
- Performance monitoring
- Optimized queries
- Rate limiting
"""

import hashlib
import logging
import sqlite3
import threading
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

try:
    import redis
except ImportError:
    redis = None

from trading_config import trading_config

from backend.utils.exceptions import DatabaseException

logger = logging.getLogger(__name__)

# Database configuration
DATABASE_PATH = "mystic_trading.db"
MAX_CONNECTIONS = 10
CACHE_TTL = trading_config.PORTFOLIO_CACHE_TTL  # Use config TTL
QUERY_TIMEOUT = trading_config.DEFAULT_REQUEST_TIMEOUT * 6  # 6x request timeout
MAX_QUERIES_PER_SECOND = trading_config.DEFAULT_REQUEST_TIMEOUT * 20  # 20x request timeout

# Redis configuration
REDIS_HOST = trading_config.DEFAULT_REDIS_HOST
REDIS_PORT = trading_config.DEFAULT_REDIS_PORT
REDIS_DB = trading_config.DEFAULT_REDIS_DB
REDIS_PASSWORD = None


@dataclass
class QueryMetrics:
    """Query performance metrics"""

    query_hash: str
    query_type: str
    execution_time: float
    timestamp: float
    success: bool
    rows_affected: int = 0


class DatabaseConnectionPool:
    """SQLite connection pool for better performance"""

    def __init__(self, max_connections: int = MAX_CONNECTIONS):
        self.max_connections = max_connections
        self.connections: deque = deque()
        self.in_use: set = set()
        self.lock = threading.Lock()
        self.stats = {
            "total_connections": 0,
            "active_connections": 0,
            "connection_errors": 0,
            "query_count": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

    def _create_connection(self) -> sqlite3.Connection:
        """Create a new database connection"""
        try:
            conn = sqlite3.connect(DATABASE_PATH, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            # Enable WAL mode for better concurrency
            conn.execute("PRAGMA journal_mode=WAL")
            # Set timeout for busy database
            conn.execute("PRAGMA busy_timeout=5000")
            # Enable foreign keys
            conn.execute("PRAGMA foreign_keys=ON")
            # Optimize for performance
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute(f"PRAGMA cache_size={trading_config.DEFAULT_REQUEST_TIMEOUT * 2000}")
            conn.execute("PRAGMA temp_store=MEMORY")

            self.stats["total_connections"] += 1
            return conn
        except Exception as e:
            self.stats["connection_errors"] += 1
            logger.error(f"Failed to create database connection: {e}")
            raise DatabaseException(f"Database connection failed: {e}")

    def get_connection(self) -> sqlite3.Connection:
        """Get a connection from the pool"""
        with self.lock:
            if self.connections:
                conn = self.connections.popleft()
                self.in_use.add(conn)
                self.stats["active_connections"] = len(self.in_use)
                return conn
            elif len(self.in_use) < self.max_connections:
                conn = self._create_connection()
                self.in_use.add(conn)
                self.stats["active_connections"] = len(self.in_use)
                return conn
            else:
                # Wait for a connection to become available
                raise DatabaseException("No available database connections")

    def return_connection(self, conn: sqlite3.Connection):
        """Return a connection to the pool"""
        with self.lock:
            if conn in self.in_use:
                self.in_use.remove(conn)
                # Reset connection state
                try:
                    conn.rollback()
                    self.connections.append(conn)
                except Exception:
                    # Connection is broken, create a new one
                    try:
                        conn.close()
                    except Exception:
                        pass
                self.stats["active_connections"] = len(self.in_use)


class QueryCache:
    """Query result caching for improved performance"""

    def __init__(self, ttl: int = CACHE_TTL):
        self.ttl = ttl
        self.cache: dict[str, tuple[Any, float]] = {}
        self.lock = threading.Lock()
        self.stats = {"hits": 0, "misses": 0, "evictions": 0}

    def _get_cache_key(self, query: str, params: tuple | None = None) -> str:
        """Generate cache key for query"""
        key_data = f"{query}:{params}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def get(self, query: str, params: tuple | None = None) -> Any | None:
        """Get cached result"""
        key = self._get_cache_key(query, params)
        with self.lock:
            if key in self.cache:
                result, timestamp = self.cache[key]
                if time.time() - timestamp < self.ttl:
                    self.stats["hits"] += 1
                    return result
                else:
                    # Expired, remove from cache
                    del self.cache[key]
                    self.stats["evictions"] += 1
            self.stats["misses"] += 1
            return None

    def set(self, query: str, result: Any, params: tuple | None = None):
        """Cache query result"""
        key = self._get_cache_key(query, params)
        with self.lock:
            self.cache[key] = (result, time.time())

    def clear(self):
        """Clear all cached results"""
        with self.lock:
            self.cache.clear()


class RateLimiter:
    """Rate limiter for database queries"""

    def __init__(self, max_queries_per_second: int = MAX_QUERIES_PER_SECOND):
        self.max_queries = max_queries_per_second
        self.query_times: deque = deque()
        self.lock = threading.Lock()
        self.stats = {
            "total_queries": 0,
            "rate_limited": 0,
            "average_rate": 0.0,
        }

    def can_proceed(self) -> bool:
        """Check if query can proceed"""
        now = time.time()
        with self.lock:
            # Remove old timestamps (older than 1 second)
            while self.query_times and now - self.query_times[0] >= 1.0:
                self.query_times.popleft()

            if len(self.query_times) < self.max_queries:
                self.query_times.append(now)
                self.stats["total_queries"] += 1
                return True
            else:
                self.stats["rate_limited"] += 1
                return False

    def get_stats(self) -> dict[str, Any]:
        """Get rate limiter statistics"""
        with self.lock:
            current_rate = len(self.query_times)
            self.stats["average_rate"] = current_rate
            return self.stats.copy()


class OptimizedDatabaseManager:
    """High-performance database manager with optimizations"""

    def __init__(self):
        self.connection_pool = DatabaseConnectionPool()
        self.query_cache = QueryCache()
        self.rate_limiter = RateLimiter()
        self.query_metrics: list[QueryMetrics] = []
        self.metrics_lock = threading.Lock()

        # Initialize database
        self._initialize_database()
        logger.info("âœ… OptimizedDatabaseManager initialized")

    def _initialize_database(self):
        """Initialize database with optimized schema"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Create optimized tables with indexes
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

            # Create indexes for better query performance
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_market_data_symbol ON market_data(symbol)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_market_data_timestamp ON market_data(timestamp)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_market_data_symbol_timestamp ON market_data(symbol, timestamp)"
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS user_settings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    setting_name TEXT NOT NULL,
                    setting_value TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_id, setting_name)
                )
            """
            )

            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_user_settings_user_id ON user_settings(user_id)"
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
                "CREATE INDEX IF NOT EXISTS idx_trading_signals_symbol ON trading_signals(symbol)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_trading_signals_timestamp ON trading_signals(timestamp)"
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

            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status)")

            conn.commit()
            logger.info("âœ… Database initialized with optimized schema")

    @contextmanager
    def get_connection(self):
        """Get database connection from pool"""
        conn = None
        try:
            conn = self.connection_pool.get_connection()
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            raise DatabaseException(f"Database operation failed: {e}")
        finally:
            if conn:
                self.connection_pool.return_connection(conn)

    def execute_query(
        self,
        query: str,
        params: tuple | None = None,
        use_cache: bool = True,
    ) -> Any:
        """Execute query with caching and rate limiting"""
        # Check rate limit
        if not self.rate_limiter.can_proceed():
            raise DatabaseException("Query rate limit exceeded")

        # Check cache first
        if use_cache:
            cached_result = self.query_cache.get(query, params)
            if cached_result is not None:
                return cached_result

        # Execute query
        start_time = time.time()
        success = False
        rows_affected = 0

        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)

                # Determine query type and get results
                query_upper = query.strip().upper()
                if query_upper.startswith("SELECT"):
                    result = cursor.fetchall()
                elif query_upper.startswith("INSERT"):
                    result = cursor.lastrowid
                    rows_affected = 1
                else:
                    result = cursor.rowcount
                    rows_affected = cursor.rowcount

                conn.commit()
                success = True

                # Cache result if it's a SELECT query
                if use_cache and query_upper.startswith("SELECT"):
                    self.query_cache.set(query, result, params)

        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise DatabaseException(f"Query failed: {e}")
        finally:
            # Record metrics
            execution_time = time.time() - start_time
            query_hash = hashlib.md5(f"{query}:{params}".encode()).hexdigest()

            with self.metrics_lock:
                self.query_metrics.append(
                    QueryMetrics(
                        query_hash=query_hash,
                        query_type=query.strip().split()[0].upper(),
                        execution_time=execution_time,
                        timestamp=time.time(),
                        success=success,
                        rows_affected=rows_affected,
                    )
                )

                # Keep only last metrics based on config
                max_metrics = trading_config.DEFAULT_REQUEST_TIMEOUT * 200
                if len(self.query_metrics) > max_metrics:
                    self.query_metrics = self.query_metrics[-max_metrics:]

        return result

    def get_market_data(self, symbol: str, limit: int = 100) -> list[dict[str, Any]]:
        """Get market data with optimized query"""
        query = """
            SELECT symbol, price, volume, timestamp, created_at
            FROM market_data
            WHERE symbol = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """

        try:
            # Use parameterized query for better performance and security
            result = self.execute_query(query, (symbol, limit))
            return [dict(row) for row in result]
        except Exception as e:
            logger.error(f"Failed to get market data for {symbol}: {e}")
            return []

    def get_market_data_by_time_range(self, symbol: str, start_time: float, end_time: float) -> list[dict[str, Any]]:
        """Get market data within a specific time range with optimized query"""
        query = """
            SELECT timestamp, price, volume, symbol, change_24h, market_cap
            FROM market_data
            WHERE symbol = ? AND timestamp BETWEEN ? AND ?
            ORDER BY timestamp DESC
            LIMIT 1000
        """
        return self.execute_query(query, (symbol, start_time, end_time), use_cache=False)

    def get_latest_market_data(self, symbols: list[str]) -> list[dict[str, Any]]:
        """Get latest market data for multiple symbols with optimized batch query"""
        if not symbols:
            return []

        placeholders = ','.join(['?' for _ in symbols])
        query = f"""
            SELECT m1.*
            FROM market_data m1
            INNER JOIN (
                SELECT symbol, MAX(timestamp) as max_timestamp
                FROM market_data
                WHERE symbol IN ({placeholders})
                GROUP BY symbol
            ) m2 ON m1.symbol = m2.symbol AND m1.timestamp = m2.max_timestamp
            ORDER BY m1.symbol
        """
        return self.execute_query(query, tuple(symbols), use_cache=True)

    def get_market_data_batch(self, symbols: list[str], limit: int = 100) -> dict[str, list[dict[str, Any]]]:
        """Get market data for multiple symbols in a single optimized query"""
        if not symbols:
            return {}

        placeholders = ','.join(['?' for _ in symbols])
        query = f"""
            SELECT symbol, timestamp, price, volume, change_24h, market_cap
            FROM market_data
            WHERE symbol IN ({placeholders})
            ORDER BY symbol, timestamp DESC
        """

        results = self.execute_query(query, tuple(symbols), use_cache=False)

        # Group results by symbol
        grouped_data = {}
        for row in results:
            symbol = row['symbol']
            if symbol not in grouped_data:
                grouped_data[symbol] = []
            grouped_data[symbol].append(dict(row))

        # Limit each symbol's data
        for symbol in grouped_data:
            grouped_data[symbol] = grouped_data[symbol][:limit]

        return grouped_data

    def get_aggregated_market_stats(self, symbols: list[str], timeframe: str = '24h') -> dict[str, dict[str, Any]]:
        """Get aggregated market statistics for multiple symbols"""
        if not symbols:
            return {}

        placeholders = ','.join(['?' for _ in symbols])

        if timeframe == '24h':
            query = f"""
                SELECT
                    symbol,
                    AVG(price) as avg_price,
                    MAX(price) as max_price,
                    MIN(price) as min_price,
                    SUM(volume) as total_volume,
                    COUNT(*) as data_points,
                    (MAX(price) - MIN(price)) / AVG(price) * 100 as volatility
                FROM market_data
                WHERE symbol IN ({placeholders})
                AND timestamp >= strftime('%s', 'now', '-1 day')
                GROUP BY symbol
            """
        else:  # 1h
            query = f"""
                SELECT
                    symbol,
                    AVG(price) as avg_price,
                    MAX(price) as max_price,
                    MIN(price) as min_price,
                    SUM(volume) as total_volume,
                    COUNT(*) as data_points,
                    (MAX(price) - MIN(price)) / AVG(price) * 100 as volatility
                FROM market_data
                WHERE symbol IN ({placeholders})
                AND timestamp >= strftime('%s', 'now', '-1 hour')
                GROUP BY symbol
            """

        results = self.execute_query(query, tuple(symbols), use_cache=True)

        stats = {}
        for row in results:
            stats[row['symbol']] = dict(row)

        return stats

    def get_trending_symbols(self, limit: int = 10, timeframe: str = '24h') -> list[dict[str, Any]]:
        """Get trending symbols based on volume and price change"""
        if timeframe == '24h':
            query = """
                SELECT
                    symbol,
                    AVG(price) as avg_price,
                    SUM(volume) as total_volume,
                    (MAX(price) - MIN(price)) / AVG(price) * 100 as price_change,
                    COUNT(*) as data_points
                FROM market_data
                WHERE timestamp >= strftime('%s', 'now', '-1 day')
                GROUP BY symbol
                HAVING data_points > 10
                ORDER BY total_volume DESC, price_change DESC
                LIMIT ?
            """
        else:  # 1h
            query = """
                SELECT
                    symbol,
                    AVG(price) as avg_price,
                    SUM(volume) as total_volume,
                    (MAX(price) - MIN(price)) / AVG(price) * 100 as price_change,
                    COUNT(*) as data_points
                FROM market_data
                WHERE timestamp >= strftime('%s', 'now', '-1 hour')
                GROUP BY symbol
                HAVING data_points > 5
                ORDER BY total_volume DESC, price_change DESC
                LIMIT ?
            """

        return self.execute_query(query, (limit,), use_cache=True)

    def get_correlation_matrix(self, symbols: list[str], timeframe: str = '24h') -> dict[str, dict[str, float]]:
        """Calculate correlation matrix for multiple symbols"""
        if len(symbols) < 2:
            return {}

        # Get price data for all symbols
        placeholders = ','.join(['?' for _ in symbols])

        if timeframe == '24h':
            query = f"""
                SELECT symbol, timestamp, price
                FROM market_data
                WHERE symbol IN ({placeholders})
                AND timestamp >= strftime('%s', 'now', '-1 day')
                ORDER BY symbol, timestamp
            """
        else:  # 1h
            query = f"""
                SELECT symbol, timestamp, price
                FROM market_data
                WHERE symbol IN ({placeholders})
                AND timestamp >= strftime('%s', 'now', '-1 hour')
                ORDER BY symbol, timestamp
            """

        results = self.execute_query(query, tuple(symbols), use_cache=False)

        # Group by symbol and calculate returns
        price_data = {}
        for row in results:
            symbol = row['symbol']
            if symbol not in price_data:
                price_data[symbol] = []
            price_data[symbol].append(row['price'])

        # Calculate correlations
        correlation_matrix = {}
        for i, symbol1 in enumerate(symbols):
            correlation_matrix[symbol1] = {}
            for j, symbol2 in enumerate(symbols):
                if i == j:
                    correlation_matrix[symbol1][symbol2] = 1.0
                else:
                    # Simple correlation calculation
                    prices1 = price_data.get(symbol1, [])
                    prices2 = price_data.get(symbol2, [])

                    if len(prices1) > 1 and len(prices2) > 1:
                        # Calculate returns
                        returns1 = [(prices1[i] - prices1[i-1])/prices1[i-1] for i in range(1, len(prices1))]
                        returns2 = [(prices2[i] - prices2[i-1])/prices2[i-1] for i in range(1, len(prices2))]

                        # Calculate correlation
                        min_len = min(len(returns1), len(returns2))
                        if min_len > 0:
                            returns1 = returns1[:min_len]
                            returns2 = returns2[:min_len]

                            mean1 = sum(returns1) / len(returns1)
                            mean2 = sum(returns2) / len(returns2)

                            numerator = sum((r1 - mean1) * (r2 - mean2) for r1, r2 in zip(returns1, returns2, strict=False))
                            denominator1 = sum((r1 - mean1) ** 2 for r1 in returns1)
                            denominator2 = sum((r2 - mean2) ** 2 for r2 in returns2)

                            if denominator1 > 0 and denominator2 > 0:
                                correlation = numerator / (denominator1 * denominator2) ** 0.5
                                correlation_matrix[symbol1][symbol2] = max(-1.0, min(1.0, correlation))
                            else:
                                correlation_matrix[symbol1][symbol2] = 0.0
                        else:
                            correlation_matrix[symbol1][symbol2] = 0.0
                    else:
                        correlation_matrix[symbol1][symbol2] = 0.0

        return correlation_matrix

    def get_volatility_analysis(self, symbol: str, timeframe: str = '24h') -> dict[str, Any]:
        """Get detailed volatility analysis for a symbol"""
        if timeframe == '24h':
            query = """
                SELECT
                    price,
                    volume,
                    timestamp,
                    (price - LAG(price) OVER (ORDER BY timestamp)) / LAG(price) OVER (ORDER BY timestamp) as returns
                FROM market_data
                WHERE symbol = ?
                AND timestamp >= strftime('%s', 'now', '-1 day')
                ORDER BY timestamp
            """
        else:  # 1h
            query = """
                SELECT
                    price,
                    volume,
                    timestamp,
                    (price - LAG(price) OVER (ORDER BY timestamp)) / LAG(price) OVER (ORDER BY timestamp) as returns
                FROM market_data
                WHERE symbol = ?
                AND timestamp >= strftime('%s', 'now', '-1 hour')
                ORDER BY timestamp
            """

        results = self.execute_query(query, (symbol,), use_cache=False)

        if not results:
            return {}

        returns = [row['returns'] for row in results if row['returns'] is not None]
        prices = [row['price'] for row in results]
        volumes = [row['volume'] for row in results if row['volume'] is not None]

        if not returns:
            return {}

        # Calculate volatility metrics
        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
        volatility = variance ** 0.5

        # Calculate additional metrics
        max_price = max(prices) if prices else 0
        min_price = min(prices) if prices else 0
        avg_volume = sum(volumes) / len(volumes) if volumes else 0

        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'volatility': volatility,
            'mean_return': mean_return,
            'max_price': max_price,
            'min_price': min_price,
            'price_range': max_price - min_price,
            'avg_volume': avg_volume,
            'data_points': len(results),
            'timestamp': time.time()
        }

    def insert_market_data(self, symbol: str, price: float, volume: float | None = None) -> bool:
        """Insert market data with optimized query"""
        query = """
            INSERT INTO market_data (symbol, price, volume, timestamp)
            VALUES (?, ?, ?, ?)
        """

        try:
            self.execute_query(query, (symbol, price, volume, time.time()), use_cache=False)
            return True
        except Exception as e:
            logger.error(f"Failed to insert market data for {symbol}: {e}")
            return False

    def get_performance_stats(self) -> dict[str, Any]:
        """Get comprehensive performance statistics"""
        with self.metrics_lock:
            if not self.query_metrics:
                return {"message": "No query metrics available"}

            # Calculate query performance metrics
            execution_times = [m.execution_time for m in self.query_metrics]
            successful_queries = [m for m in self.query_metrics if m.success]

            stats = {
                "total_queries": len(self.query_metrics),
                "successful_queries": len(successful_queries),
                "failed_queries": (len(self.query_metrics) - len(successful_queries)),
                "success_rate": (
                    len(successful_queries) / len(self.query_metrics) if self.query_metrics else 0
                ),
                "average_execution_time": (
                    sum(execution_times) / len(execution_times) if execution_times else 0
                ),
                "max_execution_time": (max(execution_times) if execution_times else 0),
                "min_execution_time": (min(execution_times) if execution_times else 0),
                "cache_stats": self.query_cache.stats.copy(),
                "connection_pool_stats": self.connection_pool.stats.copy(),
                "rate_limiter_stats": self.rate_limiter.get_stats(),
                "query_types": defaultdict(int),
            }

            # Count query types
            for metric in self.query_metrics:
                stats["query_types"][metric.query_type] += 1

            return stats

    def clear_cache(self):
        """Clear query cache"""
        self.query_cache.clear()
        logger.info("âœ… Query cache cleared")

    def optimize_database(self):
        """Run database optimization"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                # Run database maintenance
                cursor.execute("VACUUM")
                cursor.execute("ANALYZE")

                # Add missing indexes if they don't exist
                additional_indexes = [
                    "CREATE INDEX IF NOT EXISTS idx_market_data_symbol_timestamp ON market_data(symbol, timestamp)",
                    "CREATE INDEX IF NOT EXISTS idx_positions_portfolio_symbol ON positions(portfolio_id, symbol)",
                    "CREATE INDEX IF NOT EXISTS idx_trade_logs_symbol_timestamp ON trade_logs(symbol, timestamp)",
                    "CREATE INDEX IF NOT EXISTS idx_strategies_active ON strategies(is_active)",
                    "CREATE INDEX IF NOT EXISTS idx_system_logs_level ON system_logs(level)",
                    "CREATE INDEX IF NOT EXISTS idx_system_logs_component ON system_logs(component)",
                ]

                for index_sql in additional_indexes:
                    cursor.execute(index_sql)

                conn.commit()
            logger.info("âœ… Database optimization completed")
        except Exception as e:
            logger.error(f"Database optimization failed: {e}")

    def get_query_performance_stats(self) -> dict[str, Any]:
        """Get detailed query performance statistics"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                # Get table sizes
                cursor.execute("""
                    SELECT name, sql FROM sqlite_master
                    WHERE type='table' AND name NOT LIKE 'sqlite_%'
                """)
                tables = cursor.fetchall()

                stats = {
                    "table_count": len(tables),
                    "tables": {},
                    "indexes": {},
                    "query_metrics": self.get_performance_stats()
                }

                # Get table row counts and sizes
                for table_name, _ in tables:
                    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                    row_count = cursor.fetchone()[0]

                    cursor.execute(f"PRAGMA table_info({table_name})")
                    columns = cursor.fetchall()

                    stats["tables"][table_name] = {
                        "row_count": row_count,
                        "column_count": len(columns)
                    }

                # Get index information
                cursor.execute("""
                    SELECT name, tbl_name, sql FROM sqlite_master
                    WHERE type='index' AND name NOT LIKE 'sqlite_%'
                """)
                indexes = cursor.fetchall()

                for index_name, table_name, index_sql in indexes:
                    stats["indexes"][index_name] = {
                        "table": table_name,
                        "sql": index_sql
                    }

                return stats

        except Exception as e:
            logger.error(f"Failed to get query performance stats: {e}")
            return {"error": str(e)}


# Global optimized database manager instance
optimized_db_manager = OptimizedDatabaseManager()


