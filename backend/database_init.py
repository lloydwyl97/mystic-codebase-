"""
Database Initialization and Management for Mystic Trading Platform
Handles database creation, migrations, and connection management
"""

import json
import os
from datetime import datetime, timedelta

import structlog
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Configure logging
logger = structlog.get_logger()

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./mystic_trading.db")
DATABASE_PATH = DATABASE_URL.replace("sqlite:///", "")

# Create SQLAlchemy engine
engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class MarketData(Base):
    """Market data table"""

    __tablename__ = "market_data"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True)
    price = Column(Float)
    volume = Column(Float)
    change_24h = Column(Float)
    timestamp = Column(DateTime, default=datetime.now(datetime.timezone.utc))
    exchange = Column(String, default="binance")


class TradeLog(Base):
    """Trade execution log"""

    __tablename__ = "trade_logs"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True)
    side = Column(String)  # buy/sell
    amount = Column(Float)
    price = Column(Float)
    timestamp = Column(DateTime, default=datetime.now(datetime.timezone.utc))
    strategy = Column(String)
    portfolio_id = Column(String, index=True)
    status = Column(String, default="executed")


class Portfolio(Base):
    """Portfolio management"""

    __tablename__ = "portfolios"

    id = Column(Integer, primary_key=True, index=True)
    portfolio_id = Column(String, unique=True, index=True)
    name = Column(String)
    total_value = Column(Float)
    cash = Column(Float)
    created_at = Column(DateTime, default=datetime.now(datetime.timezone.utc))
    updated_at = Column(
        DateTime,
        default=datetime.now(datetime.timezone.utc),
        onupdate=datetime.now(datetime.timezone.utc),
    )


class Position(Base):
    """Portfolio positions"""

    __tablename__ = "positions"

    id = Column(Integer, primary_key=True, index=True)
    portfolio_id = Column(String, index=True)
    symbol = Column(String, index=True)
    amount = Column(Float)
    average_price = Column(Float)
    current_value = Column(Float)
    timestamp = Column(DateTime, default=datetime.now(datetime.timezone.utc))


class Strategy(Base):
    """Trading strategies"""

    __tablename__ = "strategies"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    description = Column(Text)
    parameters = Column(Text)  # JSON string
    performance_metrics = Column(Text)  # JSON string
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.now(datetime.timezone.utc))
    updated_at = Column(
        DateTime,
        default=datetime.now(datetime.timezone.utc),
        onupdate=datetime.now(datetime.timezone.utc),
    )


class RiskAssessment(Base):
    """Risk assessment records"""

    __tablename__ = "risk_assessments"

    id = Column(Integer, primary_key=True, index=True)
    portfolio_id = Column(String, index=True)
    risk_metrics = Column(Text)  # JSON string
    alerts = Column(Text)  # JSON string
    risk_score = Column(Float)
    timestamp = Column(DateTime, default=datetime.now(datetime.timezone.utc))


class PerformanceMetrics(Base):
    """Performance metrics history"""

    __tablename__ = "performance_metrics"

    id = Column(Integer, primary_key=True, index=True)
    portfolio_id = Column(String, index=True)
    metrics = Column(Text)  # JSON string
    grade = Column(String)
    trend = Column(String)
    timestamp = Column(DateTime, default=datetime.now(datetime.timezone.utc))


class SystemLog(Base):
    """System operation logs"""

    __tablename__ = "system_logs"

    id = Column(Integer, primary_key=True, index=True)
    level = Column(String)  # INFO, WARNING, ERROR, CRITICAL
    component = Column(String)
    message = Column(Text)
    timestamp = Column(DateTime, default=datetime.now(datetime.timezone.utc))
    metadata = Column(Text)  # JSON string


def ensure_database_directory():
    """Ensure database directory exists"""
    try:
        db_dir = os.path.dirname(DATABASE_PATH)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
            logger.info(f"Created database directory: {db_dir}")
    except Exception as e:
        logger.error(f"Error creating database directory: {e}")


def create_database():
    """Create database and tables"""
    try:
        ensure_database_directory()

        # Create all tables
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")

        # Initialize with sample data
        initialize_sample_data()

        return True
    except Exception as e:
        logger.error(f"Error creating database: {e}")
        return False


def initialize_sample_data():
    """Initialize database with sample data"""
    try:
        db = SessionLocal()

        # Check if data already exists
        if db.query(Portfolio).first():
            logger.info("Sample data already exists, skipping initialization")
            return

        # Create sample portfolio
        portfolio = Portfolio(
            portfolio_id="default",
            name="Default Trading Portfolio",
            total_value=100000.0,
            cash=15000.0,
        )
        db.add(portfolio)

        # Create sample positions
        positions = [
            Position(
                portfolio_id="default",
                symbol="BTC/USDT",
                amount=0.5,
                average_price=50000.0,
                current_value=25000.0,
            ),
            Position(
                portfolio_id="default",
                symbol="ETH/USDT",
                amount=5.0,
                average_price=3000.0,
                current_value=15000.0,
            ),
            Position(
                portfolio_id="default",
                symbol="BNB/USDT",
                amount=50.0,
                average_price=300.0,
                current_value=15000.0,
            ),
        ]

        for position in positions:
            db.add(position)

        # Create sample strategies
        strategies = [
            Strategy(
                name="Momentum_AI_v1",
                description="AI-powered momentum trading strategy",
                parameters=json.dumps(
                    {
                        "lookback_period": 14,
                        "threshold": 0.02,
                        "max_position_size": 0.2,
                    }
                ),
                performance_metrics=json.dumps(
                    {
                        "sharpe_ratio": 1.85,
                        "total_return": 0.23,
                        "max_drawdown": 0.08,
                    }
                ),
            ),
            Strategy(
                name="Mean_Reversion_AI_v2",
                description="AI-powered mean reversion strategy",
                parameters=json.dumps(
                    {
                        "window_size": 20,
                        "std_dev_threshold": 2.0,
                        "reversion_strength": 0.5,
                    }
                ),
                performance_metrics=json.dumps(
                    {
                        "sharpe_ratio": 1.42,
                        "total_return": 0.18,
                        "max_drawdown": 0.12,
                    }
                ),
            ),
        ]

        for strategy in strategies:
            db.add(strategy)

        # Create sample market data
        symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "ADA/USDT", "SOL/USDT"]
        for symbol in symbols:
            market_data = MarketData(
                symbol=symbol,
                price=(50000.0 if "BTC" in symbol else 3000.0 if "ETH" in symbol else 100.0),
                volume=10000.0,
                change_24h=2.5,
                exchange="binance",
            )
            db.add(market_data)

        db.commit()
        logger.info("Sample data initialized successfully")

    except Exception as e:
        logger.error(f"Error initializing sample data: {e}")
        db.rollback()
    finally:
        db.close()


def check_database_health():
    """Check database health and connectivity"""
    try:
        db = SessionLocal()

        # Test basic queries
        portfolio_count = db.query(Portfolio).count()
        strategy_count = db.query(Strategy).count()
        market_data_count = db.query(MarketData).count()

        health_status = {
            "status": "healthy",
            "timestamp": datetime.now(datetime.timezone.utc).isoformat(),
            "portfolio_count": portfolio_count,
            "strategy_count": strategy_count,
            "market_data_count": market_data_count,
            "database_path": DATABASE_PATH,
            "database_size_mb": get_database_size(),
        }

        logger.info(f"Database health check passed: {health_status}")
        return health_status

    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now(datetime.timezone.utc).isoformat(),
        }
    finally:
        db.close()


def get_database_size():
    """Get database file size in MB"""
    try:
        if os.path.exists(DATABASE_PATH):
            size_bytes = os.path.getsize(DATABASE_PATH)
            return round(size_bytes / (1024 * 1024), 2)
        return 0
    except Exception:
        return 0


def cleanup_old_data(days_to_keep: int = 30):
    """Clean up old data from database"""
    try:
        db = SessionLocal()
        cutoff_date = datetime.now(datetime.timezone.utc) - timedelta(days=days_to_keep)

        # Clean up old market data
        old_market_data = db.query(MarketData).filter(MarketData.timestamp < cutoff_date).delete()

        # Clean up old trade logs
        old_trade_logs = db.query(TradeLog).filter(TradeLog.timestamp < cutoff_date).delete()

        # Clean up old system logs
        old_system_logs = db.query(SystemLog).filter(SystemLog.timestamp < cutoff_date).delete()

        db.commit()

        cleanup_results = {
            "market_data_deleted": old_market_data,
            "trade_logs_deleted": old_trade_logs,
            "system_logs_deleted": old_system_logs,
            "cutoff_date": cutoff_date.isoformat(),
        }

        logger.info(f"Database cleanup completed: {cleanup_results}")
        return cleanup_results

    except Exception as e:
        logger.error(f"Database cleanup failed: {e}")
        db.rollback()
        return None
    finally:
        db.close()


def get_database_session():
    """Get database session with proper error handling"""
    try:
        return SessionLocal()
    except Exception as e:
        logger.error(f"Error creating database session: {e}")
        return None


if __name__ == "__main__":
    # Initialize database when run directly
    logger.info("Initializing Mystic Trading Database...")
    success = create_database()
    if success:
        logger.info("Database initialization completed successfully")
        health = check_database_health()
        logger.info(f"Database health: {health}")
    else:
        logger.error("Database initialization failed")


