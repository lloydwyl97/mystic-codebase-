# models.py
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime, timezone

Base = declarative_base()


class Strategy(Base):
    __tablename__ = "strategies"
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False, unique=True)
    description = Column(Text)
    win_rate = Column(Float, default=0.0)
    avg_profit = Column(Float, default=0.0)
    trades_executed = Column(Integer, default=0)
    total_profit = Column(Float, default=0.0)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )


class Trade(Base):
    __tablename__ = "trades"
    id = Column(Integer, primary_key=True)
    coin = Column(String(20), nullable=False)
    strategy_id = Column(Integer, nullable=False)
    strategy_name = Column(String(255))
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float)
    quantity = Column(Float, default=1.0)
    profit = Column(Float)
    profit_percentage = Column(Float)
    duration_minutes = Column(Float)
    success = Column(Boolean)
    trade_type = Column(String(20), default="spot")  # spot, futures, etc.
    status = Column(String(20), default="completed")  # pending, completed, cancelled
    entry_reason = Column(Text)
    exit_reason = Column(Text)
    risk_level = Column(String(20), default="medium")
    tags = Column(Text)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class StrategyPerformance(Base):
    __tablename__ = "strategy_performance"
    id = Column(Integer, primary_key=True)
    strategy_id = Column(Integer, nullable=False)
    strategy_name = Column(String(255))
    date = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    win_rate = Column(Float, default=0.0)
    avg_profit = Column(Float, default=0.0)
    total_trades = Column(Integer, default=0)
    total_profit = Column(Float, default=0.0)
    max_drawdown = Column(Float, default=0.0)
    sharpe_ratio = Column(Float, default=0.0)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
