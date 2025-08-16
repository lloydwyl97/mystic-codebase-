import os
from dotenv import load_dotenv
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Float,
    DateTime,
    Boolean,
    ForeignKey,
    func,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, scoped_session
from datetime import datetime

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///db.sqlite3")

engine = create_engine(
    DATABASE_URL,
    connect_args=({"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}),
)
SessionLocal = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))
Base = declarative_base()


class Strategy(Base):
    __tablename__ = "strategies"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    version = Column(String, default="1.0")
    active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    trades = relationship("Trade", back_populates="strategy")


class Trade(Base):
    __tablename__ = "trades"
    id = Column(Integer, primary_key=True, index=True)
    strategy_id = Column(Integer, ForeignKey("strategies.id"))
    pair = Column(String)
    side = Column(String)
    entry_price = Column(Float)
    exit_price = Column(Float, nullable=True)
    profit = Column(Float, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    closed_at = Column(DateTime, nullable=True)
    strategy = relationship("Strategy", back_populates="trades")


def init_db():
    Base.metadata.create_all(bind=engine)


def get_session():
    return SessionLocal()


def register_strategy(name, version="1.0"):
    session = get_session()
    strategy = session.query(Strategy).filter_by(name=name).first()
    if not strategy:
        strategy = Strategy(name=name, version=version)
        session.add(strategy)
        session.commit()
    return strategy.id


def get_strategy_id(name):
    session = get_session()
    strategy = session.query(Strategy).filter_by(name=name).first()
    return strategy.id if strategy else None


def log_trade(strategy_name, pair, side, entry_price):
    session = get_session()
    strategy_id = get_strategy_id(strategy_name)
    if not strategy_id:
        strategy_id = register_strategy(strategy_name)
    trade = Trade(strategy_id=strategy_id, pair=pair, side=side, entry_price=entry_price)
    session.add(trade)
    session.commit()
    return trade.id


def update_trade_exit(trade_id, exit_price):
    session = get_session()
    trade = session.query(Trade).filter_by(id=trade_id).first()
    if trade and exit_price:
        trade.exit_price = exit_price
        trade.profit = (
            exit_price - trade.entry_price
            if trade.side == "BUY"
            else trade.entry_price - exit_price
        )
        trade.closed_at = datetime.utcnow()
        session.commit()


def get_recent_trades(limit=20):
    session = get_session()
    return session.query(Trade).order_by(Trade.timestamp.desc()).limit(limit).all()


def get_active_strategies():
    session = get_session()
    return session.query(Strategy).filter_by(active=True).all()


def get_strategy_stats():
    session = get_session()
    result = (
        session.query(
            Strategy.name,
            func.count(Trade.id).label("trade_count"),
            func.avg(Trade.profit).label("avg_profit"),
        )
        .join(Trade)
        .group_by(Strategy.id)
        .all()
    )
    return {name: {"trades": count, "avg_profit": avg} for name, count, avg in result}


