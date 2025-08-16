"""
Data Module for Mystic Trading Platform

Contains all data-related functionality including market data,
price fetching, and data processing.
"""

from .binance_data import BinanceDataFetcher, BinanceMarketData
from .market_data import MarketData, MarketDataManager, market_data_manager

__all__ = [
    "MarketData",
    "MarketDataManager",
    "market_data_manager",
    "BinanceDataFetcher",
    "BinanceMarketData",
]


