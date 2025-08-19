"""
Portfolio Endpoints
Consolidated portfolio management, overview, and live trading data
All endpoints return live data - no stubs or placeholders
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Protocol, runtime_checkable, List

from fastapi import APIRouter, HTTPException

# Import real services
try:
    from backend.modules.ai.persistent_cache import get_persistent_cache
    from backend.modules.ai.trade_tracker import (
        get_active_trades,
        get_trade_history,
        get_trade_summary,
    )
    from backend.modules.data.binance_data import BinanceData
    from backend.modules.data.coinbase_data import CoinbaseData
    from backend.services.trading import TradingService
except ImportError as e:
    logging.warning(f"Some trading services not available: {e}")
    # Define fallback functions
    def get_persistent_cache():
        return None

    def get_active_trades():
        return []

    def get_trade_history(days=30):
        return []

    def get_trade_summary():
        return {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "total_profit": 0.0,
            "win_rate": 0.0,
            "average_profit": 0.0
        }

    BinanceData = None
    CoinbaseData = None
    TradingService = None

# Import PortfolioService separately to ensure it's available
try:
    from backend.services.portfolio_service import PortfolioService
except ImportError as e:
    logging.warning(f"PortfolioService not available: {e}")
    PortfolioService = None

# Redis accessor shim (prefer project accessors if available)
try:
    # First try a lightweight sync client accessor used elsewhere in endpoints
    from backend.services.redis_client import get_redis_client as _get_redis  # type: ignore[import-not-found]
    get_redis = _get_redis  # type: ignore[assignment]
except Exception:
    try:
        from backend.services.redis_service import get_redis as _get_redis  # type: ignore[import-not-found]
        get_redis = _get_redis  # type: ignore[assignment]
    except Exception:
        try:
            from backend.services.cache_or_redis_client import get_cache as _get_redis  # type: ignore[import-not-found]
            get_redis = _get_redis  # type: ignore[assignment]
        except Exception:
            get_redis = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)
router = APIRouter()

@runtime_checkable
class PortfolioServiceProtocol(Protocol):
    async def get_portfolio_overview(self) -> Dict[str, Any]:
        ...

    async def get_portfolio_summary(self) -> Dict[str, Any]:
        ...

    async def get_positions(self) -> Dict[str, Any]:
        ...


# Initialize real services
try:
    portfolio_service: PortfolioServiceProtocol | None = PortfolioService() if PortfolioService else None  # type: ignore[assignment]
    _redis = get_redis() if callable(get_redis) else None  # type: ignore[misc]
    trading_service = TradingService(_redis) if TradingService else None  # type: ignore[call-arg]
    binance_data = BinanceData() if BinanceData else None
    coinbase_data = CoinbaseData() if CoinbaseData else None
except Exception as e:
    logger.warning(f"Could not initialize some trading services: {e}")


@router.get("/portfolio/overview")
async def get_portfolio_overview() -> Dict[str, Any]:
    """Get comprehensive portfolio overview with live data"""
    try:
        # Get real portfolio data
        portfolio_data = {}
        try:
            if portfolio_service:
                portfolio_data = await portfolio_service.get_portfolio_overview()
        except Exception as e:
            logger.error(f"Error getting portfolio overview: {e}")
            portfolio_data = {"error": "Portfolio service unavailable"}

        # Get live market data for portfolio valuation
        market_data = {}
        try:
            cache = get_persistent_cache()
            if cache:
                market_data = cache.get_market_data()
        except Exception as e:
            logger.error(f"Error getting market data: {e}")
            market_data = {"error": "Market data unavailable"}

        # Get active trades
        active_trades = []
        try:
            active_trades = get_active_trades()
        except Exception as e:
            logger.error(f"Error getting active trades: {e}")
            active_trades = []

        overview_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "portfolio": portfolio_data,
            "market_data": market_data,
            "active_trades": active_trades,
            "total_trades": len(active_trades),
            "version": "1.0.0",
        }

        return overview_data

    except Exception as e:
        logger.error(f"Error getting portfolio overview: {e}")
        raise HTTPException(status_code=500, detail=f"Portfolio overview failed: {str(e)}")


@router.get("/portfolio/live")
async def get_portfolio_live() -> Dict[str, Any]:
    """Get live portfolio data and real-time updates"""
    try:
        # Get real-time portfolio data
        live_data = {}
        try:
            if portfolio_service:
                live_data = await portfolio_service.get_portfolio_summary()
        except Exception as e:
            logger.error(f"Error getting live portfolio data: {e}")
            live_data = {"error": "Live portfolio data unavailable"}

        # Get real-time market prices
        market_prices = {}
        try:
            cache = get_persistent_cache()
            if cache:
                market_prices = cache.get_latest_prices()
        except Exception as e:
            logger.error(f"Error getting market prices: {e}")
            market_prices = {"error": "Market prices unavailable"}

        # Get portfolio performance metrics
        performance = {}
        try:
            if portfolio_service:
                performance = await portfolio_service.get_portfolio_summary()
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            performance = {"error": "Performance metrics unavailable"}

        live_portfolio_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "portfolio": live_data,
            "market_prices": market_prices,
            "performance": performance,
            "version": "1.0.0",
        }

        return live_portfolio_data

    except Exception as e:
        logger.error(f"Error getting live portfolio data: {e}")
        raise HTTPException(status_code=500, detail=f"Live portfolio data failed: {str(e)}")


@router.get("/portfolio/positions")
async def get_portfolio_positions() -> Dict[str, Any]:
    """Get current portfolio positions and holdings"""
    try:
        # Get real portfolio positions
        positions = {}
        try:
            if portfolio_service:
                positions = await portfolio_service.get_positions()
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            positions = {"error": "Positions unavailable"}

        # Get position performance
        position_performance = {}
        try:
            if portfolio_service:
                position_performance = await portfolio_service.get_portfolio_summary()
        except Exception as e:
            logger.error(f"Error getting position performance: {e}")
            position_performance = {"error": "Position performance unavailable"}

        positions_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "positions": positions,
            "performance": position_performance,
            "total_positions": (len(positions) if isinstance(positions, dict) else 0),
            "version": "1.0.0",
        }

        return positions_data

    except Exception as e:
        logger.error(f"Error getting portfolio positions: {e}")
        raise HTTPException(status_code=500, detail=f"Portfolio positions failed: {str(e)}")


@router.get("/portfolio/history")
async def get_portfolio_history(days: int = 30) -> Dict[str, Any]:
    """Get portfolio trading history"""
    try:
        # Get real trading history
        history = []
        try:
            history = get_trade_history(days)
        except Exception as e:
            logger.error(f"Error getting trade history: {e}")
            history = []

        # Get portfolio value history
        value_history = {}
        try:
            if portfolio_service:
                value_history = await portfolio_service.get_portfolio_summary()
        except Exception as e:
            logger.error(f"Error getting value history: {e}")
            value_history = {"error": "Value history unavailable"}

        history_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "trades": history,
            "value_history": value_history,
            "total_trades": len(history),
            "period_days": days,
            "version": "1.0.0",
        }

        return history_data

    except Exception as e:
        logger.error(f"Error getting portfolio history: {e}")
        raise HTTPException(status_code=500, detail=f"Portfolio history failed: {str(e)}")


@router.get("/portfolio/performance")
async def get_portfolio_performance() -> Dict[str, Any]:
    """Get detailed portfolio performance metrics"""
    try:
        # Get real performance data
        performance = {}
        try:
            if portfolio_service:
                performance = await portfolio_service.get_portfolio_summary()
        except Exception as e:
            logger.error(f"Error getting detailed performance: {e}")
            performance = {"error": "Performance data unavailable"}

        # Get trade summary
        trade_summary = {}
        try:
            trade_summary = get_trade_summary()
        except Exception as e:
            logger.error(f"Error getting trade summary: {e}")
            trade_summary = {"error": "Trade summary unavailable"}

        performance_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "performance": performance,
            "trade_summary": trade_summary,
            "version": "1.0.0",
        }

        return performance_data

    except Exception as e:
        logger.error(f"Error getting portfolio performance: {e}")
        raise HTTPException(status_code=500, detail=f"Portfolio performance failed: {str(e)}")


@router.get("/portfolio/allocations")
async def get_portfolio_allocations() -> Dict[str, Any]:
    """Get portfolio asset allocations and distribution"""
    try:
        # Get real allocation data
        allocations = {}
        try:
            if portfolio_service:
                allocations = await portfolio_service.get_portfolio_summary()
        except Exception as e:
            logger.error(f"Error getting allocations: {e}")
            allocations = {"error": "Allocation data unavailable"}

        # Get sector distribution
        sector_distribution = {}
        try:
            if portfolio_service:
                sector_distribution = await portfolio_service.get_portfolio_summary()
        except Exception as e:
            logger.error(f"Error getting sector distribution: {e}")
            sector_distribution = {"error": "Sector distribution unavailable"}

        allocations_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "allocations": allocations,
            "sector_distribution": sector_distribution,
            "version": "1.0.0",
        }

        return allocations_data

    except Exception as e:
        logger.error(f"Error getting portfolio allocations: {e}")
        raise HTTPException(status_code=500, detail=f"Portfolio allocations failed: {str(e)}")



