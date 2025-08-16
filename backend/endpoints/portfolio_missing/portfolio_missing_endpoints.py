"""
Missing Portfolio Endpoints

Provides missing portfolio endpoints that return live data:
- Portfolio Performance
- Portfolio Balance
- Portfolio Transactions
"""

import logging
from typing import Any, Dict

from fastapi import APIRouter, HTTPException
from backend.services.portfolio_service import PortfolioService

logger = logging.getLogger(__name__)
router = APIRouter()
portfolio_service = PortfolioService()


@router.get("/api/portfolio/performance")
async def get_portfolio_performance() -> Dict[str, Any]:
    """
    Get portfolio performance metrics with live data

    Returns comprehensive performance metrics including:
    - Total return
    - Daily/weekly/monthly performance
    - Sharpe ratio
    - Maximum drawdown
    - Win rate
    """
    try:
        result = await portfolio_service.get_portfolio_summary()
        return result
    except Exception as e:
        logger.error(f"Error getting portfolio performance: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting portfolio performance: {str(e)}",
        )


@router.get("/api/portfolio/balance")
async def get_portfolio_balance() -> Dict[str, Any]:
    """
    Get portfolio balance with live data

    Returns detailed balance information including:
    - Total balance
    - Asset breakdown
    - Available vs locked funds
    - Exchange balances
    """
    try:
        result = await portfolio_service.get_wallet_balance()
        return result
    except Exception as e:
        logger.error(f"Error getting portfolio balance: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting portfolio balance: {str(e)}",
        )


@router.get("/api/portfolio/transactions")
async def get_portfolio_transactions(limit: int = 50, offset: int = 0) -> Dict[str, Any]:
    """
    Get portfolio transaction history with live data

    Returns transaction history including:
    - Buy/sell orders
    - Deposits/withdrawals
    - Transaction details
    """
    try:
        transactions = await portfolio_service.get_transactions(limit)
        return {"transactions": transactions, "limit": limit, "offset": offset}
    except Exception as e:
        logger.error(f"Error getting portfolio transactions: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting portfolio transactions: {str(e)}",
        )



