"""
Portfolio Service for Mystic AI Trading Platform
Provides comprehensive portfolio tracking and analysis using cached trade data.
"""

import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any

# Add backend to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from backend.modules.ai.persistent_cache import PersistentCache

logger = logging.getLogger(__name__)


class PortfolioService:
    """Service for managing portfolio data and operations using cached trade data."""

    def __init__(self):
        """Initialize portfolio service with cache integration"""
        self.cache = PersistentCache()
        self.portfolio_data = {}
        self.positions = {}
        self.transactions = []

        logger.info("âœ… PortfolioService initialized")

    def _get_trade_history(self, limit: int = 1000) -> list[dict[str, Any]]:
        """Get trade history from cache"""
        try:
            # Get recent trade signals from cache
            signals = self.cache.get_signals_by_type("TRADE_EXECUTED", limit=limit)

            trades = []
            for signal in signals:
                trade_data = signal.get("metadata", {})
                if trade_data:
                    trades.append({
                        "symbol": signal.get("symbol", ""),
                        "trade_type": trade_data.get("trade_type", ""),
                        "quantity": trade_data.get("quantity", 0.0),
                        "price": trade_data.get("price", 0.0),
                        "amount_usd": trade_data.get("amount_usd", 0.0),
                        "exchange": trade_data.get("exchange", ""),
                        "timestamp": signal.get("timestamp", ""),
                        "trade_id": signal.get("signal_id", "")
                    })

            return trades

        except Exception as e:
            logger.error(f"Failed to get trade history: {e}")
            return []

    def _get_latest_prices(self) -> dict[str, float]:
        """Get latest prices for all symbols from cache"""
        try:
            # Get recent price signals from cache
            signals = self.cache.get_signals_by_type("PRICE_UPDATE", limit=100)

            latest_prices = {}
            for signal in signals:
                symbol = signal.get("symbol", "")
                price = signal.get("metadata", {}).get("price", 0.0)

                if symbol and price > 0 and symbol not in latest_prices:
                    # Keep the most recent price for each symbol
                    latest_prices[symbol] = price

            return latest_prices

        except Exception as e:
            logger.error(f"Failed to get latest prices: {e}")
            return {}

    def _calculate_holdings(self, trades: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        """Calculate holdings per symbol from trade history"""
        try:
            holdings = {}

            for trade in trades:
                symbol = trade.get("symbol", "")
                trade_type = trade.get("trade_type", "")
                quantity = trade.get("quantity", 0.0)
                trade.get("price", 0.0)
                amount_usd = trade.get("amount_usd", 0.0)

                if not symbol or quantity <= 0:
                    continue

                if symbol not in holdings:
                    holdings[symbol] = {
                        "quantity": 0.0,
                        "total_buy_amount": 0.0,
                        "total_sell_amount": 0.0,
                        "total_buy_quantity": 0.0,
                        "total_sell_quantity": 0.0,
                        "average_buy_price": 0.0,
                        "trades": []
                    }

                holding = holdings[symbol]
                holding["trades"].append(trade)

                if trade_type == "BUY":
                    holding["quantity"] += quantity
                    holding["total_buy_amount"] += amount_usd
                    holding["total_buy_quantity"] += quantity
                elif trade_type == "SELL":
                    holding["quantity"] -= quantity
                    holding["total_sell_amount"] += amount_usd
                    holding["total_sell_quantity"] += quantity

                # Calculate average buy price
                if holding["total_buy_quantity"] > 0:
                    holding["average_buy_price"] = holding["total_buy_amount"] / holding["total_buy_quantity"]

            return holdings

        except Exception as e:
            logger.error(f"Failed to calculate holdings: {e}")
            return {}

    def _calculate_unrealized_pnl(self, holdings: dict[str, dict[str, Any]], latest_prices: dict[str, float]) -> dict[str, Any]:
        """Calculate unrealized PnL for all holdings"""
        try:
            total_pnl = 0.0
            total_value = 0.0
            pnl_by_symbol = {}

            for symbol, holding in holdings.items():
                quantity = holding.get("quantity", 0.0)
                avg_buy_price = holding.get("average_buy_price", 0.0)
                current_price = latest_prices.get(symbol, 0.0)

                if quantity > 0 and avg_buy_price > 0 and current_price > 0:
                    # Calculate unrealized PnL
                    current_value = quantity * current_price
                    cost_basis = quantity * avg_buy_price
                    unrealized_pnl = current_value - cost_basis
                    pnl_percentage = (unrealized_pnl / cost_basis * 100) if cost_basis > 0 else 0.0

                    pnl_by_symbol[symbol] = {
                        "quantity": quantity,
                        "average_buy_price": avg_buy_price,
                        "current_price": current_price,
                        "current_value": current_value,
                        "cost_basis": cost_basis,
                        "unrealized_pnl": unrealized_pnl,
                        "pnl_percentage": pnl_percentage
                    }

                    total_pnl += unrealized_pnl
                    total_value += current_value

            return {
                "total_pnl": total_pnl,
                "total_value": total_value,
                "pnl_by_symbol": pnl_by_symbol
            }

        except Exception as e:
            logger.error(f"Failed to calculate unrealized PnL: {e}")
            return {
                "total_pnl": 0.0,
                "total_value": 0.0,
                "pnl_by_symbol": {}
            }

    def get_portfolio_overview(self) -> dict[str, Any]:
        """Get comprehensive portfolio overview with live data from cache"""
        try:
            logger.info("ðŸ“Š Getting portfolio overview...")

            # Get trade history and latest prices
            trades = self._get_trade_history()
            latest_prices = self._get_latest_prices()

            # Calculate holdings
            holdings = self._calculate_holdings(trades)

            # Calculate unrealized PnL
            pnl_data = self._calculate_unrealized_pnl(holdings, latest_prices)

            # Build portfolio overview
            portfolio_overview = {
                "total_value": pnl_data["total_value"],
                "total_pnl": pnl_data["total_pnl"],
                "positions_count": len([h for h in holdings.values() if h.get("quantity", 0) > 0]),
                "total_trades": len(trades),
                "holdings": {},
                "performance": {
                    "total_pnl": pnl_data["total_pnl"],
                    "total_value": pnl_data["total_value"],
                    "pnl_percentage": (
                        pnl_data["total_pnl"] / pnl_data["total_value"] * 100
                    )
                    if pnl_data["total_value"] > 0
                    else 0.0
                },
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "source": "cached_trade_data"
            }

            # Add detailed holdings information
            for symbol, holding in holdings.items():
                quantity = holding.get("quantity", 0.0)
                if quantity > 0:
                    current_price = latest_prices.get(symbol, 0.0)
                    pnl_info = pnl_data["pnl_by_symbol"].get(symbol, {})

                    portfolio_overview["holdings"][symbol] = {
                        "quantity": quantity,
                        "average_buy_price": holding.get("average_buy_price", 0.0),
                        "current_price": current_price,
                        "current_value": pnl_info.get("current_value", 0.0),
                        "cost_basis": pnl_info.get("cost_basis", 0.0),
                        "unrealized_pnl": pnl_info.get("unrealized_pnl", 0.0),
                        "pnl_percentage": pnl_info.get("pnl_percentage", 0.0),
                        "total_buy_amount": holding.get("total_buy_amount", 0.0),
                        "total_sell_amount": holding.get("total_sell_amount", 0.0),
                        "trade_count": len(holding.get("trades", []))
                    }

            logger.info(f"âœ… Portfolio overview generated: {portfolio_overview['positions_count']} positions, ${portfolio_overview['total_value']:.2f} total value")
            return portfolio_overview

        except Exception as e:
            logger.error(f"Failed to get portfolio overview: {e}")
            return {
                "total_value": 0.0,
                "total_pnl": 0.0,
                "positions_count": 0,
                "total_trades": 0,
                "holdings": {},
                "performance": {
                    "total_pnl": 0.0,
                    "total_value": 0.0,
                    "pnl_percentage": 0.0
                },
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "source": "error",
                "error": str(e)
            }

    async def get_overview(self) -> dict[str, Any]:
        """Alias for get_portfolio_overview() to match API expectations"""
        return self.get_portfolio_overview()

    async def get_positions(self) -> list[dict[str, Any]]:
        """Get portfolio positions for API endpoint"""
        try:
            overview = self.get_portfolio_overview()
            holdings = overview.get("holdings", {})

            positions = []
            for symbol, holding in holdings.items():
                if holding.get("quantity", 0) > 0:
                    positions.append({
                        "symbol": symbol,
                        "quantity": holding.get("quantity", 0.0),
                        "average_buy_price": holding.get("average_buy_price", 0.0),
                        "current_price": holding.get("current_price", 0.0),
                        "current_value": holding.get("current_value", 0.0),
                        "cost_basis": holding.get("cost_basis", 0.0),
                        "unrealized_pnl": holding.get("unrealized_pnl", 0.0),
                        "pnl_percentage": holding.get("pnl_percentage", 0.0),
                        "total_buy_amount": holding.get("total_buy_amount", 0.0),
                        "total_sell_amount": holding.get("total_sell_amount", 0.0),
                        "trade_count": holding.get("trade_count", 0)
                    })

            return positions

        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []

    async def get_portfolio_summary(self) -> dict[str, Any]:
        """Get portfolio summary for API endpoints"""
        try:
            overview = self.get_portfolio_overview()
            return {
                "total_value": overview.get("total_value", 0.0),
                "total_pnl": overview.get("total_pnl", 0.0),
                "positions_count": overview.get("positions_count", 0),
                "total_trades": overview.get("total_trades", 0),
                "performance": overview.get("performance", {}),
                "last_updated": overview.get("last_updated", ""),
                "source": overview.get("source", "")
            }
        except Exception as e:
            logger.error(f"Failed to get portfolio summary: {e}")
            return {
                "total_value": 0.0,
                "total_pnl": 0.0,
                "positions_count": 0,
                "total_trades": 0,
                "performance": {},
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "source": "error"
            }

    async def get_transactions(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return recent trades as transactions for API endpoints."""
        try:
            history = self._get_trade_history(limit=limit)
            transactions: list[dict[str, Any]] = []
            for t in history:
                transactions.append(
                    {
                        "timestamp": t.get("timestamp"),
                        "symbol": t.get("symbol"),
                        "side": t.get("trade_type") or t.get("side") or t.get("type"),
                        "quantity": t.get("quantity") or t.get("qty") or 0.0,
                        "price": t.get("price", 0.0),
                        "amount_usd": t.get("amount_usd", 0.0),
                        "exchange": t.get("exchange", ""),
                        "trade_id": t.get("trade_id") or t.get("id"),
                    }
                )
            return transactions[: max(0, int(limit))]
        except Exception as e:
            logger.error(f"Failed to get transactions: {e}")
            return []

    def get_usdt_balance(self) -> dict[str, float]:
        """Get USDT balance from portfolio"""
        try:
            # Get portfolio overview
            overview = self.get_portfolio_overview()
            
            # Calculate USDT balance components
            total_value = overview.get("total_value", 0.0)
            total_invested = overview.get("total_invested", 0.0)
            available_usdt = max(0.0, total_value - total_invested)
            
            # Calculate parking efficiency (percentage of USDT that's allocated)
            parking_efficiency = (total_invested / total_value * 100) if total_value > 0 else 0.0
            
            return {
                "total": total_value,
                "allocated": total_invested,
                "available": available_usdt,
                "efficiency": parking_efficiency
            }
            
        except Exception as e:
            logger.error(f"Failed to get USDT balance: {e}")
            return {
                "total": 0.0,
                "allocated": 0.0,
                "available": 0.0,
                "efficiency": 0.0
            }

    def get_total_value(self) -> float:
        """Get overall portfolio value in USD"""
        try:
            overview = self.get_portfolio_overview()
            return overview.get("total_value", 0.0)

        except Exception as e:
            logger.error(f"Failed to get total value: {e}")
            return 0.0

    def store_portfolio_snapshot(self) -> dict[str, Any]:
        """Store portfolio snapshot in cache for dashboard use"""
        try:
            logger.info("ðŸ’¾ Storing portfolio snapshot...")

            # Get current portfolio overview
            overview = self.get_portfolio_overview()

            # Create snapshot with additional metadata
            snapshot = {
                "portfolio_overview": overview,
                "snapshot_timestamp": datetime.now(timezone.utc).isoformat(),
                "metadata": {
                    "total_positions": overview.get("positions_count", 0),
                    "total_trades": overview.get("total_trades", 0),
                    "total_value": overview.get("total_value", 0.0),
                    "total_pnl": overview.get("total_pnl", 0.0),
                    "source": "portfolio_service_snapshot"
                }
            }

            # Store in cache
            snapshot_id = f"portfolio_snapshot_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
            self.cache.store_signal(
                signal_id=snapshot_id,
                symbol="PORTFOLIO_SNAPSHOT",
                signal_type="PORTFOLIO_SNAPSHOT",
                confidence=1.0,
                strategy="portfolio_tracking",
                metadata=snapshot
            )

            logger.info(f"âœ… Portfolio snapshot stored: {snapshot_id}")
            return {
                "snapshot_id": snapshot_id,
                "timestamp": snapshot["snapshot_timestamp"],
                "total_value": overview.get("total_value", 0.0),
                "positions_count": overview.get("positions_count", 0)
            }

        except Exception as e:
            logger.error(f"Failed to store portfolio snapshot: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

    def get_portfolio_status(self) -> dict[str, Any]:
        """Get current portfolio service status"""
        try:
            return {
                "service": "PortfolioService",
                "status": "active",
                "cache_connected": True,
                "last_snapshot": datetime.now(timezone.utc).isoformat(),
                "total_value": self.get_total_value(),
                "positions_count": len([h for h in self._calculate_holdings(self._get_trade_history()).values() if h.get("quantity", 0) > 0]),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            logger.error(f"Failed to get portfolio status: {e}")
            return {"success": False, "error": str(e)}


# Global portfolio service instance
portfolio_service = PortfolioService()


def get_portfolio_service() -> PortfolioService:
    """Get the global portfolio service instance"""
    return portfolio_service


if __name__ == "__main__":
    # Test the portfolio service
    service = PortfolioService()
    print(f"âœ… PortfolioService initialized: {service}")

    # Test portfolio overview
    overview = service.get_portfolio_overview()
    print(f"Portfolio overview: {overview}")

    # Test total value
    total_value = service.get_total_value()
    print(f"Total value: ${total_value:.2f}")

    # Test snapshot
    snapshot = service.store_portfolio_snapshot()
    print(f"Snapshot: {snapshot}")

    # Test status
    status = service.get_portfolio_status()
    print(f"Service status: {status['status']}")


