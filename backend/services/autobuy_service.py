"""
Autobuy Service for Mystic AI Trading Platform
Monitors cached prices and AI signals to execute automated buy orders.
"""

import logging
import uuid
from typing import Dict, Any, Optional, Tuple, List, cast
from datetime import datetime, timezone
import sys
import os

# Add backend to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from modules.ai.persistent_cache import PersistentCache
from services.market_data_router import MarketDataRouter
from utils.symbols import normalize_symbol_to_dash
from modules.notifications.alert_manager import AlertManager  # type: ignore[import-not-found]
from modules.ai.analytics_engine import AnalyticsEngine  # type: ignore[import-not-found]
import asyncio

logger = logging.getLogger(__name__)


class SignalEngine:
    """SignalEngine backed by PersistentCache"""

    def __init__(self):
        self.cache: Any = PersistentCache()  # type: ignore[no-redef]

    def get_latest_signal(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get the latest AI signal for a symbol"""
        try:
            signals = cast(List[Dict[str, Any]], self.cache.get_signals(symbol=symbol, limit=1))  # type: ignore[attr-defined]
            return signals[0] if signals else None
        except Exception as e:
            logger.error(f"Failed to get signal for {symbol}: {e}")
            return None

    def calculate_rsi(self, symbol: str, period: int = 14) -> Optional[float]:
        """Calculate RSI for a symbol using cached price data"""
        try:
            # Get price history for RSI calculation
            price_history: List[Dict[str, Any]] = cast(List[Dict[str, Any]], self.cache.get_price_history('aggregated', symbol, limit=period + 1))  # type: ignore[attr-defined]
            if len(price_history) < period + 1:
                return None

            prices = [float(p['price']) for p in price_history]
            prices.reverse()  # Oldest first

            # Calculate RSI
            gains: List[float] = []
            losses: List[float] = []

            for i in range(1, len(prices)):
                change = prices[i] - prices[i-1]
                if change > 0:
                    gains.append(change)
                    losses.append(0)
                else:
                    gains.append(0)
                    losses.append(abs(change))

            if len(gains) < period:
                return None

            # Calculate average gain and loss
            avg_gain: float = sum(gains[-period:]) / period
            avg_loss: float = sum(losses[-period:]) / period

            if avg_loss == 0:
                return 100.0

            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

            return rsi

        except Exception as e:
            logger.error(f"Failed to calculate RSI for {symbol}: {e}")
            return None


class AutoExecutionService:
    """AutoExecutionService using PersistentCache for trade logs"""

    def __init__(self):
        self.cache: Any = PersistentCache()
        self.simulation_mode = True  # Default to simulation mode

    def execute_buy_order(self, exchange: str, symbol: str, quantity: float,
                         price: float, order_type: str = "market") -> Dict[str, Any]:
        """Execute a buy order (simulated or real)"""
        try:
            order_id = str(uuid.uuid4())
            timestamp = datetime.now(timezone.utc)

            # Calculate total value
            total_value = quantity * price

            # Log the trade
            trade_success = self.cache.log_trade(  # type: ignore[attr-defined]
                trade_id=order_id,
                symbol=symbol,
                side="BUY",
                quantity=quantity,
                price=price,
                exchange=exchange,
                status="completed" if self.simulation_mode else "pending"
            )

            if trade_success:
                logger.info(f"✅ Buy order executed: {symbol} {quantity} @ ${price:.2f} = ${total_value:.2f}")

                return {
                    "success": True,
                    "order_id": order_id,
                    "symbol": symbol,
                    "side": "BUY",
                    "quantity": quantity,
                    "price": price,
                    "total_value": total_value,
                    "exchange": exchange,
                    "simulation_mode": self.simulation_mode,
                    "timestamp": timestamp.isoformat()
                }
            else:
                return {"success": False, "error": "Failed to log trade"}

        except Exception as e:
            logger.error(f"❌ Failed to execute buy order: {e}")
            return {"success": False, "error": str(e)}

    def set_simulation_mode(self, enabled: bool):
        """Enable or disable simulation mode"""
        self.simulation_mode = enabled
        logger.info(f"Simulation mode: {'ENABLED' if enabled else 'DISABLED'}")


class AutobuyService:
    """Main autobuy service that monitors signals and executes trades"""

    def __init__(self):
        self.cache: Any = PersistentCache()
        self.signal_engine = SignalEngine()
        self.execution_service = AutoExecutionService()
        self.active_orders: List[Dict[str, Any]] = []
        self.router = MarketDataRouter()
        self.alerts = AlertManager()
        self.analytics: Any = AnalyticsEngine()

        # Configuration
        self.rsi_threshold = 30  # Buy when RSI is below this
        self.min_quantity = 10.0  # Minimum trade amount in USD
        self.max_quantity = 1000.0  # Maximum trade amount in USD
        self.quantity_percentage = 0.1  # Percentage of available balance to use

        logger.info("✅ AutobuyService initialized")

    async def warmup(self, symbols: List[str]) -> Dict[str, Any]:
        try:
            tasks = [self.router.fanout_tickers(s) for s in symbols]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            warmed: Dict[str, Any] = {}
            for s, r in zip(symbols, results):
                warmed[normalize_symbol_to_dash(s)] = r if not isinstance(r, Exception) else {"error": str(r)}
            warm_event: Dict[str, Any] = {"symbols": [normalize_symbol_to_dash(s) for s in symbols]}
            self.analytics.track_event("autobuy_warmup", warm_event)
            return {"warmed": warmed, "ts": datetime.now(timezone.utc).isoformat()}
        except Exception as e:
            return {"error": str(e)}

    async def eval_signals(self, symbol: str) -> Dict[str, Any]:
        try:
            symbol_dash = normalize_symbol_to_dash(symbol)
            ohlcv = await self.router.get_ohlcv("binanceus", symbol_dash, interval="1h", limit=100)
            closes = [c.close for c in ohlcv[-50:]]
            if len(closes) < 14:
                return {"error": "insufficient_data"}
            rsi = self.signal_engine.calculate_rsi(symbol_dash) or 50.0
            sma_20 = sum(closes[-20:]) / 20 if len(closes) >= 20 else sum(closes) / len(closes)
            sma_50 = sum(closes[-50:]) / 50 if len(closes) >= 50 else sma_20
            latest = closes[-1]
            features = {"rsi": rsi, "sma_20": sma_20, "sma_50": sma_50, "price": latest}
            return {"symbol": symbol_dash, "features": features, "ts": datetime.now(timezone.utc).isoformat()}
        except Exception as e:
            return {"error": str(e)}

    async def decide_and_route(self, symbol: str) -> Dict[str, Any]:
        sig = await self.eval_signals(symbol)
        if sig.get("error"):
            return {"action": "hold", "reason": sig["error"], "signal": sig}
        f: Dict[str, float] = cast(Dict[str, float], sig["features"])
        action: str = "buy" if (f["rsi"] < self.rsi_threshold and f["sma_20"] > f["sma_50"]) else "hold"
        self.analytics.set_metric("last_decision_rsi", f["rsi"])  # type: ignore[arg-type]
        decision_event: Dict[str, Any] = {"symbol": sig["symbol"], "action": action, "features": f}
        self.analytics.track_event("autobuy_decision", decision_event)
        return {"action": action, "signal": sig}

    async def execute_decision(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if decision.get("action") != "buy":
                return {"skipped": True}
            symbol = decision["signal"]["symbol"]
            tickers = await self.router.fanout_tickers(symbol)
            if not tickers:
                return {"error": "no_market"}
            # Select best price among adapters
            best_ex, best = min(tickers.items(), key=lambda kv: kv[1].price)
            qty = max(self.min_quantity, self.max_quantity * self.quantity_percentage) / max(best.price, 1e-9)
            result: Dict[str, Any] = {
                "exchange": best_ex,
                "symbol": symbol,
                "qty": qty,
                "price": best.price,
                "ts": datetime.now(timezone.utc).isoformat(),
                "dry_run": True,
            }
            try:
                self.alerts.create_alert(
                    alert_type="trade_alert",
                    symbol=symbol,
                    condition=">",
                    threshold=best.price,
                    message=f"Autobuy decision BUY {symbol} @ {best.price} on {best_ex}",
                )
            except Exception:
                pass
            return result
        except Exception as e:
            return {"error": str(e)}

    async def heartbeat(self) -> Dict[str, Any]:
        try:
            adapters = await self.router.get_enabled_adapters()
            return {"status": "ready", "adapters": adapters, "active_orders": len(self.active_orders)}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def check_buy_conditions(self, symbol: str) -> Tuple[bool, Dict[str, Any]]:
        """Check if buy conditions are met for a symbol"""
        try:
            # Get latest price
            latest_price = self.cache.get_latest_price('aggregated', symbol)  # type: ignore[attr-defined]
            if not latest_price:
                return False, {"error": "No price data available"}

            price = float(latest_price['price'])

            # Get latest signal
            signal = self.signal_engine.get_latest_signal(symbol)
            if not signal:
                return False, {"error": "No signal data available"}

            # Calculate RSI
            rsi = self.signal_engine.calculate_rsi(symbol)
            if rsi is None:
                return False, {"error": "Unable to calculate RSI"}

            # Check buy conditions
            conditions_met = (
                rsi < self.rsi_threshold and
                signal.get('confidence', 0) > 0.7 and
                signal.get('action') == 'BUY'
            )

            return conditions_met, {
                "price": price,
                "rsi": rsi,
                "signal": signal,
                "conditions_met": conditions_met
            }

        except Exception as e:
            logger.error(f"❌ Failed to check buy conditions for {symbol}: {e}")
            return False, {"error": str(e)}

    def calculate_trade_quantity(self, symbol: str, price: float) -> float:
        """Calculate the quantity to trade based on available balance and limits"""
        try:
            # Get available balance (simplified - would normally check exchange balance)
            available_balance = 10000.0  # Mock balance

            # Calculate quantity based on percentage and limits
            target_amount = available_balance * self.quantity_percentage
            target_amount = max(self.min_quantity, min(target_amount, self.max_quantity))

            quantity = target_amount / price

            return quantity

        except Exception as e:
            logger.error(f"❌ Failed to calculate trade quantity for {symbol}: {e}")
            return 0.0

    def execute_autobuy(self, exchange: str, symbol: str) -> Dict[str, Any]:
        """Execute an automated buy order for a symbol"""
        try:
            # Check buy conditions
            should_buy, conditions = self.check_buy_conditions(symbol)

            if not should_buy:
                return {
                    "success": False,
                    "reason": "Buy conditions not met",
                    "conditions": conditions
                }

            price = conditions['price']
            quantity = self.calculate_trade_quantity(symbol, price)

            if quantity <= 0:
                return {
                    "success": False,
                    "reason": "Invalid quantity calculated",
                    "quantity": quantity
                }

            # Execute the buy order
            result = self.execution_service.execute_buy_order(
                exchange=exchange,
                symbol=symbol,
                quantity=quantity,
                price=price
            )

            if result.get("success"):
                # Add to active orders
                self.active_orders.append({
                    "order_id": result["order_id"],
                    "symbol": symbol,
                    "exchange": exchange,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })

            return result

        except Exception as e:
            logger.error(f"❌ Failed to execute autobuy for {symbol}: {e}")
            return {"success": False, "error": str(e)}

    def execute_all_autobuys(self) -> Dict[str, Any]:
        """Execute autobuy for all monitored symbols"""
        try:
            # List of symbols to monitor
            symbols: List[str] = [
                "BTC-USD", "ETH-USD", "ADA-USD", "SOL-USD", "DOT-USD",
                "LINK-USD", "MATIC-USD", "AVAX-USD", "UNI-USD", "ATOM-USD"
            ]

            results: List[Dict[str, Any]] = []
            successful_trades: int = 0

            for symbol in symbols:
                try:
                    # Try multiple exchanges
                    exchanges: List[str] = ["coinbase_us", "binance_us", "kraken_us"]

                    for exchange in exchanges:
                        result = self.execute_autobuy(exchange, symbol)
                        results.append({
                            "symbol": symbol,
                            "exchange": exchange,
                            "result": result
                        })

                        if result.get("success"):
                            successful_trades += 1
                            break  # Move to next symbol if successful

                except Exception as e:
                    logger.error(f"❌ Failed to process {symbol}: {e}")
                    results.append({
                        "symbol": symbol,
                        "exchange": "unknown",
                        "result": {"success": False, "error": str(e)}
                    })

            summary: Dict[str, Any] = {
                "success": True,
                "total_symbols": len(symbols),
                "successful_trades": successful_trades,
                "results": results,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

            logger.info(f"✅ Autobuy execution completed: {successful_trades}/{len(symbols)} successful")
            return summary

        except Exception as e:
            logger.error(f"❌ Failed to execute all autobuys: {e}")
            return {"success": False, "error": str(e)}

    def get_open_orders(self) -> List[Dict[str, Any]]:
        """Get all open autobuy orders"""
        try:
            return [
                {
                    "order_id": order.get("order_id", ""),
                    "symbol": order.get("symbol", ""),
                    "quantity": order.get("quantity", 0.0),
                    "price": order.get("price", 0.0),
                    "status": order.get("status", "pending"),
                    "exchange": order.get("exchange", ""),
                    "timestamp": order.get("timestamp", ""),
                    "order_type": "autobuy"
                }
                for order in self.active_orders
            ]
        except Exception as e:
            logger.error(f"❌ Failed to get open orders: {e}")
            return []

    def get_trigger_stats(self) -> Dict[str, Any]:
        """Get autobuy trigger statistics"""
        try:
            # Get recent autobuy signals
            autobuy_signals = cast(List[Dict[str, Any]], self.cache.get_signals_by_type("AUTOBUY_TRIGGER", limit=100))  # type: ignore[attr-defined]
            
            total_triggers = len(autobuy_signals)
            successful_triggers = len([s for s in autobuy_signals if s.get("metadata", {}).get("success", False)])
            failed_triggers = total_triggers - successful_triggers
            
            return {
                "total_triggers": total_triggers,
                "successful_triggers": successful_triggers,
                "failed_triggers": failed_triggers,
                "success_rate": (successful_triggers / total_triggers * 100) if total_triggers > 0 else 0,
                "recent_triggers": autobuy_signals[:10]
            }
        except Exception as e:
            logger.error(f"❌ Failed to get trigger stats: {e}")
            return {
                "total_triggers": 0,
                "successful_triggers": 0,
                "failed_triggers": 0,
                "success_rate": 0,
                "recent_triggers": []
            }

    def get_service_status(self) -> Dict[str, Any]:
        """Get current service status and statistics"""
        try:
            # Get cache stats
            cache_stats = cast(Dict[str, Any], self.cache.get_cache_stats())  # type: ignore[attr-defined]

            # Get recent trades
            recent_trades = cast(List[Dict[str, Any]], self.cache.get_trades(limit=10))  # type: ignore[attr-defined]

            # Get recent signals
            recent_signals = cast(List[Dict[str, Any]], self.cache.get_signals(limit=10))  # type: ignore[attr-defined]

            return {
                "service": "AutobuyService",
                "status": "active",
                "simulation_mode": self.execution_service.simulation_mode,
                "active_orders": len(self.active_orders),
                "cache_stats": cache_stats,
                "recent_trades": recent_trades,
                "recent_signals": recent_signals,
                "configuration": {
                    "rsi_threshold": self.rsi_threshold,
                    "min_quantity": self.min_quantity,
                    "max_quantity": self.max_quantity,
                    "quantity_percentage": self.quantity_percentage
                }
            }

        except Exception as e:
            logger.error(f"❌ Failed to get service status: {e}")
            return {"success": False, "error": str(e)}

    # Methods expected by endpoints
    async def get_configuration(self) -> Dict[str, Any]:
        return {
            "rsi_threshold": self.rsi_threshold,
            "min_quantity": self.min_quantity,
            "max_quantity": self.max_quantity,
            "quantity_percentage": self.quantity_percentage,
        }

    async def get_status(self) -> Dict[str, Any]:
        return self.get_service_status()

    async def get_statistics(self) -> Dict[str, Any]:
        return self.get_trigger_stats()

    async def get_performance_metrics(self) -> Dict[str, Any]:
        return {
            "active_orders": len(self.active_orders),
            "simulation_mode": self.execution_service.simulation_mode,
        }

    async def get_recent_trades(self, limit: int = 50) -> List[Dict[str, Any]]:
        try:
            return cast(List[Dict[str, Any]], self.cache.get_trades(limit=limit))  # type: ignore[attr-defined]
        except Exception:
            return []

    async def get_trade_summary(self) -> Dict[str, Any]:
        trades = cast(List[Dict[str, Any]], self.cache.get_trades(limit=100))  # type: ignore[attr-defined]
        return {"total_trades": len(trades)}

    async def get_recent_signals(self, limit: int = 50) -> List[Dict[str, Any]]:
        try:
            return cast(List[Dict[str, Any]], self.cache.get_signals(limit=limit))  # type: ignore[attr-defined]
        except Exception:
            return []

    async def get_signal_analysis(self) -> Dict[str, Any]:
        return {"quality": "unknown"}

    async def get_ai_status(self) -> Dict[str, Any]:
        return {"status": "active"}

    async def get_ai_performance(self) -> Dict[str, Any]:
        return {"accuracy": 0.0}

    async def start(self) -> Dict[str, Any]:
        return {"started": True}

    async def stop(self) -> Dict[str, Any]:
        return {"stopped": True}

    async def update_configuration(self, config: Dict[str, Any]) -> Dict[str, Any]:
        # Update only known keys
        self.rsi_threshold = config.get("rsi_threshold", self.rsi_threshold)
        self.min_quantity = config.get("min_quantity", self.min_quantity)
        self.max_quantity = config.get("max_quantity", self.max_quantity)
        self.quantity_percentage = config.get("quantity_percentage", self.quantity_percentage)
        return {"updated": True, "config": await self.get_configuration()}


# Global service instance
autobuy_service = AutobuyService()


def get_autobuy_service() -> AutobuyService:
    """Get the global autobuy service instance"""
    return autobuy_service


if __name__ == "__main__":
    # Test the service
    service = AutobuyService()
    print(f"✅ AutobuyService initialized: {service}")

    # Test status
    status = service.get_service_status()
    print(f"Service status: {status['status']}")
    print(f"Simulation mode: {status['simulation_mode']}")
    print(f"Active orders: {status['active_orders']}")
