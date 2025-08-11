#!/usr/bin/env python3
"""
Unified Signal Manager
Coordinates all three tiers of signal fetchers and provides central interface
"""

import asyncio
import json
import logging
from dataclasses import asdict, dataclass
from datetime import timezone, datetime
from typing import Any, Dict, List, Optional

# Fix import paths to use proper relative imports
try:
    from .cosmic_fetcher import CosmicFetcher
    from .indicators import IndicatorsFetcher
    from .price_fetcher import PriceFetcher
    from .trade_engine import TradeEngine
except ImportError:
    # Fallback to absolute imports if relative imports fail
    from cosmic_fetcher import CosmicFetcher
    from indicators import IndicatorsFetcher
    from price_fetcher import PriceFetcher
    from trade_engine import TradeEngine

logger = logging.getLogger(__name__)


@dataclass
class UnifiedSignal:
    symbol: str
    tier1_data: Dict[str, Any]
    tier2_data: Dict[str, Any]
    tier3_data: Dict[str, Any]
    trade_decision: Dict[str, Any]
    timestamp: str


class UnifiedSignalManager:
    def __init__(self, redis_client: Any):
        self.redis_client = redis_client
        self.is_running = False

        # Initialize all tier components
        self.price_fetcher = PriceFetcher(redis_client)
        self.indicators_fetcher = IndicatorsFetcher(redis_client)
        self.cosmic_fetcher = CosmicFetcher(redis_client)
        self.trade_engine = TradeEngine(redis_client)

        # Manager configuration
        self.config = {
            "sync_interval": 10,  # Sync all tiers every 10 seconds
            "health_check_interval": 60,  # Health check every minute
            "cache_ttl": 300,  # 5 minutes
            "auto_restart": True,
            "max_restart_attempts": 3,
        }

        # Component tasks
        self.tasks = []

        # Performance tracking
        self.stats: Dict[str, Any] = {
            "start_time": None,
            "last_sync": None,
            "tier1_updates": 0,
            "tier2_updates": 0,
            "tier3_updates": 0,
            "trade_decisions": 0,
        }

        logger.info("Unified Signal Manager initialized")

    async def start_tier1_fetcher(self):
        """Start Tier 1 price fetcher"""
        try:
            logger.info("Starting Tier 1 Price Fetcher...")
            await self.price_fetcher.run()
        except Exception as e:
            logger.error(f"Tier 1 fetcher error: {e}")
            if self.config["auto_restart"]:
                logger.info("Restarting Tier 1 fetcher...")
                await asyncio.sleep(5)
                await self.start_tier1_fetcher()

    async def start_tier2_fetcher(self):
        """Start Tier 2 indicators fetcher"""
        try:
            logger.info("Starting Tier 2 Indicators Fetcher...")
            await self.indicators_fetcher.run()
        except Exception as e:
            logger.error(f"Tier 2 fetcher error: {e}")
            if self.config["auto_restart"]:
                logger.info("Restarting Tier 2 fetcher...")
                await asyncio.sleep(5)
                await self.start_tier2_fetcher()

    async def start_tier3_fetcher(self):
        """Start Tier 3 cosmic fetcher"""
        try:
            logger.info("Starting Tier 3 Cosmic Fetcher...")
            await self.cosmic_fetcher.run()
        except Exception as e:
            logger.error(f"Tier 3 fetcher error: {e}")
            if self.config["auto_restart"]:
                logger.info("Restarting Tier 3 fetcher...")
                await asyncio.sleep(5)
                await self.start_tier3_fetcher()

    async def start_trade_engine(self):
        """Start trade decision engine"""
        try:
            logger.info("Starting Trade Decision Engine...")
            await self.trade_engine.run()
        except Exception as e:
            logger.error(f"Trade engine error: {e}")
            if self.config["auto_restart"]:
                logger.info("Restarting trade engine...")
                await asyncio.sleep(5)
                await self.start_trade_engine()

    async def sync_all_tiers(self):
        """Synchronize all tiers and generate unified signals"""
        try:
            logger.debug("Synchronizing all signal tiers...")

            # Get data from all tiers
            tier1_signals = await self.price_fetcher.fetch_all_tier1_signals()
            tier2_signals = await self.indicators_fetcher.fetch_all_tier2_indicators()
            tier3_signals = await self.cosmic_fetcher.fetch_all_tier3_signals()

            # Generate trade decisions
            trade_decisions = await self.trade_engine.generate_trade_decisions()

            # Create unified signals
            unified_signals: Dict[str, Any] = {}

            # Process each symbol
            for symbol in tier1_signals.get("prices", {}):
                try:
                    unified_signal = UnifiedSignal(
                        symbol=symbol,
                        tier1_data=tier1_signals["prices"].get(symbol, {}),
                        tier2_data=tier2_signals.get("indicators", {}).get(symbol, {}),
                        tier3_data=tier3_signals.get("cosmic_signals", {}),
                        trade_decision=next(
                            (asdict(d) for d in trade_decisions if d.symbol == symbol),
                            {},  # type: ignore
                        ),
                        timestamp=datetime.now(timezone.utc).isoformat(),
                    )

                    unified_signals[symbol] = asdict(unified_signal)

                except Exception as e:
                    logger.error(f"Error creating unified signal for {symbol}: {e}")
                    continue

            # Cache unified signals
            await self._cache_unified_signals(unified_signals)

            # Update stats
            self.stats["last_sync"] = datetime.now(timezone.utc).isoformat()
            self.stats["tier1_updates"] += 1
            self.stats["tier2_updates"] += 1
            self.stats["tier3_updates"] += 1
            self.stats["trade_decisions"] += len(trade_decisions)

            logger.debug(f"Synchronized {len(unified_signals)} unified signals")

        except Exception as e:
            logger.error(f"Error synchronizing tiers: {e}")

    async def _cache_unified_signals(self, signals: Dict[str, Any]):
        """Cache unified signals"""
        try:
            self.redis_client.setex(
                "unified_signals",
                self.config["cache_ttl"],
                json.dumps(signals),
            )
        except Exception as e:
            logger.error(f"Error caching unified signals: {e}")

    async def health_check(self):
        """Periodic health check of all components"""
        while self.is_running:
            try:
                logger.info("=== Unified Signal Manager Health Check ===")

                # Check Tier 1
                tier1_status = self.price_fetcher.get_status()
                logger.info(f"Tier 1 (Price Fetcher): {tier1_status['status']}")

                # Check Tier 2
                tier2_status = self.indicators_fetcher.get_status()
                logger.info(f"Tier 2 (Indicators): {tier2_status['status']}")

                # Check Tier 3
                tier3_status = self.cosmic_fetcher.get_status()
                logger.info(f"Tier 3 (Cosmic): {tier3_status['status']}")

                # Check Trade Engine
                trade_status = self.trade_engine.get_status()
                logger.info(
                    f"Trade Engine: {trade_status['status']} - {trade_status['coin_states_count']} coins"
                )

                # Check cache health
                cache_keys = [
                    "tier1_signals",
                    "tier2_indicators",
                    "cosmic_signals",
                    "unified_signals",
                ]
                for key in cache_keys:
                    exists = self.redis_client.exists(key)
                    logger.info(f"Cache {key}: {'OK' if exists else 'MISSING'}")

                logger.info(f"Manager Stats: {self.stats}")

            except Exception as e:
                logger.error(f"Error in health check: {e}")

            await asyncio.sleep(self.config["health_check_interval"])

    async def sync_loop(self):
        """Main synchronization loop"""
        while self.is_running:
            try:
                await self.sync_all_tiers()
                await asyncio.sleep(self.config["sync_interval"])
            except Exception as e:
                logger.error(f"Error in sync loop: {e}")
                await asyncio.sleep(30)  # Wait 30 seconds on error

    async def start_all_components(self):
        """Start all signal components concurrently"""
        logger.info("Starting all signal components...")
        self.is_running = True
        self.stats["start_time"] = datetime.now(timezone.utc).isoformat()

        # Create tasks for all components
        tier1_task = asyncio.create_task(self.start_tier1_fetcher())
        tier2_task = asyncio.create_task(self.start_tier2_fetcher())
        tier3_task = asyncio.create_task(self.start_tier3_fetcher())
        trade_task = asyncio.create_task(self.start_trade_engine())
        sync_task = asyncio.create_task(self.sync_loop())
        health_task = asyncio.create_task(self.health_check())

        self.tasks = [
            tier1_task,
            tier2_task,
            tier3_task,
            trade_task,
            sync_task,
            health_task,
        ]

        try:
            # Wait for all tasks to complete
            await asyncio.gather(*self.tasks, return_exceptions=True)
        except Exception as e:
            logger.error(f"Error in component management: {e}")
        finally:
            await self.stop_all_components()

    async def stop_all_components(self):
        """Stop all components gracefully"""
        logger.info("Stopping all signal components...")
        self.is_running = False

        # Stop individual components
        try:
            await self.price_fetcher.close()
        except Exception as e:
            logger.error(f"Error closing price fetcher: {e}")

        # Cancel all tasks
        for task in self.tasks:
            if not task.done():
                task.cancel()

        logger.info("All components stopped")

    async def get_unified_signals(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Get unified signals for all symbols or a specific symbol"""
        try:
            signals_data = self.redis_client.get("unified_signals")
            if not signals_data:
                return {}

            signals = json.loads(signals_data)

            if symbol:
                return signals.get(symbol, {})
            else:
                return signals

        except Exception as e:
            logger.error(f"Error getting unified signals: {e}")
            return {}

    async def get_signal_summary(self) -> Dict[str, Any]:
        """Get summary of all signal tiers"""
        try:
            summary = {
                "tier1": {
                    "status": self.price_fetcher.get_status()["status"],
                    "coins_count": len(
                        self.price_fetcher.binance_coins + self.price_fetcher.coinbase_coins
                    ),
                },
                "tier2": {
                    "status": self.indicators_fetcher.get_status()["status"],
                    "indicators_count": len(self.trade_engine.coin_states),
                },
                "tier3": {
                    "status": self.cosmic_fetcher.get_status()["status"],
                    "cosmic_signals": [
                        "schumann_resonance",
                        "solar_flare_index",
                        "pineal_alignment",
                    ],
                },
                "trade_engine": {
                    "status": self.trade_engine.get_status()["status"],
                    "decisions_count": len(await self.trade_engine.get_trade_decisions()),
                },
                "manager_stats": self.stats,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            return summary

        except Exception as e:
            logger.error(f"Error getting signal summary: {e}")
            return {}

    async def get_active_signals(self) -> List[Dict[str, Any]]:
        """Get all active trading signals"""
        try:
            decisions = await self.trade_engine.get_trade_decisions()
            active_signals: List[Dict[str, Any]] = []

            for decision in decisions:
                if (
                    decision.get("action") in ["BUY", "SELL"]
                    and decision.get("confidence", 0) > 0.7
                ):
                    active_signals.append(decision)

            return active_signals

        except Exception as e:
            logger.error(f"Error getting active signals: {e}")
            return []

    async def run(self):
        """Main unified signal manager loop"""
        logger.info("Unified Signal Manager starting...")

        try:
            await self.start_all_components()
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
        except Exception as e:
            logger.error(f"Fatal error in Unified Signal Manager: {e}")
        finally:
            await self.stop_all_components()
            logger.info("Unified Signal Manager stopped")

    def get_status(self) -> Dict[str, Any]:
        """Get overall status of the unified signal manager"""
        return {
            "manager_status": "running" if self.is_running else "stopped",
            "config": self.config,
            "stats": self.stats,
            "components": {
                "tier1": self.price_fetcher.get_status(),
                "tier2": self.indicators_fetcher.get_status(),
                "tier3": self.cosmic_fetcher.get_status(),
                "trade_engine": self.trade_engine.get_status(),
            },
        }


# Global manager instance
unified_signal_manager = None


def get_unified_signal_manager(redis_client: Any) -> UnifiedSignalManager:
    """Get or create unified signal manager instance"""
    global unified_signal_manager
    if unified_signal_manager is None:
        unified_signal_manager = UnifiedSignalManager(redis_client)
    return unified_signal_manager


async def main():
    """Main function to run the unified signal manager"""
    # This would be called with a Redis client
    # For now, we'll just show the structure
    logger.info("Unified Signal Manager - Main function")
    logger.info("Use get_unified_signal_manager(redis_client) to get instance")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Unified Signal Manager stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
