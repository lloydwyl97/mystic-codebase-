#!/usr/bin/env python3
"""
AI Training Data Pipeline
Collects real trade data and market signals for AI learning and model training
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional
import pandas as pd

from shared_cache import SharedCache
from services.mystic_signal_engine import mystic_signal_engine

logger = logging.getLogger(__name__)


class AITrainingDataPipeline:
    """Collects and processes training data for AI models"""

    def __init__(self, cache: SharedCache):
        self.cache = cache
        self.is_running = False

        # Training data storage
        self.training_data_dir = "data/ai_training"
        self.model_versions_dir = "data/model_versions"

        # Ensure directories exist
        os.makedirs(self.training_data_dir, exist_ok=True)
        os.makedirs(self.model_versions_dir, exist_ok=True)

        # Data collection settings
        self.collection_interval = 60  # Collect data every minute
        self.feature_window = 100  # Number of data points for features
        self.prediction_horizon = 24  # Hours to predict ahead

        # Training data cache
        self.training_cache: Dict[str, List[Dict[str, Any]]] = {}
        self.last_collection = {}

        logger.info("✅ AI Training Data Pipeline initialized")

    async def start(self):
        """Start the training data collection pipeline"""
        self.is_running = True
        logger.info("🚀 Starting AI Training Data Pipeline")

        while self.is_running:
            try:
                await self.collect_training_data()
                await asyncio.sleep(self.collection_interval)
            except Exception as e:
                logger.error(f"❌ Error in training data collection: {e}")
                await asyncio.sleep(30)

    async def stop(self):
        """Stop the training data collection pipeline"""
        self.is_running = False
        logger.info("🛑 AI Training Data Pipeline stopped")

    async def collect_training_data(self):
        """Collect comprehensive training data from all sources"""
        try:
            current_time = datetime.now(timezone.utc)

            # Collect market data
            market_data = await self._collect_market_data()

            # Collect mystic signals
            mystic_data = await self._collect_mystic_data()

            # Collect trade execution data
            trade_data = await self._collect_trade_data()

            # Collect strategy performance data
            strategy_data = await self._collect_strategy_data()

            # Combine all data into training sample
            training_sample = {
                "timestamp": current_time.isoformat(),
                "market_data": market_data,
                "mystic_data": mystic_data,
                "trade_data": trade_data,
                "strategy_data": strategy_data,
                "features": await self._extract_features(market_data, mystic_data, trade_data),
                "targets": await self._extract_targets(trade_data, strategy_data),
            }

            # Store training sample
            await self._store_training_sample(training_sample)

            # Update cache
            await self._update_training_cache(training_sample)

            logger.debug(f"✅ Collected training data for {current_time}")

        except Exception as e:
            logger.error(f"❌ Error collecting training data: {e}")

    async def _collect_market_data(self) -> Dict[str, Any]:
        """Collect market data for training"""
        try:
            # Get market data from cache
            market_data = self.cache.get_market_data()

            # Get indicator data for all symbols
            indicator_data = {}
            symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT", "AVAXUSDT"]

            for symbol in symbols:
                indicator_data[symbol] = self.cache.get_indicator_data(symbol)

            return {
                "market_data": market_data,
                "indicator_data": indicator_data,
                "volume_data": {symbol: self.cache.get_volume_data(symbol) for symbol in symbols},
            }

        except Exception as e:
            logger.error(f"❌ Error collecting market data: {e}")
            return {}

    async def _collect_mystic_data(self) -> Dict[str, Any]:
        """Collect mystic signal data for training"""
        try:
            # Generate comprehensive mystic signal
            mystic_signal = await mystic_signal_engine.generate_comprehensive_signal()

            return {
                "mystic_signal": {
                    "signal_type": mystic_signal.signal_type.value,
                    "confidence": mystic_signal.confidence,
                    "strength": mystic_signal.strength,
                    "factors": mystic_signal.factors,
                    "reasoning": mystic_signal.reasoning,
                },
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            logger.error(f"❌ Error collecting mystic data: {e}")
            return {}

    async def _collect_trade_data(self) -> Dict[str, Any]:
        """Collect trade execution data for training"""
        try:
            # This would normally come from the autobuy system
            # For now, return sample data structure
            return {
                "recent_trades": [],
                "trade_performance": {
                    "total_trades": 0,
                    "successful_trades": 0,
                    "failed_trades": 0,
                    "total_profit": 0.0,
                    "win_rate": 0.0,
                },
                "active_positions": [],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            logger.error(f"❌ Error collecting trade data: {e}")
            return {}

    async def _collect_strategy_data(self) -> Dict[str, Any]:
        """Collect strategy performance data for training"""
        try:
            # This would normally come from the strategy system
            # For now, return sample data structure
            return {
                "strategy_performance": {},
                "signal_accuracy": {},
                "confidence_scores": {},
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            logger.error(f"❌ Error collecting strategy data: {e}")
            return {}

    async def _extract_features(
        self,
        market_data: Dict[str, Any],
        mystic_data: Dict[str, Any],
        trade_data: Dict[str, Any],
    ) -> Dict[str, float]:
        """Extract numerical features for AI training"""
        try:
            features = {}

            # Market features
            if market_data and "market_data" in market_data:
                for symbol, data in market_data.get("market_data", {}).items():
                    if isinstance(data, dict):
                        features[f"{symbol}_price"] = data.get("price", 0.0)
                        features[f"{symbol}_volume"] = data.get("volume", 0.0)
                        features[f"{symbol}_change_24h"] = data.get("change_24h", 0.0)

            # Indicator features
            if market_data and "indicator_data" in market_data:
                for symbol, data in market_data.get("indicator_data", {}).items():
                    if isinstance(data, dict):
                        features[f"{symbol}_rsi"] = data.get("rsi", 50.0)
                        features[f"{symbol}_macd"] = data.get("macd", {}).get("histogram", 0.0)

            # Mystic features
            if mystic_data and "mystic_signal" in mystic_data:
                mystic_signal = mystic_data["mystic_signal"]
                features["mystic_confidence"] = mystic_signal.get("confidence", 0.5)
                features["mystic_strength"] = mystic_signal.get("strength", 0.5)
                features["mystic_signal_type"] = self._encode_signal_type(
                    mystic_signal.get("signal_type", "HOLD")
                )

            # Trade features
            if trade_data and "trade_performance" in trade_data:
                performance = trade_data["trade_performance"]
                features["win_rate"] = performance.get("win_rate", 0.0)
                features["total_profit"] = performance.get("total_profit", 0.0)
                features["total_trades"] = performance.get("total_trades", 0)

            return features

        except Exception as e:
            logger.error(f"❌ Error extracting features: {e}")
            return {}

    async def _extract_targets(
        self, trade_data: Dict[str, Any], strategy_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """Extract target variables for AI training"""
        try:
            targets = {}

            # Trade performance targets
            if trade_data and "trade_performance" in trade_data:
                performance = trade_data["trade_performance"]
                targets["next_trade_profit"] = performance.get("total_profit", 0.0) / max(
                    performance.get("total_trades", 1), 1
                )
                targets["next_trade_success"] = (
                    1.0 if performance.get("win_rate", 0.0) > 0.5 else 0.0
                )

            # Strategy performance targets
            if strategy_data and "signal_accuracy" in strategy_data:
                accuracy = strategy_data["signal_accuracy"]
                targets["signal_accuracy"] = sum(accuracy.values()) / max(len(accuracy), 1)

            return targets

        except Exception as e:
            logger.error(f"❌ Error extracting targets: {e}")
            return {}

    def _encode_signal_type(self, signal_type: str) -> float:
        """Encode signal type as numerical value"""
        encoding = {
            "STRONG_BUY": 1.0,
            "BUY": 0.75,
            "HOLD": 0.5,
            "SELL": 0.25,
            "STRONG_SELL": 0.0,
        }
        return encoding.get(signal_type, 0.5)

    async def _store_training_sample(self, sample: Dict[str, Any]):
        """Store training sample to file"""
        try:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            filename = f"{self.training_data_dir}/training_sample_{timestamp}.json"

            with open(filename, "w") as f:
                json.dump(sample, f, indent=2, default=str)

            logger.debug(f"✅ Stored training sample: {filename}")

        except Exception as e:
            logger.error(f"❌ Error storing training sample: {e}")

    async def _update_training_cache(self, sample: Dict[str, Any]):
        """Update in-memory training cache"""
        try:
            timestamp = sample["timestamp"]

            # Store in cache by date
            date_key = datetime.fromisoformat(timestamp).strftime("%Y-%m-%d")

            if date_key not in self.training_cache:
                self.training_cache[date_key] = []

            self.training_cache[date_key].append(sample)

            # Keep only recent data (last 30 days)
            if len(self.training_cache[date_key]) > 1440:  # 24 hours * 60 minutes
                self.training_cache[date_key] = self.training_cache[date_key][-1440:]

            self.last_collection[date_key] = timestamp

        except Exception as e:
            logger.error(f"❌ Error updating training cache: {e}")

    async def get_training_dataset(self, days: int = 7) -> pd.DataFrame:
        """Get training dataset for the specified number of days"""
        try:
            all_samples = []
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=days)

            for date_key, samples in self.training_cache.items():
                sample_date = datetime.strptime(date_key, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                if start_date <= sample_date <= end_date:
                    all_samples.extend(samples)

            if not all_samples:
                logger.warning(f"⚠️ No training samples found for last {days} days")
                return pd.DataFrame()

            # Convert to DataFrame
            df = pd.DataFrame(all_samples)

            # Extract features and targets
            features_df = pd.json_normalize(df["features"])
            targets_df = pd.json_normalize(df["targets"])

            # Combine features and targets
            training_df = pd.concat([features_df, targets_df], axis=1)

            logger.info(f"✅ Created training dataset with {len(training_df)} samples")
            return training_df

        except Exception as e:
            logger.error(f"❌ Error creating training dataset: {e}")
            return pd.DataFrame()

    async def save_model_version(self, model_data: Dict[str, Any], version: str):
        """Save model version with training metadata"""
        try:
            model_file = f"{self.model_versions_dir}/model_v{version}.json"

            model_metadata = {
                "version": version,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "training_samples": len(
                    self.training_cache.get(datetime.now(timezone.utc).strftime("%Y-%m-%d"), [])
                ),
                "model_data": model_data,
            }

            with open(model_file, "w") as f:
                json.dump(model_metadata, f, indent=2, default=str)

            logger.info(f"✅ Saved model version {version}")

        except Exception as e:
            logger.error(f"❌ Error saving model version: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Get pipeline status"""
        try:
            total_samples = sum(len(samples) for samples in self.training_cache.values())

            return {
                "is_running": self.is_running,
                "total_samples": total_samples,
                "cache_dates": list(self.training_cache.keys()),
                "last_collection": self.last_collection,
                "collection_interval": self.collection_interval,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            logger.error(f"❌ Error getting pipeline status: {e}")
            return {"error": str(e)}


# Global instance
ai_training_pipeline: Optional[AITrainingDataPipeline] = None


def get_ai_training_pipeline(cache: SharedCache) -> AITrainingDataPipeline:
    """Get or create AI training pipeline instance"""
    global ai_training_pipeline
    if ai_training_pipeline is None:
        ai_training_pipeline = AITrainingDataPipeline(cache)
    return ai_training_pipeline
