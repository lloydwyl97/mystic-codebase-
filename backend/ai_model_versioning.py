#!/usr/bin/env python3
"""
AI Model Versioning System
Manages AI model versions, performance tracking, and rollback capabilities
"""

import json
import logging
import os
import shutil
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


class AIModelVersioning:
    """Manages AI model versions and performance tracking"""

    def __init__(self):
        self.models_dir = "data/model_versions"
        self.backup_dir = "data/model_backups"
        self.performance_dir = "data/model_performance"

        # Ensure directories exist
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.backup_dir, exist_ok=True)
        os.makedirs(self.performance_dir, exist_ok=True)

        # Model registry
        self.model_registry: dict[str, dict[str, Any]] = {}
        self.active_model: str | None = None
        self.model_performance: dict[str, dict[str, Any]] = {}

        # Performance thresholds
        self.performance_thresholds = {
            "min_accuracy": 0.6,
            "min_profit": 0.0,
            "max_drawdown": -0.1,
            "min_win_rate": 0.5,
        }

        # Load existing models
        self._load_model_registry()

        logger.info("âœ… AI Model Versioning System initialized")

    def _load_model_registry(self):
        """Load existing model registry from file"""
        try:
            registry_file = f"{self.models_dir}/model_registry.json"
            if os.path.exists(registry_file):
                with open(registry_file) as f:
                    self.model_registry = json.load(f)

                # Set active model
                for version, model_data in self.model_registry.items():
                    if model_data.get("is_active", False):
                        self.active_model = version
                        break

                logger.info(f"âœ… Loaded model registry with {len(self.model_registry)} models")
                logger.info(f"âœ… Active model: {self.active_model}")

        except Exception as e:
            logger.error(f"âŒ Error loading model registry: {e}")

    def _save_model_registry(self):
        """Save model registry to file"""
        try:
            registry_file = f"{self.models_dir}/model_registry.json"
            with open(registry_file, "w") as f:
                json.dump(self.model_registry, f, indent=2, default=str)

            logger.debug("âœ… Saved model registry")

        except Exception as e:
            logger.error(f"âŒ Error saving model registry: {e}")

    async def create_model_version(
        self, model_data: dict[str, Any], version: str, description: str = ""
    ) -> bool:
        """Create a new model version"""
        try:
            # Validate model data
            if not self._validate_model_data(model_data):
                logger.error("âŒ Invalid model data")
                return False

            # Create model metadata
            model_metadata = {
                "version": version,
                "description": description,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "is_active": False,
                "performance_metrics": {},
                "training_samples": model_data.get("training_samples", 0),
                "model_type": model_data.get("model_type", "unknown"),
                "hyperparameters": model_data.get("hyperparameters", {}),
                "features_used": model_data.get("features_used", []),
                "targets_used": model_data.get("targets_used", []),
            }

            # Save model file
            model_file = f"{self.models_dir}/model_v{version}.json"
            with open(model_file, "w") as f:
                json.dump(
                    {"metadata": model_metadata, "model_data": model_data},
                    f,
                    indent=2,
                    default=str,
                )

            # Add to registry
            self.model_registry[version] = model_metadata

            # Save registry
            self._save_model_registry()

            logger.info(f"âœ… Created model version {version}")
            return True

        except Exception as e:
            logger.error(f"âŒ Error creating model version {version}: {e}")
            return False

    def _validate_model_data(self, model_data: dict[str, Any]) -> bool:
        """Validate model data structure"""
        required_fields = [
            "model_type",
            "hyperparameters",
            "features_used",
            "targets_used",
        ]

        for field in required_fields:
            if field not in model_data:
                logger.error(f"âŒ Missing required field: {field}")
                return False

        return True

    async def activate_model(self, version: str) -> bool:
        """Activate a specific model version"""
        try:
            if version not in self.model_registry:
                logger.error(f"âŒ Model version {version} not found")
                return False

            # Deactivate current active model
            if self.active_model:
                self.model_registry[self.active_model]["is_active"] = False

            # Activate new model
            self.model_registry[version]["is_active"] = True
            self.model_registry[version]["activated_at"] = datetime.now(timezone.utc).isoformat()
            self.active_model = version

            # Save registry
            self._save_model_registry()

            logger.info(f"âœ… Activated model version {version}")
            return True

        except Exception as e:
            logger.error(f"âŒ Error activating model version {version}: {e}")
            return False

    async def deactivate_model(self, version: str) -> bool:
        """Deactivate a specific model version"""
        try:
            if version not in self.model_registry:
                logger.error(f"âŒ Model version {version} not found")
                return False

            self.model_registry[version]["is_active"] = False
            self.model_registry[version]["deactivated_at"] = datetime.now(timezone.utc).isoformat()

            if self.active_model == version:
                self.active_model = None

            # Save registry
            self._save_model_registry()

            logger.info(f"âœ… Deactivated model version {version}")
            return True

        except Exception as e:
            logger.error(f"âŒ Error deactivating model version {version}: {e}")
            return False

    async def rollback_model(self, target_version: str) -> bool:
        """Rollback to a previous model version"""
        try:
            if target_version not in self.model_registry:
                logger.error(f"âŒ Target model version {target_version} not found")
                return False

            # Create backup of current active model
            if self.active_model:
                await self._create_model_backup(self.active_model)

            # Activate target model
            success = await self.activate_model(target_version)

            if success:
                logger.info(f"âœ… Successfully rolled back to model version {target_version}")
                return True
            else:
                logger.error(f"âŒ Failed to rollback to model version {target_version}")
                return False

        except Exception as e:
            logger.error(f"âŒ Error rolling back to model version {target_version}: {e}")
            return False

    async def _create_model_backup(self, version: str):
        """Create backup of a model version"""
        try:
            source_file = f"{self.models_dir}/model_v{version}.json"
            backup_file = f"{self.backup_dir}/model_v{version}_backup_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"

            if os.path.exists(source_file):
                shutil.copy2(source_file, backup_file)
                logger.info(f"âœ… Created backup: {backup_file}")

        except Exception as e:
            logger.error(f"âŒ Error creating model backup: {e}")

    async def update_model_performance(
        self, version: str, performance_data: dict[str, Any]
    ) -> bool:
        """Update performance metrics for a model version"""
        try:
            if version not in self.model_registry:
                logger.error(f"âŒ Model version {version} not found")
                return False

            # Update performance metrics
            self.model_registry[version]["performance_metrics"] = performance_data
            self.model_registry[version]["last_performance_update"] = datetime.now(
                timezone.utc
            ).isoformat()

            # Store detailed performance data
            performance_file = f"{self.performance_dir}/performance_v{version}.json"
            with open(performance_file, "w") as f:
                json.dump(
                    {
                        "version": version,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "performance_data": performance_data,
                    },
                    f,
                    indent=2,
                    default=str,
                )

            # Save registry
            self._save_model_registry()

            logger.info(f"âœ… Updated performance for model version {version}")
            return True

        except Exception as e:
            logger.error(f"âŒ Error updating performance for model version {version}: {e}")
            return False

    async def evaluate_model_performance(self, version: str) -> dict[str, Any]:
        """Evaluate if a model meets performance thresholds"""
        try:
            if version not in self.model_registry:
                return {"error": f"Model version {version} not found"}

            performance = self.model_registry[version].get("performance_metrics", {})

            evaluation = {
                "version": version,
                "meets_thresholds": True,
                "failed_thresholds": [],
                "recommendation": "keep_active",
            }

            # Check accuracy threshold
            accuracy = performance.get("accuracy", 0.0)
            if accuracy < self.performance_thresholds["min_accuracy"]:
                evaluation["meets_thresholds"] = False
                evaluation["failed_thresholds"].append(
                    f"accuracy: {accuracy} < {self.performance_thresholds['min_accuracy']}"
                )

            # Check profit threshold
            profit = performance.get("total_profit", 0.0)
            if profit < self.performance_thresholds["min_profit"]:
                evaluation["meets_thresholds"] = False
                evaluation["failed_thresholds"].append(
                    f"profit: {profit} < {self.performance_thresholds['min_profit']}"
                )

            # Check drawdown threshold
            drawdown = performance.get("max_drawdown", 0.0)
            if drawdown < self.performance_thresholds["max_drawdown"]:
                evaluation["meets_thresholds"] = False
                evaluation["failed_thresholds"].append(
                    f"drawdown: {drawdown} < {self.performance_thresholds['max_drawdown']}"
                )

            # Check win rate threshold
            win_rate = performance.get("win_rate", 0.0)
            if win_rate < self.performance_thresholds["min_win_rate"]:
                evaluation["meets_thresholds"] = False
                evaluation["failed_thresholds"].append(
                    f"win_rate: {win_rate} < {self.performance_thresholds['min_win_rate']}"
                )

            # Set recommendation
            if not evaluation["meets_thresholds"]:
                evaluation["recommendation"] = "deactivate"
            elif self.active_model == version:
                evaluation["recommendation"] = "keep_active"
            else:
                evaluation["recommendation"] = "activate"

            return evaluation

        except Exception as e:
            logger.error(f"âŒ Error evaluating model performance: {e}")
            return {"error": str(e)}

    async def get_best_performing_model(self) -> str | None:
        """Get the best performing model version"""
        try:
            best_model = None
            best_score = -float("inf")

            for version, model_data in self.model_registry.items():
                performance = model_data.get("performance_metrics", {})

                # Calculate composite score
                accuracy = performance.get("accuracy", 0.0)
                profit = performance.get("total_profit", 0.0)
                win_rate = performance.get("win_rate", 0.0)
                drawdown = abs(performance.get("max_drawdown", 0.0))

                # Weighted score (higher is better)
                score = (accuracy * 0.3) + (profit * 0.4) + (win_rate * 0.2) - (drawdown * 0.1)

                if score > best_score:
                    best_score = score
                    best_model = version

            return best_model

        except Exception as e:
            logger.error(f"âŒ Error finding best performing model: {e}")
            return None

    async def auto_optimize_models(self) -> dict[str, Any]:
        """Automatically optimize model selection based on performance"""
        try:
            optimization_result = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "actions_taken": [],
                "recommendations": [],
            }

            # Evaluate current active model
            if self.active_model:
                evaluation = await self.evaluate_model_performance(self.active_model)

                if evaluation.get("recommendation") == "deactivate":
                    # Find best performing model
                    best_model = await self.get_best_performing_model()

                    if best_model and best_model != self.active_model:
                        # Rollback to best model
                        success = await self.rollback_model(best_model)
                        if success:
                            optimization_result["actions_taken"].append(
                                f"rolled_back_to_v{best_model}"
                            )
                            optimization_result["recommendations"].append(
                                f"Model v{self.active_model} underperforming, rolled back to v{best_model}"
                            )

            # Check for new models that might be better
            best_model = await self.get_best_performing_model()
            if best_model and best_model != self.active_model:
                best_evaluation = await self.evaluate_model_performance(best_model)

                if best_evaluation.get("recommendation") == "activate":
                    success = await self.activate_model(best_model)
                    if success:
                        optimization_result["actions_taken"].append(f"activated_v{best_model}")
                        optimization_result["recommendations"].append(
                            f"Activated better performing model v{best_model}"
                        )

            logger.info(
                f"âœ… Auto-optimization completed: {len(optimization_result['actions_taken'])} actions taken"
            )
            return optimization_result

        except Exception as e:
            logger.error(f"âŒ Error in auto-optimization: {e}")
            return {"error": str(e)}

    def get_model_registry(self) -> dict[str, Any]:
        """Get the complete model registry"""
        return {
            "models": self.model_registry,
            "active_model": self.active_model,
            "total_models": len(self.model_registry),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def get_model_performance_history(self, version: str) -> list[dict[str, Any]]:
        """Get performance history for a specific model version"""
        try:
            performance_file = f"{self.performance_dir}/performance_v{version}.json"

            if os.path.exists(performance_file):
                with open(performance_file) as f:
                    return json.load(f)
            else:
                return []

        except Exception as e:
            logger.error(f"âŒ Error getting performance history for {version}: {e}")
            return []

    def get_status(self) -> dict[str, Any]:
        """Get system status"""
        try:
            return {
                "total_models": len(self.model_registry),
                "active_model": self.active_model,
                "models_dir": self.models_dir,
                "backup_dir": self.backup_dir,
                "performance_dir": self.performance_dir,
                "performance_thresholds": self.performance_thresholds,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            logger.error(f"âŒ Error getting system status: {e}")
            return {"error": str(e)}


# Global instance
ai_model_versioning: AIModelVersioning | None = None


def get_ai_model_versioning() -> AIModelVersioning:
    """Get or create AI model versioning instance"""
    global ai_model_versioning
    if ai_model_versioning is None:
        ai_model_versioning = AIModelVersioning()
    return ai_model_versioning


