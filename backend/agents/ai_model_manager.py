"""
AI Model Manager
Handles model versioning, deployment, and lifecycle management
"""

import asyncio
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import joblib  # type: ignore[reportMissingTypeStubs]
import numpy as np
import pandas as pd
import torch

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # noqa: E402

from backend.agents.base_agent import BaseAgent  # noqa: E402


class ModelVersion:
    """Model version information"""

    def __init__(
        self,
        model_id: str,
        version: str,
        model_type: str,
        metadata: dict[str, Any],
    ):
        self.model_id = model_id
        self.version = version
        self.model_type = model_type
        self.metadata = metadata
        self.created_at = datetime.now().isoformat()
        self.status = "created"
        self.performance_metrics = {}
        self.deployment_status = "not_deployed"
        self.model_size: int | None = None
        self.last_updated: datetime | None = None
        self.retention_period: timedelta = timedelta(days=30)


class ModelRegistry:
    """Model registry for versioning and tracking"""

    def __init__(self, registry_path: str):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self.models = {}
        self.load_registry()

    def load_registry(self):
        """Load model registry from disk"""
        try:
            registry_file = self.registry_path / "registry.json"
            if registry_file.exists():
                with open(registry_file) as f:
                    registry_data = json.load(f)
                    self.models = registry_data
        except Exception as e:
            print(f"âŒ Error loading model registry: {e}")

    def save_registry(self):
        """Save model registry to disk"""
        try:
            registry_file = self.registry_path / "registry.json"
            with open(registry_file, "w") as f:
                json.dump(self.models, f, indent=2)
        except Exception as e:
            print(f"âŒ Error saving model registry: {e}")

    def register_model(
        self,
        model_id: str,
        version: str,
        model_type: str,
        metadata: dict[str, Any],
    ) -> ModelVersion:
        """Register a new model version"""
        try:
            model_version = ModelVersion(model_id, version, model_type, metadata)

            if model_id not in self.models:
                self.models[model_id] = {}

            self.models[model_id][version] = {
                "model_type": model_type,
                "metadata": metadata,
                "created_at": model_version.created_at,
                "status": model_version.status,
                "performance_metrics": model_version.performance_metrics,
                "deployment_status": model_version.deployment_status,
            }

            self.save_registry()
            return model_version

        except Exception as e:
            print(f"âŒ Error registering model: {e}")
            return None

    def get_model_versions(self, model_id: str) -> list[str]:
        """Get all versions of a model"""
        try:
            if model_id in self.models:
                return list(self.models[model_id].keys())
            return []
        except Exception as e:
            print(f"âŒ Error getting model versions: {e}")
            return []

    def get_latest_version(self, model_id: str) -> str | None:
        """Get the latest version of a model"""
        try:
            versions = self.get_model_versions(model_id)
            if versions:
                return max(
                    versions,
                    key=lambda v: self.models[model_id][v]["created_at"],
                )
            return None
        except Exception as e:
            print(f"âŒ Error getting latest version: {e}")
            return None

    def update_model_status(self, model_id: str, version: str, status: str):
        """Update model status"""
        try:
            if model_id in self.models and version in self.models[model_id]:
                self.models[model_id][version]["status"] = status
                self.save_registry()
        except Exception as e:
            print(f"âŒ Error updating model status: {e}")

    def update_performance_metrics(self, model_id: str, version: str, metrics: dict[str, Any]):
        """Update model performance metrics"""
        try:
            if model_id in self.models and version in self.models[model_id]:
                self.models[model_id][version]["performance_metrics"] = metrics
                self.save_registry()
        except Exception as e:
            print(f"âŒ Error updating performance metrics: {e}")


class AIModelManager(BaseAgent):
    """AI Model Manager - Handles model versioning and deployment"""

    def __init__(self, agent_id: str = "ai_model_manager_001"):
        super().__init__(agent_id, "ai_model_manager")

        # Model manager-specific state
        self.state.update(
            {
                "models_managed": {},
                "deployments_active": {},
                "model_metrics": {},
                "last_deployment": None,
                "deployment_count": 0,
            }
        )

        # Model management configuration
        self.model_config = {
            "model_types": {
                "deep_learning": {
                    "extensions": [".pth", ".pt"],
                    "framework": "pytorch",
                    "supported_formats": ["state_dict", "full_model"],
                },
                "reinforcement_learning": {
                    "extensions": [".pth", ".pt"],
                    "framework": "pytorch",
                    "supported_formats": ["state_dict", "full_model"],
                },
                "machine_learning": {
                    "extensions": [".pkl", ".joblib"],
                    "framework": "sklearn",
                    "supported_formats": ["pickle", "joblib"],
                },
                "computer_vision": {
                    "extensions": [".pth", ".pt", ".onnx"],
                    "framework": "pytorch",
                    "supported_formats": ["state_dict", "full_model", "onnx"],
                },
            },
            "deployment_settings": {
                "auto_deploy": True,
                "deployment_threshold": (0.8),  # Performance threshold for auto-deployment
                "rollback_threshold": (0.6),  # Performance threshold for rollback
                "max_versions": 10,  # Maximum versions to keep
                "backup_enabled": True,
            },
            "monitoring_settings": {
                "performance_tracking": True,
                "drift_detection": True,
                "health_checks": True,
                "alerting": True,
            },
        }

        # Initialize model registry
        models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
        self.registry = ModelRegistry(models_dir)

        # Model storage paths
        self.models_path = Path(models_dir)
        self.models_path.mkdir(parents=True, exist_ok=True)

        # Register model manager-specific handlers
        self.register_handler("register_model", self.handle_register_model)
        self.register_handler("deploy_model", self.handle_deploy_model)
        self.register_handler("get_model_info", self.handle_get_model_info)
        self.register_handler("update_metrics", self.handle_update_metrics)

        print(f"ðŸ“¦ AI Model Manager {agent_id} initialized")

    async def validate_model_file(self, model_path: str, model_type: str) -> tuple[bool, str]:
        """Validate model file using appropriate framework"""
        try:
            model_file = Path(model_path)
            if not model_file.exists():
                return False, "Model file does not exist"

            # Check file extension
            file_ext = model_file.suffix.lower()
            supported_extensions = self.model_config["model_types"][model_type]["extensions"]

            if file_ext not in supported_extensions:
                return False, f"Unsupported file extension: {file_ext}"

            # Validate based on model type
            if model_type in ["deep_learning", "reinforcement_learning", "computer_vision"]:
                # PyTorch model validation
                try:
                    model_data = torch.load(model_path, map_location='cpu')
                    if isinstance(model_data, dict):
                        # State dict format
                        return True, "Valid PyTorch state dict"
                    else:
                        # Full model format
                        return True, "Valid PyTorch model"
                except Exception as e:
                    return False, f"PyTorch model validation failed: {e}"

            elif model_type == "machine_learning":
                # Scikit-learn model validation
                try:
                    model = joblib.load(model_path)
                    # Use the model to verify it's valid
                    if hasattr(model, 'predict'):
                        return True, "Valid scikit-learn model"
                    else:
                        return False, "Invalid scikit-learn model: missing predict method"
                except Exception as e:
                    return False, f"Scikit-learn model validation failed: {e}"

            return True, "Model validation passed"

        except Exception as e:
            return False, f"Model validation error: {e}"

    async def calculate_model_metrics(self, model_path: str, model_type: str) -> dict[str, Any]:
        """Calculate model metrics using numpy and pandas"""
        try:
            model_file = Path(model_path)
            file_size = model_file.stat().st_size

            # Create sample data for metrics calculation
            sample_data = np.random.randn(100, 10)  # 100 samples, 10 features
            df = pd.DataFrame(sample_data, columns=[f'feature_{i}' for i in range(10)])

            # Calculate basic metrics
            metrics = {
                "file_size_mb": file_size / (1024 * 1024),
                "feature_count": df.shape[1],
                "sample_count": df.shape[0],
                "data_type": str(df.dtypes.to_dict()),
                "memory_usage_mb": df.memory_usage(deep=True).sum() / (1024 * 1024),
                "timestamp": datetime.now().isoformat()
            }

            # Add model-specific metrics
            if model_type in ["deep_learning", "reinforcement_learning", "computer_vision"]:
                try:
                    model_data = torch.load(model_path, map_location='cpu')
                    if isinstance(model_data, dict):
                        metrics["parameter_count"] = sum(p.numel() for p in model_data.values() if hasattr(p, 'numel'))
                    else:
                        metrics["parameter_count"] = sum(p.numel() for p in model_data.parameters())
                except Exception:
                    metrics["parameter_count"] = "unknown"

            return metrics

        except Exception as e:
            print(f"âŒ Error calculating model metrics: {e}")
            return {"error": str(e)}

    async def load_model_for_inference(self, model_path: str, model_type: str):
        """Load model for inference using appropriate framework"""
        try:
            if model_type in ["deep_learning", "reinforcement_learning", "computer_vision"]:
                # Load PyTorch model
                model = torch.load(model_path, map_location='cpu')
                return model
            elif model_type == "machine_learning":
                # Load scikit-learn model
                model = joblib.load(model_path)
                return model
            else:
                raise ValueError(f"Unsupported model type: {model_type}")

        except Exception as e:
            print(f"âŒ Error loading model for inference: {e}")
            return None

    async def initialize(self):
        """Initialize AI model manager resources"""
        try:
            # Load model configuration
            await self.load_model_config()

            # Initialize model storage
            await self.initialize_model_storage()

            # Load existing models
            await self.load_existing_models()

            # Start model monitoring
            await self.start_model_monitoring()

            print(f"âœ… AI Model Manager {self.agent_id} initialized successfully")

        except Exception as e:
            print(f"âŒ Error initializing AI Model Manager: {e}")
            self.update_health_status("error")

    async def process_loop(self):
        """Main model management processing loop"""
        while self.running:
            try:
                # Monitor model performance
                await self.monitor_model_performance()

                # Manage model deployments
                await self.manage_deployments()

                # Clean up old models
                await self.cleanup_old_models()

                # Update model metrics
                await self.update_model_metrics()

                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                print(f"âŒ Error in model management processing loop: {e}")
                await asyncio.sleep(120)

    async def load_model_config(self):
        """Load model configuration from Redis"""
        try:
            # Load model configuration
            config_data = self.redis_client.get("ai_model_manager_config")
            if config_data:
                self.model_config = json.loads(config_data)

            print(
                f"ðŸ“‹ Model configuration loaded: {len(self.model_config['model_types'])} model types"
            )

        except Exception as e:
            print(f"âŒ Error loading model configuration: {e}")

    async def initialize_model_storage(self):
        """Initialize model storage structure"""
        try:
            # Create model storage directories
            for model_type in self.model_config["model_types"].keys():
                model_type_path = self.models_path / model_type
                model_type_path.mkdir(exist_ok=True)

                # Create subdirectories
                (model_type_path / "versions").mkdir(exist_ok=True)
                (model_type_path / "deployed").mkdir(exist_ok=True)
                (model_type_path / "backups").mkdir(exist_ok=True)

            print("ðŸ“ Model storage initialized")

        except Exception as e:
            print(f"âŒ Error initializing model storage: {e}")

    async def load_existing_models(self):
        """Load existing models from storage"""
        try:
            # Load models from registry
            for model_id, versions in self.registry.models.items():
                self.state["models_managed"][model_id] = {
                    "versions": list(versions.keys()),
                    "latest_version": self.registry.get_latest_version(model_id),
                    "model_type": (
                        versions[list(versions.keys())[0]]["model_type"] if versions else "unknown"
                    ),
                }

            print(f"ðŸ“¦ Loaded {len(self.state['models_managed'])} existing models")

        except Exception as e:
            print(f"âŒ Error loading existing models: {e}")

    async def start_model_monitoring(self):
        """Start model monitoring"""
        try:
            # Subscribe to model updates
            pubsub = self.redis_client.pubsub()
            pubsub.subscribe("model_updates")
            pubsub.subscribe("model_metrics")

            # Start monitoring listener
            asyncio.create_task(self.listen_model_updates(pubsub))

            print("ðŸ“¡ Model monitoring started")

        except Exception as e:
            print(f"âŒ Error starting model monitoring: {e}")

    async def listen_model_updates(self, pubsub):
        """Listen for model updates"""
        try:
            for message in pubsub.listen():
                if not self.running:
                    break

                if message["type"] == "message":
                    update_data = json.loads(message["data"])
                    await self.process_model_update(update_data)

        except Exception as e:
            print(f"âŒ Error in model updates listener: {e}")
        finally:
            pubsub.close()

    async def process_model_update(self, update_data: dict[str, Any]):
        """Process model update"""
        try:
            update_type = update_data.get("type")

            if update_type == "model_trained":
                await self.handle_model_trained(update_data)
            elif update_type == "model_metrics":
                await self.handle_model_metrics(update_data)
            elif update_type == "model_deployed":
                await self.handle_model_deployed(update_data)

        except Exception as e:
            print(f"âŒ Error processing model update: {e}")

    async def handle_model_trained(self, update_data: dict[str, Any]):
        """Handle model trained event"""
        try:
            model_id = update_data.get("model_id")
            model_type = update_data.get("model_type")
            model_path = update_data.get("model_path")
            metadata = update_data.get("metadata", {})

            if model_id and model_path:
                # Validate model file
                is_valid, validation_msg = await self.validate_model_file(model_path, model_type)
                if not is_valid:
                    print(f"âŒ Model validation failed: {validation_msg}")
                    return

                # Calculate model metrics
                metrics = await self.calculate_model_metrics(model_path, model_type)
                metadata.update(metrics)

                # Register new model version
                version = await self.generate_version_number(model_id)
                model_version = self.registry.register_model(
                    model_id, version, model_type, metadata
                )

                if model_version:
                    # Copy model to storage
                    await self.store_model(model_id, version, model_type, model_path)

                    # Update state
                    if model_id not in self.state["models_managed"]:
                        self.state["models_managed"][model_id] = {
                            "versions": [],
                            "latest_version": None,
                            "model_type": model_type,
                        }

                    self.state["models_managed"][model_id]["versions"].append(version)
                    self.state["models_managed"][model_id]["latest_version"] = version

                    print(f"âœ… Registered new model version: {model_id} v{version}")

        except Exception as e:
            print(f"âŒ Error handling model trained: {e}")

    async def handle_model_metrics(self, update_data: dict[str, Any]):
        """Handle model metrics update"""
        try:
            model_id = update_data.get("model_id")
            version = update_data.get("version")
            metrics = update_data.get("metrics", {})

            if model_id and version:
                # Update performance metrics
                self.registry.update_performance_metrics(model_id, version, metrics)

                # Check if model should be deployed
                if self.should_deploy_model(model_id, version, metrics):
                    await self.deploy_model(model_id, version)

                # Check if model should be rolled back
                elif self.should_rollback_model(model_id, version, metrics):
                    await self.rollback_model(model_id)

        except Exception as e:
            print(f"âŒ Error handling model metrics: {e}")

    async def handle_model_deployed(self, update_data: dict[str, Any]):
        """Handle model deployed event"""
        try:
            model_id = update_data.get("model_id")
            version = update_data.get("version")

            if model_id and version:
                # Update deployment status
                self.registry.update_model_status(model_id, version, "deployed")

                # Update state
                if model_id not in self.state["deployments_active"]:
                    self.state["deployments_active"][model_id] = {}

                self.state["deployments_active"][model_id] = {
                    "version": version,
                    "deployed_at": datetime.now().isoformat(),
                }

                print(f"âœ… Model deployed: {model_id} v{version}")

        except Exception as e:
            print(f"âŒ Error handling model deployed: {e}")

    async def generate_version_number(self, model_id: str) -> str:
        """Generate next version number for model"""
        try:
            existing_versions = self.registry.get_model_versions(model_id)
            version_numbers = []

            for version in existing_versions:
                try:
                    parts = version.split(".")
                    if len(parts) == 3:
                        version_numbers.append([int(parts[0]), int(parts[1]), int(parts[2])])
                except (ValueError, IndexError):
                    continue

            if version_numbers:
                latest = max(version_numbers)
                return f"{latest[0]}.{latest[1]}.{latest[2] + 1}"
            else:
                return "1.0.0"

        except Exception as e:
            print(f"âŒ Error generating version number: {e}")
            return "1.0.0"

    async def store_model(self, model_id: str, version: str, model_type: str, source_path: str):
        """Store model in versioned storage"""
        try:
            # Create version directory
            version_path = self.models_path / model_type / "versions" / f"{model_id}_v{version}"
            version_path.mkdir(parents=True, exist_ok=True)

            # Copy model file
            source_file = Path(source_path)
            if source_file.exists():
                dest_file = version_path / source_file.name
                shutil.copy2(source_file, dest_file)

                # Create metadata file
                metadata = {
                    "model_id": model_id,
                    "version": version,
                    "model_type": model_type,
                    "original_path": str(source_path),
                    "stored_at": datetime.now().isoformat(),
                    "file_size": dest_file.stat().st_size,
                    "file_hash": await self.calculate_file_hash(dest_file),
                }

                with open(version_path / "metadata.json", "w") as f:
                    json.dump(metadata, f, indent=2)

                print(f"âœ… Model stored: {model_id} v{version}")

        except Exception as e:
            print(f"âŒ Error storing model: {e}")

    async def calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file"""
        try:
            hash_sha256 = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            print(f"âŒ Error calculating file hash: {e}")
            return ""

    def should_deploy_model(self, model_id: str, version: str, metrics: dict[str, Any]) -> bool:
        """Check if model should be deployed"""
        try:
            if not self.model_config["deployment_settings"]["auto_deploy"]:
                return False

            # Check performance threshold
            performance_score = metrics.get("performance_score", 0)
            threshold = self.model_config["deployment_settings"]["deployment_threshold"]

            return performance_score >= threshold

        except Exception as e:
            print(f"âŒ Error checking deployment condition: {e}")
            return False

    def should_rollback_model(self, model_id: str, version: str, metrics: dict[str, Any]) -> bool:
        """Check if model should be rolled back"""
        try:
            # Check performance threshold
            performance_score = metrics.get("performance_score", 0)
            threshold = self.model_config["deployment_settings"]["rollback_threshold"]

            return performance_score < threshold

        except Exception as e:
            print(f"âŒ Error checking rollback condition: {e}")
            return False

    async def deploy_model(self, model_id: str, version: str):
        """Deploy model to production"""
        try:
            print(f"ðŸš€ Deploying model: {model_id} v{version}")

            # Get model info
            if (
                model_id not in self.registry.models
                or version not in self.registry.models[model_id]
            ):
                print(f"âŒ Model not found: {model_id} v{version}")
                return

            model_info = self.registry.models[model_id][version]
            model_type = model_info["model_type"]

            # Create deployment directory
            deployed_path = self.models_path / model_type / "deployed" / model_id
            deployed_path.mkdir(parents=True, exist_ok=True)

            # Copy model to deployment directory
            version_path = self.models_path / model_type / "versions" / f"{model_id}_v{version}"
            if version_path.exists():
                # Copy all files from version directory
                for file_path in version_path.iterdir():
                    if file_path.is_file():
                        shutil.copy2(file_path, deployed_path / file_path.name)

                # Update deployment status
                self.registry.update_model_status(model_id, version, "deployed")

                # Broadcast deployment
                await self.broadcast_model_deployment(model_id, version, model_type)

                # Update state
                self.state["deployments_active"][model_id] = {
                    "version": version,
                    "deployed_at": datetime.now().isoformat(),
                }

                self.state["deployment_count"] += 1
                self.state["last_deployment"] = datetime.now().isoformat()

                print(f"âœ… Model deployed successfully: {model_id} v{version}")

        except Exception as e:
            print(f"âŒ Error deploying model: {e}")

    async def rollback_model(self, model_id: str):
        """Rollback model to previous version"""
        try:
            print(f"ðŸ”„ Rolling back model: {model_id}")

            # Get previous version
            versions = self.registry.get_model_versions(model_id)
            if len(versions) < 2:
                print(f"âŒ No previous version available for rollback: {model_id}")
                return

            # Get second latest version
            versions.sort(key=lambda v: self.registry.models[model_id][v]["created_at"])
            previous_version = versions[-2]

            # Deploy previous version
            await self.deploy_model(model_id, previous_version)

            print(f"âœ… Model rolled back: {model_id} to v{previous_version}")

        except Exception as e:
            print(f"âŒ Error rolling back model: {e}")

    async def broadcast_model_deployment(self, model_id: str, version: str, model_type: str):
        """Broadcast model deployment to other agents"""
        try:
            deployment_update = {
                "type": "model_deployment_update",
                "model_id": model_id,
                "version": version,
                "model_type": model_type,
                "deployed_at": datetime.now().isoformat(),
            }

            # Broadcast to all agents
            await self.broadcast_message(deployment_update)

            # Send to specific agents
            await self.send_message("strategy_agent", deployment_update)
            await self.send_message("execution_agent", deployment_update)

        except Exception as e:
            print(f"âŒ Error broadcasting model deployment: {e}")

    async def monitor_model_performance(self):
        """Monitor model performance"""
        try:
            # Check deployed models performance
            for model_id, deployment_info in self.state["deployments_active"].items():
                version = deployment_info["version"]

                # Get performance metrics from Redis
                metrics_key = f"model_metrics:{model_id}:{version}"
                metrics_data = self.redis_client.get(metrics_key)

                if metrics_data:
                    metrics = json.loads(metrics_data)

                    # Check for performance degradation
                    if self.should_rollback_model(model_id, version, metrics):
                        await self.rollback_model(model_id)

        except Exception as e:
            print(f"âŒ Error monitoring model performance: {e}")

    async def manage_deployments(self):
        """Manage model deployments"""
        try:
            # Check for new models that should be deployed
            for model_id, model_info in self.state["models_managed"].items():
                latest_version = model_info["latest_version"]

                if latest_version and model_id not in self.state["deployments_active"]:
                    # Check if model meets deployment criteria
                    metrics_key = f"model_metrics:{model_id}:{latest_version}"
                    metrics_data = self.redis_client.get(metrics_key)

                    if metrics_data:
                        metrics = json.loads(metrics_data)
                        if self.should_deploy_model(model_id, latest_version, metrics):
                            await self.deploy_model(model_id, latest_version)

        except Exception as e:
            print(f"âŒ Error managing deployments: {e}")

    async def cleanup_old_models(self):
        """Clean up old model versions"""
        try:
            max_versions = self.model_config["deployment_settings"]["max_versions"]

            for model_id, model_info in self.state["models_managed"].items():
                versions = model_info["versions"]

                if len(versions) > max_versions:
                    # Keep only the latest versions
                    versions_to_keep = sorted(
                        versions,
                        key=lambda v: self.registry.models[model_id][v]["created_at"],
                    )[-max_versions:]
                    versions_to_remove = [v for v in versions if v not in versions_to_keep]

                    for version in versions_to_remove:
                        await self.remove_model_version(model_id, version)

        except Exception as e:
            print(f"âŒ Error cleaning up old models: {e}")

    async def remove_model_version(self, model_id: str, version: str):
        """Remove a model version"""
        try:
            # Get model info
            if model_id in self.registry.models and version in self.registry.models[model_id]:
                model_type = self.registry.models[model_id][version]["model_type"]

                # Remove from storage
                version_path = self.models_path / model_type / "versions" / f"{model_id}_v{version}"
                if version_path.exists():
                    shutil.rmtree(version_path)

                # Remove from registry
                del self.registry.models[model_id][version]
                self.registry.save_registry()

                # Update state
                if model_id in self.state["models_managed"]:
                    self.state["models_managed"][model_id]["versions"].remove(version)

                print(f"ðŸ—‘ï¸ Removed model version: {model_id} v{version}")

        except Exception as e:
            print(f"âŒ Error removing model version: {e}")

    async def handle_register_model(self, message: dict[str, Any]):
        """Handle model registration request"""
        try:
            model_id = message.get("model_id")
            model_type = message.get("model_type")
            model_path = message.get("model_path")
            metadata = message.get("metadata", {})

            print(f"ðŸ“ Model registration requested for {model_id}")

            if model_id and model_path:
                # Validate model file
                is_valid, validation_msg = await self.validate_model_file(model_path, model_type)
                if not is_valid:
                    response = {
                        "type": "model_registration_response",
                        "model_id": model_id,
                        "status": "failed",
                        "error": f"Model validation failed: {validation_msg}",
                        "timestamp": datetime.now().isoformat(),
                    }
                else:
                    # Calculate model metrics
                    metrics = await self.calculate_model_metrics(model_path, model_type)
                    metadata.update(metrics)

                    # Register new model version
                    version = await self.generate_version_number(model_id)
                    model_version = self.registry.register_model(
                        model_id, version, model_type, metadata
                    )

                    if model_version:
                        # Store model
                        await self.store_model(model_id, version, model_type, model_path)

                        response = {
                            "type": "model_registration_response",
                            "model_id": model_id,
                            "version": version,
                            "status": "registered",
                            "timestamp": datetime.now().isoformat(),
                        }
                    else:
                        response = {
                            "type": "model_registration_response",
                            "model_id": model_id,
                            "status": "failed",
                            "error": "Failed to register model",
                            "timestamp": datetime.now().isoformat(),
                        }
            else:
                response = {
                    "type": "model_registration_response",
                    "model_id": model_id,
                    "status": "failed",
                    "error": "Missing required parameters",
                    "timestamp": datetime.now().isoformat(),
                }

            sender = message.get("from_agent")
            if sender:
                await self.send_message(sender, response)

        except Exception as e:
            print(f"âŒ Error handling model registration: {e}")
            await self.broadcast_error(f"Model registration error: {e}")

    async def handle_deploy_model(self, message: dict[str, Any]):
        """Handle model deployment request"""
        try:
            model_id = message.get("model_id")
            version = message.get("version")

            print(f"ðŸš€ Model deployment requested for {model_id} v{version}")

            if model_id and version:
                await self.deploy_model(model_id, version)

                response = {
                    "type": "model_deployment_response",
                    "model_id": model_id,
                    "version": version,
                    "status": "deployed",
                    "timestamp": datetime.now().isoformat(),
                }
            else:
                response = {
                    "type": "model_deployment_response",
                    "model_id": model_id,
                    "version": version,
                    "status": "failed",
                    "error": "Missing required parameters",
                    "timestamp": datetime.now().isoformat(),
                }

            sender = message.get("from_agent")
            if sender:
                await self.send_message(sender, response)

        except Exception as e:
            print(f"âŒ Error handling model deployment: {e}")
            await self.broadcast_error(f"Model deployment error: {e}")

    async def handle_get_model_info(self, message: dict[str, Any]):
        """Handle model info request"""
        try:
            model_id = message.get("model_id")

            print(f"ðŸ“Š Model info requested for {model_id}")

            if model_id and model_id in self.state["models_managed"]:
                model_info = self.state["models_managed"][model_id]
                deployment_info = self.state["deployments_active"].get(model_id, {})

                response = {
                    "type": "model_info_response",
                    "model_id": model_id,
                    "model_info": model_info,
                    "deployment_info": deployment_info,
                    "registry_info": self.registry.models.get(model_id, {}),
                    "timestamp": datetime.now().isoformat(),
                }
            else:
                response = {
                    "type": "model_info_response",
                    "model_id": model_id,
                    "model_info": None,
                    "timestamp": datetime.now().isoformat(),
                }

            sender = message.get("from_agent")
            if sender:
                await self.send_message(sender, response)

        except Exception as e:
            print(f"âŒ Error handling model info request: {e}")
            await self.broadcast_error(f"Model info error: {e}")

    async def handle_update_metrics(self, message: dict[str, Any]):
        """Handle metrics update request"""
        try:
            model_id = message.get("model_id")
            version = message.get("version")
            metrics = message.get("metrics", {})

            print(f"ðŸ“ˆ Metrics update for {model_id} v{version}")

            if model_id and version and metrics:
                # Update performance metrics
                self.registry.update_performance_metrics(model_id, version, metrics)

                # Store in Redis for monitoring
                metrics_key = f"model_metrics:{model_id}:{version}"
                self.redis_client.set(metrics_key, json.dumps(metrics), ex=3600)

                response = {
                    "type": "metrics_update_response",
                    "model_id": model_id,
                    "version": version,
                    "status": "updated",
                    "timestamp": datetime.now().isoformat(),
                }
            else:
                response = {
                    "type": "metrics_update_response",
                    "model_id": model_id,
                    "version": version,
                    "status": "failed",
                    "error": "Missing required parameters",
                    "timestamp": datetime.now().isoformat(),
                }

            sender = message.get("from_agent")
            if sender:
                await self.send_message(sender, response)

        except Exception as e:
            print(f"âŒ Error handling metrics update: {e}")
            await self.broadcast_error(f"Metrics update error: {e}")

    async def update_model_metrics(self):
        """Update model manager metrics"""
        try:
            metrics = {
                "agent_id": self.agent_id,
                "models_managed": len(self.state["models_managed"]),
                "deployments_active": len(self.state["deployments_active"]),
                "deployment_count": self.state["deployment_count"],
                "last_deployment": self.state["last_deployment"],
                "registry_size": len(self.registry.models),
                "timestamp": datetime.now().isoformat(),
            }

            # Store metrics in Redis
            self.redis_client.set(f"agent_metrics:{self.agent_id}", json.dumps(metrics), ex=300)

        except Exception as e:
            print(f"âŒ Error updating model metrics: {e}")

    async def process_market_data(self, market_data: dict[str, Any]):
        """Process incoming market data for model management"""
        try:
            print("ðŸ“Š Processing market data for model management")

            # Update market data in state
            self.state["last_market_data"] = market_data
            self.state["last_market_update"] = datetime.now().isoformat()

            # Check if any models need retraining based on market conditions
            for model_id, model_info in self.state["models_managed"].items():
                if model_info.get("auto_retrain", False):
                    # Analyze market data for model performance
                    performance_metrics = await self.analyze_model_performance(model_id, market_data)
                    
                    if performance_metrics.get("needs_retraining", False):
                        print(f"ðŸ”„ Model {model_id} needs retraining based on market data")
                        await self.handle_model_retraining_request(model_id, performance_metrics)

            # Update model metrics
            await self.update_model_metrics()

            print("âœ… Market data processed for model management")

        except Exception as e:
            print(f"âŒ Error processing market data for model management: {e}")
            await self.broadcast_error(f"Model management market data error: {e}")

    async def analyze_model_performance(self, model_id: str, market_data: dict[str, Any]) -> dict[str, Any]:
        """Analyze model performance based on market data"""
        try:
            # Mock performance analysis
            # In a real implementation, this would analyze model predictions vs actual market movements
            return {
                "model_id": model_id,
                "accuracy": 0.75,
                "needs_retraining": False,
                "confidence": 0.8,
                "market_conditions": "stable"
            }
        except Exception as e:
            print(f"âŒ Error analyzing model performance: {e}")
            return {
                "model_id": model_id,
                "accuracy": 0.0,
                "needs_retraining": False,
                "confidence": 0.0,
                "market_conditions": "unknown"
            }

    async def handle_model_retraining_request(self, model_id: str, performance_metrics: dict[str, Any]):
        """Handle model retraining request"""
        try:
            print(f"ðŸ”„ Initiating retraining for model {model_id}")
            
            # Broadcast retraining request
            await self.broadcast_message({
                "type": "model_retraining_request",
                "model_id": model_id,
                "performance_metrics": performance_metrics,
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            print(f"âŒ Error handling model retraining request: {e}")
            await self.broadcast_error(f"Model retraining error: {e}")


async def main():
    """Main function"""
    agent = AIModelManager()
    await agent.start()


if __name__ == "__main__":
    asyncio.run(main())


