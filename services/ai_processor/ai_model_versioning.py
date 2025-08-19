"""
AI Model Versioning Service
Advanced model versioning, tracking, and management system
"""

import asyncio
import hashlib
import json
import os
import shutil
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, cast

import redis
from dotenv import load_dotenv

from utils.redis_helpers import to_str_list

# Load environment variables
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))


@dataclass
class ModelVersion:
    """Model version representation"""

    version_id: str
    model_name: str
    model_type: str
    file_path: str
    metadata: dict[str, Any]
    performance_metrics: dict[str, Any]
    created_at: str
    created_by: str
    status: str = "ACTIVE"
    parent_version: str | None = None
    tags: list[str] = field(default_factory=list)
    description: str = ""
    file_hash: str = ""
    file_size: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary"""
        return {
            "version_id": self.version_id,
            "model_name": self.model_name,
            "model_type": self.model_type,
            "file_path": self.file_path,
            "metadata": self.metadata,
            "performance_metrics": self.performance_metrics,
            "created_at": self.created_at,
            "created_by": self.created_by,
            "status": self.status,
            "parent_version": self.parent_version,
            "tags": self.tags,
            "description": self.description,
            "file_hash": self.file_hash,
            "file_size": self.file_size,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelVersion":
        """Create from dictionary"""
        return cls(**data)


class ModelVersioningService:
    def __init__(self):
        """Initialize Model Versioning Service"""
        self.redis_client = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            db=int(os.getenv("REDIS_DB", 0)),
            decode_responses=True,
        )
        self.running = False
        self.db_path = "model_versions.db"
        self.models_dir = "model_versions"

        # Create directories
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs("models", exist_ok=True)
        os.makedirs("scalers", exist_ok=True)

        # Initialize database
        self.init_database()

    def init_database(self):
        """Initialize SQLite database for model versioning"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Create tables
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS model_versions (
                    version_id TEXT PRIMARY KEY,
                    model_name TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    metadata TEXT,
                    performance_metrics TEXT,
                    created_at TEXT NOT NULL,
                    created_by TEXT NOT NULL,
                    status TEXT DEFAULT 'ACTIVE',
                    parent_version TEXT,
                    tags TEXT,
                    description TEXT,
                    file_hash TEXT,
                    file_size INTEGER
                )
            """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS model_lineage (
                    version_id TEXT,
                    parent_version TEXT,
                    relationship_type TEXT,
                    created_at TEXT,
                    PRIMARY KEY (version_id, parent_version)
                )
            """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS model_deployments (
                    deployment_id TEXT PRIMARY KEY,
                    version_id TEXT,
                    environment TEXT,
                    deployed_at TEXT,
                    status TEXT,
                    performance_snapshot TEXT,
                    FOREIGN KEY (version_id) REFERENCES model_versions (version_id)
                )
            """
            )

            conn.commit()
            conn.close()
            print("✅ Database initialized successfully")

        except Exception as e:
            print(f"❌ Database initialization error: {e}")

    async def start(self):
        """Start the Model Versioning Service"""
        print("📦 Starting Model Versioning Service...")
        self.running = True

        # Start monitoring
        await self.monitor_models()

    async def monitor_models(self):
        """Monitor for new models and version them"""
        print("👀 Monitoring for new models...")

        while self.running:
            try:
                # Check for new models in Redis
                new_model_raw = self.redis_client.lpop("new_models_queue")
                new_model = cast(str | None, new_model_raw if isinstance(new_model_raw, str) else None)

                if new_model:
                    model_data = json.loads(new_model)
                    await self.version_model(model_data)

                # Check for model updates
                await self.check_model_updates()

                # Cleanup old versions
                await self.cleanup_old_versions()

                await asyncio.sleep(30)  # Check every 30 seconds

            except Exception as e:
                print(f"❌ Error in model monitoring: {e}")
                await asyncio.sleep(60)

    async def version_model(self, model_data: dict[str, Any]):
        """Create a new version of a model"""
        try:
            model_name = model_data.get("name", "Unknown")
            model_type = model_data.get("type", "unknown")
            file_path = model_data.get("model_path", "")
            scaler_path = model_data.get("scaler_path", "")

            print(f"📦 Versioning model: {model_name}")

            # Generate version ID
            version_id = self.generate_version_id(model_name, model_type)

            # Calculate file hash
            file_hash = await self.calculate_file_hash(file_path)
            file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0

            # Create metadata
            metadata = {
                "model_type": model_type,
                "scaler_path": scaler_path,
                "parameters": model_data.get("parameters", {}),
                "symbol": model_data.get("symbol", ""),
                "training_config": model_data.get("training_config", {}),
            }

            # Performance metrics
            performance_metrics = model_data.get("performance", {})

            # Create model version
            model_version = ModelVersion(
                version_id=version_id,
                model_name=model_name,
                model_type=model_type,
                file_path=file_path,
                metadata=metadata,
                performance_metrics=performance_metrics,
                created_at=datetime.now().isoformat(),
                created_by="AI_Strategy_Generator",
                file_hash=file_hash,
                file_size=file_size,
            )

            # Store version
            await self.store_model_version(model_version)

            # Copy model file to versioned location
            await self.backup_model_file(model_version)

            # Update Redis
            self.redis_client.set(
                f"model_version:{version_id}",
                json.dumps(model_version.to_dict()),
                ex=86400,
            )
            self.redis_client.lpush("model_versions", version_id)

            print(f"✅ Versioned model: {version_id}")

        except Exception as e:
            print(f"❌ Error versioning model: {e}")

    def generate_version_id(self, model_name: str, model_type: str) -> str:
        """Generate unique version ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{model_name}_{model_type}_{timestamp}"

    async def calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of model file"""
        try:
            if not os.path.exists(file_path):
                return ""

            hash_sha256 = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)

            return hash_sha256.hexdigest()

        except Exception as e:
            print(f"Error calculating file hash: {e}")
            return ""

    async def store_model_version(self, model_version: ModelVersion):
        """Store model version in database and Redis"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT OR REPLACE INTO model_versions 
                (version_id, model_name, model_type, file_path, metadata, performance_metrics,
                 created_at, created_by, status, parent_version, tags, description, file_hash, file_size)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    model_version.version_id,
                    model_version.model_name,
                    model_version.model_type,
                    model_version.file_path,
                    json.dumps(model_version.metadata),
                    json.dumps(model_version.performance_metrics),
                    model_version.created_at,
                    model_version.created_by,
                    model_version.status,
                    model_version.parent_version,
                    json.dumps(model_version.tags),
                    model_version.description,
                    model_version.file_hash,
                    model_version.file_size,
                ),
            )

            conn.commit()
            conn.close()

            # Store in Redis for quick access
            self.redis_client.set(
                f"model_version:{model_version.version_id}",
                json.dumps(model_version.to_dict()),
                ex=86400,
            )

            # Add to version list
            self.redis_client.lpush("model_versions", model_version.version_id)

            # Broadcast version update
            await self.broadcast_model_versions()

        except Exception as e:
            print(f"Error storing model version: {e}")

    async def broadcast_model_versions(self):
        """Broadcast model version updates"""
        try:
            # Get all model versions
            all_versions = []

            # Get all version IDs
            version_ids = to_str_list(self.redis_client.lrange("model_versions", 0, -1))

            for version_id in version_ids:
                version_data = self.redis_client.get(f"model_version:{version_id}")
                if version_data:
                    version = json.loads(version_data)
                    all_versions.append(version)

            # Store in Redis for dashboard access
            self.redis_client.set("model_versions", json.dumps(all_versions), ex=300)

            # Publish to Redis channel
            self.redis_client.publish("model_versions", json.dumps(all_versions))

        except Exception as e:
            print(f"Error broadcasting model versions: {e}")

    async def backup_model_file(self, model_version: ModelVersion):
        """Backup model file to versioned location"""
        try:
            if not os.path.exists(model_version.file_path):
                return

            # Create versioned file path
            versioned_path = os.path.join(
                self.models_dir, f"{model_version.version_id}.pth"
            )

            # Copy file
            shutil.copy2(model_version.file_path, versioned_path)

            # Update file path
            model_version.file_path = versioned_path

            print(f"✅ Backed up model to: {versioned_path}")

        except Exception as e:
            print(f"Error backing up model file: {e}")

    async def check_model_updates(self):
        """Check for model updates and create new versions"""
        try:
            # Get active models
            active_models = to_str_list(self.redis_client.lrange("ai_strategies", 0, -1))

            for model_id in active_models:
                model_data = self.redis_client.get(f"ai_strategy:{model_id}")
                if model_data:
                    model = json.loads(model_data)

                    # Check if model needs versioning
                    if await self.should_version_model(model):
                        await self.version_model(model)

        except Exception as e:
            print(f"Error checking model updates: {e}")

    async def should_version_model(self, model: dict[str, Any]) -> bool:
        """Check if model should be versioned"""
        try:
            model_path = model.get("model_path", "")
            if not model_path or not os.path.exists(model_path):
                return False

            # Check if model has been modified
            last_modified = os.path.getmtime(model_path)
            last_version_time = self.redis_client.get(f"last_version:{model['id']}")

            if last_version_time:
                last_version_timestamp = float(last_version_time)
                if last_modified > last_version_timestamp:
                    return True

            # Check if performance has improved significantly
            current_performance = model.get("performance", {})
            last_performance = self.redis_client.get(f"last_performance:{model['id']}")

            if last_performance:
                last_perf = json.loads(last_performance)
                if self.has_significant_improvement(current_performance, last_perf):
                    return True

            return False

        except Exception as e:
            print(f"Error checking if model should be versioned: {e}")
            return False

    def has_significant_improvement(
        self, current: dict[str, Any], previous: dict[str, Any]
    ) -> bool:
        """Check if performance has improved significantly"""
        try:
            # Check accuracy improvement
            current_acc = current.get("accuracy", 0)
            previous_acc = previous.get("accuracy", 0)

            if current_acc - previous_acc > 0.05:  # 5% improvement
                return True

            # Check other metrics
            current_return = current.get("total_return", 0)
            previous_return = previous.get("total_return", 0)

            if current_return - previous_return > 0.02:  # 2% return improvement
                return True

            return False

        except Exception as e:
            print(f"Error checking performance improvement: {e}")
            return False

    async def cleanup_old_versions(self):
        """Clean up old model versions"""
        try:
            # Keep only last 10 versions per model
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Get all model names
            cursor.execute(
                "SELECT DISTINCT model_name FROM model_versions WHERE status = 'ACTIVE'"
            )
            model_names = [row[0] for row in cursor.fetchall()]

            for model_name in model_names:
                # Get versions for this model
                cursor.execute(
                    """
                    SELECT version_id, created_at FROM model_versions 
                    WHERE model_name = ? AND status = 'ACTIVE'
                    ORDER BY created_at DESC
                """,
                    (model_name,),
                )

                versions = cursor.fetchall()

                if len(versions) > 10:
                    # Mark old versions as archived
                    old_versions = versions[10:]
                    for version_id, _ in old_versions:
                        cursor.execute(
                            """
                            UPDATE model_versions SET status = 'ARCHIVED' 
                            WHERE version_id = ?
                        """,
                            (version_id,),
                        )

                        # Remove from Redis
                        self.redis_client.delete(f"model_version:{version_id}")

            conn.commit()
            conn.close()

        except Exception as e:
            print(f"Error cleaning up old versions: {e}")

    async def get_model_versions(self, model_name: str | None = None) -> list[ModelVersion]:
        """Get model versions"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            if model_name:
                cursor.execute(
                    """
                    SELECT * FROM model_versions 
                    WHERE model_name = ? AND status = 'ACTIVE'
                    ORDER BY created_at DESC
                """,
                    (model_name,),
                )
            else:
                cursor.execute(
                    """
                    SELECT * FROM model_versions 
                    WHERE status = 'ACTIVE'
                    ORDER BY created_at DESC
                """
                )

            rows = cursor.fetchall()
            conn.close()

            versions = []
            for row in rows:
                version = ModelVersion(
                    version_id=row[0],
                    model_name=row[1],
                    model_type=row[2],
                    file_path=row[3],
                    metadata=json.loads(row[4]) if row[4] else {},
                    performance_metrics=json.loads(row[5]) if row[5] else {},
                    created_at=row[6],
                    created_by=row[7],
                    status=row[8],
                    parent_version=row[9],
                    tags=json.loads(row[10]) if row[10] else [],
                    description=row[11],
                    file_hash=row[12],
                    file_size=row[13],
                )
                versions.append(version)

            return versions

        except Exception as e:
            print(f"Error getting model versions: {e}")
            return []

    async def get_model_lineage(self, version_id: str) -> list[dict[str, Any]]:
        """Get model lineage"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT mv1.version_id, mv1.model_name, mv1.created_at,
                       ml.relationship_type, mv1.performance_metrics
                FROM model_versions mv1
                LEFT JOIN model_lineage ml ON mv1.version_id = ml.version_id
                WHERE ml.parent_version = ? OR mv1.version_id = ?
                ORDER BY mv1.created_at
            """,
                (version_id, version_id),
            )

            rows = cursor.fetchall()
            conn.close()

            lineage = []
            for row in rows:
                lineage.append(
                    {
                        "version_id": row[0],
                        "model_name": row[1],
                        "created_at": row[2],
                        "relationship_type": row[3],
                        "performance_metrics": (json.loads(row[4]) if row[4] else {}),
                    }
                )

            return lineage

        except Exception as e:
            print(f"Error getting model lineage: {e}")
            return []

    async def deploy_model(
        self, version_id: str, environment: str = "production"
    ) -> str | None:
        """Deploy a model version"""
        try:
            # Get model version
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT * FROM model_versions WHERE version_id = ?
            """,
                (version_id,),
            )

            row = cursor.fetchone()
            if not row:
                return None

            # Create deployment record
            deployment_id = (
                f"DEPLOY_{version_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )

            cursor.execute(
                """
                INSERT INTO model_deployments 
                (deployment_id, version_id, environment, deployed_at, status, performance_snapshot)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    deployment_id,
                    version_id,
                    environment,
                    datetime.now().isoformat(),
                    "DEPLOYED",
                    row[5],  # performance_metrics
                ),
            )

            conn.commit()
            conn.close()

            # Update Redis
            self.redis_client.set(f"deployed_model:{environment}", version_id, ex=86400)

            print(f"✅ Deployed model {version_id} to {environment}")
            return deployment_id

        except Exception as e:
            print(f"Error deploying model: {e}")
            return None

    async def compare_versions(
        self, version1_id: str, version2_id: str
    ) -> dict[str, Any]:
        """Compare two model versions"""
        try:
            versions = await self.get_model_versions()
            v1 = next((v for v in versions if v.version_id == version1_id), None)
            v2 = next((v for v in versions if v.version_id == version2_id), None)

            if not v1 or not v2:
                return {}

            comparison = {
                "version1": v1.to_dict(),
                "version2": v2.to_dict(),
                "differences": {},
            }

            # Compare performance metrics
            perf1 = v1.performance_metrics
            perf2 = v2.performance_metrics

            for metric in [
                "accuracy",
                "total_return",
                "sharpe_ratio",
                "win_rate",
            ]:
                val1 = perf1.get(metric, 0)
                val2 = perf2.get(metric, 0)
                diff = val2 - val1
                comparison["differences"][metric] = {
                    "version1": val1,
                    "version2": val2,
                    "difference": diff,
                    "improvement": diff > 0,
                }

            # Compare metadata
            comparison["differences"]["parameters"] = {
                "version1": v1.metadata.get("parameters", {}),
                "version2": v2.metadata.get("parameters", {}),
                "changed": (
                    v1.metadata.get("parameters", {})
                    != v2.metadata.get("parameters", {})
                ),
            }

            return comparison

        except Exception as e:
            print(f"Error comparing versions: {e}")
            return {}

    async def stop(self):
        """Stop the Model Versioning Service"""
        print("🛑 Stopping Model Versioning Service...")
        self.running = False


async def main():
    """Main function"""
    versioning_service = ModelVersioningService()

    try:
        await versioning_service.start()
    except KeyboardInterrupt:
        print("🛑 Received interrupt signal")
    except Exception as e:
        print(f"❌ Error in main: {e}")
    finally:
        await versioning_service.stop()


if __name__ == "__main__":
    asyncio.run(main())
