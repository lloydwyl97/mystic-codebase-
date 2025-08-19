# strat_versions.py
"""
Strategy Versioning System for AI Trading
Manages strategy versions, saves/loads configurations, and tracks evolution.
Built for Windows 11 Home + PowerShell + Docker.
"""

import json
import os
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import hashlib
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Directory for storing strategy versions
STRATEGY_VERSIONS_DIR = "strategy_versions"
os.makedirs(STRATEGY_VERSIONS_DIR, exist_ok=True)


def generate_version_id(config: Dict[str, Any], strategy_type: str) -> str:
    """
    Generate a unique version ID for a strategy configuration.

    Args:
        config: Strategy configuration
        strategy_type: Type of strategy

    Returns:
        Unique version ID
    """
    # Create a hash of the configuration
    config_str = json.dumps(config, sort_keys=True)
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]

    # Add timestamp for uniqueness
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    return f"{strategy_type}_v{timestamp}_{config_hash}"


def save_strategy_version(
    config: Dict[str, Any],
    strategy_type: str,
    performance: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Save a strategy version with configuration and performance data.

    Args:
        config: Strategy configuration
        strategy_type: Type of strategy
        performance: Performance metrics (optional)
        metadata: Additional metadata (optional)

    Returns:
        Version ID of saved strategy
    """
    try:
        version_id = generate_version_id(config, strategy_type)

        version_data = {
            "version_id": version_id,
            "strategy_type": strategy_type,
            "config": config,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "performance": performance or {},
            "metadata": metadata or {},
        }

        # Save to file
        filename = os.path.join(STRATEGY_VERSIONS_DIR, f"{version_id}.json")
        with open(filename, "w") as f:
            json.dump(version_data, f, indent=2)

        logger.info(f"Saved strategy version: {version_id}")
        return version_id

    except Exception as e:
        logger.error(f"Error saving strategy version: {e}")
        return ""


def load_strategy_version(version_id: str) -> Optional[Dict[str, Any]]:
    """
    Load a strategy version by version ID.

    Args:
        version_id: Version ID to load

    Returns:
        Strategy version data or None if not found
    """
    try:
        filename = os.path.join(STRATEGY_VERSIONS_DIR, f"{version_id}.json")

        if not os.path.exists(filename):
            logger.warning(f"Strategy version not found: {version_id}")
            return None

        with open(filename, "r") as f:
            version_data = json.load(f)

        logger.info(f"Loaded strategy version: {version_id}")
        return version_data

    except Exception as e:
        logger.error(f"Error loading strategy version {version_id}: {e}")
        return None


def list_strategy_versions(
    strategy_type: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    List all available strategy versions.

    Args:
        strategy_type: Filter by strategy type (optional)

    Returns:
        List of strategy version summaries
    """
    try:
        versions = []

        if not os.path.exists(STRATEGY_VERSIONS_DIR):
            return versions

        for filename in os.listdir(STRATEGY_VERSIONS_DIR):
            if filename.endswith(".json"):
                try:
                    filepath = os.path.join(STRATEGY_VERSIONS_DIR, filename)
                    with open(filepath, "r") as f:
                        version_data = json.load(f)

                    # Filter by strategy type if specified
                    if strategy_type and version_data.get("strategy_type") != strategy_type:
                        continue

                    # Create summary
                    summary = {
                        "version_id": version_data.get("version_id"),
                        "strategy_type": version_data.get("strategy_type"),
                        "created_at": version_data.get("created_at"),
                        "performance": version_data.get("performance", {}),
                        "config_summary": {
                            k: v
                            for k, v in version_data.get("config", {}).items()
                            if isinstance(v, (int, float, str))
                        },
                    }
                    versions.append(summary)

                except Exception as e:
                    logger.warning(f"Error reading version file {filename}: {e}")
                    continue

        # Sort by creation date (newest first)
        versions.sort(key=lambda x: x.get("created_at", ""), reverse=True)

        return versions

    except Exception as e:
        logger.error(f"Error listing strategy versions: {e}")
        return []


def get_best_performing_version(
    strategy_type: str, metric: str = "total_profit"
) -> Optional[Dict[str, Any]]:
    """
    Get the best performing version of a strategy type.

    Args:
        strategy_type: Type of strategy
        metric: Performance metric to optimize for

    Returns:
        Best performing strategy version or None
    """
    try:
        versions = list_strategy_versions(strategy_type)

        if not versions:
            return None

        # Filter versions with performance data
        versions_with_performance = [
            v for v in versions if v.get("performance") and metric in v["performance"]
        ]

        if not versions_with_performance:
            return None

        # Find best performing version
        best_version = max(
            versions_with_performance,
            key=lambda x: x["performance"].get(metric, float("-inf")),
        )

        # Load full version data
        return load_strategy_version(best_version["version_id"])

    except Exception as e:
        logger.error(f"Error getting best performing version: {e}")
        return None


def delete_strategy_version(version_id: str) -> bool:
    """
    Delete a strategy version.

    Args:
        version_id: Version ID to delete

    Returns:
        True if deleted successfully, False otherwise
    """
    try:
        filename = os.path.join(STRATEGY_VERSIONS_DIR, f"{version_id}.json")

        if not os.path.exists(filename):
            logger.warning(f"Strategy version not found for deletion: {version_id}")
            return False

        os.remove(filename)
        logger.info(f"Deleted strategy version: {version_id}")
        return True

    except Exception as e:
        logger.error(f"Error deleting strategy version {version_id}: {e}")
        return False


def compare_versions(version_id1: str, version_id2: str) -> Dict[str, Any]:
    """
    Compare two strategy versions.

    Args:
        version_id1: First version ID
        version_id2: Second version ID

    Returns:
        Comparison results
    """
    try:
        version1 = load_strategy_version(version_id1)
        version2 = load_strategy_version(version_id2)

        if not version1 or not version2:
            return {"error": "One or both versions not found"}

        comparison = {
            "version1": {
                "version_id": version1.get("version_id"),
                "strategy_type": version1.get("strategy_type"),
                "created_at": version1.get("created_at"),
                "performance": version1.get("performance", {}),
                "config": version1.get("config", {}),
            },
            "version2": {
                "version_id": version2.get("version_id"),
                "strategy_type": version2.get("strategy_type"),
                "created_at": version2.get("created_at"),
                "performance": version2.get("performance", {}),
                "config": version2.get("config", {}),
            },
            "differences": {
                "config_differences": {},
                "performance_differences": {},
            },
        }

        # Compare configurations
        config1 = version1.get("config", {})
        config2 = version2.get("config", {})

        all_keys = set(config1.keys()) | set(config2.keys())
        for key in all_keys:
            if config1.get(key) != config2.get(key):
                comparison["differences"]["config_differences"][key] = {
                    "version1": config1.get(key),
                    "version2": config2.get(key),
                }

        # Compare performance
        perf1 = version1.get("performance", {})
        perf2 = version2.get("performance", {})

        all_perf_keys = set(perf1.keys()) | set(perf2.keys())
        for key in all_perf_keys:
            if perf1.get(key) != perf2.get(key):
                comparison["differences"]["performance_differences"][key] = {
                    "version1": perf1.get(key),
                    "version2": perf2.get(key),
                }

        return comparison

    except Exception as e:
        logger.error(f"Error comparing versions: {e}")
        return {"error": str(e)}


def export_strategy_version(version_id: str, export_path: str) -> bool:
    """
    Export a strategy version to a specific path.

    Args:
        version_id: Version ID to export
        export_path: Path to export to

    Returns:
        True if exported successfully, False otherwise
    """
    try:
        version_data = load_strategy_version(version_id)

        if not version_data:
            return False

        # Ensure export directory exists
        os.makedirs(os.path.dirname(export_path), exist_ok=True)

        # Export to specified path
        with open(export_path, "w") as f:
            json.dump(version_data, f, indent=2)

        logger.info(f"Exported strategy version {version_id} to {export_path}")
        return True

    except Exception as e:
        logger.error(f"Error exporting strategy version: {e}")
        return False


def import_strategy_version(import_path: str) -> str:
    """
    Import a strategy version from a file.

    Args:
        import_path: Path to import from

    Returns:
        Version ID of imported strategy
    """
    try:
        with open(import_path, "r") as f:
            version_data = json.load(f)

        # Validate required fields
        required_fields = ["strategy_type", "config"]
        for field in required_fields:
            if field not in version_data:
                raise ValueError(f"Missing required field: {field}")

        # Generate new version ID if not present
        if "version_id" not in version_data:
            version_data["version_id"] = generate_version_id(
                version_data["config"], version_data["strategy_type"]
            )

        # Save to versions directory
        return save_strategy_version(
            version_data["config"],
            version_data["strategy_type"],
            version_data.get("performance"),
            version_data.get("metadata"),
        )

    except Exception as e:
        logger.error(f"Error importing strategy version: {e}")
        return ""


# Convenience functions for hyperparameter tuner
def save_optimized_config(
    config: Dict[str, Any], strategy_type: str, performance: Dict[str, Any]
) -> str:
    """
    Save an optimized configuration with performance data.

    Args:
        config: Optimized configuration
        strategy_type: Strategy type
        performance: Performance metrics

    Returns:
        Version ID
    """
    metadata = {
        "source": "hyperparameter_optimization",
        "optimization_method": "auto_tuned",
    }

    return save_strategy_version(config, strategy_type, performance, metadata)


def load_latest_version(strategy_type: str) -> Optional[Dict[str, Any]]:
    """
    Load the latest version of a strategy type.

    Args:
        strategy_type: Strategy type

    Returns:
        Latest strategy version or None
    """
    versions = list_strategy_versions(strategy_type)

    if not versions:
        return None

    # Return the most recent version
    latest_version_id = versions[0]["version_id"]
    return load_strategy_version(latest_version_id)


if __name__ == "__main__":
    # Test the module
    print("Testing Strategy Versioning System...")

    # Test configuration
    test_config = {
        "rsi_period": 14,
        "ema_fast": 12,
        "ema_slow": 26,
        "stop_loss_pct": 0.02,
    }

    # Save test version
    version_id = save_strategy_version(
        test_config,
        "rsi_ema_breakout",
        {"total_profit": 150.0, "win_rate": 0.65},
    )

    print(f"Saved test version: {version_id}")

    # Load test version
    loaded_version = load_strategy_version(version_id)
    print(f"Loaded version: {loaded_version is not None}")

    # List versions
    versions = list_strategy_versions("rsi_ema_breakout")
    print(f"Found {len(versions)} versions")

    print("Strategy versioning system test completed!")

