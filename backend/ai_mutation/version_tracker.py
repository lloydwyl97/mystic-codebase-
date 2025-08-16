"""
Version Tracker Module

Tracks strategy versions, manages version history, and provides
version comparison and rollback capabilities.
"""

import json
import logging
import sqlite3
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone

from .strategy_locker import get_live_strategy

logger = logging.getLogger(__name__)


def get_strategy_versions(strategy_name: str = None) -> List[Dict[str, Any]]:
    """Get version history for strategies"""
    try:
        conn = sqlite3.connect("simulation_trades.db")
        cursor = conn.cursor()

        if strategy_name:
            cursor.execute(
                """
                SELECT strategy_file, strategy_name, strategy_type, simulated_profit,
                       win_rate, num_trades, max_drawdown, sharpe_ratio, promoted,
                       cycle_number, created_at
                FROM ai_mutations
                WHERE strategy_name LIKE ?
                ORDER BY created_at DESC
            """,
                (f"%{strategy_name}%",),
            )
        else:
            cursor.execute(
                """
                SELECT strategy_file, strategy_name, strategy_type, simulated_profit,
                       win_rate, num_trades, max_drawdown, sharpe_ratio, promoted,
                       cycle_number, created_at
                FROM ai_mutations
                ORDER BY created_at DESC
                LIMIT 50
            """
            )

        rows = cursor.fetchall()
        conn.close()

        versions = []
        for row in rows:
            versions.append(
                {
                    "file": row[0],
                    "name": row[1],
                    "type": row[2],
                    "profit": row[3],
                    "win_rate": row[4],
                    "num_trades": row[5],
                    "max_drawdown": row[6],
                    "sharpe_ratio": row[7],
                    "promoted": bool(row[8]),
                    "cycle_number": row[9],
                    "created_at": row[10],
                    "version": _extract_version_from_filename(row[0]),
                }
            )

        return versions

    except Exception as e:
        logger.error(f"âŒ Error getting strategy versions: {e}")
        return []


def _extract_version_from_filename(filename: str) -> str:
    """Extract version number from filename"""
    try:
        # Look for version patterns like v1, v2, etc.
        if "_v" in filename:
            parts = filename.split("_v")
            if len(parts) > 1:
                version_part = parts[-1].replace(".json", "")
                return f"v{version_part}"

        # Look for version patterns like v1.0.0
        if "v" in filename and "." in filename:
            start = filename.find("v")
            end = filename.find(".json")
            if start != -1 and end != -1:
                return filename[start:end]

        return "v1.0.0"  # Default version

    except Exception:
        return "v1.0.0"


def get_version_comparison(strategy_name: str, version1: str, version2: str) -> Dict[str, Any]:
    """Compare two versions of a strategy"""
    try:
        conn = sqlite3.connect("simulation_trades.db")
        cursor = conn.cursor()

        # Get both versions
        cursor.execute(
            """
            SELECT strategy_file, simulated_profit, win_rate, num_trades,
                   max_drawdown, sharpe_ratio, promoted, created_at
            FROM ai_mutations
            WHERE strategy_name LIKE ? AND strategy_file LIKE ?
            ORDER BY created_at DESC
            LIMIT 1
        """,
            (f"%{strategy_name}%", f"%{version1}%"),
        )

        version1_data = cursor.fetchone()

        cursor.execute(
            """
            SELECT strategy_file, simulated_profit, win_rate, num_trades,
                   max_drawdown, sharpe_ratio, promoted, created_at
            FROM ai_mutations
            WHERE strategy_name LIKE ? AND strategy_file LIKE ?
            ORDER BY created_at DESC
            LIMIT 1
        """,
            (f"%{strategy_name}%", f"%{version2}%"),
        )

        version2_data = cursor.fetchone()

        conn.close()

        if not version1_data or not version2_data:
            return {"error": "One or both versions not found"}

        # Calculate differences
        comparison = {
            "version1": {
                "file": version1_data[0],
                "profit": version1_data[1],
                "win_rate": version1_data[2],
                "num_trades": version1_data[3],
                "max_drawdown": version1_data[4],
                "sharpe_ratio": version1_data[5],
                "promoted": bool(version1_data[6]),
                "created_at": version1_data[7],
            },
            "version2": {
                "file": version2_data[0],
                "profit": version2_data[1],
                "win_rate": version2_data[2],
                "num_trades": version2_data[3],
                "max_drawdown": version2_data[4],
                "sharpe_ratio": version2_data[5],
                "promoted": bool(version2_data[6]),
                "created_at": version2_data[7],
            },
            "differences": {
                "profit_change": version2_data[1] - version1_data[1],
                "win_rate_change": version2_data[2] - version1_data[2],
                "trades_change": version2_data[3] - version1_data[3],
                "drawdown_change": version2_data[4] - version1_data[4],
                "sharpe_change": version2_data[5] - version1_data[5],
            },
        }

        return comparison

    except Exception as e:
        logger.error(f"âŒ Error comparing versions: {e}")
        return {"error": str(e)}


def rollback_to_version(strategy_name: str, target_version: str) -> Dict[str, Any]:
    """Rollback to a specific version of a strategy"""
    try:
        # Find the target version file
        target_file = None
        versions = get_strategy_versions(strategy_name)

        for version in versions:
            if target_version in version["file"]:
                target_file = version["file"]
                break

        if not target_file:
            return {
                "success": False,
                "error": f"Target version {target_version} not found",
            }

        # Find the target file in the filesystem
        target_path = None
        search_paths = [
            Path("strategies") / target_file,
            Path("strategies/promoted") / target_file,
            Path("mutated_strategies") / target_file,
        ]

        for path in search_paths:
            if path.exists():
                target_path = path
                break

        if not target_path:
            return {
                "success": False,
                "error": f"Target file not found: {target_file}",
            }

        # Create backup of current live strategy
        live_strategy = get_live_strategy()
        if live_strategy:
            from .strategy_locker import backup_strategy

            backup_path = backup_strategy(live_strategy)

        # Copy target version to main strategies directory
        import shutil

        new_strategy_path = Path("strategies") / target_file
        shutil.copy2(target_path, new_strategy_path)

        # Set as live strategy
        from .strategy_locker import set_live_strategy

        set_live_strategy(target_file)

        logger.info(f"ðŸ”„ Rolled back to version: {target_version}")

        return {
            "success": True,
            "target_version": target_version,
            "target_file": target_file,
            "new_path": str(new_strategy_path),
            "backup_created": backup_path if live_strategy else None,
        }

    except Exception as e:
        logger.error(f"âŒ Error rolling back to version: {e}")
        return {"success": False, "error": str(e)}


def get_version_history(strategy_name: str) -> List[Dict[str, Any]]:
    """Get detailed version history for a specific strategy"""
    try:
        conn = sqlite3.connect("simulation_trades.db")
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT strategy_file, simulated_profit, win_rate, num_trades,
                   max_drawdown, sharpe_ratio, promoted, cycle_number, created_at
            FROM ai_mutations
            WHERE strategy_name LIKE ?
            ORDER BY created_at ASC
        """,
            (f"%{strategy_name}%",),
        )

        rows = cursor.fetchall()
        conn.close()

        history = []
        for i, row in enumerate(rows):
            history.append(
                {
                    "version_number": i + 1,
                    "file": row[0],
                    "profit": row[1],
                    "win_rate": row[2],
                    "num_trades": row[3],
                    "max_drawdown": row[4],
                    "sharpe_ratio": row[5],
                    "promoted": bool(row[6]),
                    "cycle_number": row[7],
                    "created_at": row[8],
                    "version": _extract_version_from_filename(row[0]),
                }
            )

        return history

    except Exception as e:
        logger.error(f"âŒ Error getting version history: {e}")
        return []


def get_latest_version(strategy_name: str) -> Optional[Dict[str, Any]]:
    """Get the latest version of a strategy"""
    try:
        versions = get_strategy_versions(strategy_name)
        if versions:
            return versions[0]  # First one is the latest (ordered by created_at DESC)
        return None

    except Exception as e:
        logger.error(f"âŒ Error getting latest version: {e}")
        return None


def get_promoted_versions() -> List[Dict[str, Any]]:
    """Get all promoted versions"""
    try:
        conn = sqlite3.connect("simulation_trades.db")
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT strategy_file, strategy_name, strategy_type, simulated_profit,
                   win_rate, num_trades, max_drawdown, sharpe_ratio, created_at
            FROM ai_mutations
            WHERE promoted = 1
            ORDER BY created_at DESC
        """
        )

        rows = cursor.fetchall()
        conn.close()

        promoted_versions = []
        for row in rows:
            promoted_versions.append(
                {
                    "file": row[0],
                    "name": row[1],
                    "type": row[2],
                    "profit": row[3],
                    "win_rate": row[4],
                    "num_trades": row[5],
                    "max_drawdown": row[6],
                    "sharpe_ratio": row[7],
                    "created_at": row[8],
                    "version": _extract_version_from_filename(row[0]),
                }
            )

        return promoted_versions

    except Exception as e:
        logger.error(f"âŒ Error getting promoted versions: {e}")
        return []


def create_version_tag(strategy_file: str, tag: str, description: str = "") -> bool:
    """Create a version tag for a strategy"""
    try:
        # Create version tags directory
        tags_dir = Path("strategies/version_tags")
        tags_dir.mkdir(parents=True, exist_ok=True)

        # Create tag file
        tag_file = tags_dir / f"{tag}.json"

        tag_data = {
            "tag": tag,
            "strategy_file": strategy_file,
            "description": description,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "created_by": "ai_mutation",
        }

        with open(tag_file, "w") as f:
            json.dump(tag_data, f, indent=2)

        logger.info(f"ðŸ·ï¸ Created version tag: {tag}")
        return True

    except Exception as e:
        logger.error(f"âŒ Error creating version tag: {e}")
        return False


def get_version_tags() -> List[Dict[str, Any]]:
    """Get all version tags"""
    try:
        tags_dir = Path("strategies/version_tags")
        if not tags_dir.exists():
            return []

        tags = []
        for tag_file in tags_dir.glob("*.json"):
            try:
                with open(tag_file, "r") as f:
                    tag_data = json.load(f)
                tags.append(tag_data)
            except Exception as e:
                logger.error(f"âŒ Error loading tag {tag_file}: {e}")

        return tags

    except Exception as e:
        logger.error(f"âŒ Error getting version tags: {e}")
        return []


