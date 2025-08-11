"""
Strategy Locker Module

Manages the current live strategy and provides access to strategy files.
Handles strategy locking, unlocking, and status tracking.
"""

import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


def get_live_strategy() -> Optional[str]:
    """Get the currently active live strategy file"""
    try:
        # Check for live strategy lock file
        lock_file = Path("strategies/live_strategy.lock")
        if lock_file.exists():
            with open(lock_file, "r") as f:
                strategy_file = f.read().strip()

            # Verify the strategy file exists
            strategy_path = Path("strategies") / strategy_file
            if strategy_path.exists():
                return strategy_file
            else:
                logger.warning(f"Live strategy file not found: {strategy_file}")
                return None

        # Fallback: look for promoted strategies
        promoted_dir = Path("strategies/promoted")
        if promoted_dir.exists():
            promoted_files = list(promoted_dir.glob("*.json"))
            if promoted_files:
                # Return the most recently modified promoted strategy
                latest_strategy = max(promoted_files, key=lambda f: f.stat().st_mtime)
                return latest_strategy.name

        # Fallback: look for any strategy in main strategies directory
        strategies_dir = Path("strategies")
        if strategies_dir.exists():
            strategy_files = list(strategies_dir.glob("*.json"))
            if strategy_files:
                # Return the most recently modified strategy
                latest_strategy = max(strategy_files, key=lambda f: f.stat().st_mtime)
                return latest_strategy.name

        return None

    except Exception as e:
        logger.error(f"❌ Error getting live strategy: {e}")
        return None


def set_live_strategy(strategy_file: str) -> bool:
    """Set the current live strategy"""
    try:
        # Verify strategy file exists
        strategy_path = Path("strategies") / strategy_file
        if not strategy_path.exists():
            logger.error(f"Strategy file not found: {strategy_file}")
            return False

        # Create lock file
        lock_file = Path("strategies/live_strategy.lock")
        lock_file.parent.mkdir(parents=True, exist_ok=True)

        with open(lock_file, "w") as f:
            f.write(strategy_file)

        logger.info(f"🔒 Live strategy set to: {strategy_file}")
        return True

    except Exception as e:
        logger.error(f"❌ Error setting live strategy: {e}")
        return False


def unlock_strategy() -> bool:
    """Remove the live strategy lock"""
    try:
        lock_file = Path("strategies/live_strategy.lock")
        if lock_file.exists():
            lock_file.unlink()
            logger.info("🔓 Live strategy lock removed")
            return True
        return False

    except Exception as e:
        logger.error(f"❌ Error unlocking strategy: {e}")
        return False


def get_strategy_info(strategy_file: str) -> Optional[Dict[str, Any]]:
    """Get detailed information about a strategy"""
    try:
        strategy_path = Path("strategies") / strategy_file
        if not strategy_path.exists():
            return None

        with open(strategy_path, "r") as f:
            strategy_data = json.load(f)

        # Get file stats
        stat = strategy_path.stat()

        return {
            "file": strategy_file,
            "name": strategy_data.get("strategy_name", strategy_file.replace(".json", "")),
            "type": strategy_data.get("strategy_type", "unknown"),
            "parameters": strategy_data.get("parameters", {}),
            "performance": strategy_data.get("performance", {}),
            "metadata": strategy_data.get("metadata", {}),
            "promoted": strategy_data.get("promoted", False),
            "created_at": strategy_data.get("created_at"),
            "modified_at": (datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()),
            "file_size": stat.st_size,
            "is_live": get_live_strategy() == strategy_file,
        }

    except Exception as e:
        logger.error(f"❌ Error getting strategy info: {e}")
        return None


def list_available_strategies() -> list:
    """List all available strategies"""
    strategies = []

    try:
        # Check main strategies directory
        strategies_dir = Path("strategies")
        if strategies_dir.exists():
            for strategy_file in strategies_dir.glob("*.json"):
                if strategy_file.name != "live_strategy.lock":
                    info = get_strategy_info(strategy_file.name)
                    if info:
                        strategies.append(info)

        # Check promoted strategies directory
        promoted_dir = Path("strategies/promoted")
        if promoted_dir.exists():
            for strategy_file in promoted_dir.glob("*.json"):
                info = get_strategy_info(f"promoted/{strategy_file.name}")
                if info:
                    strategies.append(info)

        # Check mutated strategies directory
        mutated_dir = Path("mutated_strategies")
        if mutated_dir.exists():
            for strategy_file in mutated_dir.glob("*.json"):
                info = get_strategy_info(f"mutated_strategies/{strategy_file.name}")
                if info:
                    strategies.append(info)

        return strategies

    except Exception as e:
        logger.error(f"❌ Error listing strategies: {e}")
        return []


def is_strategy_locked(strategy_file: str) -> bool:
    """Check if a strategy is currently locked as live"""
    live_strategy = get_live_strategy()
    return live_strategy == strategy_file


def get_strategy_status(strategy_file: str) -> Dict[str, Any]:
    """Get comprehensive status of a strategy"""
    try:
        info = get_strategy_info(strategy_file)
        if not info:
            return {"error": f"Strategy not found: {strategy_file}"}

        # Check if strategy is live
        is_live = is_strategy_locked(strategy_file)

        # Get performance metrics
        performance = info.get("performance", {})

        status = {
            "file": strategy_file,
            "name": info.get("name"),
            "type": info.get("type"),
            "is_live": is_live,
            "promoted": info.get("promoted", False),
            "performance": {
                "profit": performance.get("profit", 0),
                "win_rate": performance.get("win_rate", 0),
                "num_trades": performance.get("num_trades", 0),
                "max_drawdown": performance.get("max_drawdown", 0),
                "sharpe_ratio": performance.get("sharpe_ratio", 0),
            },
            "metadata": info.get("metadata", {}),
            "created_at": info.get("created_at"),
            "modified_at": info.get("modified_at"),
        }

        return status

    except Exception as e:
        logger.error(f"❌ Error getting strategy status: {e}")
        return {"error": str(e)}


def backup_strategy(strategy_file: str) -> Optional[str]:
    """Create a backup of a strategy file"""
    try:
        strategy_path = Path("strategies") / strategy_file
        if not strategy_path.exists():
            logger.error(f"Strategy file not found: {strategy_file}")
            return None

        # Create backup directory
        backup_dir = Path("strategies/backup")
        backup_dir.mkdir(parents=True, exist_ok=True)

        # Create backup filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"{strategy_file.replace('.json', '')}_backup_{timestamp}.json"
        backup_path = backup_dir / backup_filename

        # Copy file
        import shutil

        shutil.copy2(strategy_path, backup_path)

        logger.info(f"💾 Strategy backed up: {backup_filename}")
        return str(backup_path)

    except Exception as e:
        logger.error(f"❌ Error backing up strategy: {e}")
        return None
