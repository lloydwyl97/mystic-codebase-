"""
Strategy Promotion Module

Handles the promotion of strategies from testing to live trading,
including validation, risk assessment, and deployment.
"""

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class StrategyPromoter:
    """Handles strategy promotion and deployment"""

    def __init__(self):
        self.promoted_strategies_dir = Path("strategies/promoted")
        self.testing_strategies_dir = Path("strategies/testing")
        self.backup_strategies_dir = Path("strategies/backup")

        # Ensure directories exist
        self.promoted_strategies_dir.mkdir(parents=True, exist_ok=True)
        self.testing_strategies_dir.mkdir(parents=True, exist_ok=True)
        self.backup_strategies_dir.mkdir(parents=True, exist_ok=True)

    def promote_strategy(self, strategy_file: str, justification: str = "") -> Dict[str, Any]:
        """Promote a strategy from testing to live"""
        try:
            # Find strategy file
            strategy_path = self._find_strategy_file(strategy_file)
            if not strategy_path:
                return {
                    "success": False,
                    "error": f"Strategy file not found: {strategy_file}",
                }

            # Load strategy data
            with open(strategy_path, "r") as f:
                strategy_data = json.load(f)

            # Validate strategy
            validation_result = self._validate_strategy(strategy_data)
            if not validation_result["valid"]:
                return {
                    "success": False,
                    "error": (f"Strategy validation failed: {validation_result['error']}"),
                }

            # Create backup
            backup_path = (
                self.backup_strategies_dir
                / f"{strategy_file}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            shutil.copy2(strategy_path, backup_path)

            # Move to promoted directory
            promoted_path = self.promoted_strategies_dir / strategy_file
            shutil.move(strategy_path, promoted_path)

            # Update strategy metadata
            strategy_data["promoted"] = True
            strategy_data["promoted_at"] = datetime.now(timezone.utc).isoformat()
            strategy_data["promoted_by"] = "ai_mutation"
            strategy_data["promotion_justification"] = justification

            with open(promoted_path, "w") as f:
                json.dump(strategy_data, f, indent=2)

            logger.info(f"ðŸŽ‰ Strategy promoted: {strategy_file}")

            return {
                "success": True,
                "strategy_file": strategy_file,
                "promoted_path": str(promoted_path),
                "backup_path": str(backup_path),
                "promoted_at": strategy_data["promoted_at"],
            }

        except Exception as e:
            logger.error(f"âŒ Error promoting strategy: {e}")
            return {"success": False, "error": str(e)}

    def demote_strategy(self, strategy_file: str, reason: str = "") -> Dict[str, Any]:
        """Demote a strategy from live to testing"""
        try:
            promoted_path = self.promoted_strategies_dir / strategy_file
            if not promoted_path.exists():
                return {
                    "success": False,
                    "error": f"Promoted strategy not found: {strategy_file}",
                }

            # Load strategy data
            with open(promoted_path, "r") as f:
                strategy_data = json.load(f)

            # Update metadata
            strategy_data["promoted"] = False
            strategy_data["demoted_at"] = datetime.now(timezone.utc).isoformat()
            strategy_data["demoted_by"] = "ai_mutation"
            strategy_data["demotion_reason"] = reason

            # Move back to testing
            testing_path = self.testing_strategies_dir / strategy_file
            shutil.move(promoted_path, testing_path)

            # Update file
            with open(testing_path, "w") as f:
                json.dump(strategy_data, f, indent=2)

            logger.info(f"ðŸ“‰ Strategy demoted: {strategy_file}")

            return {
                "success": True,
                "strategy_file": strategy_file,
                "testing_path": str(testing_path),
                "demoted_at": strategy_data["demoted_at"],
            }

        except Exception as e:
            logger.error(f"âŒ Error demoting strategy: {e}")
            return {"success": False, "error": str(e)}

    def get_promoted_strategies(self) -> List[Dict[str, Any]]:
        """Get list of promoted strategies"""
        strategies = []

        for strategy_file in self.promoted_strategies_dir.glob("*.json"):
            try:
                with open(strategy_file, "r") as f:
                    strategy_data = json.load(f)

                strategies.append(
                    {
                        "file": strategy_file.name,
                        "name": strategy_data.get("strategy_name", strategy_file.stem),
                        "type": strategy_data.get("strategy_type", "unknown"),
                        "promoted_at": strategy_data.get("promoted_at"),
                        "performance": strategy_data.get("performance", {}),
                        "metadata": strategy_data.get("metadata", {}),
                    }
                )

            except Exception as e:
                logger.error(f"âŒ Error loading promoted strategy {strategy_file}: {e}")

        return strategies

    def _find_strategy_file(self, strategy_file: str) -> Optional[Path]:
        """Find strategy file in various directories"""
        search_paths = [
            Path("strategies") / strategy_file,
            self.testing_strategies_dir / strategy_file,
            self.promoted_strategies_dir / strategy_file,
            Path("mutated_strategies") / strategy_file,
        ]

        for path in search_paths:
            if path.exists():
                return path

        return None

    def _validate_strategy(self, strategy_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate strategy before promotion"""
        try:
            # Check required fields
            required_fields = ["strategy_name", "strategy_type", "parameters"]
            for field in required_fields:
                if field not in strategy_data:
                    return {
                        "valid": False,
                        "error": f"Missing required field: {field}",
                    }

            # Check performance metrics
            performance = strategy_data.get("performance", {})
            if performance:
                profit = performance.get("profit", 0)
                win_rate = performance.get("win_rate", 0)
                sharpe = performance.get("sharpe_ratio", 0)

                # Basic validation criteria
                if profit < 0.05:  # Less than 5% profit
                    return {
                        "valid": False,
                        "error": "Insufficient profit for promotion",
                    }

                if win_rate < 0.5:  # Less than 50% win rate
                    return {
                        "valid": False,
                        "error": "Insufficient win rate for promotion",
                    }

                if sharpe < 0.5:  # Less than 0.5 Sharpe ratio
                    return {
                        "valid": False,
                        "error": "Insufficient Sharpe ratio for promotion",
                    }

            return {"valid": True}

        except Exception as e:
            return {"valid": False, "error": f"Validation error: {str(e)}"}

    def rollback_promotion(self, strategy_file: str, backup_file: str) -> Dict[str, Any]:
        """Rollback a promotion using backup file"""
        try:
            backup_path = Path(backup_file)
            if not backup_path.exists():
                return {
                    "success": False,
                    "error": f"Backup file not found: {backup_file}",
                }

            # Remove current promoted version
            promoted_path = self.promoted_strategies_dir / strategy_file
            if promoted_path.exists():
                promoted_path.unlink()

            # Restore from backup
            restored_path = self.promoted_strategies_dir / strategy_file
            shutil.copy2(backup_path, restored_path)

            logger.info(f"ðŸ”„ Strategy rollback completed: {strategy_file}")

            return {
                "success": True,
                "strategy_file": strategy_file,
                "restored_path": str(restored_path),
                "backup_used": str(backup_path),
            }

        except Exception as e:
            logger.error(f"âŒ Error rolling back promotion: {e}")
            return {"success": False, "error": str(e)}


# Global instance
strategy_promoter = StrategyPromoter()


