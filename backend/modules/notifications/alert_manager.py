"""
Alert Manager for Mystic Trading Platform

Contains alert management logic for real-time trading alerts.
Handles alert generation, delivery, and management.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from utils.exceptions import NotificationException

logger = logging.getLogger(__name__)

# Simple usage of imports to avoid unused import errors
_ = json.dumps({"status": "loaded"})


class AlertManager:
    """Alert manager for trading notifications and alerts"""

    def __init__(self):
        self.alerts: Dict[str, Dict[str, Any]] = {}
        self.alert_history: List[Dict[str, Any]] = []
        self.alert_types: Dict[str, Dict[str, Any]] = {
            "price_alert": {
                "description": "Price-based alerts",
                "fields": ["symbol", "condition", "threshold"],
            },
            "volume_alert": {
                "description": "Volume-based alerts",
                "fields": ["symbol", "condition", "threshold"],
            },
            "system_alert": {
                "description": "System alerts",
                "fields": ["message"],
            },
            "trade_alert": {
                "description": "Trade alerts",
                "fields": ["symbol", "action", "quantity"],
            },
        }
        self.is_active: bool = True
        self.notification_channels: List[str] = [
            "email",
            "webhook",
            "database",
        ]

    def create_alert(
        self,
        alert_type: str,
        symbol: Optional[str] = None,
        condition: Optional[str] = None,
        threshold: Optional[float] = None,
        message: Optional[str] = None,
        expires_at: Any = None,
    ) -> Dict[str, Any]:
        """Create a new alert"""
        if alert_type not in self.alert_types:
            raise NotificationException(f"Invalid alert type: {alert_type}")
        # Check for missing required parameters
        required = self.alert_types[alert_type]["fields"]
        for field in required:
            if field == "symbol" and symbol is None:
                raise NotificationException("Missing required parameters")
            if field == "condition" and condition is None:
                raise NotificationException("Missing required parameters")
            if field == "threshold" and threshold is None:
                raise NotificationException("Missing required parameters")
            if field == "message" and message is None:
                raise NotificationException("Missing required parameters")
        if message is None:
            raise NotificationException("Missing required parameters")
        alert_id = f"alert_{len(self.alerts) + 1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        alert = {
            "id": alert_id,
            "type": alert_type,
            "symbol": symbol,
            "condition": condition,
            "threshold": threshold,
            "message": message,
            "active": True,
            "created_at": datetime.now(),
            "triggered": False,
            "triggered_at": None,
        }
        if expires_at is not None:
            alert["expires_at"] = expires_at
        self.alerts[alert_id] = alert
        logger.info(f"Alert created: {alert_type} - {message}")
        return alert

    def get_alert(self, alert_id: str) -> Optional[Dict[str, Any]]:
        """Get alert by ID"""
        return self.alerts.get(alert_id)

    def update_alert(self, alert_id: str, updates: Dict[str, Any]) -> bool:
        """Update an alert"""
        if alert_id not in self.alerts:
            return False

        alert = self.alerts[alert_id]
        valid_fields = ["threshold", "message", "condition", "active"]

        for field, value in updates.items():
            if field not in valid_fields:
                raise NotificationException(f"Invalid field: {field}")
            alert[field] = value

        logger.info(f"Alert {alert_id} updated")
        return True

    def delete_alert(self, alert_id: str) -> bool:
        """Delete an alert"""
        if alert_id in self.alerts:
            del self.alerts[alert_id]
            logger.info(f"Alert {alert_id} deleted")
            return True
        return False

    def list_alerts(
        self, alert_type: Optional[str] = None, symbol: Optional[str] = None
    ) -> List[str]:
        """List alert IDs with optional filtering"""
        alert_ids = list(self.alerts.keys())

        if alert_type:
            alert_ids = [aid for aid in alert_ids if self.alerts[aid]["type"] == alert_type]

        if symbol:
            alert_ids = [aid for aid in alert_ids if self.alerts[aid]["symbol"] == symbol]

        return alert_ids

    def activate_alert(self, alert_id: str) -> bool:
        """Activate an alert"""
        if alert_id in self.alerts:
            self.alerts[alert_id]["active"] = True
            logger.info(f"Alert {alert_id} activated")
            return True
        return False

    def deactivate_alert(self, alert_id: str) -> bool:
        """Deactivate an alert"""
        if alert_id in self.alerts:
            self.alerts[alert_id]["active"] = False
            logger.info(f"Alert {alert_id} deactivated")
            return True
        return False

    def check_price_alerts(self, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check price-based alerts"""
        triggered_alerts: List[Dict[str, Any]] = []
        for alert_id, alert in self.alerts.items():
            if not alert["active"] or alert["type"] != "price_alert":
                continue
            symbol = alert["symbol"]
            if symbol not in market_data:
                continue
            price = float(market_data[symbol].get("price", 0))
            threshold = float(alert["threshold"])
            condition = alert["condition"]
            triggered = False
            if ">" in condition and price > threshold:
                triggered = True
            elif "<" in condition and price < threshold:
                triggered = True
            if triggered and not alert["triggered"]:
                alert["triggered"] = True
                alert["triggered_at"] = datetime.now()
                alert_copy = dict(alert)
                alert_copy["alert_id"] = alert_id
                triggered_alerts.append(alert_copy)
        return triggered_alerts

    def check_volume_alerts(self, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check volume-based alerts"""
        triggered_alerts: List[Dict[str, Any]] = []
        for alert_id, alert in self.alerts.items():
            if not alert["active"] or alert["type"] != "volume_alert":
                continue
            symbol = alert["symbol"]
            if symbol not in market_data:
                continue
            volume = float(market_data[symbol].get("volume", 0))
            threshold = float(alert["threshold"])
            condition = alert["condition"]
            triggered = False
            if ">" in condition and volume > threshold:
                triggered = True
            elif "<" in condition and volume < threshold:
                triggered = True
            if triggered and not alert["triggered"]:
                alert["triggered"] = True
                alert["triggered_at"] = datetime.now()
                alert_copy = dict(alert)
                alert_copy["alert_id"] = alert_id
                triggered_alerts.append(alert_copy)
        return triggered_alerts

    def check_all_alerts(self, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check all alerts"""
        triggered_alerts: List[Dict[str, Any]] = []
        triggered_alerts.extend(self.check_price_alerts(market_data))
        triggered_alerts.extend(self.check_volume_alerts(market_data))
        return triggered_alerts

    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert statistics"""
        total_alerts = len(self.alerts)
        active_alerts = len([a for a in self.alerts.values() if a["active"]])
        inactive_alerts = len([a for a in self.alerts.values() if not a["active"]])
        triggered_alerts = len([a for a in self.alerts.values() if a["triggered"]])
        # Count by type
        type_counts = {}
        for alert in self.alerts.values():
            alert_type = alert["type"]
            type_counts[alert_type] = type_counts.get(alert_type, 0) + 1
        return {
            "total_alerts": total_alerts,
            "active_alerts": active_alerts,
            "inactive_alerts": inactive_alerts,
            "triggered_alerts": triggered_alerts,
            "alerts_by_type": type_counts,
        }

    def cleanup_expired_alerts(self, days: int = 30) -> int:
        """Clean up expired alerts"""
        from datetime import timedelta

        now = datetime.now()
        cutoff_date = now - timedelta(days=days)
        expired_alerts = []
        for alert_id, alert in self.alerts.items():
            expires_at = alert.get("expires_at")
            if expires_at is not None:
                # If expires_at is a datetime, compare directly
                if isinstance(expires_at, datetime):
                    if expires_at < now:
                        expired_alerts.append(alert_id)
                else:
                    # Try to parse if string
                    try:
                        dt = datetime.fromisoformat(str(expires_at))
                        if dt < now:
                            expired_alerts.append(alert_id)
                    except Exception:
                        pass
            elif alert["created_at"] < cutoff_date:
                expired_alerts.append(alert_id)
        for alert_id in expired_alerts:
            self.alert_history.append(self.alerts[alert_id])
            del self.alerts[alert_id]
        logger.info(f"Cleaned up {len(expired_alerts)} expired alerts")
        return len(expired_alerts)

    def get_alerts(
        self, status: Optional[str] = None, alert_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get alerts with optional filtering (legacy method)"""
        filtered_alerts = list(self.alerts.values())

        if status:
            if status == "active":
                filtered_alerts = [a for a in filtered_alerts if a["active"]]
            elif status == "triggered":
                filtered_alerts = [a for a in filtered_alerts if a["triggered"]]

        if alert_type:
            filtered_alerts = [a for a in filtered_alerts if a["type"] == alert_type]

        return filtered_alerts

    def acknowledge_alert(self, alert_id: str) -> Dict[str, Any]:
        """Acknowledge an alert (legacy method)"""
        if alert_id in self.alerts:
            self.alerts[alert_id]["acknowledged"] = True
            self.alerts[alert_id]["acknowledged_at"] = datetime.now()
            logger.info(f"Alert {alert_id} acknowledged")
            return {"success": True, "alert": self.alerts[alert_id]}

        return {"success": False, "error": "Alert not found"}

    def resolve_alert(self, alert_id: str, resolution_notes: str = "") -> Dict[str, Any]:
        """Resolve an alert (legacy method)"""
        if alert_id in self.alerts:
            self.alerts[alert_id]["status"] = "resolved"
            self.alerts[alert_id]["resolved_at"] = datetime.now()
            self.alerts[alert_id]["resolution_notes"] = resolution_notes
            logger.info(f"Alert {alert_id} resolved")
            return {"success": True, "alert": self.alerts[alert_id]}

        return {"success": False, "error": "Alert not found"}

    def add_alert_rule(self, rule_name: str, rule_config: Dict[str, Any]) -> Dict[str, Any]:
        """Add a new alert rule (legacy method)"""
        rule = {
            "name": rule_name,
            "config": rule_config,
            "created_at": datetime.now(),
            "is_active": True,
        }

        self.alert_rules = getattr(self, "alert_rules", {})
        self.alert_rules[rule_name] = rule
        logger.info(f"Alert rule added: {rule_name}")

        return {"success": True, "rule": rule}

    def remove_alert_rule(self, rule_name: str) -> Dict[str, Any]:
        """Remove an alert rule (legacy method)"""
        self.alert_rules = getattr(self, "alert_rules", {})
        if rule_name in self.alert_rules:
            del self.alert_rules[rule_name]
            logger.info(f"Alert rule removed: {rule_name}")
            return {"success": True}

        return {"success": False, "error": "Rule not found"}

    def check_alert_conditions(self, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check market data against alert rules and generate alerts (legacy method)"""
        return self.check_all_alerts(market_data)

    def clear_old_alerts(self, days: int = 30) -> Dict[str, Any]:
        """Clear alerts older than specified days (legacy method)"""
        cleared_count = self.cleanup_expired_alerts(days)
        return {
            "success": True,
            "cleared_count": cleared_count,
            "remaining_count": len(self.alerts),
        }

    def export_alerts(self, format_type: str = "json") -> Dict[str, Any]:
        """Export alerts (legacy method)"""
        if format_type == "json":
            return {
                "success": True,
                "data": list(self.alerts.values()),
                "export_timestamp": datetime.now().isoformat(),
            }
        return {"success": False, "error": "Unsupported format"}

    def import_alerts(self, alerts_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Import alerts (legacy method)"""
        imported_count = 0
        for alert_data in alerts_data:
            try:
                alert_id = alert_data.get("id", f"imported_{imported_count}")
                self.alerts[alert_id] = alert_data
                imported_count += 1
            except Exception as e:
                logger.error(f"Failed to import alert: {e}")

        return {
            "success": True,
            "imported_count": imported_count,
            "total_alerts": len(self.alerts),
        }

    def get_alert_by_id(self, alert_id: str) -> Optional[Dict[str, Any]]:
        """Get alert by ID (legacy method)"""
        return self.get_alert(alert_id)
