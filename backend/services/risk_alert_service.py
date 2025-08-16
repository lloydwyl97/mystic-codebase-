"""
Risk Alert Service for Mystic AI Trading Platform
Provides real-time risk monitoring and alerting for trading operations.
"""

import logging
import os
import requests
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone, timedelta
import sys

# Add backend to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from backend.modules.ai.persistent_cache import PersistentCache

logger = logging.getLogger(__name__)


class RiskAlertService:
    def __init__(self):
        """Initialize risk alert service with monitoring parameters"""
        self.cache = PersistentCache()

        # Risk thresholds
        self.risk_thresholds = {
            "drawdown_percentage": 10.0,  # 10% drawdown from last trade
            "volatility_spike_percentage": 5.0,  # 5% volatility spike in 5 minutes
            "max_exposure_percentage": 20.0,  # 20% max exposure per asset
            "api_delay_threshold_seconds": 30,  # 30 seconds max API delay
            "missing_data_threshold_minutes": 5  # 5 minutes max missing data
        }

        # Alert configuration
        self.alert_levels = {
            "LOW": "ðŸŸ¡",
            "MEDIUM": "ðŸŸ ",
            "HIGH": "ðŸ”´",
            "CRITICAL": "ðŸš¨"
        }

        # Discord webhook support
        self.discord_webhook_url = os.getenv("RISK_WEBHOOK_URL")
        self.discord_enabled = bool(self.discord_webhook_url)

        # Risk monitoring state
        self.last_check_time = datetime.now(timezone.utc)
        self.active_alerts = []
        self.alert_history = []

        # Portfolio tracking
        self.portfolio_exposure = {}
        self.last_trade_prices = {}

        logger.info("âœ… RiskAlertService initialized")

    def _calculate_drawdown(self, current_price: float, last_trade_price: float) -> float:
        """Calculate drawdown percentage from last trade"""
        try:
            if last_trade_price <= 0:
                return 0.0

            drawdown = ((last_trade_price - current_price) / last_trade_price) * 100
            return max(0.0, drawdown)

        except Exception as e:
            logger.error(f"Failed to calculate drawdown: {e}")
            return 0.0

    def _calculate_volatility(self, prices: List[float], window_minutes: int = 5) -> float:
        """Calculate volatility over a time window"""
        try:
            if len(prices) < 2:
                return 0.0

            # Calculate percentage changes
            changes = []
            for i in range(1, len(prices)):
                if prices[i-1] > 0:
                    change = ((prices[i] - prices[i-1]) / prices[i-1]) * 100
                    changes.append(abs(change))

            if not changes:
                return 0.0

            # Calculate average volatility
            avg_volatility = sum(changes) / len(changes)
            return avg_volatility

        except Exception as e:
            logger.error(f"Failed to calculate volatility: {e}")
            return 0.0

    def _get_recent_prices(self, symbol: str, minutes: int = 5) -> List[float]:
        """Get recent price data for volatility calculation"""
        try:
            # Get recent price signals from cache
            signals = self.cache.get_signals_by_type("PRICE_UPDATE", limit=50)

            # Filter by symbol and time
            symbol_prices = []
            cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=minutes)

            for signal in signals:
                if signal.get("symbol") == symbol:
                    signal_time = datetime.fromisoformat(signal.get("timestamp", "").replace("Z", "+00:00"))
                    if signal_time >= cutoff_time:
                        price = signal.get("metadata", {}).get("price", 0.0)
                        if price > 0:
                            symbol_prices.append(price)

            return symbol_prices

        except Exception as e:
            logger.error(f"Failed to get recent prices for {symbol}: {e}")
            return []

    def _get_portfolio_exposure(self) -> Dict[str, float]:
        """Get current portfolio exposure from cache"""
        try:
            # Get recent trade signals
            signals = self.cache.get_signals_by_type("TRADE_EXECUTED", limit=100)

            exposure = {}
            for signal in signals:
                symbol = signal.get("symbol", "")
                trade_data = signal.get("metadata", {})

                if symbol and trade_data:
                    trade_type = trade_data.get("trade_type", "")
                    amount_usd = trade_data.get("amount_usd", 0.0)

                    if trade_type == "BUY":
                        exposure[symbol] = exposure.get(symbol, 0.0) + amount_usd
                    elif trade_type == "SELL":
                        exposure[symbol] = exposure.get(symbol, 0.0) - amount_usd

            return exposure

        except Exception as e:
            logger.error(f"Failed to get portfolio exposure: {e}")
            return {}

    def _get_last_trade_prices(self) -> Dict[str, float]:
        """Get last trade prices for drawdown calculation"""
        try:
            # Get recent trade signals
            signals = self.cache.get_signals_by_type("TRADE_EXECUTED", limit=100)

            last_prices = {}
            for signal in signals:
                symbol = signal.get("symbol", "")
                trade_data = signal.get("metadata", {})

                if symbol and trade_data:
                    price = trade_data.get("price", 0.0)
                    if price > 0:
                        # Keep the most recent price for each symbol
                        if symbol not in last_prices:
                            last_prices[symbol] = price

            return last_prices

        except Exception as e:
            logger.error(f"Failed to get last trade prices: {e}")
            return {}

    def _check_drawdown_risk(self, symbol: str, current_price: float) -> Optional[Dict[str, Any]]:
        """Check for drawdown risk"""
        try:
            last_trade_price = self.last_trade_prices.get(symbol, 0.0)

            if last_trade_price <= 0:
                return None

            drawdown = self._calculate_drawdown(current_price, last_trade_price)

            if drawdown >= self.risk_thresholds["drawdown_percentage"]:
                return {
                    "risk_type": "DRAWDOWN",
                    "symbol": symbol,
                    "current_price": current_price,
                    "last_trade_price": last_trade_price,
                    "drawdown_percentage": drawdown,
                    "threshold": self.risk_thresholds["drawdown_percentage"],
                    "level": "HIGH" if drawdown >= 15.0 else "MEDIUM",
                    "message": f"Drawdown alert: {symbol} down {drawdown:.2f}% from last trade"
                }

            return None

        except Exception as e:
            logger.error(f"Failed to check drawdown risk for {symbol}: {e}")
            return None

    def _check_volatility_risk(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Check for volatility spike risk"""
        try:
            recent_prices = self._get_recent_prices(symbol, minutes=5)

            if len(recent_prices) < 2:
                return None

            volatility = self._calculate_volatility(recent_prices)

            if volatility >= self.risk_thresholds["volatility_spike_percentage"]:
                return {
                    "risk_type": "VOLATILITY_SPIKE",
                    "symbol": symbol,
                    "volatility_percentage": volatility,
                    "threshold": self.risk_thresholds["volatility_spike_percentage"],
                    "price_count": len(recent_prices),
                    "level": "HIGH" if volatility >= 10.0 else "MEDIUM",
                    "message": f"Volatility spike: {symbol} showing {volatility:.2f}% volatility"
                }

            return None

        except Exception as e:
            logger.error(f"Failed to check volatility risk for {symbol}: {e}")
            return None

    def _check_exposure_risk(self, symbol: str, current_exposure: float) -> Optional[Dict[str, Any]]:
        """Check for portfolio exposure risk"""
        try:
            # Calculate total portfolio value (simplified)
            total_exposure = sum(self.portfolio_exposure.values())

            if total_exposure <= 0:
                return None

            exposure_percentage = (current_exposure / total_exposure) * 100

            if exposure_percentage >= self.risk_thresholds["max_exposure_percentage"]:
                return {
                    "risk_type": "EXPOSURE_LIMIT",
                    "symbol": symbol,
                    "exposure_usd": current_exposure,
                    "exposure_percentage": exposure_percentage,
                    "total_portfolio": total_exposure,
                    "threshold": self.risk_thresholds["max_exposure_percentage"],
                    "level": "CRITICAL" if exposure_percentage >= 30.0 else "HIGH",
                    "message": f"Exposure limit: {symbol} at {exposure_percentage:.2f}% of portfolio"
                }

            return None

        except Exception as e:
            logger.error(f"Failed to check exposure risk for {symbol}: {e}")
            return None

    def _check_api_health(self) -> List[Dict[str, Any]]:
        """Check API health and data freshness"""
        try:
            api_alerts = []

            # Get recent data signals
            signals = self.cache.get_signals_by_type("PRICE_UPDATE", limit=10)

            if not signals:
                api_alerts.append({
                    "risk_type": "API_HEALTH",
                    "issue": "NO_RECENT_DATA",
                    "level": "CRITICAL",
                    "message": "No recent price data available - API may be down"
                })
                return api_alerts

            # Check data freshness
            latest_signal = signals[0]
            signal_time = datetime.fromisoformat(latest_signal.get("timestamp", "").replace("Z", "+00:00"))
            time_diff = (datetime.now(timezone.utc) - signal_time).total_seconds()

            if time_diff > self.risk_thresholds["api_delay_threshold_seconds"]:
                api_alerts.append({
                    "risk_type": "API_HEALTH",
                    "issue": "DATA_DELAY",
                    "delay_seconds": time_diff,
                    "threshold": self.risk_thresholds["api_delay_threshold_seconds"],
                    "level": "HIGH" if time_diff > 60 else "MEDIUM",
                    "message": f"API data delay: {time_diff:.0f}s old (threshold: {self.risk_thresholds['api_delay_threshold_seconds']}s)"
                })

            return api_alerts

        except Exception as e:
            logger.error(f"Failed to check API health: {e}")
            return [{
                "risk_type": "API_HEALTH",
                "issue": "CHECK_FAILED",
                "level": "MEDIUM",
                "message": f"API health check failed: {str(e)}"
            }]

    def _send_discord_alert(self, alert: Dict[str, Any]) -> bool:
        """Send alert to Discord webhook if configured"""
        try:
            if not self.discord_enabled or not self.discord_webhook_url:
                return False

            level_emoji = self.alert_levels.get(alert.get("level", "MEDIUM"), "ðŸŸ¡")

            embed = {
                "title": f"{level_emoji} Risk Alert: {alert.get('risk_type', 'UNKNOWN')}",
                "description": alert.get("message", "Risk alert triggered"),
                "color": {
                    "LOW": 0xFFFF00,    # Yellow
                    "MEDIUM": 0xFFA500,  # Orange
                    "HIGH": 0xFF0000,    # Red
                    "CRITICAL": 0x8B0000  # Dark Red
                }.get(alert.get("level", "MEDIUM"), 0xFFA500),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "fields": []
            }

            # Add relevant fields
            for key, value in alert.items():
                if key not in ["risk_type", "level", "message"] and value is not None:
                    embed["fields"].append({
                        "name": key.replace("_", " ").title(),
                        "value": str(value),
                        "inline": True
                    })

            payload = {
                "embeds": [embed]
            }

            response = requests.post(self.discord_webhook_url, json=payload, timeout=10)
            return response.status_code == 204

        except Exception as e:
            logger.error(f"Failed to send Discord alert: {e}")
            return False

    def _store_alert(self, alert: Dict[str, Any]) -> None:
        """Store alert in cache"""
        try:
            alert_id = f"risk_alert_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

            self.cache.store_signal(
                signal_id=alert_id,
                symbol=alert.get("symbol", "RISK_ALERT"),
                signal_type="RISK_ALERT",
                confidence=1.0,
                strategy="risk_monitoring",
                metadata={
                    **alert,
                    "alert_id": alert_id,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            )

            # Add to active alerts
            self.active_alerts.append(alert)
            self.alert_history.append(alert)

            # Keep only recent alerts
            if len(self.active_alerts) > 50:
                self.active_alerts.pop(0)
            if len(self.alert_history) > 1000:
                self.alert_history.pop(0)

        except Exception as e:
            logger.error(f"Failed to store alert: {e}")

    def _print_alert(self, alert: Dict[str, Any]) -> None:
        """Print alert to console"""
        try:
            level_emoji = self.alert_levels.get(alert.get("level", "MEDIUM"), "ðŸŸ¡")
            level = alert.get("level", "MEDIUM")

            print(f"{level_emoji} [{level}] {alert.get('message', 'Risk alert')}")

            # Print additional details for high/critical alerts
            if level in ["HIGH", "CRITICAL"]:
                for key, value in alert.items():
                    if key not in ["level", "message"] and value is not None:
                        print(f"  {key.replace('_', ' ').title()}: {value}")
                print()

        except Exception as e:
            logger.error(f"Failed to print alert: {e}")

    def check_risks(self) -> Dict[str, Any]:
        """Check all risk conditions and trigger alerts"""
        try:
            logger.info("ðŸ” Checking risk conditions...")

            # Update portfolio exposure and last trade prices
            self.portfolio_exposure = self._get_portfolio_exposure()
            self.last_trade_prices = self._get_last_trade_prices()

            all_alerts = []

            # Check API health
            api_alerts = self._check_api_health()
            all_alerts.extend(api_alerts)

            # Check risks for each symbol with exposure
            for symbol, exposure in self.portfolio_exposure.items():
                if exposure <= 0:
                    continue

                # Get current price
                current_price = 0.0
                recent_signals = self.cache.get_signals_by_type("PRICE_UPDATE", limit=5)
                for signal in recent_signals:
                    if signal.get("symbol") == symbol:
                        current_price = signal.get("metadata", {}).get("price", 0.0)
                        break

                if current_price <= 0:
                    continue

                # Check drawdown risk
                drawdown_alert = self._check_drawdown_risk(symbol, current_price)
                if drawdown_alert:
                    all_alerts.append(drawdown_alert)

                # Check volatility risk
                volatility_alert = self._check_volatility_risk(symbol)
                if volatility_alert:
                    all_alerts.append(volatility_alert)

                # Check exposure risk
                exposure_alert = self._check_exposure_risk(symbol, exposure)
                if exposure_alert:
                    all_alerts.append(exposure_alert)

            # Process alerts
            for alert in all_alerts:
                # Store alert
                self._store_alert(alert)

                # Print alert
                self._print_alert(alert)

                # Send Discord alert if enabled
                if self.discord_enabled:
                    self._send_discord_alert(alert)

            # Update last check time
            self.last_check_time = datetime.now(timezone.utc)

            result = {
                "timestamp": self.last_check_time.isoformat(),
                "alerts_generated": len(all_alerts),
                "active_alerts": len(self.active_alerts),
                "portfolio_exposure": self.portfolio_exposure,
                "risk_levels": {
                    level: len([a for a in all_alerts if a.get("level") == level])
                    for level in self.alert_levels.keys()
                }
            }

            logger.info(f"âœ… Risk check complete: {len(all_alerts)} alerts generated")
            return result

        except Exception as e:
            logger.error(f"Failed to check risks: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

    def get_latest_alerts(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get latest risk alerts"""
        try:
            # Get recent alerts from cache
            signals = self.cache.get_signals_by_type("RISK_ALERT", limit=limit)

            alerts = []
            for signal in signals:
                alert_data = signal.get("metadata", {})
                if alert_data:
                    alerts.append({
                        "alert_id": alert_data.get("alert_id"),
                        "timestamp": signal.get("timestamp"),
                        "symbol": signal.get("symbol"),
                        "risk_type": alert_data.get("risk_type"),
                        "level": alert_data.get("level"),
                        "message": alert_data.get("message"),
                        "details": {k: v for k, v in alert_data.items()
                                  if k not in ["alert_id", "risk_type", "level", "message", "timestamp"]}
                    })

            return alerts

        except Exception as e:
            logger.error(f"Failed to get latest alerts: {e}")
            return []

    def get_risk_status(self) -> Dict[str, Any]:
        """Get current risk service status"""
        try:
            return {
                "service": "RiskAlertService",
                "status": "active",
                "last_check": self.last_check_time.isoformat(),
                "active_alerts": len(self.active_alerts),
                "total_alerts": len(self.alert_history),
                "discord_enabled": self.discord_enabled,
                "risk_thresholds": self.risk_thresholds,
                "portfolio_exposure": self.portfolio_exposure,
                "alert_levels": self.alert_levels,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            logger.error(f"Failed to get risk status: {e}")
            return {"success": False, "error": str(e)}


# Global risk alert service instance
risk_alert_service = RiskAlertService()


def get_risk_alert_service() -> RiskAlertService:
    """Get the global risk alert service instance"""
    return risk_alert_service


if __name__ == "__main__":
    # Test the risk alert service
    service = RiskAlertService()
    print(f"âœ… RiskAlertService initialized: {service}")

    # Test risk check
    result = service.check_risks()
    print(f"Risk check result: {result}")

    # Test latest alerts
    alerts = service.get_latest_alerts()
    print(f"Latest alerts: {alerts}")

    # Test status
    status = service.get_risk_status()
    print(f"Service status: {status['status']}")


