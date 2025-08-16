"""
Risk Management Service
Manages portfolio risk and position limits
"""

import asyncio
import json
import os
import redis
from datetime import datetime
from typing import Dict, Any, List
from dotenv import load_dotenv

# Load environment variables
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))


class RiskManager:
    def __init__(self):
        """Initialize risk manager"""
        self.redis_client = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            db=int(os.getenv("REDIS_DB", 0)),
            decode_responses=True,
        )
        self.running = False

    async def start(self):
        """Start the risk manager"""
        print("ðŸš€ Starting Risk Management Service...")
        self.running = True

        # Start risk monitoring
        await self.monitor_risk()

    async def monitor_risk(self):
        """Monitor portfolio risk"""
        print("âš ï¸ Starting risk monitoring...")

        while self.running:
            try:
                # Calculate risk metrics
                risk_data = await self.calculate_risk_metrics()

                # Check for risk alerts
                alerts = await self.check_risk_alerts(risk_data)

                # Store risk data
                await self.store_risk_data(risk_data)

                # Publish alerts if any
                if alerts:
                    await self.publish_risk_alerts(alerts)

                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                print(f"âŒ Error in risk monitoring: {e}")
                await asyncio.sleep(120)

    async def calculate_risk_metrics(self) -> Dict[str, Any]:
        """Calculate risk metrics"""
        try:
            risk_data = {
                "portfolio_risk": {
                    "var_95": -0.0234,
                    "cvar_95": -0.0345,
                    "volatility": 0.156,
                    "beta": 1.12,
                    "correlation": 0.78,
                },
                "position_limits": {
                    "max_position_size": 0.15,
                    "max_daily_loss": 0.05,
                    "max_drawdown_limit": 0.20,
                    "current_exposure": 0.67,
                },
                "stress_tests": {
                    "market_crash_scenario": -0.089,
                    "flash_crash_scenario": -0.156,
                    "correlation_breakdown": -0.234,
                },
                "timestamp": datetime.now().isoformat(),
            }

            return risk_data

        except Exception as e:
            print(f"Error calculating risk metrics: {e}")
            return {}

    async def check_risk_alerts(self, risk_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for risk alerts"""
        try:
            alerts = []

            # Check VaR threshold
            if risk_data.get("portfolio_risk", {}).get("var_95", 0) < -0.05:
                alerts.append(
                    {
                        "type": "WARNING",
                        "message": "VaR threshold exceeded",
                        "severity": "HIGH",
                        "value": risk_data["portfolio_risk"]["var_95"],
                        "timestamp": datetime.now().isoformat(),
                    }
                )

            # Check volatility threshold
            if risk_data.get("portfolio_risk", {}).get("volatility", 0) > 0.20:
                alerts.append(
                    {
                        "type": "WARNING",
                        "message": "High volatility detected",
                        "severity": "MEDIUM",
                        "value": risk_data["portfolio_risk"]["volatility"],
                        "timestamp": datetime.now().isoformat(),
                    }
                )

            # Check position limits
            current_exposure = risk_data.get("position_limits", {}).get("current_exposure", 0)
            max_exposure = risk_data.get("position_limits", {}).get("max_position_size", 0.15)

            if current_exposure > max_exposure:
                alerts.append(
                    {
                        "type": "WARNING",
                        "message": "Position size limit exceeded",
                        "severity": "HIGH",
                        "value": current_exposure,
                        "timestamp": datetime.now().isoformat(),
                    }
                )

            return alerts

        except Exception as e:
            print(f"Error checking risk alerts: {e}")
            return []

    async def store_risk_data(self, data: Dict[str, Any]):
        """Store risk data in Redis"""
        try:
            self.redis_client.set("risk_data", json.dumps(data), ex=1800)  # 30 minutes TTL
        except Exception as e:
            print(f"Error storing risk data: {e}")

    async def publish_risk_alerts(self, alerts: List[Dict[str, Any]]):
        """Publish risk alerts to Redis channels"""
        try:
            self.redis_client.publish("risk_alerts", json.dumps(alerts))
        except Exception as e:
            print(f"Error publishing risk alerts: {e}")

    async def stop(self):
        """Stop the risk manager"""
        print("ðŸ›‘ Stopping Risk Management Service...")
        self.running = False


async def main():
    """Main function"""
    manager = RiskManager()

    try:
        await manager.start()
    except KeyboardInterrupt:
        print("ðŸ›‘ Received interrupt signal")
    except Exception as e:
        print(f"âŒ Error in main: {e}")
    finally:
        await manager.stop()


if __name__ == "__main__":
    asyncio.run(main())


