"""
Risk Agent
Handles risk management, position sizing, and portfolio risk monitoring
"""

import asyncio
import json
import numpy as np
import time
from datetime import datetime
from typing import Dict, Any
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.base_agent import BaseAgent


class RiskAgent(BaseAgent):
    """Risk Management Agent - Handles risk assessment and position sizing"""

    def __init__(self, agent_id: str = "risk_agent_001"):
        super().__init__(agent_id, "risk")

        # Risk-specific state
        self.state.update(
            {
                "portfolio_risk": {},
                "position_limits": {},
                "risk_metrics": {},
                "alerts": [],
                "risk_level": "medium",
                "max_portfolio_risk": 0.02,  # 2% max portfolio risk
                "max_position_size": 0.1,  # 10% max position size
                "stop_loss_pct": 0.05,  # 5% stop loss
                "take_profit_pct": 0.15,  # 15% take profit
            }
        )

        # Register risk-specific handlers
        self.register_handler("assess_risk", self.handle_assess_risk)
        self.register_handler("calculate_position_size", self.handle_calculate_position_size)
        self.register_handler("portfolio_update", self.handle_portfolio_update)
        self.register_handler("trading_signal", self.handle_trading_signal)
        self.register_handler("market_data", self.handle_market_data)

        print(f"🛡️ Risk Agent {agent_id} initialized")

    async def initialize(self):
        """Initialize risk agent resources"""
        try:
            # Load risk configuration
            await self.load_risk_config()

            # Initialize portfolio monitoring
            await self.initialize_portfolio_monitoring()

            # Subscribe to market data
            await self.subscribe_to_market_data()

            print(f"✅ Risk Agent {self.agent_id} initialized successfully")

        except Exception as e:
            print(f"❌ Error initializing Risk Agent: {e}")
            self.update_health_status("error")

    async def process_loop(self):
        """Main risk processing loop"""
        while self.running:
            try:
                # Monitor portfolio risk
                await self.monitor_portfolio_risk()

                # Check for risk alerts
                await self.check_risk_alerts()

                # Update risk metrics
                await self.update_risk_metrics()

                # Clean up old alerts
                await self.cleanup_alerts()

                await asyncio.sleep(30)  # Check every 30 seconds

            except Exception as e:
                print(f"❌ Error in risk processing loop: {e}")
                await asyncio.sleep(60)

    async def load_risk_config(self):
        """Load risk configuration from Redis"""
        try:
            # Load risk configuration
            risk_config = self.redis_client.get("risk_config")
            if risk_config:
                config = json.loads(risk_config)
                self.state.update(config)

            # Load position limits
            position_limits = self.redis_client.get("position_limits")
            if position_limits:
                self.state["position_limits"] = json.loads(position_limits)

            print("📋 Risk configuration loaded")

        except Exception as e:
            print(f"❌ Error loading risk configuration: {e}")

    async def initialize_portfolio_monitoring(self):
        """Initialize portfolio monitoring"""
        try:
            # Get current portfolio
            portfolio_data = self.redis_client.get("portfolio")
            if portfolio_data:
                portfolio = json.loads(portfolio_data)
                await self.analyze_portfolio_risk(portfolio)

            print("📊 Portfolio monitoring initialized")

        except Exception as e:
            print(f"❌ Error initializing portfolio monitoring: {e}")

    async def subscribe_to_market_data(self):
        """Subscribe to market data updates"""
        try:
            # Subscribe to market data channel
            pubsub = self.redis_client.pubsub()
            pubsub.subscribe("market_data")

            # Start market data listener
            asyncio.create_task(self.listen_market_data(pubsub))

            print("📡 Risk Agent subscribed to market data")

        except Exception as e:
            print(f"❌ Error subscribing to market data: {e}")

    async def listen_market_data(self, pubsub):
        """Listen for market data updates"""
        try:
            for message in pubsub.listen():
                if not self.running:
                    break

                if message["type"] == "message":
                    market_data = json.loads(message["data"])
                    await self.process_market_data(market_data)

        except Exception as e:
            print(f"❌ Error in market data listener: {e}")
        finally:
            pubsub.close()

    async def process_market_data(self, market_data: Dict[str, Any]):
        """Process incoming market data for risk assessment"""
        try:
            symbol = market_data.get("symbol")
            price = market_data.get("price")
            volume = market_data.get("volume", 0)

            # Update risk metrics for this symbol
            await self.update_symbol_risk(symbol, price, volume)

            # Check for risk alerts
            await self.check_symbol_risk_alerts(symbol, market_data)

        except Exception as e:
            print(f"❌ Error processing market data: {e}")

    async def handle_assess_risk(self, message: Dict[str, Any]):
        """Handle risk assessment request"""
        try:
            symbol = message.get("symbol")
            position_size = message.get("position_size", 0)
            current_price = message.get("current_price", 0)

            print(f"🛡️ Assessing risk for {symbol}")

            # Perform risk assessment
            risk_assessment = await self.assess_trade_risk(symbol, position_size, current_price)

            # Send response
            response = {
                "type": "risk_assessment",
                "symbol": symbol,
                "assessment": risk_assessment,
                "timestamp": datetime.now().isoformat(),
            }

            sender = message.get("from_agent")
            if sender:
                await self.send_message(sender, response)

        except Exception as e:
            print(f"❌ Error assessing risk: {e}")
            await self.broadcast_error(f"Risk assessment error: {e}")

    async def handle_calculate_position_size(self, message: Dict[str, Any]):
        """Handle position size calculation request"""
        try:
            symbol = message.get("symbol")
            signal_strength = message.get("signal_strength", 0.5)
            available_capital = message.get("available_capital", 0)
            current_price = message.get("current_price", 0)

            print(f"📏 Calculating position size for {symbol}")

            # Calculate optimal position size
            position_size = await self.calculate_optimal_position_size(
                symbol, signal_strength, available_capital, current_price
            )

            # Send response
            response = {
                "type": "position_size_calculation",
                "symbol": symbol,
                "position_size": position_size,
                "timestamp": datetime.now().isoformat(),
            }

            sender = message.get("from_agent")
            if sender:
                await self.send_message(sender, response)

        except Exception as e:
            print(f"❌ Error calculating position size: {e}")
            await self.broadcast_error(f"Position size calculation error: {e}")

    async def handle_portfolio_update(self, message: Dict[str, Any]):
        """Handle portfolio update"""
        try:
            portfolio = message.get("portfolio", {})

            # Analyze portfolio risk
            await self.analyze_portfolio_risk(portfolio)

            # Check for portfolio-level risk alerts
            await self.check_portfolio_risk_alerts(portfolio)

        except Exception as e:
            print(f"❌ Error handling portfolio update: {e}")

    async def handle_trading_signal(self, message: Dict[str, Any]):
        """Handle trading signal for risk validation"""
        try:
            strategy_id = message.get("strategy_id")
            signal = message.get("signal", {})
            market_data = message.get("market_data", {})

            symbol = market_data.get("symbol")
            signal.get("type")
            confidence = signal.get("confidence", 0)

            print(f"🔍 Validating trading signal for {symbol}")

            # Validate signal from risk perspective
            validation = await self.validate_trading_signal(signal, market_data)

            if validation["approved"]:
                # Calculate position size
                position_size = await self.calculate_optimal_position_size(
                    symbol,
                    confidence,
                    validation["available_capital"],
                    market_data.get("price", 0),
                )

                # Send approved signal to execution agent
                await self.send_message(
                    "execution_agent",
                    {
                        "type": "approved_trading_signal",
                        "strategy_id": strategy_id,
                        "signal": signal,
                        "market_data": market_data,
                        "position_size": position_size,
                        "risk_validation": validation,
                    },
                )
            else:
                # Send rejection to strategy agent
                await self.send_message(
                    "strategy_agent",
                    {
                        "type": "signal_rejected",
                        "strategy_id": strategy_id,
                        "reason": validation["reason"],
                        "signal": signal,
                    },
                )

        except Exception as e:
            print(f"❌ Error handling trading signal: {e}")

    async def assess_trade_risk(
        self, symbol: str, position_size: float, current_price: float
    ) -> Dict[str, Any]:
        """Assess risk for a specific trade"""
        try:
            # Get portfolio risk metrics
            portfolio_risk = self.state["portfolio_risk"]

            # Calculate trade-specific risk metrics
            trade_risk = {
                "symbol": symbol,
                "position_size": position_size,
                "current_price": current_price,
                "dollar_risk": position_size * current_price,
                "portfolio_risk_contribution": 0,
                "var_95": 0,
                "max_loss": (position_size * current_price * self.state["stop_loss_pct"]),
                "risk_score": 0,
                "recommendation": "hold",
            }

            # Calculate portfolio risk contribution
            total_portfolio_value = portfolio_risk.get("total_value", 1)
            if total_portfolio_value > 0:
                trade_risk["portfolio_risk_contribution"] = (
                    trade_risk["dollar_risk"] / total_portfolio_value
                )

            # Calculate Value at Risk (VaR)
            volatility = await self.get_symbol_volatility(symbol)
            trade_risk["var_95"] = trade_risk["dollar_risk"] * volatility * 1.645  # 95% confidence

            # Calculate risk score
            trade_risk["risk_score"] = self.calculate_trade_risk_score(trade_risk)

            # Generate recommendation
            trade_risk["recommendation"] = self.generate_risk_recommendation(trade_risk)

            return trade_risk

        except Exception as e:
            print(f"❌ Error assessing trade risk: {e}")
            return {}

    async def calculate_optimal_position_size(
        self,
        symbol: str,
        signal_strength: float,
        available_capital: float,
        current_price: float,
    ) -> Dict[str, Any]:
        """Calculate optimal position size based on risk parameters"""
        try:
            # Get risk metrics for symbol
            symbol_risk = self.state["risk_metrics"].get(symbol, {})
            volatility = symbol_risk.get("volatility", 0.2)

            # Calculate Kelly Criterion position size
            win_rate = signal_strength
            avg_win = self.state["take_profit_pct"]
            avg_loss = self.state["stop_loss_pct"]

            kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win

            # Apply risk limits
            max_position_size = min(
                kelly_fraction,
                self.state["max_position_size"],
                self.state["max_portfolio_risk"] / volatility,
            )

            # Calculate actual position size
            position_value = available_capital * max_position_size
            position_size = position_value / current_price if current_price > 0 else 0

            # Apply position limits
            symbol_limit = self.state["position_limits"].get(symbol, 1.0)
            position_size = min(position_size, available_capital * symbol_limit / current_price)

            return {
                "position_size": position_size,
                "position_value": position_value,
                "kelly_fraction": kelly_fraction,
                "max_position_size": max_position_size,
                "risk_adjusted": True,
            }

        except Exception as e:
            print(f"❌ Error calculating position size: {e}")
            return {"position_size": 0, "error": str(e)}

    async def validate_trading_signal(
        self, signal: Dict[str, Any], market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate trading signal from risk perspective"""
        try:
            symbol = market_data.get("symbol")
            signal.get("type")
            confidence = signal.get("confidence", 0)
            market_data.get("price", 0)

            validation = {
                "approved": False,
                "reason": "",
                "available_capital": 0,
                "risk_level": "high",
            }

            # Check confidence threshold
            if confidence < 0.6:
                validation["reason"] = "Signal confidence too low"
                return validation

            # Check portfolio risk limits
            portfolio_risk = self.state["portfolio_risk"]
            current_risk = portfolio_risk.get("current_risk", 0)
            max_risk = self.state["max_portfolio_risk"]

            if current_risk >= max_risk:
                validation["reason"] = "Portfolio risk limit reached"
                return validation

            # Check symbol-specific limits
            symbol_risk = self.state["risk_metrics"].get(symbol, {})
            if symbol_risk.get("risk_level", "medium") == "high":
                validation["reason"] = "Symbol risk level too high"
                return validation

            # Calculate available capital
            total_capital = portfolio_risk.get("total_value", 0)
            used_capital = portfolio_risk.get("used_capital", 0)
            validation["available_capital"] = total_capital - used_capital

            # Approve signal
            validation["approved"] = True
            validation["risk_level"] = symbol_risk.get("risk_level", "medium")

            return validation

        except Exception as e:
            print(f"❌ Error validating trading signal: {e}")
            return {"approved": False, "reason": f"Validation error: {e}"}

    async def analyze_portfolio_risk(self, portfolio: Dict[str, Any]):
        """Analyze portfolio risk metrics"""
        try:
            positions = portfolio.get("positions", [])
            total_value = portfolio.get("total_value", 0)

            # Calculate portfolio risk metrics
            portfolio_risk = {
                "total_value": total_value,
                "used_capital": sum(pos.get("value", 0) for pos in positions),
                "current_risk": 0,
                "var_95": 0,
                "max_drawdown": 0,
                "diversification_score": 0,
                "concentration_risk": 0,
            }

            # Calculate current risk
            for position in positions:
                symbol = position.get("symbol")
                value = position.get("value", 0)
                symbol_risk = self.state["risk_metrics"].get(symbol, {})
                volatility = symbol_risk.get("volatility", 0.2)

                portfolio_risk["current_risk"] += (value / total_value) * volatility

            # Calculate VaR
            portfolio_risk["var_95"] = total_value * portfolio_risk["current_risk"] * 1.645

            # Calculate diversification score
            if len(positions) > 0:
                portfolio_risk["diversification_score"] = min(1.0, len(positions) / 10)

            # Calculate concentration risk
            if total_value > 0:
                max_position_value = max((pos.get("value", 0) for pos in positions), default=0)
                portfolio_risk["concentration_risk"] = max_position_value / total_value

            # Update state
            self.state["portfolio_risk"] = portfolio_risk

            # Store in Redis
            self.redis_client.set("portfolio_risk", json.dumps(portfolio_risk), ex=300)

        except Exception as e:
            print(f"❌ Error analyzing portfolio risk: {e}")

    async def check_risk_alerts(self):
        """Check for risk alerts"""
        try:
            portfolio_risk = self.state["portfolio_risk"]

            # Check portfolio risk limits
            if portfolio_risk.get("current_risk", 0) > self.state["max_portfolio_risk"]:
                await self.create_alert(
                    "high_portfolio_risk",
                    {
                        "current_risk": portfolio_risk["current_risk"],
                        "max_risk": self.state["max_portfolio_risk"],
                    },
                )

            # Check concentration risk
            if portfolio_risk.get("concentration_risk", 0) > 0.2:  # 20% concentration limit
                await self.create_alert(
                    "high_concentration_risk",
                    {"concentration": portfolio_risk["concentration_risk"]},
                )

            # Check VaR limits
            if (
                portfolio_risk.get("var_95", 0) > portfolio_risk.get("total_value", 0) * 0.05
            ):  # 5% VaR limit
                await self.create_alert(
                    "high_var_risk",
                    {
                        "var_95": portfolio_risk["var_95"],
                        "total_value": portfolio_risk["total_value"],
                    },
                )

        except Exception as e:
            print(f"❌ Error checking risk alerts: {e}")

    async def check_symbol_risk_alerts(self, symbol: str, market_data: Dict[str, Any]):
        """Check for symbol-specific risk alerts"""
        try:
            symbol_risk = self.state["risk_metrics"].get(symbol, {})

            # Check volatility spikes
            current_volatility = symbol_risk.get("volatility", 0)
            historical_volatility = symbol_risk.get("historical_volatility", 0)

            if historical_volatility > 0 and current_volatility > historical_volatility * 2:
                await self.create_alert(
                    "volatility_spike",
                    {
                        "symbol": symbol,
                        "current_volatility": current_volatility,
                        "historical_volatility": historical_volatility,
                    },
                )

            # Check price gaps
            price = market_data.get("price", 0)
            last_price = symbol_risk.get("last_price", price)

            if last_price > 0:
                price_change = abs(price - last_price) / last_price
                if price_change > 0.1:  # 10% price gap
                    await self.create_alert(
                        "price_gap",
                        {
                            "symbol": symbol,
                            "price_change": price_change,
                            "current_price": price,
                            "last_price": last_price,
                        },
                    )

            # Update last price
            symbol_risk["last_price"] = price
            self.state["risk_metrics"][symbol] = symbol_risk

        except Exception as e:
            print(f"❌ Error checking symbol risk alerts: {e}")

    async def check_portfolio_risk_alerts(self, portfolio: Dict[str, Any]):
        """Check for portfolio-level risk alerts"""
        try:
            positions = portfolio.get("positions", [])

            # Check for large losses
            for position in positions:
                symbol = position.get("symbol")
                unrealized_pnl = position.get("unrealized_pnl", 0)
                value = position.get("value", 0)

                if value > 0 and unrealized_pnl < -value * 0.1:  # 10% loss
                    await self.create_alert(
                        "large_loss",
                        {
                            "symbol": symbol,
                            "unrealized_pnl": unrealized_pnl,
                            "value": value,
                            "loss_pct": unrealized_pnl / value,
                        },
                    )

        except Exception as e:
            print(f"❌ Error checking portfolio risk alerts: {e}")

    async def create_alert(self, alert_type: str, data: Dict[str, Any]):
        """Create a risk alert"""
        try:
            alert = {
                "id": f"alert_{int(time.time())}",
                "type": alert_type,
                "data": data,
                "timestamp": datetime.now().isoformat(),
                "severity": self.get_alert_severity(alert_type),
                "acknowledged": False,
            }

            # Add to alerts list
            self.state["alerts"].append(alert)

            # Broadcast alert
            await self.broadcast_message({"type": "risk_alert", "alert": alert})

            # Store in Redis
            self.redis_client.lpush("risk_alerts", json.dumps(alert))
            self.redis_client.ltrim("risk_alerts", 0, 99)  # Keep last 100 alerts

            print(f"🚨 Risk alert created: {alert_type}")

        except Exception as e:
            print(f"❌ Error creating alert: {e}")

    def get_alert_severity(self, alert_type: str) -> str:
        """Get alert severity level"""
        severity_map = {
            "high_portfolio_risk": "high",
            "high_concentration_risk": "high",
            "high_var_risk": "high",
            "volatility_spike": "medium",
            "price_gap": "medium",
            "large_loss": "high",
        }

        return severity_map.get(alert_type, "low")

    async def update_symbol_risk(self, symbol: str, price: float, volume: float):
        """Update risk metrics for a symbol"""
        try:
            symbol_risk = self.state["risk_metrics"].get(symbol, {})

            # Update price history
            price_history = symbol_risk.get("price_history", [])
            price_history.append({"price": price, "timestamp": datetime.now().isoformat()})

            # Keep last 100 prices
            if len(price_history) > 100:
                price_history = price_history[-100:]

            # Calculate volatility
            if len(price_history) > 1:
                prices = [p["price"] for p in price_history]
                returns = np.diff(np.log(prices))
                volatility = np.std(returns) * np.sqrt(252)  # Annualized

                symbol_risk.update(
                    {
                        "volatility": volatility,
                        "historical_volatility": symbol_risk.get("volatility", volatility),
                        "price_history": price_history,
                        "last_update": datetime.now().isoformat(),
                    }
                )

            # Determine risk level
            if symbol_risk.get("volatility", 0) > 0.5:
                symbol_risk["risk_level"] = "high"
            elif symbol_risk.get("volatility", 0) > 0.3:
                symbol_risk["risk_level"] = "medium"
            else:
                symbol_risk["risk_level"] = "low"

            # Update state
            self.state["risk_metrics"][symbol] = symbol_risk

        except Exception as e:
            print(f"❌ Error updating symbol risk: {e}")

    async def get_symbol_volatility(self, symbol: str) -> float:
        """Get volatility for a symbol"""
        try:
            symbol_risk = self.state["risk_metrics"].get(symbol, {})
            return symbol_risk.get("volatility", 0.2)  # Default 20% volatility
        except Exception as e:
            print(f"❌ Error getting symbol volatility: {e}")
            return 0.2

    def calculate_trade_risk_score(self, trade_risk: Dict[str, Any]) -> float:
        """Calculate risk score for a trade (0-100, higher is riskier)"""
        try:
            score = 0

            # Portfolio risk contribution (40% weight)
            portfolio_contribution = trade_risk.get("portfolio_risk_contribution", 0)
            score += portfolio_contribution * 40

            # VaR contribution (30% weight)
            var_contribution = trade_risk.get("var_95", 0) / trade_risk.get("dollar_risk", 1)
            score += var_contribution * 30

            # Position size (30% weight)
            position_size = trade_risk.get("position_size", 0)
            max_size = self.state["max_position_size"]
            size_ratio = min(position_size / max_size, 1.0) if max_size > 0 else 0
            score += size_ratio * 30

            return min(100, score)

        except Exception as e:
            print(f"❌ Error calculating trade risk score: {e}")
            return 50.0

    def generate_risk_recommendation(self, trade_risk: Dict[str, Any]) -> str:
        """Generate risk recommendation for a trade"""
        try:
            risk_score = trade_risk.get("risk_score", 50)

            if risk_score < 30:
                return "proceed"
            elif risk_score < 60:
                return "proceed_with_caution"
            elif risk_score < 80:
                return "reduce_size"
            else:
                return "avoid"

        except Exception as e:
            print(f"❌ Error generating risk recommendation: {e}")
            return "hold"

    async def update_risk_metrics(self):
        """Update risk metrics"""
        try:
            metrics = {
                "agent_id": self.agent_id,
                "portfolio_risk": self.state["portfolio_risk"],
                "alerts_count": len(self.state["alerts"]),
                "risk_level": self.state["risk_level"],
                "timestamp": datetime.now().isoformat(),
            }

            # Store metrics in Redis
            self.redis_client.set(f"agent_metrics:{self.agent_id}", json.dumps(metrics), ex=300)

        except Exception as e:
            print(f"❌ Error updating risk metrics: {e}")

    async def cleanup_alerts(self):
        """Clean up old alerts"""
        try:
            current_time = datetime.now()
            alerts = self.state["alerts"]

            # Remove alerts older than 24 hours
            alerts = [
                alert
                for alert in alerts
                if (current_time - datetime.fromisoformat(alert["timestamp"])).days < 1
            ]

            self.state["alerts"] = alerts

        except Exception as e:
            print(f"❌ Error cleaning up alerts: {e}")

    async def handle_market_data(self, message: Dict[str, Any]):
        """Handle market data message"""
        try:
            market_data = message.get("market_data", {})
            print(f"📊 Risk Agent received market data for {len(market_data)} symbols")
            
            # Process market data
            await self.process_market_data(market_data)
            
            # Update risk metrics for each symbol
            for symbol, data in market_data.items():
                price = data.get("price", 0)
                volume = data.get("volume", 0)
                if price > 0:
                    await self.update_symbol_risk(symbol, price, volume)
            
            # Check for risk alerts
            await self.check_risk_alerts()
            
            # Update risk metrics
            await self.update_risk_metrics()
            
        except Exception as e:
            print(f"❌ Error handling market data: {e}")
            await self.broadcast_error(f"Market data handling error: {e}")
