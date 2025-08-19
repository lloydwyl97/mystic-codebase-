"""
Compliance Agent
Handles regulatory compliance, trading limits, and audit logging
"""

import asyncio
import json
import os
import sys
import uuid
from datetime import datetime
from typing import Any

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.agents.base_agent import BaseAgent


class ComplianceAgent(BaseAgent):
    """Compliance Agent - Handles regulatory compliance and audit logging"""

    def __init__(self, agent_id: str = "compliance_agent_001"):
        super().__init__(agent_id, "compliance")

        # Compliance-specific state
        self.state.update(
            {
                "compliance_rules": {},
                "trading_limits": {},
                "audit_log": [],
                "violations": [],
                "regulatory_status": "compliant",
                "last_audit": None,
                "compliance_score": 100,
            }
        )

        # Register compliance-specific handlers
        self.register_handler("validate_trade", self.handle_validate_trade)
        self.register_handler("check_compliance", self.handle_check_compliance)
        self.register_handler("audit_request", self.handle_audit_request)
        self.register_handler("update_limits", self.handle_update_limits)
        self.register_handler("trading_signal", self.handle_trading_signal)

        print(f"âš–ï¸ Compliance Agent {agent_id} initialized")

    async def initialize(self):
        """Initialize compliance agent resources"""
        try:
            # Load compliance rules and limits
            await self.load_compliance_config()

            # Initialize audit system
            await self.initialize_audit_system()

            # Start compliance monitoring
            await self.start_compliance_monitoring()

            print(f"âœ… Compliance Agent {self.agent_id} initialized successfully")

        except Exception as e:
            print(f"âŒ Error initializing Compliance Agent: {e}")
            self.update_health_status("error")

    async def process_loop(self):
        """Main compliance processing loop"""
        while self.running:
            try:
                # Monitor compliance status
                await self.monitor_compliance_status()

                # Check for violations
                await self.check_violations()

                # Update compliance metrics
                await self.update_compliance_metrics()

                # Clean up old audit logs
                await self.cleanup_audit_logs()

                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                print(f"âŒ Error in compliance processing loop: {e}")
                await asyncio.sleep(120)

    async def load_compliance_config(self):
        """Load compliance configuration from Redis"""
        try:
            # Load compliance rules
            rules_data = self.redis_client.get("compliance_rules")
            if rules_data:
                self.state["compliance_rules"] = json.loads(rules_data)
            else:
                # Set default compliance rules
                self.state["compliance_rules"] = {
                    "max_daily_trades": 100,
                    "max_daily_volume": 100000,
                    "max_position_size": 0.1,
                    "min_rest_period": 300,  # 5 minutes
                    "max_loss_per_day": 0.05,  # 5%
                    "prohibited_symbols": [],
                    "trading_hours": {"start": "00:00", "end": "23:59"},
                }

            # Load trading limits
            limits_data = self.redis_client.get("trading_limits")
            if limits_data:
                self.state["trading_limits"] = json.loads(limits_data)
            else:
                # Set default trading limits
                self.state["trading_limits"] = {
                    "daily_trades": 0,
                    "daily_volume": 0,
                    "daily_loss": 0,
                    "last_trade_time": None,
                }

            print("ðŸ“‹ Compliance configuration loaded")

        except Exception as e:
            print(f"âŒ Error loading compliance configuration: {e}")

    async def initialize_audit_system(self):
        """Initialize audit logging system"""
        try:
            # Create audit log directory
            audit_dir = "audit_logs"
            os.makedirs(audit_dir, exist_ok=True)

            # Load existing audit logs
            from utils.redis_helpers import to_str_list
            audit_data = to_str_list(self.redis_client.lrange("audit_log", 0, -1))
            self.state["audit_log"] = [json.loads(item) for item in audit_data]

            print("ðŸ“ Audit system initialized")

        except Exception as e:
            print(f"âŒ Error initializing audit system: {e}")

    async def start_compliance_monitoring(self):
        """Start compliance monitoring"""
        try:
            # Subscribe to trading events
            pubsub = self.redis_client.pubsub()
            pubsub.subscribe("trading_events")

            # Start event listener
            asyncio.create_task(self.listen_trading_events(pubsub))

            print("ðŸ‘ï¸ Compliance monitoring started")

        except Exception as e:
            print(f"âŒ Error starting compliance monitoring: {e}")

    async def listen_trading_events(self, pubsub):
        """Listen for trading events"""
        try:
            for message in pubsub.listen():
                if not self.running:
                    break

                if message["type"] == "message":
                    event_data = json.loads(message["data"])
                    await self.process_trading_event(event_data)

        except Exception as e:
            print(f"âŒ Error in trading events listener: {e}")
        finally:
            pubsub.close()

    async def process_trading_event(self, event_data: dict[str, Any]):
        """Process trading event for compliance"""
        try:
            event_type = event_data.get("type")

            if event_type == "trade_executed":
                await self.audit_trade(event_data)
            elif event_type == "order_placed":
                await self.audit_order(event_data)
            elif event_type == "position_opened":
                await self.audit_position(event_data)

        except Exception as e:
            print(f"âŒ Error processing trading event: {e}")

    async def handle_validate_trade(self, message: dict[str, Any]):
        """Handle trade validation request"""
        try:
            trade_data = message.get("trade_data", {})

            print("âš–ï¸ Validating trade compliance")

            # Validate trade
            validation_result = await self.validate_trade_compliance(trade_data)

            # Send response
            response = {
                "type": "trade_validation",
                "approved": validation_result["approved"],
                "reason": validation_result.get("reason", ""),
                "violations": validation_result.get("violations", []),
                "timestamp": datetime.now().isoformat(),
            }

            sender = message.get("from_agent")
            if sender:
                await self.send_message(sender, response)

        except Exception as e:
            print(f"âŒ Error validating trade: {e}")
            await self.broadcast_error(f"Trade validation error: {e}")

    async def handle_check_compliance(self, message: dict[str, Any]):
        """Handle compliance check request"""
        try:
            check_type = message.get("check_type", "general")

            print(f"ðŸ” Performing compliance check: {check_type}")

            # Perform compliance check
            compliance_status = await self.perform_compliance_check(check_type)

            # Send response
            response = {
                "type": "compliance_status",
                "status": compliance_status,
                "timestamp": datetime.now().isoformat(),
            }

            sender = message.get("from_agent")
            if sender:
                await self.send_message(sender, response)

        except Exception as e:
            print(f"âŒ Error checking compliance: {e}")
            await self.broadcast_error(f"Compliance check error: {e}")

    async def handle_audit_request(self, message: dict[str, Any]):
        """Handle audit request"""
        try:
            audit_type = message.get("audit_type", "general")
            start_date = message.get("start_date")
            end_date = message.get("end_date")

            print(f"ðŸ“‹ Performing audit: {audit_type}")

            # Perform audit
            audit_result = await self.perform_audit(audit_type, start_date, end_date)

            # Send response
            response = {
                "type": "audit_result",
                "audit_type": audit_type,
                "result": audit_result,
                "timestamp": datetime.now().isoformat(),
            }

            sender = message.get("from_agent")
            if sender:
                await self.send_message(sender, response)

        except Exception as e:
            print(f"âŒ Error performing audit: {e}")
            await self.broadcast_error(f"Audit error: {e}")

    async def handle_update_limits(self, message: dict[str, Any]):
        """Handle trading limits update"""
        try:
            new_limits = message.get("limits", {})

            print("ðŸ“Š Updating trading limits")

            # Update limits
            self.state["trading_limits"].update(new_limits)

            # Store in Redis
            self.redis_client.set(
                "trading_limits",
                json.dumps(self.state["trading_limits"]),
                ex=3600,
            )

            # Send confirmation
            response = {
                "type": "limits_updated",
                "limits": self.state["trading_limits"],
                "timestamp": datetime.now().isoformat(),
            }

            sender = message.get("from_agent")
            if sender:
                await self.send_message(sender, response)

        except Exception as e:
            print(f"âŒ Error updating limits: {e}")
            await self.broadcast_error(f"Limits update error: {e}")

    async def handle_trading_signal(self, message: dict[str, Any]):
        """Handle trading signal for compliance validation"""
        try:
            signal = message.get("signal", {})
            market_data = message.get("market_data", {})

            symbol = market_data.get("symbol")
            signal.get("type")

            print(f"ðŸ” Validating signal compliance for {symbol}")

            # Validate signal compliance
            compliance_check = await self.validate_signal_compliance(signal, market_data)

            if compliance_check["approved"]:
                # Send approved signal to risk agent
                await self.send_message(
                    "risk_agent",
                    {
                        "type": "compliance_approved_signal",
                        "signal": signal,
                        "market_data": market_data,
                        "compliance_check": compliance_check,
                    },
                )
            else:
                # Send rejection to strategy agent
                await self.send_message(
                    "strategy_agent",
                    {
                        "type": "signal_compliance_rejected",
                        "reason": compliance_check["reason"],
                        "signal": signal,
                    },
                )

        except Exception as e:
            print(f"âŒ Error handling trading signal: {e}")

    async def validate_trade_compliance(self, trade_data: dict[str, Any]) -> dict[str, Any]:
        """Validate trade for compliance"""
        try:
            violations = []
            approved = True

            # Check trading hours
            if not await self.check_trading_hours():
                violations.append("Outside trading hours")
                approved = False

            # Check daily trade limit
            if not await self.check_daily_trade_limit():
                violations.append("Daily trade limit exceeded")
                approved = False

            # Check daily volume limit
            if not await self.check_daily_volume_limit(trade_data):
                violations.append("Daily volume limit exceeded")
                approved = False

            # Check position size limit
            if not await self.check_position_size_limit(trade_data):
                violations.append("Position size limit exceeded")
                approved = False

            # Check prohibited symbols
            if not await self.check_prohibited_symbols(trade_data):
                violations.append("Symbol is prohibited")
                approved = False

            # Check rest period
            if not await self.check_rest_period():
                violations.append("Rest period not met")
                approved = False

            # Check daily loss limit
            if not await self.check_daily_loss_limit(trade_data):
                violations.append("Daily loss limit exceeded")
                approved = False

            return {
                "approved": approved,
                "violations": violations,
                "reason": "; ".join(violations) if violations else "Compliant",
            }

        except Exception as e:
            print(f"âŒ Error validating trade compliance: {e}")
            return {
                "approved": False,
                "violations": [f"Validation error: {e}"],
            }

    async def validate_signal_compliance(
        self, signal: dict[str, Any], market_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Validate trading signal for compliance"""
        try:
            symbol = market_data.get("symbol")
            signal_type = signal.get("type")

            # Create mock trade data for validation
            trade_data = {
                "symbol": symbol,
                "type": signal_type,
                "quantity": 0.001,  # Default small quantity
                "value": 100,  # Default small value
            }

            # Use trade validation logic
            validation = await self.validate_trade_compliance(trade_data)

            return validation

        except Exception as e:
            print(f"âŒ Error validating signal compliance: {e}")
            return {"approved": False, "reason": f"Validation error: {e}"}

    async def check_trading_hours(self) -> bool:
        """Check if current time is within trading hours"""
        try:
            trading_hours = self.state["compliance_rules"].get("trading_hours", {})
            start_time = trading_hours.get("start", "00:00")
            end_time = trading_hours.get("end", "23:59")

            current_time = datetime.now().time()
            start = datetime.strptime(start_time, "%H:%M").time()
            end = datetime.strptime(end_time, "%H:%M").time()

            return start <= current_time <= end

        except Exception as e:
            print(f"âŒ Error checking trading hours: {e}")
            return True  # Default to allowed

    async def check_daily_trade_limit(self) -> bool:
        """Check daily trade limit"""
        try:
            max_trades = self.state["compliance_rules"].get("max_daily_trades", 100)
            current_trades = self.state["trading_limits"].get("daily_trades", 0)

            return current_trades < max_trades

        except Exception as e:
            print(f"âŒ Error checking daily trade limit: {e}")
            return True

    async def check_daily_volume_limit(self, trade_data: dict[str, Any]) -> bool:
        try:
            max_volume = self.state["compliance_rules"].get("max_daily_volume", 100000)
            current_volume = self.state["trading_limits"].get("daily_volume", 0)
            trade_value = trade_data.get("value", 0)

            return (current_volume + trade_value) <= max_volume

        except Exception as e:
            print(f"âŒ Error checking daily volume limit: {e}")
            return True

    async def check_position_size_limit(self, trade_data: dict[str, Any]) -> bool:
        try:
            max_position_size = self.state["compliance_rules"].get("max_position_size", 0.1)
            trade_quantity = trade_data.get("quantity", 0)

            # This would need portfolio context for full validation
            # For now, just check basic limits
            return trade_quantity <= max_position_size

        except Exception as e:
            print(f"âŒ Error checking position size limit: {e}")
            return True

    async def check_prohibited_symbols(self, trade_data: dict[str, Any]) -> bool:
        try:
            prohibited_symbols = self.state["compliance_rules"].get("prohibited_symbols", [])
            symbol = trade_data.get("symbol", "")

            return symbol not in prohibited_symbols

        except Exception as e:
            print(f"âŒ Error checking prohibited symbols: {e}")
            return True

    async def check_rest_period(self) -> bool:
        try:
            min_rest_period = self.state["compliance_rules"].get("min_rest_period", 300)
            last_trade_time = self.state["trading_limits"].get("last_trade_time")

            if not last_trade_time:
                return True

            last_trade = datetime.fromisoformat(last_trade_time)
            time_since_last_trade = (datetime.now() - last_trade).total_seconds()

            return time_since_last_trade >= min_rest_period

        except Exception as e:
            print(f"âŒ Error checking rest period: {e}")
            return True

    async def check_daily_loss_limit(self, trade_data: dict[str, Any]) -> bool:
        try:
            max_daily_loss = self.state["compliance_rules"].get("max_loss_per_day", 0.05)
            current_daily_loss = self.state["trading_limits"].get("daily_loss", 0)

            # This would need portfolio context for full validation
            # For now, just check basic limits
            return current_daily_loss <= max_daily_loss

        except Exception as e:
            print(f"âŒ Error checking daily loss limit: {e}")
            return True

    async def audit_trade(self, trade_data: dict[str, Any]):
        """Audit a trade"""
        try:
            audit_entry = {
                "audit_id": str(uuid.uuid4()),
                "type": "trade_audit",
                "trade_data": trade_data,
                "compliance_check": await self.validate_trade_compliance(trade_data),
                "timestamp": datetime.now().isoformat(),
                "agent_id": self.agent_id,
            }

            # Add to audit log
            self.state["audit_log"].append(audit_entry)

            # Store in Redis
            self.redis_client.lpush("audit_log", json.dumps(audit_entry))
            self.redis_client.ltrim("audit_log", 0, 9999)  # Keep last 10,000 entries

            # Update trading limits
            await self.update_trading_limits(trade_data)

            print(f"ðŸ“ Trade audited: {audit_entry['audit_id']}")

        except Exception as e:
            print(f"âŒ Error auditing trade: {e}")

    async def audit_order(self, order_data: dict[str, Any]):
        """Audit an order"""
        try:
            audit_entry = {
                "audit_id": str(uuid.uuid4()),
                "type": "order_audit",
                "order_data": order_data,
                "timestamp": datetime.now().isoformat(),
                "agent_id": self.agent_id,
            }

            # Add to audit log
            self.state["audit_log"].append(audit_entry)

            # Store in Redis
            self.redis_client.lpush("audit_log", json.dumps(audit_entry))

            print(f"ðŸ“ Order audited: {audit_entry['audit_id']}")

        except Exception as e:
            print(f"âŒ Error auditing order: {e}")

    async def audit_position(self, position_data: dict[str, Any]):
        """Audit a position"""
        try:
            audit_entry = {
                "audit_id": str(uuid.uuid4()),
                "type": "position_audit",
                "position_data": position_data,
                "timestamp": datetime.now().isoformat(),
                "agent_id": self.agent_id,
            }

            # Add to audit log
            self.state["audit_log"].append(audit_entry)

            # Store in Redis
            self.redis_client.lpush("audit_log", json.dumps(audit_entry))

            print(f"ðŸ“ Position audited: {audit_entry['audit_id']}")

        except Exception as e:
            print(f"âŒ Error auditing position: {e}")

    async def update_trading_limits(self, trade_data: dict[str, Any]):
        """Update trading limits after a trade"""
        try:
            # Update daily trade count
            self.state["trading_limits"]["daily_trades"] += 1

            # Update daily volume
            trade_value = trade_data.get("value", 0)
            self.state["trading_limits"]["daily_volume"] += trade_value

            # Update last trade time
            self.state["trading_limits"]["last_trade_time"] = datetime.now().isoformat()

            # Store in Redis
            self.redis_client.set(
                "trading_limits",
                json.dumps(self.state["trading_limits"]),
                ex=3600,
            )

        except Exception as e:
            print(f"âŒ Error updating trading limits: {e}")

    async def perform_compliance_check(self, check_type: str) -> dict[str, Any]:
        """Perform a compliance check"""
        try:
            check_result = {
                "check_type": check_type,
                "status": "compliant",
                "violations": [],
                "score": 100,
                "timestamp": datetime.now().isoformat(),
            }

            if check_type == "general":
                # Perform general compliance check
                violations = await self.check_all_compliance_rules()
                check_result["violations"] = violations
                check_result["status"] = "compliant" if not violations else "non_compliant"
                check_result["score"] = max(0, 100 - len(violations) * 10)

            elif check_type == "trading_limits":
                # Check trading limits
                limit_violations = await self.check_trading_limits()
                check_result["violations"] = limit_violations
                check_result["status"] = "compliant" if not limit_violations else "non_compliant"

            # Update compliance status
            self.state["regulatory_status"] = check_result["status"]
            self.state["compliance_score"] = check_result["score"]

            return check_result

        except Exception as e:
            print(f"âŒ Error performing compliance check: {e}")
            return {"status": "error", "error": str(e)}

    async def check_all_compliance_rules(self) -> list[str]:
        """Check all compliance rules"""
        try:
            violations = []

            # Check trading hours
            if not await self.check_trading_hours():
                violations.append("Outside trading hours")

            # Check daily limits
            if not await self.check_daily_trade_limit():
                violations.append("Daily trade limit exceeded")

            # Check volume limits
            if not await self.check_daily_volume_limit({"value": 0}):
                violations.append("Daily volume limit exceeded")

            return violations

        except Exception as e:
            print(f"âŒ Error checking compliance rules: {e}")
            return [f"Compliance check error: {e}"]

    async def check_trading_limits(self) -> list[str]:
        """Check trading limits"""
        try:
            violations = []

            limits = self.state["trading_limits"]
            rules = self.state["compliance_rules"]

            # Check daily trade limit
            if limits.get("daily_trades", 0) >= rules.get("max_daily_trades", 100):
                violations.append("Daily trade limit reached")

            # Check daily volume limit
            if limits.get("daily_volume", 0) >= rules.get("max_daily_volume", 100000):
                violations.append("Daily volume limit reached")

            return violations

        except Exception as e:
            print(f"âŒ Error checking trading limits: {e}")
            return [f"Limit check error: {e}"]

    async def perform_audit(
        self, audit_type: str, start_date: str = None, end_date: str = None
    ) -> dict[str, Any]:
        """Perform an audit"""
        try:
            audit_result = {
                "audit_type": audit_type,
                "start_date": start_date,
                "end_date": end_date,
                "total_entries": 0,
                "violations": [],
                "compliance_score": 0,
                "timestamp": datetime.now().isoformat(),
            }

            # Filter audit log by date range
            filtered_log = self.state["audit_log"]

            if start_date and end_date:
                start = datetime.fromisoformat(start_date)
                end = datetime.fromisoformat(end_date)
                filtered_log = [
                    entry
                    for entry in self.state["audit_log"]
                    if start <= datetime.fromisoformat(entry["timestamp"]) <= end
                ]

            audit_result["total_entries"] = len(filtered_log)

            # Analyze violations
            violations = []
            for entry in filtered_log:
                if entry.get("compliance_check", {}).get("violations"):
                    violations.extend(entry["compliance_check"]["violations"])

            audit_result["violations"] = list(set(violations))  # Remove duplicates
            audit_result["compliance_score"] = max(0, 100 - len(audit_result["violations"]) * 5)

            return audit_result

        except Exception as e:
            print(f"âŒ Error performing audit: {e}")
            return {"error": str(e)}

    async def monitor_compliance_status(self):
        """Monitor compliance status"""
        try:
            # Perform periodic compliance check
            compliance_status = await self.perform_compliance_check("general")

            # Update state
            self.state["regulatory_status"] = compliance_status["status"]
            self.state["compliance_score"] = compliance_status["score"]

            # Check for violations
            if compliance_status["violations"]:
                await self.create_violation_report(compliance_status["violations"])

        except Exception as e:
            print(f"âŒ Error monitoring compliance status: {e}")

    async def check_violations(self):
        """Check for compliance violations"""
        try:
            # Check for new violations
            violations = await self.check_all_compliance_rules()

            if violations:
                await self.create_violation_report(violations)

                # Update compliance status
                self.state["regulatory_status"] = "non_compliant"
                self.state["compliance_score"] = max(0, 100 - len(violations) * 10)

        except Exception as e:
            print(f"âŒ Error checking violations: {e}")

    async def create_violation_report(self, violations: list[str]):
        """Create a violation report"""
        try:
            violation_report = {
                "report_id": str(uuid.uuid4()),
                "violations": violations,
                "timestamp": datetime.now().isoformat(),
                "severity": "high" if len(violations) > 3 else "medium",
                "status": "open",
            }

            # Add to violations list
            self.state["violations"].append(violation_report)

            # Store in Redis
            self.redis_client.lpush("compliance_violations", json.dumps(violation_report))

            # Broadcast violation alert
            await self.broadcast_message(
                {"type": "compliance_violation", "report": violation_report}
            )

            print(f"ðŸš¨ Compliance violation report created: {violation_report['report_id']}")

        except Exception as e:
            print(f"âŒ Error creating violation report: {e}")

    async def update_compliance_metrics(self):
        """Update compliance metrics"""
        try:
            metrics = {
                "agent_id": self.agent_id,
                "regulatory_status": self.state["regulatory_status"],
                "compliance_score": self.state["compliance_score"],
                "audit_log_count": len(self.state["audit_log"]),
                "violations_count": len(self.state["violations"]),
                "last_audit": self.state["last_audit"],
                "timestamp": datetime.now().isoformat(),
            }

            # Store metrics in Redis
            self.redis_client.set(f"agent_metrics:{self.agent_id}", json.dumps(metrics), ex=300)

        except Exception as e:
            print(f"âŒ Error updating compliance metrics: {e}")

    async def cleanup_audit_logs(self):
        """Clean up old audit logs"""
        try:
            current_time = datetime.now()

            # Keep only last 1000 audit entries
            if len(self.state["audit_log"]) > 1000:
                self.state["audit_log"] = self.state["audit_log"][-1000:]

            # Remove violations older than 30 days
            violations = self.state["violations"]
            violations = [
                violation
                for violation in violations
                if (current_time - datetime.fromisoformat(violation["timestamp"])).days < 30
            ]
            self.state["violations"] = violations

        except Exception as e:
            print(f"âŒ Error cleaning up audit logs: {e}")

    async def process_market_data(self, market_data: dict[str, Any]):
        """Process incoming market data for compliance monitoring"""
        try:
            print("ðŸ“Š Processing market data for compliance monitoring")

            # Update market data in state
            self.state["last_market_data"] = market_data
            self.state["last_market_update"] = datetime.now().isoformat()

            # Check for compliance issues in market data
            for symbol, data in market_data.items():
                # Check for unusual price movements that might indicate market manipulation
                if "price" in data and "volume" in data:
                    price = data["price"]
                    volume = data["volume"]
                    
                    # Check for unusual volume spikes
                    if volume > self.state.get("max_normal_volume", 1000000):
                        await self.audit_market_anomaly(symbol, "high_volume", data)
                    
                    # Check for unusual price movements
                    if "change_24h" in data and abs(data["change_24h"]) > 20:
                        await self.audit_market_anomaly(symbol, "extreme_price_movement", data)

            # Update compliance metrics
            await self.update_compliance_metrics()

            print("âœ… Market data processed for compliance monitoring")

        except Exception as e:
            print(f"âŒ Error processing market data for compliance: {e}")
            await self.broadcast_error(f"Compliance market data error: {e}")

    async def audit_market_anomaly(self, symbol: str, anomaly_type: str, data: dict[str, Any]):
        """Audit market anomalies for compliance"""
        try:
            audit_entry = {
                "timestamp": datetime.now().isoformat(),
                "symbol": symbol,
                "anomaly_type": anomaly_type,
                "data": data,
                "severity": "high" if anomaly_type == "extreme_price_movement" else "medium"
            }
            
            # Add to audit log
            self.state["audit_log"].append(audit_entry)
            
            # Store in Redis
            self.redis_client.lpush("compliance_audit_log", json.dumps(audit_entry))
            
            print(f"ðŸ” Market anomaly audited: {symbol} - {anomaly_type}")
            
        except Exception as e:
            print(f"âŒ Error auditing market anomaly: {e}")


