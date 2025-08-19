# yield_rotator.py
"""
Yield Rotation Engine
Manages idle capital by parking it in yield-generating protocols
and rotating back to trading.
"""

import time
from datetime import datetime, timedelta, timezone
from typing import Any


class YieldRotator:
    """
    Advanced yield rotation engine for capital efficiency.
    """

    def __init__(self, min_park_amount: float = 100.0, max_park_percentage: float = 0.3):
        """
        Initialize yield rotator.

        Args:
            min_park_amount: Minimum amount to park in yield
            max_park_percentage: Maximum percentage of capital to park
        """
        self.min_park_amount = min_park_amount
        self.max_park_percentage = max_park_percentage
        self.parked_capital = {}
        self.yield_history = []
        self.yield_protocols = {
            "usdt_staking": {
                "name": "USDT Staking",
                "apy": 0.08,  # 8% APY
                "min_lock": 1,  # 1 day minimum
                "max_lock": 365,  # 365 days maximum
                "risk_level": "low",
            },
            "defi_lending": {
                "name": "DeFi Lending",
                "apy": 0.12,  # 12% APY
                "min_lock": 7,  # 7 days minimum
                "max_lock": 365,
                "risk_level": "medium",
            },
            "liquidity_pools": {
                "name": "Liquidity Pools",
                "apy": 0.18,  # 18% APY
                "min_lock": 30,  # 30 days minimum
                "max_lock": 365,
                "risk_level": "high",
            },
            "stablecoin_farming": {
                "name": "Stablecoin Farming",
                "apy": 0.15,  # 15% APY
                "min_lock": 14,  # 14 days minimum
                "max_lock": 365,
                "risk_level": "medium",
            },
        }

    def calculate_optimal_park_amount(
        self, total_capital: float, idle_percentage: float = 0.2
    ) -> float:
        """
        Calculate optimal amount to park in yield protocols.

        Args:
            total_capital: Total available capital
            idle_percentage: Percentage of capital that's idle

        Returns:
            Amount to park in yield
        """
        idle_amount = total_capital * idle_percentage
        max_park = total_capital * self.max_park_percentage

        park_amount = min(idle_amount, max_park)

        if park_amount < self.min_park_amount:
            return 0.0

        return round(park_amount, 2)

    def select_yield_protocol(
        self,
        amount: float,
        risk_tolerance: str = "medium",
        lock_period: int = 30,
    ) -> dict[str, Any]:
        """
        Select optimal yield protocol based on parameters.

        Args:
            amount: Amount to park
            risk_tolerance: Risk tolerance level
            lock_period: Desired lock period in days

        Returns:
            Selected protocol configuration
        """
        available_protocols = []

        for protocol_id, protocol in self.yield_protocols.items():
            if protocol["min_lock"] <= lock_period <= protocol[
                "max_lock"
            ] and self._matches_risk_tolerance(protocol["risk_level"], risk_tolerance):
                available_protocols.append((protocol_id, protocol))

        if not available_protocols:
            # Default to USDT staking if no matches
            return {
                "protocol_id": "usdt_staking",
                "protocol": self.yield_protocols["usdt_staking"],
                "amount": amount,
                "lock_period": lock_period,
            }

        # Sort by APY and select best
        available_protocols.sort(key=lambda x: x[1]["apy"], reverse=True)
        selected_id, selected_protocol = available_protocols[0]

        return {
            "protocol_id": selected_id,
            "protocol": selected_protocol,
            "amount": amount,
            "lock_period": lock_period,
        }

    def _matches_risk_tolerance(self, protocol_risk: str, user_risk: str) -> bool:
        """Check if protocol risk matches user tolerance."""
        risk_levels = {"low": 1, "medium": 2, "high": 3}
        return risk_levels.get(protocol_risk, 2) <= risk_levels.get(user_risk, 2)

    def park_capital(
        self, amount: float, protocol_id: str = None, lock_period: int = 30
    ) -> dict[str, Any]:
        """
        Park capital in yield protocol.

        Args:
            amount: Amount to park
            protocol_id: Specific protocol to use
            lock_period: Lock period in days

        Returns:
            Parking result
        """
        if amount < self.min_park_amount:
            return {
                "success": False,
                "error": (f"Amount ${amount} below minimum ${self.min_park_amount}"),
            }

        # Select protocol
        if protocol_id and protocol_id in self.yield_protocols:
            protocol = self.yield_protocols[protocol_id]
        else:
            selection = self.select_yield_protocol(amount, lock_period=lock_period)
            protocol_id = selection["protocol_id"]
            protocol = selection["protocol"]

        # Simulate parking (in production, this would call actual APIs)
        parking_id = f"park_{int(time.time())}"

        parking_record = {
            "id": parking_id,
            "amount": amount,
            "protocol_id": protocol_id,
            "protocol_name": protocol["name"],
            "apy": protocol["apy"],
            "lock_period": lock_period,
            "start_time": datetime.now(timezone.utc).isoformat(),
            "end_time": ((datetime.now(timezone.utc) + timedelta(days=lock_period)).isoformat()),
            "status": "active",
            "earned_yield": 0.0,
        }

        self.parked_capital[parking_id] = parking_record

        # Log parking action
        self._log_parking_action(parking_record)

        print(f"ðŸ’° Parked ${amount} in {protocol['name']} for {lock_period} days")
        print(f"ðŸ“ˆ Expected APY: {protocol['apy']:.1%}")

        return {
            "success": True,
            "parking_id": parking_id,
            "protocol": protocol,
            "expected_yield": amount * protocol["apy"] * (lock_period / 365),
        }

    def withdraw_capital(self, parking_id: str, force: bool = False) -> dict[str, Any]:
        """
        Withdraw capital from yield protocol.

        Args:
            parking_id: Parking record ID
            force: Force withdrawal (may incur penalties)

        Returns:
            Withdrawal result
        """
        if parking_id not in self.parked_capital:
            return {
                "success": False,
                "error": f"Parking record {parking_id} not found",
            }

        parking_record = self.parked_capital[parking_id]

        # Check if lock period has ended
        end_time = datetime.fromisoformat(parking_record["end_time"])
        current_time = datetime.now(timezone.utc)

        if current_time < end_time and not force:
            return {
                "success": False,
                "error": f"Lock period not ended. Ends at {end_time}",
            }

        # Calculate earned yield
        days_parked = (current_time - datetime.fromisoformat(parking_record["start_time"])).days
        earned_yield = parking_record["amount"] * parking_record["apy"] * (days_parked / 365)

        # Update record
        parking_record["status"] = "withdrawn"
        parking_record["earned_yield"] = round(earned_yield, 2)
        parking_record["withdrawal_time"] = current_time.isoformat()

        # Log withdrawal
        self._log_withdrawal_action(parking_record)

        print(f"ðŸ’° Withdrew ${parking_record['amount']} + ${earned_yield} yield")

        return {
            "success": True,
            "amount": parking_record["amount"],
            "earned_yield": earned_yield,
            "total_returned": parking_record["amount"] + earned_yield,
        }

    def get_parked_capital_summary(self) -> dict[str, Any]:
        """Get summary of all parked capital."""
        if not self.parked_capital:
            return {
                "total_parked": 0.0,
                "total_earned": 0.0,
                "active_positions": 0,
                "protocols": {},
            }

        total_parked = 0.0
        total_earned = 0.0
        active_positions = 0
        protocols = {}

        for parking_id, record in self.parked_capital.items():
            if record["status"] == "active":
                total_parked += record["amount"]
                active_positions += 1

                # Calculate current earned yield
                start_time = datetime.fromisoformat(record["start_time"])
                days_parked = (datetime.now(timezone.utc) - start_time).days
                earned = record["amount"] * record["apy"] * (days_parked / 365)
                total_earned += earned

                # Group by protocol
                protocol = record["protocol_id"]
                if protocol not in protocols:
                    protocols[protocol] = {
                        "amount": 0.0,
                        "positions": 0,
                        "avg_apy": 0.0,
                    }
                protocols[protocol]["amount"] += record["amount"]
                protocols[protocol]["positions"] += 1
                protocols[protocol]["avg_apy"] = record["apy"]
            else:
                total_earned += record["earned_yield"]

        return {
            "total_parked": round(total_parked, 2),
            "total_earned": round(total_earned, 2),
            "active_positions": active_positions,
            "protocols": protocols,
        }

    def auto_rotate_capital(
        self,
        total_capital: float,
        trading_signal_strength: float,
        idle_percentage: float = 0.2,
    ) -> dict[str, Any]:
        """
        Automatically rotate capital based on trading signals.

        Args:
            total_capital: Total available capital
            trading_signal_strength: Strength of trading signal (0-1)
            idle_percentage: Percentage of capital that's idle

        Returns:
            Rotation actions taken
        """
        actions = []

        # If strong trading signal, withdraw from yield
        if trading_signal_strength > 0.7:
            summary = self.get_parked_capital_summary()
            if summary["total_parked"] > 0:
                # Withdraw all parked capital
                for parking_id, record in self.parked_capital.items():
                    if record["status"] == "active":
                        result = self.withdraw_capital(parking_id, force=True)
                        if result["success"]:
                            actions.append(
                                {
                                    "action": "withdraw",
                                    "parking_id": parking_id,
                                    "amount": result["total_returned"],
                                    "reason": "strong_trading_signal",
                                }
                            )

        # If weak trading signal, park idle capital
        elif trading_signal_strength < 0.3:
            park_amount = self.calculate_optimal_park_amount(total_capital, idle_percentage)
            if park_amount > 0:
                result = self.park_capital(park_amount, lock_period=7)
                if result["success"]:
                    actions.append(
                        {
                            "action": "park",
                            "parking_id": result["parking_id"],
                            "amount": park_amount,
                            "reason": "weak_trading_signal",
                        }
                    )

        return {
            "actions_taken": len(actions),
            "actions": actions,
            "trading_signal_strength": trading_signal_strength,
        }

    def _log_parking_action(self, parking_record: dict[str, Any]):
        """Log parking action."""
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": "park",
            "parking_id": parking_record["id"],
            "amount": parking_record["amount"],
            "protocol": parking_record["protocol_name"],
            "apy": parking_record["apy"],
            "lock_period": parking_record["lock_period"],
        }
        self.yield_history.append(log_entry)

    def _log_withdrawal_action(self, parking_record: dict[str, Any]):
        """Log withdrawal action."""
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": "withdraw",
            "parking_id": parking_record["id"],
            "amount": parking_record["amount"],
            "earned_yield": parking_record["earned_yield"],
            "total_returned": (parking_record["amount"] + parking_record["earned_yield"]),
        }
        self.yield_history.append(log_entry)

    def get_yield_history(self, limit: int = 50) -> list[dict[str, Any]]:
        """Get yield operation history."""
        return self.yield_history[-limit:]


# Convenience functions
def park_in_yield(usdt_amount: float, protocol_id: str = None) -> dict[str, Any]:
    """
    Simple function to park capital in yield.

    Args:
        usdt_amount: Amount to park
        protocol_id: Protocol to use

    Returns:
        Parking result
    """
    rotator = YieldRotator()
    return rotator.park_capital(usdt_amount, protocol_id)


def exit_yield(parking_id: str) -> dict[str, Any]:
    """
    Simple function to exit yield position.

    Args:
        parking_id: Parking record ID

    Returns:
        Withdrawal result
    """
    rotator = YieldRotator()
    return rotator.withdraw_capital(parking_id)


def auto_rotate_yield(total_capital: float, trading_signal: float) -> dict[str, Any]:
    """
    Auto-rotate capital based on trading signals.

    Args:
        total_capital: Total capital
        trading_signal: Trading signal strength (0-1)

    Returns:
        Rotation actions
    """
    rotator = YieldRotator()
    return rotator.auto_rotate_capital(total_capital, trading_signal)


# Example usage
if __name__ == "__main__":
    print("ðŸ’° Yield Rotation Engine")
    print("=" * 40)

    # Test yield rotation
    rotator = YieldRotator()

    # Park some capital
    print("\nðŸ“ˆ Parking capital...")
    result = rotator.park_capital(1000, lock_period=30)
    if result["success"]:
        parking_id = result["parking_id"]

        # Check summary
        summary = rotator.get_parked_capital_summary()
        print(f"ðŸ’° Total parked: ${summary['total_parked']}")
        print(f"ðŸ“ˆ Total earned: ${summary['total_earned']}")

        # Test auto-rotation
        print("\nðŸ”„ Testing auto-rotation...")
        rotation = rotator.auto_rotate_capital(10000, trading_signal_strength=0.8)
        print(f"Actions taken: {rotation['actions_taken']}")

    print("\nðŸŽ¯ Yield rotation testing complete!")

