import json
import os
import sqlite3
from datetime import datetime
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException

router = APIRouter(tags=["wallet-dashboard"])

# Database paths
SIM_DB_PATH = os.getenv("SIM_DB_PATH", "simulation_trades.db")
MODEL_STATE_FILE = os.getenv("MODEL_STATE_PATH", "ai_model_state.json")


def get_db_connection():
    """Create database connection"""
    return sqlite3.connect(SIM_DB_PATH)


@router.get("/wallets/summary")
async def get_wallet_summary() -> List[Dict[str, Any]]:
    """Get summary of all wallet allocations and balances"""
    try:
        # Attempt to load from a real database table 'wallets'
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT wallet_name, total, allocation, last, status FROM wallets")
        rows = cursor.fetchall()
        conn.close()
        if not rows:
            # Return empty wallet data instead of raising error
            return {
                "total_balance": 0.0,
                "wallets": {},
                "last_updated": datetime.now().isoformat(),
            }
        wallets = [
            {
                "wallet_name": row[0],
                "total": row[1],
                "allocation": row[2],
                "last": row[3],
                "status": row[4],
            }
            for row in rows
        ]
        return wallets
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching wallet data: {str(e)}")


@router.get("/yield/leaderboard")
async def get_yield_leaderboard() -> List[Dict[str, Any]]:
    """Get DeFi yield leaderboard with APY comparisons"""
    try:
        # Attempt to load from a real database table 'yields'
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT provider, avg_apy, total_deployed, protocol, risk_level FROM yields")
        rows = cursor.fetchall()
        conn.close()
        if not rows:
            # Return empty yield data instead of raising error
            return {
                "total_yield": 0.0,
                "yield_sources": {},
                "last_updated": datetime.now().isoformat(),
            }
        yield_data = [
            {
                "provider": row[0],
                "avg_apy": row[1],
                "total_deployed": row[2],
                "protocol": row[3],
                "risk_level": row[4],
            }
            for row in rows
        ]
        return yield_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching yield data: {str(e)}")


@router.get("/staking/summary")
async def get_staking_summary() -> Dict[str, Any]:
    """Get staking summary across all platforms"""
    try:
        # Attempt to load from a real database table 'staking'
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT platform, amount_staked, apy, status, last_updated FROM staking")
        rows = cursor.fetchall()
        conn.close()
        if not rows:
            # Return empty staking data instead of raising error
            return {
                "total_staked": 0.0,
                "staking_sources": {},
                "last_updated": datetime.now().isoformat(),
            }
        staking_data = {
            row[0]: {
                "amount_staked": row[1],
                "apy": row[2],
                "status": row[3],
                "last_updated": row[4],
            }
            for row in rows
        }
        staking_data["total_staked"] = sum(row[1] for row in rows)
        staking_data["total_earned"] = None  # Implement real calculation
        return staking_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching staking data: {str(e)}")


@router.get("/coldwallet/status")
async def get_cold_wallet_status() -> Dict[str, Any]:
    """Get cold wallet sync status and metrics"""
    try:
        # Attempt to load from a real file or database table 'cold_wallet'
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT total_sent, last_sync, threshold, sync_count, status, address FROM cold_wallet LIMIT 1"
        )
        row = cursor.fetchone()
        conn.close()
        if not row:
            # Return empty cold wallet data instead of raising error
            return {
                "total_sent": 0.0,
                "last_sync": datetime.now().isoformat(),
                "threshold": 1000.0,
                "sync_count": 0,
                "status": "inactive",
                "address": "0x0000000000000000000000000000000000000000",
            }
        cold_wallet_data = {
            "total_sent": row[0],
            "last_sync": row[1],
            "threshold": row[2],
            "sync_count": row[3],
            "status": row[4],
            "address": row[5],
        }
        return cold_wallet_data
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching cold wallet data: {str(e)}",
        )


@router.get("/ai/dashboard")
async def get_ai_dashboard() -> Dict[str, Any]:
    """Get AI trading engine status and performance"""
    try:
        # Load AI model state
        model_state = {}
        if os.path.exists(MODEL_STATE_FILE):
            with open(MODEL_STATE_FILE, "r") as f:
                model_state = json.load(f)

        # Get performance summary from database
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT COUNT(*), AVG(simulated_profit), SUM(simulated_profit) FROM simulated_trades"
        )
        count, avg_profit, total_profit = cursor.fetchone()
        conn.close()

        ai_data = {
            "model_state": {
                "mode": model_state.get("mode", "training"),
                "confidence_threshold": model_state.get("confidence_threshold", 0.75),
                "adjustment_count": model_state.get("adjustment_count", 0),
                "last_update": model_state.get("last_update"),
            },
            "performance_summary": {
                "total_trades": count or 0,
                "avg_profit": avg_profit or 0.0,
                "total_profit": total_profit or 0.0,
            },
            "system_status": "online",
            "last_heartbeat": datetime.now(datetime.timezone.utc).isoformat(),
        }
        return ai_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching AI data: {str(e)}")


@router.get("/trades/recent")
async def get_recent_trades(limit: int = 20) -> List[Dict[str, Any]]:
    """Get recent trading activity for charting"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT timestamp, symbol, action, price, simulated_profit
            FROM simulated_trades
            ORDER BY timestamp DESC
            LIMIT ?
        """,
            (limit,),
        )

        trades = []
        for row in cursor.fetchall():
            trades.append(
                {
                    "timestamp": row[0],
                    "symbol": row[1],
                    "action": row[2],
                    "price": row[3],
                    "profit": row[4],
                }
            )
        conn.close()

        # If no real trades, return empty list
        if not trades:
            trades = []

        return trades
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching trade data: {str(e)}")


@router.get("/portfolio/overview")
async def get_portfolio_overview() -> Dict[str, Any]:
    """Get complete portfolio overview"""
    try:
        # Aggregate data from all endpoints
        wallets = await get_wallet_summary()
        yield_data = await get_yield_leaderboard()
        staking = await get_staking_summary()
        cold_status = await get_cold_wallet_status()
        ai_data = await get_ai_dashboard()

        total_value = sum(wallet["total"] for wallet in wallets)
        total_yield = sum(yield_item["total_deployed"] for yield_item in yield_data)
        total_staked = staking["total_staked"]

        portfolio_data = {
            "total_portfolio_value": total_value,
            "total_yield_deployed": total_yield,
            "total_staked": total_staked,
            "total_cold_storage": cold_status["total_sent"],
            "ai_performance": ai_data["performance_summary"]["total_profit"],
            "last_updated": datetime.now(datetime.timezone.utc).isoformat(),
            "breakdown": {
                "trading_wallets": sum(
                    w["total"] for w in wallets if "Trading" in w["wallet_name"]
                ),
                "yield_farming": total_yield,
                "staking": total_staked,
                "cold_storage": cold_status["total_sent"],
            },
        }
        return portfolio_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching portfolio data: {str(e)}")


@router.post("/wallet/allocate")
async def allocate_funds(amount: float, wallet_name: str) -> Dict[str, Any]:
    """Allocate funds to a specific wallet"""
    try:
        # Real allocation using wallet service
        from services.wallet_service import get_wallet_service

        wallet_service = get_wallet_service()
        allocation_result = await wallet_service.allocate_funds(amount, wallet_name)
        return allocation_result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error allocating funds: {str(e)}")


@router.post("/yield/rotate")
async def rotate_yield_funds() -> Dict[str, Any]:
    """Rotate funds to highest yielding protocol"""
    try:
        # Real rotation using yield service
        from services.yield_service import get_yield_service

        yield_service = get_yield_service()
        rotation_result = await yield_service.rotate_funds()
        return rotation_result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error rotating yield funds: {str(e)}")
