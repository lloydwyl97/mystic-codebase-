#!/usr/bin/env python3
"""
AI Trade Engine Service
Port 8004 - Standalone AI trading engine service
"""

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Any

import redis
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

# Add backend directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# import backend.ai as ai trade engine
from ai_trade_engine import AITradeEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/ai_trade_engine.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("ai_trade_engine_service")

# Initialize FastAPI app
app = FastAPI(
    title="AI Trade Engine Service",
    description="Standalone AI trading engine service",
    version="1.0.0",
)


class AITradeEngineService:
    """AI Trade Engine Service"""

    def __init__(self):
        """Initialize the service"""
        self.trade_engine = AITradeEngine()
        self.running = False
        self.trade_history = []

        # Initialize Redis connection
        self.redis_client = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            db=int(os.getenv("REDIS_DB", 0)),
            decode_responses=True,
        )

        logger.info("âœ… AI Trade Engine Service initialized")

    async def start(self):
        """Start the service"""
        logger.info("ðŸš€ Starting AI Trade Engine Service...")
        self.running = True

        # Start the trade engine
        await self.trade_engine.start()

        # Start trade monitoring loop
        asyncio.create_task(self.trade_monitor_loop())

    async def stop(self):
        """Stop the service"""
        logger.info("ðŸ›‘ Stopping AI Trade Engine Service...")
        self.running = False
        await self.trade_engine.stop()

    async def trade_monitor_loop(self):
        """Monitor for trade requests"""
        logger.info("ðŸ‘€ Starting trade monitor loop...")

        while self.running:
            try:
                # Check for trade requests
                from utils.redis_helpers import to_str
                request = to_str(self.redis_client.lpop("ai_trade_queue"))

                if request:
                    request_data = json.loads(request)
                    await self.process_trade_request(request_data)

                # Wait before next check
                await asyncio.sleep(5)

            except Exception as e:
                logger.error(f"âŒ Error in trade monitor loop: {e}")
                await asyncio.sleep(30)

    async def process_trade_request(self, request_data: dict[str, Any]):
        """Process a trade request"""
        try:
            symbol = request_data.get("symbol", "ETHUSDT")
            side = request_data.get("side", "BUY")
            quantity = request_data.get("quantity", 0.01)
            price = request_data.get("price")
            strategy_id = request_data.get("strategy_id", "unknown")

            logger.info(f"ðŸŽ¯ Processing trade request for {symbol} {side} {quantity}")

            # Execute trade using the trade engine
            trade_result = await self.trade_engine.execute_trade(symbol, side, quantity, price)

            # Record trade
            trade_record = {
                "strategy_id": strategy_id,
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "price": price,
                "result": trade_result,
                "timestamp": datetime.now().isoformat(),
                "status": ("success" if trade_result and "error" not in trade_result else "failed"),
            }

            # Store trade record
            self.trade_history.append(trade_record)
            self.redis_client.set(
                f"trade:{strategy_id}:{int(time.time())}",
                json.dumps(trade_record),
            )

            # Publish result
            self.redis_client.lpush("trade_results", json.dumps(trade_record))

            logger.info(f"âœ… Trade processed: {trade_record['status']}")

            return trade_record

        except Exception as e:
            logger.error(f"âŒ Error processing trade request: {e}")

            # Record failed trade
            trade_record = {
                "strategy_id": request_data.get("strategy_id", "unknown"),
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "status": "failed",
            }

            self.trade_history.append(trade_record)
            return trade_record

    async def execute_trade(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float | None = None,
    ) -> dict[str, Any]:
        """Execute a trade"""
        try:
            logger.info(f"ðŸŽ¯ Executing trade: {symbol} {side} {quantity}")

            # Execute trade using the trade engine
            result = await self.trade_engine.execute_trade(symbol, side, quantity, price)

            # Record trade
            trade_record = {
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "price": price,
                "result": result,
                "timestamp": datetime.now().isoformat(),
                "status": ("success" if result and "error" not in result else "failed"),
            }

            # Store trade record
            self.trade_history.append(trade_record)

            logger.info(f"âœ… Trade executed: {trade_record['status']}")

            return trade_record

        except Exception as e:
            logger.error(f"âŒ Error executing trade: {e}")
            raise

    async def get_trade_history(self, limit: int = 100) -> list[dict[str, Any]]:
        """Get trade history"""
        try:
            # Return recent trades
            return self.trade_history[-limit:] if self.trade_history else []
        except Exception as e:
            logger.error(f"âŒ Error getting trade history: {e}")
            return []

    async def get_strategy_trades(self, strategy_id: str) -> list[dict[str, Any]]:
        """Get trades for a specific strategy"""
        try:
            trades = []
            for key in self.redis_client.scan_iter(f"trade:{strategy_id}:*"):
                trade_data = self.redis_client.get(key)
                if trade_data:
                    trades.append(json.loads(trade_data))
            return trades
        except Exception as e:
            logger.error(f"âŒ Error getting strategy trades: {e}")
            return []

    async def get_portfolio_status(self) -> dict[str, Any]:
        """Get current portfolio status"""
        try:
            # Get portfolio from trade engine
            portfolio = await self.trade_engine.get_portfolio()

            return {
                "portfolio": portfolio,
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            logger.error(f"âŒ Error getting portfolio status: {e}")
            return {"error": str(e)}


# Global service instance
trade_service = None


@app.on_event("startup")
async def startup_event():
    """Startup event - initialize service"""
    global trade_service
    try:
        trade_service = AITradeEngineService()
        await trade_service.start()
        logger.info("âœ… AI Trade Engine Service started")
    except Exception as e:
        logger.error(f"âŒ Failed to start AI Trade Engine Service: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event - stop service"""
    global trade_service
    if trade_service:
        await trade_service.stop()
        logger.info("âœ… AI Trade Engine Service stopped")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return JSONResponse(
        status_code=200,
        content={
            "status": "healthy",
            "service": "ai-trade-engine",
            "timestamp": datetime.now().isoformat(),
            "running": trade_service.running if trade_service else False,
        },
    )


@app.get("/status")
async def service_status():
    """Get service status"""
    if not trade_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    return {
        "status": "running" if trade_service.running else "stopped",
        "redis_connected": trade_service.redis_client.ping(),
        "trade_count": len(trade_service.trade_history),
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/trade")
async def execute_trade(
    symbol: str,
    side: str,
    quantity: float,
    price: float | None = None,
    strategy_id: str | None = None,
):
    """Execute a trade"""
    if not trade_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        trade_record = await trade_service.execute_trade(symbol, side, quantity, price)

        # Add strategy_id if provided
        if strategy_id:
            trade_record["strategy_id"] = strategy_id

        return {
            "status": "success",
            "trade": trade_record,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"âŒ Error in trade endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/trades")
async def get_trades(limit: int = 100):
    """Get trade history"""
    if not trade_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        trades = await trade_service.get_trade_history(limit)
        return {
            "trades": trades,
            "count": len(trades),
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"âŒ Error getting trades: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/trades/{strategy_id}")
async def get_strategy_trades(strategy_id: str):
    """Get trades for a specific strategy"""
    if not trade_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        trades = await trade_service.get_strategy_trades(strategy_id)
        return {
            "strategy_id": strategy_id,
            "trades": trades,
            "count": len(trades),
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"âŒ Error getting strategy trades: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/portfolio")
async def get_portfolio():
    """Get portfolio status"""
    if not trade_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        portfolio = await trade_service.get_portfolio_status()
        return portfolio
    except Exception as e:
        logger.error(f"âŒ Error getting portfolio: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/queue/process")
async def process_trade_queue():
    """Process trade queue"""
    if not trade_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        processed = 0
        while True:
            from utils.redis_helpers import to_str
            request = to_str(trade_service.redis_client.lpop("ai_trade_queue"))
            if not request:
                break

            request_data = json.loads(request)
            await trade_service.process_trade_request(request_data)
            processed += 1

        return {
            "status": "success",
            "processed": processed,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"âŒ Error processing trade queue: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # Get port from environment
    port = int(os.getenv("SERVICE_PORT", 8004))

    logger.info(f"ðŸš€ Starting AI Trade Engine Service on port {port}")

    # Start the FastAPI server
    uvicorn.run(
        "ai_trade_engine_service:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
        reload=False,
    )


