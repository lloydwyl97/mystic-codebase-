#!/usr/bin/env python3
"""
AI Leaderboard Executor Service
Port 8005 - Standalone AI leaderboard execution service
"""

import asyncio
import json
import time
import os
import sys
import logging
from datetime import datetime
from typing import Dict, Any, List
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import redis

# Add backend directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import AI leaderboard executor
from ai_leaderboard_executor import AILeaderboardExecutor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/ai_leaderboard_executor.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("ai_leaderboard_executor_service")

# Initialize FastAPI app
app = FastAPI(
    title="AI Leaderboard Executor Service",
    description="Standalone AI leaderboard execution service",
    version="1.0.0",
)


class AILeaderboardExecutorService:
    """AI Leaderboard Executor Service"""

    def __init__(self):
        """Initialize the service"""
        self.executor = AILeaderboardExecutor()
        self.running = False
        self.leaderboard_history = []

        # Initialize Redis connection
        self.redis_client = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            db=int(os.getenv("REDIS_DB", 0)),
            decode_responses=True,
        )

        logger.info("✅ AI Leaderboard Executor Service initialized")

    async def start(self):
        """Start the service"""
        logger.info("🚀 Starting AI Leaderboard Executor Service...")
        self.running = True

        # Start the executor
        await self.executor.start()

        # Start leaderboard monitoring loop
        asyncio.create_task(self.leaderboard_monitor_loop())

    async def stop(self):
        """Stop the service"""
        logger.info("🛑 Stopping AI Leaderboard Executor Service...")
        self.running = False
        await self.executor.stop()

    async def leaderboard_monitor_loop(self):
        """Monitor for leaderboard execution requests"""
        logger.info("👀 Starting leaderboard monitor loop...")

        while self.running:
            try:
                # Check for leaderboard requests
                request = self.redis_client.lpop("ai_leaderboard_queue")

                if request:
                    request_data = json.loads(request)
                    await self.process_leaderboard_request(request_data)

                # Wait before next check
                await asyncio.sleep(30)

            except Exception as e:
                logger.error(f"❌ Error in leaderboard monitor loop: {e}")
                await asyncio.sleep(60)

    async def process_leaderboard_request(self, request_data: Dict[str, Any]):
        """Process a leaderboard request"""
        try:
            strategy_id = request_data.get("strategy_id", "unknown")
            symbol = request_data.get("symbol", "ETHUSDT")
            timeframe = request_data.get("timeframe", "1h")

            logger.info(f"🏆 Processing leaderboard request for strategy {strategy_id}")

            # Execute leaderboard analysis
            leaderboard_result = await self.executor.execute_leaderboard_analysis(
                strategy_id, symbol, timeframe
            )

            # Record leaderboard execution
            leaderboard_record = {
                "strategy_id": strategy_id,
                "symbol": symbol,
                "timeframe": timeframe,
                "result": leaderboard_result,
                "timestamp": datetime.now().isoformat(),
                "status": (
                    "success"
                    if leaderboard_result and "error" not in leaderboard_result
                    else "failed"
                ),
            }

            # Store leaderboard record
            self.leaderboard_history.append(leaderboard_record)
            self.redis_client.set(
                f"leaderboard:{strategy_id}:{int(time.time())}",
                json.dumps(leaderboard_record),
            )

            # Publish result
            self.redis_client.lpush("leaderboard_results", json.dumps(leaderboard_record))

            logger.info(f"✅ Leaderboard processed: {leaderboard_record['status']}")

            return leaderboard_record

        except Exception as e:
            logger.error(f"❌ Error processing leaderboard request: {e}")

            # Record failed leaderboard execution
            leaderboard_record = {
                "strategy_id": request_data.get("strategy_id", "unknown"),
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "status": "failed",
            }

            self.leaderboard_history.append(leaderboard_record)
            return leaderboard_record

    async def execute_leaderboard_analysis(
        self, strategy_id: str, symbol: str, timeframe: str
    ) -> Dict[str, Any]:
        """Execute leaderboard analysis"""
        try:
            logger.info(f"🏆 Executing leaderboard analysis for {strategy_id}")

            # Execute leaderboard analysis
            result = await self.executor.execute_leaderboard_analysis(
                strategy_id, symbol, timeframe
            )

            # Record leaderboard execution
            leaderboard_record = {
                "strategy_id": strategy_id,
                "symbol": symbol,
                "timeframe": timeframe,
                "result": result,
                "timestamp": datetime.now().isoformat(),
                "status": ("success" if result and "error" not in result else "failed"),
            }

            # Store leaderboard record
            self.leaderboard_history.append(leaderboard_record)

            logger.info(f"✅ Leaderboard analysis executed: {leaderboard_record['status']}")

            return leaderboard_record

        except Exception as e:
            logger.error(f"❌ Error executing leaderboard analysis: {e}")
            raise

    async def get_leaderboard_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get leaderboard history"""
        try:
            # Return recent leaderboard executions
            return self.leaderboard_history[-limit:] if self.leaderboard_history else []
        except Exception as e:
            logger.error(f"❌ Error getting leaderboard history: {e}")
            return []

    async def get_strategy_leaderboard(self, strategy_id: str) -> List[Dict[str, Any]]:
        """Get leaderboard for a specific strategy"""
        try:
            leaderboards = []
            for key in self.redis_client.scan_iter(f"leaderboard:{strategy_id}:*"):
                leaderboard_data = self.redis_client.get(key)
                if leaderboard_data:
                    leaderboards.append(json.loads(leaderboard_data))
            return leaderboards
        except Exception as e:
            logger.error(f"❌ Error getting strategy leaderboard: {e}")
            return []

    async def get_current_leaderboard(self) -> Dict[str, Any]:
        """Get current leaderboard status"""
        try:
            # Get current leaderboard from executor
            leaderboard = await self.executor.get_current_leaderboard()

            return {
                "leaderboard": leaderboard,
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            logger.error(f"❌ Error getting current leaderboard: {e}")
            return {"error": str(e)}


# Global service instance
leaderboard_service = None


@app.on_event("startup")
async def startup_event():
    """Startup event - initialize service"""
    global leaderboard_service
    try:
        leaderboard_service = AILeaderboardExecutorService()
        await leaderboard_service.start()
        logger.info("✅ AI Leaderboard Executor Service started")
    except Exception as e:
        logger.error(f"❌ Failed to start AI Leaderboard Executor Service: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event - stop service"""
    global leaderboard_service
    if leaderboard_service:
        await leaderboard_service.stop()
        logger.info("✅ AI Leaderboard Executor Service stopped")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return JSONResponse(
        status_code=200,
        content={
            "status": "healthy",
            "service": "ai-leaderboard-executor",
            "timestamp": datetime.now().isoformat(),
            "running": (leaderboard_service.running if leaderboard_service else False),
        },
    )


@app.get("/status")
async def service_status():
    """Get service status"""
    if not leaderboard_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    return {
        "status": "running" if leaderboard_service.running else "stopped",
        "redis_connected": leaderboard_service.redis_client.ping(),
        "leaderboard_count": len(leaderboard_service.leaderboard_history),
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/analyze")
async def execute_leaderboard_analysis(
    strategy_id: str, symbol: str = "ETHUSDT", timeframe: str = "1h"
):
    """Execute leaderboard analysis"""
    if not leaderboard_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        leaderboard_record = await leaderboard_service.execute_leaderboard_analysis(
            strategy_id, symbol, timeframe
        )

        return {
            "status": "success",
            "leaderboard": leaderboard_record,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"❌ Error in analyze endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/leaderboards")
async def get_leaderboards(limit: int = 100):
    """Get leaderboard history"""
    if not leaderboard_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        leaderboards = await leaderboard_service.get_leaderboard_history(limit)
        return {
            "leaderboards": leaderboards,
            "count": len(leaderboards),
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"❌ Error getting leaderboards: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/leaderboards/{strategy_id}")
async def get_strategy_leaderboard(strategy_id: str):
    """Get leaderboard for a specific strategy"""
    if not leaderboard_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        leaderboards = await leaderboard_service.get_strategy_leaderboard(strategy_id)
        return {
            "strategy_id": strategy_id,
            "leaderboards": leaderboards,
            "count": len(leaderboards),
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"❌ Error getting strategy leaderboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/current")
async def get_current_leaderboard():
    """Get current leaderboard"""
    if not leaderboard_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        leaderboard = await leaderboard_service.get_current_leaderboard()
        return leaderboard
    except Exception as e:
        logger.error(f"❌ Error getting current leaderboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/queue/process")
async def process_leaderboard_queue():
    """Process leaderboard queue"""
    if not leaderboard_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        processed = 0
        while True:
            request = leaderboard_service.redis_client.lpop("ai_leaderboard_queue")
            if not request:
                break

            request_data = json.loads(request)
            await leaderboard_service.process_leaderboard_request(request_data)
            processed += 1

        return {
            "status": "success",
            "processed": processed,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"❌ Error processing leaderboard queue: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # Get port from environment
    port = int(os.getenv("SERVICE_PORT", 8005))

    logger.info(f"🚀 Starting AI Leaderboard Executor Service on port {port}")

    # Start the FastAPI server
    uvicorn.run(
        "ai_leaderboard_executor_service:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
        reload=False,
    )
