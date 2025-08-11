#!/usr/bin/env python3
"""
AI Strategy Executor Service
Port 8003 - Standalone AI strategy execution service
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

# Import AI strategy execution
from ai_strategy_execution import execute_ai_strategy_signal

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/ai_strategy_executor.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("ai_strategy_executor_service")

# Initialize FastAPI app
app = FastAPI(
    title="AI Strategy Executor Service",
    description="Standalone AI strategy execution service",
    version="1.0.0",
)


class AIStrategyExecutorService:
    """AI Strategy Executor Service"""

    def __init__(self):
        """Initialize the service"""
        self.running = False
        self.execution_history = []

        # Initialize Redis connection
        self.redis_client = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            db=int(os.getenv("REDIS_DB", 0)),
            decode_responses=True,
        )

        logger.info("✅ AI Strategy Executor Service initialized")

    async def start(self):
        """Start the service"""
        logger.info("🚀 Starting AI Strategy Executor Service...")
        self.running = True

        # Start execution monitoring loop
        asyncio.create_task(self.execution_monitor_loop())

    async def stop(self):
        """Stop the service"""
        logger.info("🛑 Stopping AI Strategy Executor Service...")
        self.running = False

    async def execution_monitor_loop(self):
        """Monitor for strategy execution requests"""
        logger.info("👀 Starting execution monitor loop...")

        while self.running:
            try:
                # Check for execution requests
                request = self.redis_client.lpop("strategy_execution_queue")

                if request:
                    request_data = json.loads(request)
                    await self.execute_strategy_request(request_data)

                # Wait before next check
                await asyncio.sleep(10)

            except Exception as e:
                logger.error(f"❌ Error in execution monitor loop: {e}")
                await asyncio.sleep(30)

    async def execute_strategy_request(self, request_data: Dict[str, Any]):
        """Execute a strategy request"""
        try:
            symbol_binance = request_data.get("symbol_binance", "ETHUSDT")
            symbol_coinbase = request_data.get("symbol_coinbase", "ETH-USD")
            usd_amount = request_data.get("usd_amount", 50)
            signal = request_data.get("signal", True)
            strategy_id = request_data.get("strategy_id", "unknown")

            logger.info(f"🎯 Executing strategy {strategy_id} for {symbol_binance}")

            # Execute the strategy
            result = execute_ai_strategy_signal(symbol_binance, symbol_coinbase, usd_amount, signal)

            # Record execution
            execution_record = {
                "strategy_id": strategy_id,
                "symbol_binance": symbol_binance,
                "symbol_coinbase": symbol_coinbase,
                "usd_amount": usd_amount,
                "signal": signal,
                "result": result,
                "timestamp": datetime.now().isoformat(),
                "status": ("success" if result and "error" not in result else "failed"),
            }

            # Store execution record
            self.execution_history.append(execution_record)
            self.redis_client.set(
                f"execution:{strategy_id}:{int(time.time())}",
                json.dumps(execution_record),
            )

            # Publish result
            self.redis_client.lpush("execution_results", json.dumps(execution_record))

            logger.info(f"✅ Strategy {strategy_id} executed: {execution_record['status']}")

            return execution_record

        except Exception as e:
            logger.error(f"❌ Error executing strategy request: {e}")

            # Record failed execution
            execution_record = {
                "strategy_id": request_data.get("strategy_id", "unknown"),
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "status": "failed",
            }

            self.execution_history.append(execution_record)
            return execution_record

    async def execute_strategy(
        self,
        strategy_id: str,
        symbol_binance: str,
        symbol_coinbase: str,
        usd_amount: float,
        signal: bool,
    ) -> Dict[str, Any]:
        """Execute a specific strategy"""
        try:
            logger.info(f"🎯 Executing strategy {strategy_id}")

            # Execute the strategy
            result = execute_ai_strategy_signal(symbol_binance, symbol_coinbase, usd_amount, signal)

            # Record execution
            execution_record = {
                "strategy_id": strategy_id,
                "symbol_binance": symbol_binance,
                "symbol_coinbase": symbol_coinbase,
                "usd_amount": usd_amount,
                "signal": signal,
                "result": result,
                "timestamp": datetime.now().isoformat(),
                "status": ("success" if result and "error" not in result else "failed"),
            }

            # Store execution record
            self.execution_history.append(execution_record)
            self.redis_client.set(
                f"execution:{strategy_id}:{int(time.time())}",
                json.dumps(execution_record),
            )

            logger.info(f"✅ Strategy {strategy_id} executed: {execution_record['status']}")

            return execution_record

        except Exception as e:
            logger.error(f"❌ Error executing strategy: {e}")
            raise

    async def get_execution_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get execution history"""
        try:
            # Return recent executions
            return self.execution_history[-limit:] if self.execution_history else []
        except Exception as e:
            logger.error(f"❌ Error getting execution history: {e}")
            return []

    async def get_strategy_executions(self, strategy_id: str) -> List[Dict[str, Any]]:
        """Get executions for a specific strategy"""
        try:
            executions = []
            for key in self.redis_client.scan_iter(f"execution:{strategy_id}:*"):
                execution_data = self.redis_client.get(key)
                if execution_data:
                    executions.append(json.loads(execution_data))
            return executions
        except Exception as e:
            logger.error(f"❌ Error getting strategy executions: {e}")
            return []


# Global service instance
executor_service = None


@app.on_event("startup")
async def startup_event():
    """Startup event - initialize service"""
    global executor_service
    try:
        executor_service = AIStrategyExecutorService()
        await executor_service.start()
        logger.info("✅ AI Strategy Executor Service started")
    except Exception as e:
        logger.error(f"❌ Failed to start AI Strategy Executor Service: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event - stop service"""
    global executor_service
    if executor_service:
        await executor_service.stop()
        logger.info("✅ AI Strategy Executor Service stopped")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return JSONResponse(
        status_code=200,
        content={
            "status": "healthy",
            "service": "ai-strategy-executor",
            "timestamp": datetime.now().isoformat(),
            "running": executor_service.running if executor_service else False,
        },
    )


@app.get("/status")
async def service_status():
    """Get service status"""
    if not executor_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    return {
        "status": "running" if executor_service.running else "stopped",
        "redis_connected": executor_service.redis_client.ping(),
        "execution_count": len(executor_service.execution_history),
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/execute")
async def execute_strategy(
    strategy_id: str,
    symbol_binance: str = "ETHUSDT",
    symbol_coinbase: str = "ETH-USD",
    usd_amount: float = 50.0,
    signal: bool = True,
):
    """Execute a strategy"""
    if not executor_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        execution_record = await executor_service.execute_strategy(
            strategy_id, symbol_binance, symbol_coinbase, usd_amount, signal
        )

        return {
            "status": "success",
            "execution": execution_record,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"❌ Error in execute endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/executions")
async def get_executions(limit: int = 100):
    """Get execution history"""
    if not executor_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        executions = await executor_service.get_execution_history(limit)
        return {
            "executions": executions,
            "count": len(executions),
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"❌ Error getting executions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/executions/{strategy_id}")
async def get_strategy_executions(strategy_id: str):
    """Get executions for a specific strategy"""
    if not executor_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        executions = await executor_service.get_strategy_executions(strategy_id)
        return {
            "strategy_id": strategy_id,
            "executions": executions,
            "count": len(executions),
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"❌ Error getting strategy executions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/queue/process")
async def process_execution_queue():
    """Process execution queue"""
    if not executor_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        processed = 0
        while True:
            request = executor_service.redis_client.lpop("strategy_execution_queue")
            if not request:
                break

            request_data = json.loads(request)
            await executor_service.execute_strategy_request(request_data)
            processed += 1

        return {
            "status": "success",
            "processed": processed,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"❌ Error processing execution queue: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # Get port from environment
    port = int(os.getenv("SERVICE_PORT", 8003))

    logger.info(f"🚀 Starting AI Strategy Executor Service on port {port}")

    # Start the FastAPI server
    uvicorn.run(
        "ai_strategy_executor_service:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
        reload=False,
    )
