#!/usr/bin/env python3
"""
AI Strategy Generator Service
Port 8002 - Standalone AI strategy generation service
"""

import json
import logging
import os
import sys
from datetime import datetime
from typing import Any

import redis
import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import JSONResponse

# Add backend directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# import backend.ai as ai strategy generator
from ai_strategy_generator import AIStrategyGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/ai_strategy_generator.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("ai_strategy_generator_service")

# Initialize FastAPI app
app = FastAPI(
    title="AI Strategy Generator Service",
    description="Standalone AI strategy generation service",
    version="1.0.0",
)


class AIStrategyGeneratorService:
    """AI Strategy Generator Service"""

    def __init__(self):
        """Initialize the service"""
        self.generator = AIStrategyGenerator()
        self.running = False

        # Initialize Redis connection
        self.redis_client = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            db=int(os.getenv("REDIS_DB", 0)),
            decode_responses=True,
        )

        logger.info("âœ… AI Strategy Generator Service initialized")

    async def start(self):
        """Start the service"""
        logger.info("ðŸš€ Starting AI Strategy Generator Service...")
        self.running = True

        # Start the generator
        await self.generator.start()

    async def stop(self):
        """Stop the service"""
        logger.info("ðŸ›‘ Stopping AI Strategy Generator Service...")
        self.running = False
        await self.generator.stop()

    async def generate_strategy(
        self,
        strategy_type: str,
        symbol: str,
        parameters: dict[str, Any] = None,
    ) -> dict[str, Any] | None:
        """Generate a new AI strategy"""
        try:
            logger.info(f"ðŸŽ¯ Generating {strategy_type} strategy for {symbol}")

            # Create strategy using the generator
            strategy = await self.generator.create_ai_strategy(
                strategy_type, symbol, parameters or {}
            )

            if strategy:
                # Store strategy in Redis
                self.redis_client.set(f"strategy:{strategy['id']}", json.dumps(strategy))

                # Publish to strategy queue
                self.redis_client.lpush("strategy_queue", json.dumps(strategy))

                logger.info(f"âœ… Generated strategy: {strategy['id']}")
                return strategy
            else:
                logger.warning(f"âš ï¸ Failed to generate strategy for {symbol}")
                return None

        except Exception as e:
            logger.error(f"âŒ Error generating strategy: {e}")
            raise

    async def get_strategy_status(self, strategy_id: str) -> dict[str, Any] | None:
        """Get strategy status"""
        try:
            strategy_data = self.redis_client.get(f"strategy:{strategy_id}")
            if strategy_data:
                return json.loads(strategy_data)
            return None
        except Exception as e:
            logger.error(f"âŒ Error getting strategy status: {e}")
            return None

    async def list_strategies(self) -> list[dict[str, Any]]:
        """List all generated strategies"""
        try:
            strategies = []
            for key in self.redis_client.scan_iter("strategy:*"):
                strategy_data = self.redis_client.get(key)
                if strategy_data:
                    strategies.append(json.loads(strategy_data))
            return strategies
        except Exception as e:
            logger.error(f"âŒ Error listing strategies: {e}")
            return []


# Global service instance
strategy_service = None


@app.on_event("startup")
async def startup_event():
    """Startup event - initialize service"""
    global strategy_service
    try:
        strategy_service = AIStrategyGeneratorService()
        await strategy_service.start()
        logger.info("âœ… AI Strategy Generator Service started")
    except Exception as e:
        logger.error(f"âŒ Failed to start AI Strategy Generator Service: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event - stop service"""
    global strategy_service
    if strategy_service:
        await strategy_service.stop()
        logger.info("âœ… AI Strategy Generator Service stopped")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return JSONResponse(
        status_code=200,
        content={
            "status": "healthy",
            "service": "ai-strategy-generator",
            "timestamp": datetime.now().isoformat(),
            "running": strategy_service.running if strategy_service else False,
        },
    )


@app.get("/status")
async def service_status():
    """Get service status"""
    if not strategy_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    return {
        "status": "running" if strategy_service.running else "stopped",
        "redis_connected": strategy_service.redis_client.ping(),
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/generate")
async def generate_strategy(
    strategy_type: str,
    symbol: str,
    parameters: dict[str, Any] = None,
    background_tasks: BackgroundTasks = None,
):
    """Generate a new AI strategy"""
    if not strategy_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        # Generate strategy
        strategy = await strategy_service.generate_strategy(strategy_type, symbol, parameters or {})

        if strategy:
            return {
                "status": "success",
                "strategy": strategy,
                "timestamp": datetime.now().isoformat(),
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to generate strategy")

    except Exception as e:
        logger.error(f"âŒ Error in generate endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/strategies")
async def list_strategies():
    """List all generated strategies"""
    if not strategy_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        strategies = await strategy_service.list_strategies()
        return {
            "strategies": strategies,
            "count": len(strategies),
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"âŒ Error listing strategies: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/strategies/{strategy_id}")
async def get_strategy(strategy_id: str):
    """Get specific strategy"""
    if not strategy_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        strategy = await strategy_service.get_strategy_status(strategy_id)
        if strategy:
            return strategy
        else:
            raise HTTPException(status_code=404, detail="Strategy not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"âŒ Error getting strategy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/queue/process")
async def process_strategy_queue():
    """Process strategy generation queue"""
    if not strategy_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        # Process queue
        processed = 0
        while True:
            from utils.redis_helpers import to_str
            request = to_str(strategy_service.redis_client.lpop("ai_strategy_queue"))
            if not request:
                break

            request_data = json.loads(request)
            await strategy_service.generate_strategy(
                request_data.get("type", "lstm"),
                request_data.get("symbol", "BTC/USDT"),
                request_data.get("parameters", {}),
            )
            processed += 1

        return {
            "status": "success",
            "processed": processed,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"âŒ Error processing queue: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # Get port from environment
    port = int(os.getenv("SERVICE_PORT", 8002))

    logger.info(f"ðŸš€ Starting AI Strategy Generator Service on port {port}")

    # Start the FastAPI server
    uvicorn.run(
        "ai_strategy_generator_service:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
        reload=False,
    )


