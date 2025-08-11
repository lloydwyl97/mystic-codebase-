#!/usr/bin/env python3
"""
AI Service - Main Entry Point
Port 8001 - AI processing and analysis service
"""

import asyncio
import os
import sys
import logging
from datetime import datetime
from typing import Dict, Any
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import redis

# Add the ai directory to the path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import AI modules (compat: support both ai.ai.* and ai.* layouts)
try:
    from ai.ai.ai_brains import AIBrains  # type: ignore[import-not-found]
    from ai.ai.ai_breakouts import AIBreakouts  # type: ignore[import-not-found]
    from ai.ai.ai_mystic import AIMystic  # type: ignore[import-not-found]
    from ai.ai.ai_signals import AISignals  # type: ignore[import-not-found]
    from ai.ai.ai_volume import AIVolume  # type: ignore[import-not-found]
    from ai.poller import AIPoller  # type: ignore[import-not-found]
    from ai.persistent_cache import PersistentCache  # type: ignore[import-not-found]
except Exception:
    from ai_brains import AIBrains  # type: ignore[no-redef]
    from ai_breakouts import AIBreakouts  # type: ignore[no-redef]
    from ai_mystic import AIMystic  # type: ignore[no-redef]
    from ai_signals import AISignals  # type: ignore[no-redef]
    from ai_volume import AIVolume  # type: ignore[no-redef]
    from poller import AIPoller  # type: ignore[no-redef]
    from persistent_cache import PersistentCache  # type: ignore[no-redef]

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/ai_service.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("ai_service")

# Initialize FastAPI app
app = FastAPI(
    title="Mystic AI Service",
    description="AI processing and analysis service for Mystic Trading Platform",
    version="1.0.0",
)


class AIService:
    """Main AI Service that coordinates all AI components"""

    def __init__(self):
        """Initialize AI Service with all components"""
        self.running = False
        self.components = {}

        # Initialize Redis connection
        self.redis_client = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            db=int(os.getenv("REDIS_DB", 0)),
            decode_responses=True,
        )

        # Initialize AI components
        try:
            self.components["brains"] = AIBrains()
            self.components["breakouts"] = AIBreakouts()
            self.components["mystic"] = AIMystic()
            self.components["signals"] = AISignals()
            self.components["volume"] = AIVolume()
            self.components["poller"] = AIPoller()
            self.components["cache"] = PersistentCache()

            logger.info("✅ All AI components initialized successfully")
        except Exception as e:
            logger.error(f"❌ Error initializing AI components: {e}")
            raise

    async def start(self):
        """Start the AI service"""
        logger.info("🚀 Starting AI Service...")
        self.running = True

        # Start AI processing loop
        await self.ai_processing_loop()

    async def stop(self):
        """Stop the AI service"""
        logger.info("🛑 Stopping AI Service...")
        self.running = False

    async def ai_processing_loop(self):
        """Main AI processing loop"""
        logger.info("🧠 Starting AI processing loop...")

        while self.running:
            try:
                # Process market data with AI brains
                await self.process_market_data()

                # Analyze breakouts
                await self.analyze_breakouts()

                # Process mystic signals
                await self.process_mystic_signals()

                # Analyze volume patterns
                await self.analyze_volume()

                # Update cache
                await self.update_cache()

                # Wait before next iteration
                await asyncio.sleep(60)  # Process every minute

            except Exception as e:
                logger.error(f"❌ Error in AI processing loop: {e}")
                await asyncio.sleep(30)  # Wait before retry

    async def process_market_data(self):
        """Process market data with AI brains"""
        try:
            # Get market data from Redis
            market_data = self.redis_client.get("market_data")
            if market_data:
                # Process with AI brains
                result = self.components["brains"].process_data(market_data)

                # Store result
                self.redis_client.set("ai_brains_result", str(result))
                logger.debug("✅ Market data processed by AI brains")
        except Exception as e:
            logger.error(f"❌ Error processing market data: {e}")

    async def analyze_breakouts(self):
        """Analyze market breakouts"""
        try:
            # Get recent price data
            price_data = self.redis_client.get("price_data")
            if price_data:
                # Analyze breakouts
                breakouts = self.components["breakouts"].detect_breakouts(price_data)

                # Store results
                self.redis_client.set("ai_breakouts", str(breakouts))
                logger.debug("✅ Breakouts analyzed")
        except Exception as e:
            logger.error(f"❌ Error analyzing breakouts: {e}")

    async def process_mystic_signals(self):
        """Process mystic trading signals"""
        try:
            # Get signal data
            signal_data = self.redis_client.get("signal_data")
            if signal_data:
                # Process with mystic AI
                signals = self.components["mystic"].process_signals(signal_data)

                # Store results
                self.redis_client.set("ai_mystic_signals", str(signals))
                logger.debug("✅ Mystic signals processed")
        except Exception as e:
            logger.error(f"❌ Error processing mystic signals: {e}")

    async def analyze_volume(self):
        """Analyze volume patterns"""
        try:
            # Get volume data
            volume_data = self.redis_client.get("volume_data")
            if volume_data:
                # Analyze volume patterns
                patterns = self.components["volume"].analyze_patterns(volume_data)

                # Store results
                self.redis_client.set("ai_volume_patterns", str(patterns))
                logger.debug("✅ Volume patterns analyzed")
        except Exception as e:
            logger.error(f"❌ Error analyzing volume: {e}")

    async def update_cache(self):
        """Update persistent cache"""
        try:
            # Update cache with latest results
            self.components["cache"].update()
            logger.debug("✅ Cache updated")
        except Exception as e:
            logger.error(f"❌ Error updating cache: {e}")


# Global AI service instance
ai_service = None


@app.on_event("startup")
async def startup_event():
    """Startup event - initialize AI service"""
    global ai_service
    try:
        ai_service = AIService()
        logger.info("✅ AI Service initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize AI Service: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event - stop AI service"""
    global ai_service
    if ai_service:
        await ai_service.stop()
        logger.info("✅ AI Service stopped")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return JSONResponse(
        status_code=200,
        content={
            "status": "healthy",
            "service": "ai",
            "timestamp": datetime.now().isoformat(),
            "components": (list(ai_service.components.keys()) if ai_service else []),
        },
    )


@app.get("/ai/status")
async def ai_status():
    """Get AI service status"""
    if not ai_service:
        raise HTTPException(status_code=503, detail="AI Service not initialized")

    return {
        "status": "running" if ai_service.running else "stopped",
        "components": list(ai_service.components.keys()),
        "redis_connected": ai_service.redis_client.ping(),
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/ai/process")
async def process_ai_request(request_data: Dict[str, Any]):
    """Process AI request"""
    if not ai_service:
        raise HTTPException(status_code=503, detail="AI Service not initialized")

    try:
        # Process the request based on type
        request_type = request_data.get("type", "general")

        if request_type == "market_data":
            await ai_service.process_market_data()
        elif request_type == "breakouts":
            await ai_service.analyze_breakouts()
        elif request_type == "signals":
            await ai_service.process_mystic_signals()
        elif request_type == "volume":
            await ai_service.analyze_volume()
        else:
            # General processing
            await ai_service.process_market_data()

        return {
            "status": "success",
            "type": request_type,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"❌ Error processing AI request: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ai/components")
async def get_ai_components():
    """Get available AI components"""
    if not ai_service:
        raise HTTPException(status_code=503, detail="AI Service not initialized")

    return {
        "components": list(ai_service.components.keys()),
        "status": "available",
        "timestamp": datetime.now().isoformat(),
    }


if __name__ == "__main__":
    # Get port from environment
    port = int(os.getenv("PORT", 8001))

    logger.info(f"🚀 Starting AI Service on port {port}")

    # Start the FastAPI server
    uvicorn.run("main:app", host="0.0.0.0", port=port, log_level="info", reload=False)
