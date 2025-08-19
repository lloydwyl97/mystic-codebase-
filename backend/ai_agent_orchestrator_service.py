#!/usr/bin/env python3
"""
AI Agent Orchestrator Service
Port 8006 - Orchestrates all AI agents in a single service
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

# import backend.ai as ai agents
from backend.agents.advanced_ai_orchestrator import AdvancedAIOrchestrator
from backend.agents.agent_orchestrator import AgentOrchestrator
from backend.agents.ai_model_manager import AIModelManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/ai_agent_orchestrator.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("ai_agent_orchestrator_service")

# Initialize FastAPI app
app = FastAPI(
    title="AI Agent Orchestrator Service",
    description="Orchestrates all AI agents in a single service",
    version="1.0.0",
)


class AIAgentOrchestratorService:
    """AI Agent Orchestrator Service"""

    def __init__(self):
        """Initialize the service"""
        self.running = False
        self.agents = {}
        self.agent_history = []

        # Initialize Redis connection
        self.redis_client = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            db=int(os.getenv("REDIS_DB", 0)),
            decode_responses=True,
        )

        # Initialize AI agents
        try:
            self.agents["model_manager"] = AIModelManager()
            self.agents["orchestrator"] = AgentOrchestrator()
            self.agents["advanced_orchestrator"] = AdvancedAIOrchestrator()

            logger.info("âœ… All AI agents initialized successfully")
        except Exception as e:
            logger.error(f"âŒ Error initializing AI agents: {e}")
            raise

        logger.info("âœ… AI Agent Orchestrator Service initialized")

    async def start(self):
        """Start the service"""
        logger.info("ðŸš€ Starting AI Agent Orchestrator Service...")
        self.running = True

        # Start all agents
        for agent_name, agent in self.agents.items():
            try:
                if hasattr(agent, "start"):
                    await agent.start()
                logger.info(f"âœ… Started agent: {agent_name}")
            except Exception as e:
                logger.error(f"âŒ Error starting agent {agent_name}: {e}")

        # Start agent monitoring loop
        asyncio.create_task(self.agent_monitor_loop())

    async def stop(self):
        """Stop the service"""
        logger.info("ðŸ›‘ Stopping AI Agent Orchestrator Service...")
        self.running = False

        # Stop all agents
        for agent_name, agent in self.agents.items():
            try:
                if hasattr(agent, "stop"):
                    await agent.stop()
                logger.info(f"âœ… Stopped agent: {agent_name}")
            except Exception as e:
                logger.error(f"âŒ Error stopping agent {agent_name}: {e}")

    async def agent_monitor_loop(self):
        """Monitor for agent requests"""
        logger.info("ðŸ‘€ Starting agent monitor loop...")

        while self.running:
            try:
                # Check for agent requests
                from utils.redis_helpers import to_str
                request = to_str(self.redis_client.lpop("ai_agent_queue"))

                if request:
                    request_data = json.loads(request)
                    await self.process_agent_request(request_data)

                # Wait before next check
                await asyncio.sleep(10)

            except Exception as e:
                logger.error(f"âŒ Error in agent monitor loop: {e}")
                await asyncio.sleep(30)

    async def process_agent_request(self, request_data: dict[str, Any]):
        """Process an agent request"""
        try:
            agent_type = request_data.get("agent_type", "orchestrator")
            action = request_data.get("action", "process")
            data = request_data.get("data", {})
            request_id = request_data.get("request_id", f"req_{int(time.time())}")

            logger.info(f"ðŸ¤– Processing {agent_type} agent request: {action}")

            # Process with appropriate agent
            if agent_type == "model_manager":
                result = await self.agents["model_manager"].process_request(action, data)
            elif agent_type == "advanced_orchestrator":
                result = await self.agents["advanced_orchestrator"].process_request(action, data)
            else:
                result = await self.agents["orchestrator"].process_request(action, data)

            # Record agent execution
            agent_record = {
                "request_id": request_id,
                "agent_type": agent_type,
                "action": action,
                "data": data,
                "result": result,
                "timestamp": datetime.now().isoformat(),
                "status": ("success" if result and "error" not in result else "failed"),
            }

            # Store agent record
            self.agent_history.append(agent_record)
            self.redis_client.set(f"agent:{request_id}", json.dumps(agent_record))

            # Publish result
            self.redis_client.lpush("agent_results", json.dumps(agent_record))

            logger.info(f"âœ… Agent request processed: {agent_record['status']}")

            return agent_record

        except Exception as e:
            logger.error(f"âŒ Error processing agent request: {e}")

            # Record failed agent execution
            agent_record = {
                "request_id": request_data.get("request_id", f"req_{int(time.time())}"),
                "agent_type": request_data.get("agent_type", "unknown"),
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "status": "failed",
            }

            self.agent_history.append(agent_record)
            return agent_record

    async def execute_agent_action(
        self, agent_type: str, action: str, data: dict[str, Any] = None
    ) -> dict[str, Any]:
        """Execute an agent action"""
        try:
            logger.info(f"ðŸ¤– Executing {agent_type} agent action: {action}")

            # Execute with appropriate agent
            if agent_type == "model_manager":
                result = await self.agents["model_manager"].process_request(action, data or {})
            elif agent_type == "advanced_orchestrator":
                result = await self.agents["advanced_orchestrator"].process_request(
                    action, data or {}
                )
            else:
                result = await self.agents["orchestrator"].process_request(action, data or {})

            # Record agent execution
            agent_record = {
                "agent_type": agent_type,
                "action": action,
                "data": data or {},
                "result": result,
                "timestamp": datetime.now().isoformat(),
                "status": ("success" if result and "error" not in result else "failed"),
            }

            # Store agent record
            self.agent_history.append(agent_record)

            logger.info(f"âœ… Agent action executed: {agent_record['status']}")

            return agent_record

        except Exception as e:
            logger.error(f"âŒ Error executing agent action: {e}")
            raise

    async def get_agent_history(self, limit: int = 100) -> list[dict[str, Any]]:
        """Get agent history"""
        try:
            # Return recent agent executions
            return self.agent_history[-limit:] if self.agent_history else []
        except Exception as e:
            logger.error(f"âŒ Error getting agent history: {e}")
            return []

    async def get_agent_status(self) -> dict[str, Any]:
        """Get status of all agents"""
        try:
            agent_status = {}
            for agent_name, agent in self.agents.items():
                try:
                    if hasattr(agent, "get_status"):
                        status = await agent.get_status()
                    else:
                        status = {"status": "running"}
                    agent_status[agent_name] = status
                except Exception as e:
                    agent_status[agent_name] = {
                        "status": "error",
                        "error": str(e),
                    }

            return {
                "agents": agent_status,
                "total_agents": len(self.agents),
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            logger.error(f"âŒ Error getting agent status: {e}")
            return {"error": str(e)}


# Global service instance
agent_service = None


@app.on_event("startup")
async def startup_event():
    """Startup event - initialize service"""
    global agent_service
    try:
        agent_service = AIAgentOrchestratorService()
        await agent_service.start()
        logger.info("âœ… AI Agent Orchestrator Service started")
    except Exception as e:
        logger.error(f"âŒ Failed to start AI Agent Orchestrator Service: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event - stop service"""
    global agent_service
    if agent_service:
        await agent_service.stop()
        logger.info("âœ… AI Agent Orchestrator Service stopped")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return JSONResponse(
        status_code=200,
        content={
            "status": "healthy",
            "service": "ai-agent-orchestrator",
            "timestamp": datetime.now().isoformat(),
            "running": agent_service.running if agent_service else False,
        },
    )


@app.get("/status")
async def service_status():
    """Get service status"""
    if not agent_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        agent_status = await agent_service.get_agent_status()
        return {
            "status": "running" if agent_service.running else "stopped",
            "redis_connected": agent_service.redis_client.ping(),
            "agent_count": len(agent_service.agents),
            "agent_history_count": len(agent_service.agent_history),
            "agents": agent_status,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"âŒ Error getting service status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/execute")
async def execute_agent_action(agent_type: str, action: str, data: dict[str, Any] = None):
    """Execute an agent action"""
    if not agent_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        agent_record = await agent_service.execute_agent_action(agent_type, action, data or {})

        return {
            "status": "success",
            "agent": agent_record,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"âŒ Error in execute endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/agents")
async def get_agents():
    """Get all agents"""
    if not agent_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        agent_status = await agent_service.get_agent_status()
        return agent_status
    except Exception as e:
        logger.error(f"âŒ Error getting agents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history")
async def get_agent_history(limit: int = 100):
    """Get agent history"""
    if not agent_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        history = await agent_service.get_agent_history(limit)
        return {
            "history": history,
            "count": len(history),
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"âŒ Error getting agent history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/queue/process")
async def process_agent_queue():
    """Process agent queue"""
    if not agent_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        processed = 0
        while True:
            from utils.redis_helpers import to_str
            request = to_str(agent_service.redis_client.lpop("ai_agent_queue"))
            if not request:
                break

            request_data = json.loads(request)
            await agent_service.process_agent_request(request_data)
            processed += 1

        return {
            "status": "success",
            "processed": processed,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"âŒ Error processing agent queue: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # Get port from environment
    port = int(os.getenv("SERVICE_PORT", 8006))

    logger.info(f"ðŸš€ Starting AI Agent Orchestrator Service on port {port}")

    # Start the FastAPI server
    uvicorn.run(
        "ai_agent_orchestrator_service:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
        reload=False,
    )


