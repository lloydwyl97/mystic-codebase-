"""
Agent Orchestrator
Coordinates all AI agents in the Mystic AI Trading Platform
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from typing import Dict, Any, List, Optional
import redis

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class AgentOrchestrator:
    """Main orchestrator for all AI agents"""

    def __init__(self) -> None:
        self.agents: Dict[str, Any] = {}
        self.running: bool = False
        self.redis_client: Optional[redis.Redis] = None
        self.agent_tasks: List[asyncio.Task] = []

        # Initialize Redis connection
        try:
            self.redis_client = redis.Redis(
                host=os.getenv("REDIS_HOST", "localhost"),
                port=int(os.getenv("REDIS_PORT", 6379)),
                db=int(os.getenv("REDIS_DB", 0)),
                decode_responses=True,
            )
            self.redis_client.ping()
            print("âœ… Redis connection established")
        except Exception as e:
            print(f"âš ï¸ Redis not available: {e}")
            self.redis_client = None

    async def initialize_agents(self) -> None:
        """Initialize all AI agents"""
        try:
            print("ðŸš€ Initializing AI Agents...")

            # Phase 5 Agents
            await self.initialize_phase5_agents()

            # Other AI Agents
            await self.initialize_other_agents()

            print(f"âœ… Initialized {len(self.agents)} agents")

        except Exception as e:
            print(f"âŒ Error initializing agents: {e}")

    async def initialize_phase5_agents(self) -> None:
        """Initialize Phase 5 interdimensional agents"""
        try:
            # Import Phase 5 agents
            from backend.agents.interdimensional_signal_decoder import (
                InterdimensionalSignalDecoder,
            )  # noqa: E402
            from backend.agents.neuro_synchronization_engine import (
                NeuroSynchronizationEngine,
            )  # noqa: E402
            from backend.agents.cosmic_pattern_recognizer import (
                CosmicPatternRecognizer,
            )  # noqa: E402
            from backend.agents.auranet_channel_interface import (
                AuraNetChannelInterface,
            )  # noqa: E402

            # Create Phase 5 agents
            self.agents["interdimensional_signal_decoder"] = InterdimensionalSignalDecoder()
            self.agents["neuro_synchronization_engine"] = NeuroSynchronizationEngine()
            self.agents["cosmic_pattern_recognizer"] = CosmicPatternRecognizer()
            self.agents["auranet_channel_interface"] = AuraNetChannelInterface()

            print("âœ… Phase 5 agents initialized")

        except Exception as e:
            print(f"âš ï¸ Phase 5 agents not available: {e}")

    async def initialize_other_agents(self) -> None:
        """Initialize other AI agents"""
        try:
            # Import other agents (if available)
            agent_modules = [
                "agents.strategy_agent",
                "agents.market_sentiment_agent",
                "agents.news_sentiment_agent",
                "agents.social_media_agent",
                "agents.chart_pattern_agent",
                "agents.technical_indicator_agent",
                "agents.market_visualization_agent",
                "agents.deep_learning_agent",
                "agents.reinforcement_learning_agent",
                "agents.ai_model_manager",
                "agents.quantum_algorithm_engine",
                "agents.quantum_machine_learning_agent",
                "agents.quantum_optimization_agent",
            ]

            for module_name in agent_modules:
                try:
                    module = __import__(module_name, fromlist=["*"])
                    # Try to find the main agent class
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        if hasattr(attr, "__bases__") and "BaseAgent" in [
                            base.__name__ for base in attr.__bases__
                        ]:
                            agent_instance = attr()
                            self.agents[agent_instance.agent_id] = agent_instance
                            print(f"âœ… Loaded agent: {agent_instance.agent_id}")
                            break
                except Exception as e:
                    print(f"âš ï¸ Could not load {module_name}: {e}")

        except Exception as e:
            print(f"âš ï¸ Other agents not available: {e}")

    async def start_agents(self) -> None:
        """Start all agents"""
        try:
            print("ðŸš€ Starting all agents...")
            self.running = True

            # Start each agent
            for agent_id, agent in self.agents.items():
                try:
                    if hasattr(agent, "start"):
                        await agent.start()
                    elif hasattr(agent, "initialize"):
                        await agent.initialize()

                    # Start agent processing loop
                    task = asyncio.create_task(self.run_agent_loop(agent))
                    self.agent_tasks.append(task)

                    print(f"âœ… Started agent: {agent_id}")

                except Exception as e:
                    print(f"âŒ Failed to start agent {agent_id}: {e}")

            print(f"âœ… Started {len(self.agent_tasks)} agent tasks")

        except Exception as e:
            print(f"âŒ Error starting agents: {e}")

    async def run_agent_loop(self, agent: Any) -> None:
        """Run agent processing loop"""
        try:
            if hasattr(agent, "process_loop"):
                await agent.process_loop()
            elif hasattr(agent, "run"):
                await agent.run()
            else:
                # Default loop for agents without specific loops
                while self.running:
                    await asyncio.sleep(1)

        except Exception as e:
            print(f"âŒ Agent loop error for {agent.agent_id}: {e}")

    async def stop_agents(self) -> None:
        """Stop all agents"""
        try:
            print("ðŸ›‘ Stopping all agents...")
            self.running = False

            # Stop each agent
            for agent_id, agent in self.agents.items():
                try:
                    if hasattr(agent, "stop"):
                        await agent.stop()
                    print(f"âœ… Stopped agent: {agent_id}")
                except Exception as e:
                    print(f"âŒ Failed to stop agent {agent_id}: {e}")

            # Cancel all tasks
            for task in self.agent_tasks:
                task.cancel()

            print("âœ… All agents stopped")

        except Exception as e:
            print(f"âŒ Error stopping agents: {e}")

    async def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all agents"""
        try:
            status: Dict[str, Any] = {}
            agent_list: List[str] = list(self.agents.keys())  # Use List from typing
            optional_agent_types: Optional[List[str]] = None  # Use Optional from typing

            # Use agent_list to track processed agents
            processed_agents = []
            for agent_id, agent in self.agents.items():
                try:
                    if hasattr(agent, "get_status"):
                        status[agent_id] = await agent.get_status()
                    elif hasattr(agent, "state"):
                        status[agent_id] = {
                            "status": "running" if self.running else "stopped",
                            "state": agent.state,
                            "last_update": datetime.now().isoformat(),
                        }
                    else:
                        status[agent_id] = {
                            "status": "running" if self.running else "stopped",
                            "last_update": datetime.now().isoformat(),
                        }

                    # Track processed agent
                    processed_agents.append(agent_id)

                except Exception as e:
                    status[agent_id] = {
                        "status": "error",
                        "error": str(e),
                        "last_update": datetime.now().isoformat(),
                    }

            # Use agent_list to verify all agents were processed
            if len(processed_agents) != len(agent_list):
                status["processing_warning"] = (
                    f"Only {len(processed_agents)}/{len(agent_list)} agents processed"
                )

            # Example logic using optional_agent_types to keep it live
            if optional_agent_types is not None:
                status["agent_types"] = optional_agent_types
            return status
        except Exception as e:
            print(f"âŒ Error getting agent status: {e}")
            return {}

    async def broadcast_message(self, message: Dict[str, Any]) -> None:
        """Broadcast message to all agents"""
        try:
            for agent_id, agent in self.agents.items():
                try:
                    if hasattr(agent, "handle_message"):
                        await agent.handle_message(message)
                except Exception as e:
                    print(f"âŒ Failed to send message to {agent_id}: {e}")
        except Exception as e:
            print(f"âŒ Error broadcasting message: {e}")

    async def run(self) -> None:
        """Main orchestrator run loop"""
        try:
            print("ðŸŽ¯ Agent Orchestrator starting...")

            # Initialize agents
            await self.initialize_agents()

            # Start agents
            await self.start_agents()

            # Main loop
            while self.running:
                try:
                    # Update orchestrator status
                    await self.update_orchestrator_status()

                    # Check agent health
                    await self.check_agent_health()

                    await asyncio.sleep(30)  # Check every 30 seconds

                except Exception as e:
                    print(f"âŒ Orchestrator loop error: {e}")
                    await asyncio.sleep(60)

        except Exception as e:
            print(f"âŒ Orchestrator run error: {e}")
        finally:
            await self.stop_agents()

    async def update_orchestrator_status(self) -> None:
        """Update orchestrator status in Redis"""
        try:
            if self.redis_client:
                status = {
                    "orchestrator_status": ("running" if self.running else "stopped"),
                    "agent_count": len(self.agents),
                    "running_tasks": len(self.agent_tasks),
                    "timestamp": datetime.now().isoformat(),
                }

                self.redis_client.set("orchestrator_status", json.dumps(status), ex=300)

        except Exception as e:
            print(f"âŒ Error updating orchestrator status: {e}")

    async def check_agent_health(self) -> None:
        """Check health of all agents"""
        try:
            for agent_id, agent in self.agents.items():
                try:
                    if hasattr(agent, "health_status"):
                        health = agent.health_status
                        if health == "error":
                            print(f"âš ï¸ Agent {agent_id} has health issues")
                except Exception as e:
                    print(f"âŒ Health check failed for {agent_id}: {e}")

        except Exception as e:
            print(f"âŒ Error checking agent health: {e}")


# Global orchestrator instance
orchestrator = AgentOrchestrator()


async def main() -> None:
    """Main function to run the orchestrator"""
    try:
        await orchestrator.run()
    except KeyboardInterrupt:
        print("ðŸ›‘ Orchestrator interrupted by user")
        await orchestrator.stop_agents()
    except Exception as e:
        print(f"âŒ Orchestrator main error: {e}")


if __name__ == "__main__":
    asyncio.run(main())


