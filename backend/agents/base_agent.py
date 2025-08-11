"""
Base Agent Class
Foundation for all AI trading agents
"""

import asyncio
import json
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, Any, Callable, Awaitable
import redis
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../.env"))


class BaseAgent(ABC):
    """Base class for all AI trading agents"""

    def __init__(self, agent_id: str, agent_type: str):
        """Initialize base agent"""
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.running = False
        self.health_status = "healthy"
        self.last_heartbeat = datetime.now()
        self.message_handlers = {}
        self.state = {}

        # Initialize Redis connection
        self.redis_client = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            db=int(os.getenv("REDIS_DB", 0)),
            decode_responses=True,
        )

        # Agent-specific channels
        self.input_channel = f"agent:{agent_id}:input"
        self.output_channel = f"agent:{agent_id}:output"
        self.heartbeat_channel = f"agent:{agent_id}:heartbeat"
        self.status_channel = f"agent:{agent_id}:status"

        # Register message handlers
        self.register_default_handlers()

        print(f"🤖 Agent {self.agent_id} ({self.agent_type}) initialized")

    def register_default_handlers(self):
        """Register default message handlers"""
        self.register_handler("ping", self.handle_ping)
        self.register_handler("status", self.handle_status_request)
        self.register_handler("shutdown", self.handle_shutdown)
        self.register_handler("update_state", self.handle_state_update)

    def register_handler(self, message_type: str, handler: Callable[[Dict[str, Any]], Awaitable[None]]):
        """Register a message handler"""
        self.message_handlers[message_type] = handler

    async def start(self):
        """Start the agent"""
        print(f"🚀 Starting agent {self.agent_id}")
        self.running = True

        # Start message listener
        asyncio.create_task(self.listen_for_messages())

        # Start heartbeat
        asyncio.create_task(self.heartbeat_loop())

        # Start agent-specific processing
        asyncio.create_task(self.process_loop())

        # Initialize agent state
        await self.initialize()

        # Broadcast agent startup
        await self.broadcast_status("started")

        print(f"✅ Agent {self.agent_id} started successfully")

    async def stop(self):
        """Stop the agent"""
        print(f"🛑 Stopping agent {self.agent_id}")
        self.running = False

        # Broadcast agent shutdown
        await self.broadcast_status("stopped")

        print(f"✅ Agent {self.agent_id} stopped")

    async def listen_for_messages(self):
        """Listen for incoming messages"""
        pubsub = self.redis_client.pubsub()
        pubsub.subscribe(self.input_channel)

        print(f"👂 Agent {self.agent_id} listening on {self.input_channel}")

        try:
            for message in pubsub.listen():
                if not self.running:
                    break

                if message["type"] == "message":
                    await self.process_message(json.loads(message["data"]))

        except Exception as e:
            print(f"❌ Error in message listener for {self.agent_id}: {e}")
        finally:
            pubsub.close()

    async def process_message(self, message: Dict[str, Any]):
        """Process incoming message"""
        try:
            message_type = message.get("type")
            handler = self.message_handlers.get(message_type)

            if handler:
                await handler(message)
            else:
                print(f"⚠️ No handler for message type: {message_type}")

        except Exception as e:
            print(f"❌ Error processing message in {self.agent_id}: {e}")
            await self.broadcast_error(f"Message processing error: {e}")

    async def send_message(self, target_agent: str, message: Dict[str, Any]):
        """Send message to another agent"""
        try:
            message.update(
                {
                    "from_agent": self.agent_id,
                    "timestamp": datetime.now().isoformat(),
                }
            )

            target_channel = f"agent:{target_agent}:input"
            self.redis_client.publish(target_channel, json.dumps(message))

        except Exception as e:
            print(f"❌ Error sending message to {target_agent}: {e}")

    async def broadcast_message(self, message: Dict[str, Any]):
        """Broadcast message to all agents"""
        try:
            message.update(
                {
                    "from_agent": self.agent_id,
                    "timestamp": datetime.now().isoformat(),
                }
            )

            self.redis_client.publish("agent:broadcast", json.dumps(message))

        except Exception as e:
            print(f"❌ Error broadcasting message: {e}")

    async def broadcast_status(self, status: str):
        """Broadcast agent status"""
        try:
            status_message = {
                "agent_id": self.agent_id,
                "agent_type": self.agent_type,
                "status": status,
                "timestamp": datetime.now().isoformat(),
                "health": self.health_status,
            }

            self.redis_client.publish(self.status_channel, json.dumps(status_message))
            self.redis_client.set(
                f"agent_status:{self.agent_id}",
                json.dumps(status_message),
                ex=300,
            )

        except Exception as e:
            print(f"❌ Error broadcasting status: {e}")

    async def broadcast_error(self, error_message: str):
        """Broadcast error message"""
        try:
            error_data = {
                "agent_id": self.agent_id,
                "agent_type": self.agent_type,
                "error": error_message,
                "timestamp": datetime.now().isoformat(),
            }

            self.redis_client.publish("agent:errors", json.dumps(error_data))

        except Exception as e:
            print(f"❌ Error broadcasting error: {e}")

    async def heartbeat_loop(self):
        """Send periodic heartbeat"""
        while self.running:
            try:
                heartbeat = {
                    "agent_id": self.agent_id,
                    "agent_type": self.agent_type,
                    "timestamp": datetime.now().isoformat(),
                    "health": self.health_status,
                    "state": self.state,
                }

                self.redis_client.publish(self.heartbeat_channel, json.dumps(heartbeat))
                self.redis_client.set(
                    f"agent_heartbeat:{self.agent_id}",
                    json.dumps(heartbeat),
                    ex=60,
                )

                self.last_heartbeat = datetime.now()

                # Periodic memory cleanup every 10 heartbeats (5 minutes)
                if hasattr(self, '_heartbeat_count'):
                    self._heartbeat_count += 1
                else:
                    self._heartbeat_count = 1

                if self._heartbeat_count % 10 == 0:
                    self.cleanup_memory()

                await asyncio.sleep(30)  # Heartbeat every 30 seconds

            except Exception as e:
                print(f"❌ Error in heartbeat loop for {self.agent_id}: {e}")
                await asyncio.sleep(60)

    async def handle_ping(self, message: Dict[str, Any]):
        """Handle ping message"""
        response = {
            "type": "pong",
            "agent_id": self.agent_id,
            "timestamp": datetime.now().isoformat(),
        }

        sender = message.get("from_agent")
        if sender:
            await self.send_message(sender, response)

    async def handle_status_request(self, message: Dict[str, Any]):
        """Handle status request"""
        status = {
            "type": "status_response",
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "status": "running" if self.running else "stopped",
            "health": self.health_status,
            "state": self.state,
            "timestamp": datetime.now().isoformat(),
        }

        sender = message.get("from_agent")
        if sender:
            await self.send_message(sender, status)

    async def handle_shutdown(self, message: Dict[str, Any]):
        """Handle shutdown request"""
        print(f"🛑 Agent {self.agent_id} received shutdown request")
        await self.stop()

    async def handle_state_update(self, message: Dict[str, Any]):
        """Handle state update request"""
        new_state = message.get("state", {})
        self.state.update(new_state)
        print(f"📝 Agent {self.agent_id} state updated: {new_state}")

    @abstractmethod
    async def initialize(self):
        """Initialize agent-specific resources"""
        pass

    @abstractmethod
    async def process_loop(self):
        """Main processing loop for agent-specific logic"""
        pass

    @abstractmethod
    async def process_market_data(self, market_data: Dict[str, Any]):
        """Process market data (to be implemented by subclasses)"""
        pass

    def update_health_status(self, status: str):
        """Update agent health status"""
        self.health_status = status
        print(f"🏥 Agent {self.agent_id} health status: {status}")

    def get_state(self) -> Dict[str, Any]:
        """Get current agent state"""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "running": self.running,
            "health": self.health_status,
            "last_heartbeat": self.last_heartbeat.isoformat(),
            "state": self.state,
        }

    def is_healthy(self) -> bool:
        """Check if agent is healthy"""
        return (
            self.running
            and self.health_status == "healthy"
            and (datetime.now() - self.last_heartbeat) < timedelta(minutes=2)
        )

    def cleanup_memory(self):
        """Clean up memory to prevent leaks"""
        try:
            # Clean up large data structures in state
            if hasattr(self, 'state') and isinstance(self.state, dict):
                for key, value in list(self.state.items()):
                    if isinstance(value, list) and len(value) > 1000:
                        # Keep only the last 500 items for large lists
                        self.state[key] = value[-500:]
                    elif isinstance(value, dict) and len(value) > 100:
                        # For large dictionaries, keep only recent entries
                        if key in ['performance_metrics', 'training_history', 'sentiment_history']:
                            # Keep only the last 100 entries for performance data
                            recent_items = dict(list(value.items())[-100:])
                            self.state[key] = recent_items

            # Clean up message handlers if too many
            if len(self.message_handlers) > 50:
                # Keep only essential handlers
                essential_handlers = ['ping', 'status', 'shutdown', 'update_state']
                self.message_handlers = {k: v for k, v in self.message_handlers.items()
                                       if k in essential_handlers}

        except Exception as e:
            print(f"⚠️ Memory cleanup error in {self.agent_id}: {e}")

    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage statistics"""
        try:
            memory_stats = {
                "agent_id": self.agent_id,
                "state_size": len(self.state) if hasattr(self, 'state') else 0,
                "message_handlers": len(self.message_handlers),
                "health_status": self.health_status,
            }

            # Calculate state memory usage
            if hasattr(self, 'state'):
                total_items = 0
                large_structures = 0
                for key, value in self.state.items():
                    if isinstance(value, list):
                        total_items += len(value)
                        if len(value) > 100:
                            large_structures += 1
                    elif isinstance(value, dict):
                        total_items += len(value)
                        if len(value) > 50:
                            large_structures += 1

                memory_stats.update({
                    "total_state_items": total_items,
                    "large_structures": large_structures,
                })

            return memory_stats

        except Exception as e:
            print(f"⚠️ Error getting memory usage for {self.agent_id}: {e}")
            return {"error": str(e)}
