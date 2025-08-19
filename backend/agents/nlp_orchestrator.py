"""
NLP Orchestrator Agent
Coordinates all NLP agents and provides unified NLP services
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from typing import Any, Optional

import numpy as np

# Make all imports live (F401 = timedelta(hours=1
_ = list[str]
_ = Optional[str]

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from backend.agents.base_agent import BaseAgent
except ImportError:
    # Fallback if the path modification didn't work
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from backend.agents.base_agent import BaseAgent


class NLPOrchestrator(BaseAgent):
    """NLP Orchestrator - Coordinates all NLP agents"""

    def __init__(self, agent_id: str = "nlp_orchestrator_001"):
        super().__init__(agent_id, "nlp_orchestrator")

        # NLP orchestrator-specific state
        self.state.update(
            {
                "nlp_agents": {},
                "unified_sentiment": {},
                "nlp_services": {},
                "coordination_tasks": [],
                "last_coordination": None,
                "coordination_count": 0,
            }
        )

        # NLP agents configuration
        self.nlp_agents = {
            "news_sentiment": {
                "agent_id": "news_sentiment_agent_001",
                "status": "unknown",
                "last_heartbeat": None,
                "capabilities": [
                    "news_analysis",
                    "sentiment_extraction",
                    "symbol_detection",
                ],
            },
            "social_media": {
                "agent_id": "social_media_agent_001",
                "status": "unknown",
                "last_heartbeat": None,
                "capabilities": [
                    "social_monitoring",
                    "trending_topics",
                    "influencer_tracking",
                ],
            },
            "market_sentiment": {
                "agent_id": "market_sentiment_agent_001",
                "status": "unknown",
                "last_heartbeat": None,
                "capabilities": [
                    "sentiment_aggregation",
                    "fear_greed_index",
                    "signal_generation",
                ],
            },
        }

        # NLP services configuration
        self.nlp_services = {
            "real_time_sentiment": {
                "enabled": True,
                "update_interval": 60,  # seconds
                "sources": ["news", "social", "market"],
            },
            "trending_analysis": {
                "enabled": True,
                "update_interval": 300,
                "sources": ["social", "news"],
            },  # seconds
            "sentiment_forecasting": {
                "enabled": True,
                "update_interval": 600,
                "sources": ["all"],
            },  # seconds
            "market_intelligence": {
                "enabled": True,
                "update_interval": 180,  # seconds
                "sources": ["news", "social", "market"],
            },
        }

        # Register NLP orchestrator-specific handlers
        self.register_handler("coordinate_nlp", self.handle_coordinate_nlp)
        self.register_handler("get_unified_sentiment", self.handle_get_unified_sentiment)
        self.register_handler("nlp_service_request", self.handle_nlp_service_request)
        self.register_handler("agent_heartbeat", self.handle_agent_heartbeat)
        self.register_handler("market_data", self.handle_market_data)

        print(f"ðŸ§  NLP Orchestrator {agent_id} initialized")

    async def initialize(self):
        """Initialize NLP orchestrator resources"""
        try:
            # Load NLP configuration
            await self.load_nlp_config()

            # Initialize coordination systems
            await self.initialize_coordination()

            # Start NLP monitoring
            await self.start_nlp_monitoring()

            print(f"âœ… NLP Orchestrator {self.agent_id} initialized successfully")

        except Exception as e:
            print(f"âŒ Error initializing NLP Orchestrator: {e}")
            self.update_health_status("error")

    async def process_loop(self):
        """Main NLP coordination loop"""
        while self.running:
            try:
                # Coordinate NLP agents
                await self.coordinate_nlp_agents()

                # Update unified sentiment
                await self.update_unified_sentiment()

                # Manage NLP services
                await self.manage_nlp_services()

                # Update coordination metrics
                await self.update_coordination_metrics()

                # Clean up old tasks
                await self.cleanup_tasks()

                await asyncio.sleep(30)  # Check every 30 seconds

            except Exception as e:
                print(f"âŒ Error in NLP coordination loop: {e}")
                await asyncio.sleep(60)

    async def load_nlp_config(self):
        """Load NLP configuration from Redis"""
        try:
            # Load NLP agents configuration
            agents_data = self.redis_client.get("nlp_agents_config")
            if agents_data:
                self.nlp_agents = json.loads(agents_data)

            # Load NLP services configuration
            services_data = self.redis_client.get("nlp_services_config")
            if services_data:
                self.nlp_services = json.loads(services_data)

            print(
                f"ðŸ“‹ NLP configuration loaded: {len(self.nlp_agents)} agents, {len(self.nlp_services)} services"
            )

        except Exception as e:
            print(f"âŒ Error loading NLP configuration: {e}")

    async def initialize_coordination(self):
        """Initialize NLP coordination systems"""
        try:
            # Initialize agent monitoring
            await self.initialize_agent_monitoring()

            # Initialize service management
            await self.initialize_service_management()

            print("ðŸ§  NLP coordination systems initialized")

        except Exception as e:
            print(f"âŒ Error initializing coordination: {e}")

    async def initialize_agent_monitoring(self):
        """Initialize agent monitoring"""
        try:
            # Start monitoring each NLP agent
            for agent_name, agent_config in self.nlp_agents.items():
                await self.monitor_agent_health(agent_name, agent_config)

            print("ðŸ“¡ Agent monitoring initialized")

        except Exception as e:
            print(f"âŒ Error initializing agent monitoring: {e}")

    async def initialize_service_management(self):
        """Initialize service management"""
        try:
            # Start service management tasks
            for service_name, service_config in self.nlp_services.items():
                if service_config.get("enabled", False):
                    await self.start_service(service_name, service_config)

            print("âš™ï¸ Service management initialized")

        except Exception as e:
            print(f"âŒ Error initializing service management: {e}")

    async def start_nlp_monitoring(self):
        """Start NLP monitoring"""
        try:
            # Subscribe to agent heartbeats and market data
            pubsub = self.redis_client.pubsub()
            pubsub.subscribe("agent_heartbeat")
            pubsub.subscribe("market_data")
            pubsub.subscribe("aggregated_sentiment_update")

            # Start monitoring listener
            asyncio.create_task(self.listen_nlp_updates(pubsub))

            print("ðŸ“¡ NLP monitoring started")

        except Exception as e:
            print(f"âŒ Error starting NLP monitoring: {e}")

    async def listen_nlp_updates(self, pubsub):
        """Listen for NLP updates"""
        try:
            for message in pubsub.listen():
                if not self.running:
                    break

                if message["type"] == "message":
                    try:
                        data = json.loads(message["data"])
                        channel = message["channel"]

                        if channel == "agent_heartbeat":
                            await self.handle_agent_heartbeat(data)
                        elif channel == "market_data":
                            await self.handle_market_data(data)
                        elif channel == "aggregated_sentiment_update":
                            await self.handle_sentiment_update(data)

                    except json.JSONDecodeError:
                        print(f"âŒ Error decoding NLP update: {message['data']}")

        except Exception as e:
            print(f"âŒ Error in NLP listener: {e}")
        finally:
            pubsub.close()

    async def handle_agent_heartbeat(self, data: dict[str, Any]):
        """Handle agent heartbeat updates"""
        try:
            agent_id = data.get("agent_id")
            data.get("agent_type")
            status = data.get("status", "unknown")
            timestamp = data.get("timestamp")

            # Update agent status
            for agent_name, agent_config in self.nlp_agents.items():
                if agent_config["agent_id"] == agent_id:
                    agent_config["status"] = status
                    agent_config["last_heartbeat"] = timestamp
                    break

            print(f"ðŸ’“ Agent heartbeat: {agent_id} ({status})")

        except Exception as e:
            print(f"âŒ Error handling agent heartbeat: {e}")

    async def handle_market_data(self, data: dict[str, Any]):
        """Handle market data for NLP context"""
        try:
            symbol = data.get("symbol")
            price = data.get("price")

            # Update market context for NLP analysis
            if symbol and price:
                await self.update_market_context(symbol, price)

        except Exception as e:
            print(f"âŒ Error handling market data: {e}")

    async def handle_sentiment_update(self, data: dict[str, Any]):
        """Handle aggregated sentiment updates"""
        try:
            symbol = data.get("symbol")
            sentiment = data.get("sentiment", {})

            # Update unified sentiment
            if symbol:
                self.state["unified_sentiment"][symbol] = sentiment

        except Exception as e:
            print(f"âŒ Error handling sentiment update: {e}")

    async def update_market_context(self, symbol: str, price: float):
        """Update market context for NLP analysis"""
        try:
            # Store market context for NLP correlation
            market_context = {
                "symbol": symbol,
                "price": price,
                "timestamp": datetime.now().isoformat(),
            }

            # Store in Redis
            self.redis_client.set(
                f"nlp_market_context:{symbol}",
                json.dumps(market_context),
                ex=300,
            )

        except Exception as e:
            print(f"âŒ Error updating market context: {e}")

    async def coordinate_nlp_agents(self):
        """Coordinate all NLP agents"""
        try:
            print(f"ðŸ§  Coordinating {len(self.nlp_agents)} NLP agents...")

            # Check agent health
            await self.check_agent_health()

            # Distribute coordination tasks
            await self.distribute_coordination_tasks()

            # Synchronize agent states
            await self.synchronize_agent_states()

            # Update coordination count
            self.state["coordination_count"] += 1
            self.state["last_coordination"] = datetime.now().isoformat()

            print("âœ… NLP coordination complete")

        except Exception as e:
            print(f"âŒ Error coordinating NLP agents: {e}")

    async def check_agent_health(self):
        """Check health of all NLP agents"""
        try:
            for agent_name, agent_config in self.nlp_agents.items():
                agent_id = agent_config["agent_id"]

                # Check if agent is responding
                try:
                    # Send health check message
                    health_check = {
                        "type": "health_check",
                        "from_agent": self.agent_id,
                        "timestamp": datetime.now().isoformat(),
                    }

                    await self.send_message(agent_id, health_check)

                    # Update agent status based on response
                    # For now, assume agent is healthy if we can send message
                    agent_config["status"] = "healthy"

                except Exception as e:
                    print(f"âŒ Agent {agent_id} health check failed: {e}")
                    agent_config["status"] = "unhealthy"

        except Exception as e:
            print(f"âŒ Error checking agent health: {e}")

    async def distribute_coordination_tasks(self):
        """Distribute coordination tasks to agents"""
        try:
            # Create coordination tasks
            tasks = [
                {
                    "type": "sentiment_analysis",
                    "priority": "high",
                    "target_agents": ["news_sentiment", "social_media"],
                    "parameters": {
                        "timeframe": "1h",
                        "symbols": ["BTC", "ETH"],
                    },
                },
                {
                    "type": "trending_analysis",
                    "priority": "medium",
                    "target_agents": ["social_media"],
                    "parameters": {
                        "platforms": ["twitter", "reddit"],
                        "keywords": ["bitcoin", "crypto"],
                    },
                },
                {
                    "type": "market_intelligence",
                    "priority": "high",
                    "target_agents": ["market_sentiment"],
                    "parameters": {
                        "update_fear_greed": True,
                        "generate_signals": True,
                    },
                },
            ]

            # Distribute tasks to agents
            for task in tasks:
                await self.distribute_task(task)

            # Store tasks in state
            self.state["coordination_tasks"] = tasks

        except Exception as e:
            print(f"âŒ Error distributing coordination tasks: {e}")

    async def distribute_task(self, task: dict[str, Any]):
        """Distribute a single task to target agents"""
        try:
            task_type = task["type"]
            target_agents = task["target_agents"]
            parameters = task.get("parameters", {})

            for agent_name in target_agents:
                if agent_name in self.nlp_agents:
                    agent_id = self.nlp_agents[agent_name]["agent_id"]

                    # Create task message
                    task_message = {
                        "type": f"coordination_task_{task_type}",
                        "from_agent": self.agent_id,
                        "task": task,
                        "parameters": parameters,
                        "timestamp": datetime.now().isoformat(),
                    }

                    # Send task to agent
                    await self.send_message(agent_id, task_message)

                    print(f"ðŸ“‹ Task {task_type} sent to {agent_name}")

        except Exception as e:
            print(f"âŒ Error distributing task {task.get('type', 'unknown')}: {e}")

    async def synchronize_agent_states(self):
        """Synchronize states between NLP agents"""
        try:
            # Get unified configuration
            unified_config = {
                "trading_symbols": [
                    "BTC",
                    "ETH",
                    "ADA",
                    "DOT",
                    "LINK",
                    "UNI",
                    "AAVE",
                ],
                "sentiment_weights": {
                    "news": 0.3,
                    "social": 0.25,
                    "market": 0.25,
                    "technical": 0.2,
                },
                "update_intervals": {
                    "news": 300,
                    "social": 180,
                    "market": 60,
                },  # 5 minutes  # 3 minutes  # 1 minute
            }

            # Send configuration to all agents
            for agent_name, agent_config in self.nlp_agents.items():
                agent_id = agent_config["agent_id"]

                sync_message = {
                    "type": "synchronize_config",
                    "from_agent": self.agent_id,
                    "config": unified_config,
                    "timestamp": datetime.now().isoformat(),
                }

                await self.send_message(agent_id, sync_message)

            print("ðŸ”„ Agent states synchronized")

        except Exception as e:
            print(f"âŒ Error synchronizing agent states: {e}")

    async def update_unified_sentiment(self):
        """Update unified sentiment from all sources"""
        try:
            # Collect sentiment from all agents
            unified_sentiment = {}

            for agent_name, agent_config in self.nlp_agents.items():
                if agent_config["status"] == "healthy":
                    agent_id = agent_config["agent_id"]

                    # Request sentiment data from agent
                    sentiment_request = {
                        "type": "get_sentiment_summary",
                        "from_agent": self.agent_id,
                        "timestamp": datetime.now().isoformat(),
                    }

                    try:
                        # Send request and wait for response
                        await self.send_message(agent_id, sentiment_request)

                        # For now, use mock data (in production, you'd wait for actual response)
                        mock_sentiment = {
                            "agent": agent_name,
                            "symbols_analyzed": 7,
                            "avg_sentiment": np.random.uniform(-0.5, 0.8),
                            "last_update": datetime.now().isoformat(),
                        }

                        unified_sentiment[agent_name] = mock_sentiment

                    except Exception as e:
                        print(f"âŒ Error getting sentiment from {agent_name}: {e}")

            # Store unified sentiment
            self.state["unified_sentiment"] = unified_sentiment

            # Broadcast unified sentiment
            await self.broadcast_unified_sentiment(unified_sentiment)

        except Exception as e:
            print(f"âŒ Error updating unified sentiment: {e}")

    async def broadcast_unified_sentiment(self, unified_sentiment: dict[str, Any]):
        """Broadcast unified sentiment to other agents"""
        try:
            sentiment_broadcast = {
                "type": "unified_sentiment_update",
                "sentiment": unified_sentiment,
                "timestamp": datetime.now().isoformat(),
            }

            # Broadcast to all agents
            await self.broadcast_message(sentiment_broadcast)

            # Send to specific agents
            await self.send_message("strategy_agent", sentiment_broadcast)
            await self.send_message("risk_agent", sentiment_broadcast)
            await self.send_message("execution_agent", sentiment_broadcast)

        except Exception as e:
            print(f"âŒ Error broadcasting unified sentiment: {e}")

    async def manage_nlp_services(self):
        """Manage NLP services"""
        try:
            for service_name, service_config in self.nlp_services.items():
                if service_config.get("enabled", False):
                    await self.update_service(service_name, service_config)

        except Exception as e:
            print(f"âŒ Error managing NLP services: {e}")

    async def start_service(self, service_name: str, service_config: dict[str, Any]):
        """Start an NLP service"""
        try:
            print(f"ðŸš€ Starting NLP service: {service_name}")

            # Create service task
            service_task = {
                "name": service_name,
                "config": service_config,
                "status": "running",
                "start_time": datetime.now().isoformat(),
            }

            # Store service task
            if "services" not in self.state["nlp_services"]:
                self.state["nlp_services"]["services"] = {}

            self.state["nlp_services"]["services"][service_name] = service_task

        except Exception as e:
            print(f"âŒ Error starting service {service_name}: {e}")

    async def update_service(self, service_name: str, service_config: dict[str, Any]):
        """Update an NLP service"""
        try:
            # Check if service needs update
            update_interval = service_config.get("update_interval", 300)

            # For now, just log service status
            print(f"âš™ï¸ Service {service_name} running (interval: {update_interval}s)")

        except Exception as e:
            print(f"âŒ Error updating service {service_name}: {e}")

    async def monitor_agent_health(self, agent_name: str, agent_config: dict[str, Any]):
        """Monitor health of a specific agent"""
        try:
            agent_id = agent_config["agent_id"]

            # Send health check
            health_check = {
                "type": "health_check",
                "from_agent": self.agent_id,
                "timestamp": datetime.now().isoformat(),
            }

            await self.send_message(agent_id, health_check)

            print(f"ðŸ’“ Monitoring agent: {agent_name} ({agent_id})")

        except Exception as e:
            print(f"âŒ Error monitoring agent {agent_name}: {e}")

    async def handle_coordinate_nlp(self, message: dict[str, Any]):
        """Handle manual NLP coordination request"""
        try:
            coordination_type = message.get("type", "full")

            print(f"ðŸ§  Manual NLP coordination requested: {coordination_type}")

            if coordination_type == "full":
                await self.coordinate_nlp_agents()
            elif coordination_type == "health_check":
                await self.check_agent_health()
            elif coordination_type == "sentiment_update":
                await self.update_unified_sentiment()

            # Send response
            response = {
                "type": "nlp_coordination_complete",
                "coordination_type": coordination_type,
                "timestamp": datetime.now().isoformat(),
            }

            sender = message.get("from_agent")
            if sender:
                await self.send_message(sender, response)

        except Exception as e:
            print(f"âŒ Error handling NLP coordination request: {e}")
            await self.broadcast_error(f"NLP coordination error: {e}")

    async def handle_get_unified_sentiment(self, message: dict[str, Any]):
        """Handle unified sentiment request"""
        try:
            symbols = message.get("symbols", [])
            timeframe = message.get("timeframe", "1h")

            print(f"ðŸ“Š Unified sentiment request for {symbols} ({timeframe})")

            # Get unified sentiment data
            sentiment_data = {
                "unified_sentiment": self.state["unified_sentiment"],
                "agent_status": {
                    name: config["status"] for name, config in self.nlp_agents.items()
                },
                "services_status": (self.state["nlp_services"].get("services", {})),
                "timestamp": datetime.now().isoformat(),
            }

            # Send response
            response = {
                "type": "unified_sentiment_response",
                "symbols": symbols,
                "timeframe": timeframe,
                "sentiment": sentiment_data,
                "timestamp": datetime.now().isoformat(),
            }

            sender = message.get("from_agent")
            if sender:
                await self.send_message(sender, response)

        except Exception as e:
            print(f"âŒ Error handling unified sentiment request: {e}")
            await self.broadcast_error(f"Unified sentiment request error: {e}")

    async def handle_nlp_service_request(self, message: dict[str, Any]):
        """Handle NLP service request"""
        try:
            service_name = message.get("service")
            parameters = message.get("parameters", {})

            print(f"ðŸ”§ NLP service request: {service_name}")

            # Route service request to appropriate agent
            service_response = await self.route_service_request(service_name, parameters)

            # Send response
            response = {
                "type": "nlp_service_response",
                "service": service_name,
                "result": service_response,
                "timestamp": datetime.now().isoformat(),
            }

            sender = message.get("from_agent")
            if sender:
                await self.send_message(sender, response)

        except Exception as e:
            print(f"âŒ Error handling NLP service request: {e}")
            await self.broadcast_error(f"NLP service request error: {e}")

    async def route_service_request(
        self, service_name: str, parameters: dict[str, Any]
    ) -> dict[str, Any]:
        """Route service request to appropriate agent"""
        try:
            # Define service routing
            service_routing = {
                "news_analysis": "news_sentiment_agent_001",
                "social_monitoring": "social_media_agent_001",
                "sentiment_aggregation": "market_sentiment_agent_001",
                "trending_analysis": "social_media_agent_001",
                "market_intelligence": "market_sentiment_agent_001",
            }

            target_agent = service_routing.get(service_name)

            if target_agent:
                # Forward request to target agent
                service_message = {
                    "type": f"service_request_{service_name}",
                    "from_agent": self.agent_id,
                    "parameters": parameters,
                    "timestamp": datetime.now().isoformat(),
                }

                await self.send_message(target_agent, service_message)

                # For now, return mock response
                return {
                    "status": "routed",
                    "target_agent": target_agent,
                    "service": service_name,
                }
            else:
                return {
                    "status": "error",
                    "message": f"Service {service_name} not found",
                }

        except Exception as e:
            print(f"âŒ Error routing service request: {e}")
            return {"status": "error", "message": str(e)}

    async def update_coordination_metrics(self):
        """Update coordination metrics"""
        try:
            metrics = {
                "agent_id": self.agent_id,
                "agents_count": len(self.nlp_agents),
                "services_count": len(self.nlp_services),
                "coordination_count": self.state["coordination_count"],
                "last_coordination": self.state["last_coordination"],
                "agent_status": {
                    name: config["status"] for name, config in self.nlp_agents.items()
                },
                "services_status": (self.state["nlp_services"].get("services", {})),
                "timestamp": datetime.now().isoformat(),
            }

            # Store metrics in Redis
            self.redis_client.set(f"agent_metrics:{self.agent_id}", json.dumps(metrics), ex=300)

        except Exception as e:
            print(f"âŒ Error updating coordination metrics: {e}")

    async def cleanup_tasks(self):
        """Clean up old coordination tasks"""
        try:
            datetime.now()

            # Clean up old tasks (keep only last 100)
            if len(self.state["coordination_tasks"]) > 100:
                self.state["coordination_tasks"] = self.state["coordination_tasks"][-100:]

        except Exception as e:
            print(f"âŒ Error cleaning up tasks: {e}")

    async def process_market_data(self, market_data: dict[str, Any]):
        """Process incoming market data for NLP coordination"""
        try:
            print("ðŸ“Š Processing market data for NLP coordination")

            # Update market data in state
            self.state["last_market_data"] = market_data
            self.state["last_market_update"] = datetime.now().isoformat()

            # Update market context for each symbol
            for symbol, data in market_data.items():
                price = data.get("price", 0)
                await self.update_market_context(symbol, price)

            # Coordinate NLP agents with new market data
            await self.coordinate_nlp_agents()

            # Update unified sentiment with new market context
            await self.update_unified_sentiment()

            # Update coordination metrics
            await self.update_coordination_metrics()

            print("âœ… Market data processed for NLP coordination")

        except Exception as e:
            print(f"âŒ Error processing market data for NLP coordination: {e}")
            await self.broadcast_error(f"NLP coordination market data error: {e}")


