"""
Agent Orchestrator
Manages all AI agents, coordinates communication, and provides central control
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # noqa: E402

from agents.base_agent import BaseAgent  # noqa: E402
from agents.strategy_agent import StrategyAgent  # noqa: E402
from agents.risk_agent import RiskAgent  # noqa: E402
from agents.execution_agent import ExecutionAgent  # noqa: E402
from agents.compliance_agent import ComplianceAgent  # noqa: E402
from agents.news_sentiment_agent import NewsSentimentAgent  # noqa: E402
from agents.social_media_agent import SocialMediaAgent  # noqa: E402
from agents.market_sentiment_agent import MarketSentimentAgent  # noqa: E402
from agents.deep_learning_agent import DeepLearningAgent  # noqa: E402
from agents.reinforcement_learning_agent import ReinforcementLearningAgent  # noqa: E402
from agents.ai_model_manager import AIModelManager  # noqa: E402
from agents.advanced_ai_orchestrator import AdvancedAIOrchestrator  # noqa: E402
from agents.nlp_orchestrator import NLPOrchestrator  # noqa: E402
from agents.chart_pattern_agent import ChartPatternAgent  # noqa: E402
from agents.technical_indicator_agent import TechnicalIndicatorAgent  # noqa: E402
from agents.market_visualization_agent import MarketVisualizationAgent  # noqa: E402
from agents.computer_vision_orchestrator import ComputerVisionOrchestrator  # noqa: E402
from agents.quantum_algorithm_engine import QuantumAlgorithmEngine  # noqa: E402
from agents.quantum_machine_learning_agent import QuantumMachineLearningAgent  # noqa: E402
from agents.quantum_optimization_agent import QuantumOptimizationAgent  # noqa: E402
from agents.quantum_trading_engine import QuantumTradingEngine  # noqa: E402
from agents.interdimensional_signal_decoder import (  # noqa: E402
    InterdimensionalSignalDecoder,
)
from agents.neuro_synchronization_engine import NeuroSynchronizationEngine  # noqa: E402
from agents.cosmic_pattern_recognizer import CosmicPatternRecognizer  # noqa: E402
from agents.auranet_channel import AuraNetChannel  # noqa: E402


class AgentOrchestrator(BaseAgent):
    """Agent Orchestrator - Coordinates all AI agents"""

    def __init__(self, agent_id: str = "orchestrator_001"):
        super().__init__(agent_id, "orchestrator")

        # Orchestrator-specific state
        self.state.update(
            {
                "agents": {},
                "agent_status": {},
                "system_status": "initializing",
                "agent_communication": {},
                "performance_metrics": {},
                "last_coordination": None,
            }
        )

        # Initialize agents
        self.strategy_agent = None
        self.risk_agent = None
        self.execution_agent = None
        self.compliance_agent = None
        self.news_sentiment_agent = None
        self.social_media_agent = None
        self.market_sentiment_agent = None
        self.deep_learning_agent = None
        self.reinforcement_learning_agent = None
        self.ai_model_manager = None
        self.advanced_ai_orchestrator = None
        self.nlp_orchestrator = None
        self.chart_pattern_agent = None
        self.technical_indicator_agent = None
        self.market_visualization_agent = None
        self.computer_vision_orchestrator = None
        self.quantum_algorithm_engine = None
        self.quantum_machine_learning_agent = None
        self.quantum_optimization_agent = None
        self.quantum_trading_engine = None
        self.interdimensional_signal_decoder = None
        self.neuro_synchronization_engine = None
        self.cosmic_pattern_recognizer = None
        self.auranet_channel = None

        # Register orchestrator-specific handlers
        self.register_handler("agent_status_update", self.handle_agent_status_update)
        self.register_handler("system_command", self.handle_system_command)
        self.register_handler("performance_report", self.handle_performance_report)
        self.register_handler("agent_communication", self.handle_agent_communication)

        print(f"🎼 Agent Orchestrator {agent_id} initialized")

    async def initialize(self):
        """Initialize orchestrator and all agents"""
        try:
            # Initialize all agents
            await self.initialize_agents()

            # Start agent monitoring
            await self.start_agent_monitoring()

            # Initialize system coordination
            await self.initialize_system_coordination()

            print(f"✅ Agent Orchestrator {self.agent_id} initialized successfully")

        except Exception as e:
            print(f"❌ Error initializing Agent Orchestrator: {e}")
            self.update_health_status("error")

    async def process_loop(self):
        """Main orchestrator processing loop"""
        while self.running:
            try:
                # Monitor all agents
                await self.monitor_agents()

                # Coordinate agent communication
                await self.coordinate_agents()

                # Update system metrics
                await self.update_system_metrics()

                # Check system health
                await self.check_system_health()

                await asyncio.sleep(30)  # Check every 30 seconds

            except Exception as e:
                print(f"❌ Error in orchestrator processing loop: {e}")
                await asyncio.sleep(60)

    async def initialize_agents(self):
        """Initialize all AI agents"""
        try:
            print("🤖 Initializing AI agents...")

            # Initialize Strategy Agent
            self.strategy_agent = StrategyAgent("strategy_agent_001")
            await self.strategy_agent.start()
            self.state["agents"]["strategy"] = self.strategy_agent
            self.state["agent_status"]["strategy"] = "running"

            # Initialize Risk Agent
            self.risk_agent = RiskAgent("risk_agent_001")
            await self.risk_agent.start()
            self.state["agents"]["risk"] = self.risk_agent
            self.state["agent_status"]["risk"] = "running"

            # Initialize Execution Agent
            self.execution_agent = ExecutionAgent("execution_agent_001")
            await self.execution_agent.start()
            self.state["agents"]["execution"] = self.execution_agent
            self.state["agent_status"]["execution"] = "running"

            # Initialize Compliance Agent
            self.compliance_agent = ComplianceAgent("compliance_agent_001")
            await self.compliance_agent.start()
            self.state["agents"]["compliance"] = self.compliance_agent
            self.state["agent_status"]["compliance"] = "running"

            # Initialize News Sentiment Agent
            self.news_sentiment_agent = NewsSentimentAgent("news_sentiment_agent_001")
            await self.news_sentiment_agent.start()
            self.state["agents"]["news_sentiment"] = self.news_sentiment_agent
            self.state["agent_status"]["news_sentiment"] = "running"

            # Initialize Social Media Agent
            self.social_media_agent = SocialMediaAgent("social_media_agent_001")
            await self.social_media_agent.start()
            self.state["agents"]["social_media"] = self.social_media_agent
            self.state["agent_status"]["social_media"] = "running"

            # Initialize Market Sentiment Agent
            self.market_sentiment_agent = MarketSentimentAgent("market_sentiment_agent_001")
            await self.market_sentiment_agent.start()
            self.state["agents"]["market_sentiment"] = self.market_sentiment_agent
            self.state["agent_status"]["market_sentiment"] = "running"

            # Initialize Deep Learning Agent
            self.deep_learning_agent = DeepLearningAgent("deep_learning_agent_001")
            await self.deep_learning_agent.start()
            self.state["agents"]["deep_learning"] = self.deep_learning_agent
            self.state["agent_status"]["deep_learning"] = "running"

            # Initialize Reinforcement Learning Agent
            self.reinforcement_learning_agent = ReinforcementLearningAgent(
                "reinforcement_learning_agent_001"
            )
            await self.reinforcement_learning_agent.start()
            self.state["agents"]["reinforcement_learning"] = self.reinforcement_learning_agent
            self.state["agent_status"]["reinforcement_learning"] = "running"

            # Initialize AI Model Manager
            self.ai_model_manager = AIModelManager("ai_model_manager_001")
            await self.ai_model_manager.start()
            self.state["agents"]["ai_model_manager"] = self.ai_model_manager
            self.state["agent_status"]["ai_model_manager"] = "running"

            # Initialize Advanced AI Orchestrator
            self.advanced_ai_orchestrator = AdvancedAIOrchestrator("advanced_ai_orchestrator_001")
            await self.advanced_ai_orchestrator.start()
            self.state["agents"]["advanced_ai_orchestrator"] = self.advanced_ai_orchestrator
            self.state["agent_status"]["advanced_ai_orchestrator"] = "running"

            # Initialize NLP Orchestrator
            self.nlp_orchestrator = NLPOrchestrator("nlp_orchestrator_001")
            await self.nlp_orchestrator.start()
            self.state["agents"]["nlp_orchestrator"] = self.nlp_orchestrator
            self.state["agent_status"]["nlp_orchestrator"] = "running"

            # Initialize Chart Pattern Agent
            self.chart_pattern_agent = ChartPatternAgent("chart_pattern_agent_001")
            await self.chart_pattern_agent.start()
            self.state["agents"]["chart_pattern"] = self.chart_pattern_agent
            self.state["agent_status"]["chart_pattern"] = "running"

            # Initialize Technical Indicator Agent
            self.technical_indicator_agent = TechnicalIndicatorAgent(
                "technical_indicator_agent_001"
            )
            await self.technical_indicator_agent.start()
            self.state["agents"]["technical_indicator"] = self.technical_indicator_agent
            self.state["agent_status"]["technical_indicator"] = "running"

            # Initialize Market Visualization Agent
            self.market_visualization_agent = MarketVisualizationAgent(
                "market_visualization_agent_001"
            )
            await self.market_visualization_agent.start()
            self.state["agents"]["market_visualization"] = self.market_visualization_agent
            self.state["agent_status"]["market_visualization"] = "running"

            # Initialize Computer Vision Orchestrator
            self.computer_vision_orchestrator = ComputerVisionOrchestrator(
                "computer_vision_orchestrator_001"
            )
            await self.computer_vision_orchestrator.start()
            self.state["agents"]["computer_vision_orchestrator"] = self.computer_vision_orchestrator
            self.state["agent_status"]["computer_vision_orchestrator"] = "running"

            # Initialize Quantum Algorithm Engine
            self.quantum_algorithm_engine = QuantumAlgorithmEngine("quantum_algorithm_engine_001")
            await self.quantum_algorithm_engine.start()
            self.state["agents"]["quantum_algorithm_engine"] = self.quantum_algorithm_engine
            self.state["agent_status"]["quantum_algorithm_engine"] = "running"

            # Initialize Quantum Machine Learning Agent
            self.quantum_machine_learning_agent = QuantumMachineLearningAgent(
                "quantum_machine_learning_agent_001"
            )
            await self.quantum_machine_learning_agent.start()
            self.state["agents"][
                "quantum_machine_learning_agent"
            ] = self.quantum_machine_learning_agent
            self.state["agent_status"]["quantum_machine_learning_agent"] = "running"

            # Initialize Quantum Optimization Agent
            self.quantum_optimization_agent = QuantumOptimizationAgent(
                "quantum_optimization_agent_001"
            )
            await self.quantum_optimization_agent.start()
            self.state["agents"]["quantum_optimization_agent"] = self.quantum_optimization_agent
            self.state["agent_status"]["quantum_optimization_agent"] = "running"

            # Initialize Quantum Trading Engine
            self.quantum_trading_engine = QuantumTradingEngine("quantum_trading_engine_001")
            await self.quantum_trading_engine.start()
            self.state["agents"]["quantum_trading_engine"] = self.quantum_trading_engine
            self.state["agent_status"]["quantum_trading_engine"] = "running"

            # Initialize Interdimensional Signal Decoder
            self.interdimensional_signal_decoder = InterdimensionalSignalDecoder(
                "interdimensional_signal_decoder_001"
            )
            await self.interdimensional_signal_decoder.start()
            self.state["agents"][
                "interdimensional_signal_decoder"
            ] = self.interdimensional_signal_decoder
            self.state["agent_status"]["interdimensional_signal_decoder"] = "running"

            # Initialize Neuro-Synchronization Engine
            self.neuro_synchronization_engine = NeuroSynchronizationEngine(
                "neuro_synchronization_engine_001"
            )
            await self.neuro_synchronization_engine.start()
            self.state["agents"]["neuro_synchronization_engine"] = self.neuro_synchronization_engine
            self.state["agent_status"]["neuro_synchronization_engine"] = "running"

            # Initialize Cosmic Pattern Recognizer
            self.cosmic_pattern_recognizer = CosmicPatternRecognizer(
                "cosmic_pattern_recognizer_001"
            )
            await self.cosmic_pattern_recognizer.start()
            self.state["agents"]["cosmic_pattern_recognizer"] = self.cosmic_pattern_recognizer
            self.state["agent_status"]["cosmic_pattern_recognizer"] = "running"

            # Initialize AuraNet Channel Interface
            self.auranet_channel = AuraNetChannel("auranet_channel_001")
            await self.auranet_channel.start()
            self.state["agents"]["auranet_channel"] = self.auranet_channel
            self.state["agent_status"]["auranet_channel"] = "running"

            print("✅ All agents initialized successfully")

        except Exception as e:
            print(f"❌ Error initializing agents: {e}")
            raise

    async def start_agent_monitoring(self):
        """Start monitoring all agents"""
        try:
            # Subscribe to agent status updates
            pubsub = self.redis_client.pubsub()
            pubsub.subscribe("agent:status")
            pubsub.subscribe("agent:errors")
            pubsub.subscribe("agent:broadcast")

            # Start monitoring listener
            asyncio.create_task(self.listen_agent_updates(pubsub))

            print("👁️ Agent monitoring started")

        except Exception as e:
            print(f"❌ Error starting agent monitoring: {e}")

    async def listen_agent_updates(self, pubsub):
        """Listen for agent updates"""
        try:
            for message in pubsub.listen():
                if not self.running:
                    break

                if message["type"] == "message":
                    update_data = json.loads(message["data"])
                    await self.process_agent_update(update_data)

        except Exception as e:
            print(f"❌ Error in agent updates listener: {e}")
        finally:
            pubsub.close()

    async def process_agent_update(self, update_data: Dict[str, Any]):
        """Process agent update"""
        try:
            update_type = update_data.get("type")
            agent_id = update_data.get("agent_id")

            if update_type == "status_update":
                await self.handle_agent_status_update(update_data)
            elif update_type == "error":
                await self.handle_agent_error(update_data)
            elif update_type == "performance_update":
                await self.handle_performance_report(update_data)

        except Exception as e:
            print(f"❌ Error processing agent update: {e}")

    async def initialize_system_coordination(self):
        """Initialize system coordination"""
        try:
            # Set up agent communication channels
            await self.setup_agent_communication()

            # Initialize system status
            self.state["system_status"] = "operational"

            # Broadcast system ready
            await self.broadcast_message(
                {
                    "type": "system_ready",
                    "timestamp": datetime.now().isoformat(),
                }
            )

            print("🎼 System coordination initialized")

        except Exception as e:
            print(f"❌ Error initializing system coordination: {e}")

    async def setup_agent_communication(self):
        """Set up agent communication channels"""
        try:
            # Define communication patterns
            communication_patterns = {
                "strategy_to_risk": {
                    "from": "strategy_agent_001",
                    "to": "risk_agent_001",
                    "channel": "trading_signal",
                },
                "risk_to_execution": {
                    "from": "risk_agent_001",
                    "to": "execution_agent_001",
                    "channel": "approved_trading_signal",
                },
                "execution_to_compliance": {
                    "from": "execution_agent_001",
                    "to": "compliance_agent_001",
                    "channel": "trade_executed",
                },
                "compliance_to_strategy": {
                    "from": "compliance_agent_001",
                    "to": "strategy_agent_001",
                    "channel": "compliance_approved_signal",
                },
            }

            self.state["agent_communication"] = communication_patterns

            # Store in Redis
            self.redis_client.set(
                "agent_communication",
                json.dumps(communication_patterns),
                ex=3600,
            )

        except Exception as e:
            print(f"❌ Error setting up agent communication: {e}")

    async def monitor_agents(self):
        """Monitor all agents"""
        try:
            for agent_name, agent in self.state["agents"].items():
                # Check agent health
                is_healthy = agent.is_healthy()

                # Update status
                if is_healthy:
                    self.state["agent_status"][agent_name] = "healthy"
                else:
                    self.state["agent_status"][agent_name] = "unhealthy"

                    # Attempt to restart unhealthy agent
                    await self.restart_agent(agent_name)

            # Update system status based on agent health
            healthy_agents = sum(
                1 for status in self.state["agent_status"].values() if status == "healthy"
            )
            total_agents = len(self.state["agent_status"])

            if healthy_agents == total_agents:
                self.state["system_status"] = "operational"
            elif healthy_agents > total_agents // 2:
                self.state["system_status"] = "degraded"
            else:
                self.state["system_status"] = "critical"

        except Exception as e:
            print(f"❌ Error monitoring agents: {e}")

    async def restart_agent(self, agent_name: str):
        """Restart an unhealthy agent"""
        try:
            print(f"🔄 Restarting {agent_name} agent")

            agent = self.state["agents"].get(agent_name)
            if agent:
                # Stop agent
                await agent.stop()

                # Wait a moment
                await asyncio.sleep(5)

                # Restart agent
                await agent.start()

                print(f"✅ {agent_name} agent restarted")

        except Exception as e:
            print(f"❌ Error restarting {agent_name} agent: {e}")

    async def coordinate_agents(self):
        """Coordinate agent communication and activities"""
        try:
            # Ensure proper message routing
            await self.route_agent_messages()

            # Coordinate agent activities
            await self.coordinate_agent_activities()

            # Update coordination timestamp
            self.state["last_coordination"] = datetime.now().isoformat()

        except Exception as e:
            print(f"❌ Error coordinating agents: {e}")

    async def route_agent_messages(self):
        """Route messages between agents"""
        try:
            # This would handle message routing logic
            # For now, agents communicate directly via Redis channels

            # Monitor message flow
            message_stats = {
                "strategy_to_risk": 0,
                "risk_to_execution": 0,
                "execution_to_compliance": 0,
                "total_messages": 0,
            }

            # Update message statistics
            self.state["agent_communication"]["message_stats"] = message_stats

        except Exception as e:
            print(f"❌ Error routing agent messages: {e}")

    async def coordinate_agent_activities(self):
        """Coordinate agent activities"""
        try:
            # Ensure agents are working together effectively

            # Check if all agents are ready
            all_ready = all(status == "healthy" for status in self.state["agent_status"].values())

            if all_ready:
                # Broadcast coordination signal
                await self.broadcast_message(
                    {
                        "type": "coordination_signal",
                        "status": "all_agents_ready",
                        "timestamp": datetime.now().isoformat(),
                    }
                )

        except Exception as e:
            print(f"❌ Error coordinating agent activities: {e}")

    async def handle_agent_status_update(self, message: Dict[str, Any]):
        """Handle agent status update"""
        try:
            agent_id = message.get("agent_id")
            status = message.get("status")
            health = message.get("health", "unknown")

            # Update agent status
            for agent_name, agent in self.state["agents"].items():
                if agent.agent_id == agent_id:
                    self.state["agent_status"][agent_name] = health
                    break

            print(f"📊 Agent {agent_id} status: {status} ({health})")

        except Exception as e:
            print(f"❌ Error handling agent status update: {e}")

    async def handle_agent_error(self, message: Dict[str, Any]):
        """Handle agent error"""
        try:
            agent_id = message.get("agent_id")
            error = message.get("error")

            print(f"❌ Agent {agent_id} error: {error}")

            # Log error
            error_log = {
                "agent_id": agent_id,
                "error": error,
                "timestamp": datetime.now().isoformat(),
            }

            self.redis_client.lpush("agent_errors", json.dumps(error_log))

            # Attempt recovery
            await self.handle_agent_recovery(agent_id, error)

        except Exception as e:
            print(f"❌ Error handling agent error: {e}")

    async def handle_agent_recovery(self, agent_id: str, error: str):
        """Handle agent recovery"""
        try:
            print(f"🔄 Attempting recovery for agent {agent_id}")

            # Find agent by ID
            for agent_name, agent in self.state["agents"].items():
                if agent.agent_id == agent_id:
                    # Attempt restart
                    await self.restart_agent(agent_name)
                    break

        except Exception as e:
            print(f"❌ Error handling agent recovery: {e}")

    async def handle_system_command(self, message: Dict[str, Any]):
        """Handle system command"""
        try:
            command = message.get("command")
            params = message.get("params", {})

            print(f"🎛️ Executing system command: {command}")

            if command == "restart_agent":
                agent_name = params.get("agent_name")
                if agent_name:
                    await self.restart_agent(agent_name)

            elif command == "stop_system":
                await self.stop_system()

            elif command == "start_system":
                await self.start_system()

            elif command == "get_status":
                await self.send_system_status(message.get("from_agent"))

            elif command == "update_config":
                await self.update_system_config(params)

        except Exception as e:
            print(f"❌ Error handling system command: {e}")

    async def handle_performance_report(self, message: Dict[str, Any]):
        """Handle performance report from agent"""
        try:
            agent_id = message.get("agent_id")
            performance = message.get("performance", {})

            # Update performance metrics
            self.state["performance_metrics"][agent_id] = performance

            # Store in Redis
            self.redis_client.set(
                f"agent_performance:{agent_id}",
                json.dumps(performance),
                ex=300,
            )

        except Exception as e:
            print(f"❌ Error handling performance report: {e}")

    async def handle_agent_communication(self, message: Dict[str, Any]):
        """Handle agent communication"""
        try:
            from_agent = message.get("from_agent")
            to_agent = message.get("to_agent")
            message_type = message.get("type")

            # Log communication
            comm_log = {
                "from": from_agent,
                "to": to_agent,
                "type": message_type,
                "timestamp": datetime.now().isoformat(),
            }

            self.redis_client.lpush("agent_communication_log", json.dumps(comm_log))

        except Exception as e:
            print(f"❌ Error handling agent communication: {e}")

    async def stop_system(self):
        """Stop the entire system"""
        try:
            print("🛑 Stopping AI trading system...")

            # Stop all agents
            for agent_name, agent in self.state["agents"].items():
                await agent.stop()

            # Update system status
            self.state["system_status"] = "stopped"

            # Broadcast system stopped
            await self.broadcast_message(
                {
                    "type": "system_stopped",
                    "timestamp": datetime.now().isoformat(),
                }
            )

            print("✅ AI trading system stopped")

        except Exception as e:
            print(f"❌ Error stopping system: {e}")

    async def start_system(self):
        """Start the entire system"""
        try:
            print("🚀 Starting AI trading system...")

            # Start all agents
            for agent_name, agent in self.state["agents"].items():
                await agent.start()

            # Update system status
            self.state["system_status"] = "operational"

            # Broadcast system started
            await self.broadcast_message(
                {
                    "type": "system_started",
                    "timestamp": datetime.now().isoformat(),
                }
            )

            print("✅ AI trading system started")

        except Exception as e:
            print(f"❌ Error starting system: {e}")

    async def send_system_status(self, target_agent: str = None):
        """Send system status"""
        try:
            status = {
                "type": "system_status",
                "system_status": self.state["system_status"],
                "agent_status": self.state["agent_status"],
                "performance_metrics": self.state["performance_metrics"],
                "timestamp": datetime.now().isoformat(),
            }

            if target_agent:
                await self.send_message(target_agent, status)
            else:
                await self.broadcast_message(status)

        except Exception as e:
            print(f"❌ Error sending system status: {e}")

    async def update_system_config(self, config: Dict[str, Any]):
        """Update system configuration"""
        try:
            # Update orchestrator configuration
            if "orchestrator" in config:
                self.state.update(config["orchestrator"])

            # Update agent configurations
            for agent_name, agent_config in config.get("agents", {}).items():
                if agent_name in self.state["agents"]:
                    agent = self.state["agents"][agent_name]
                    agent.state.update(agent_config)

            # Store configuration in Redis
            self.redis_client.set("system_config", json.dumps(config), ex=3600)

            print("✅ System configuration updated")

        except Exception as e:
            print(f"❌ Error updating system config: {e}")

    async def check_system_health(self):
        """Check overall system health"""
        try:
            # Calculate system health score
            health_score = 0
            total_agents = len(self.state["agent_status"])

            for status in self.state["agent_status"].values():
                if status == "healthy":
                    health_score += 100
                elif status == "degraded":
                    health_score += 50
                else:
                    health_score += 0

            health_score = health_score // total_agents

            # Update system health
            if health_score >= 90:
                self.state["system_status"] = "operational"
            elif health_score >= 50:
                self.state["system_status"] = "degraded"
            else:
                self.state["system_status"] = "critical"

            # Broadcast health update
            await self.broadcast_message(
                {
                    "type": "system_health_update",
                    "health_score": health_score,
                    "system_status": self.state["system_status"],
                    "timestamp": datetime.now().isoformat(),
                }
            )

        except Exception as e:
            print(f"❌ Error checking system health: {e}")

    async def update_system_metrics(self):
        """Update system metrics"""
        try:
            metrics = {
                "agent_id": self.agent_id,
                "system_status": self.state["system_status"],
                "agent_status": self.state["agent_status"],
                "performance_metrics": self.state["performance_metrics"],
                "last_coordination": self.state["last_coordination"],
                "timestamp": datetime.now().isoformat(),
            }

            # Store metrics in Redis
            self.redis_client.set(f"agent_metrics:{self.agent_id}", json.dumps(metrics), ex=300)

        except Exception as e:
            print(f"❌ Error updating system metrics: {e}")

    async def process_market_data(self, market_data: Dict[str, Any]):
        """Process incoming market data and distribute to agents"""
        try:
            print(f"📊 Processing market data for agent orchestration")

            # Update market data in state
            self.state["last_market_data"] = market_data
            self.state["last_market_update"] = datetime.now().isoformat()

            # Distribute market data to all agents
            for agent_name, agent in self.state["agents"].items():
                try:
                    if hasattr(agent, 'process_market_data'):
                        await agent.process_market_data(market_data)
                        print(f"✅ Market data sent to {agent_name}")
                    else:
                        print(f"⚠️ Agent {agent_name} doesn't have process_market_data method")
                except Exception as e:
                    print(f"❌ Error sending market data to {agent_name}: {e}")

            # Update system metrics
            await self.update_system_metrics()

            # Check system health after processing
            await self.check_system_health()

            print(f"✅ Market data processed for agent orchestration")

        except Exception as e:
            print(f"❌ Error processing market data for orchestration: {e}")
            await self.broadcast_error(f"Agent orchestration market data error: {e}")

    async def stop(self):
        """Stop the orchestrator and all agents"""
        try:
            print("🛑 Stopping Agent Orchestrator...")

            # Stop all agents
            for agent_name, agent in self.state["agents"].items():
                await agent.stop()

            # Stop orchestrator
            await super().stop()

            print("✅ Agent Orchestrator stopped")

        except Exception as e:
            print(f"❌ Error stopping orchestrator: {e}")


async def main():
    """Main function to run the agent orchestrator"""
    orchestrator = None
    try:
        # Create and start orchestrator
        orchestrator = AgentOrchestrator()
        await orchestrator.start()

        # Keep running with memory management
        while True:
            await asyncio.sleep(1)

            # Periodic memory cleanup
            if hasattr(orchestrator, 'state') and 'performance_metrics' in orchestrator.state:
                # Clean up old performance metrics to prevent memory accumulation
                if len(orchestrator.state['performance_metrics']) > 1000:
                    # Keep only the last 500 metrics
                    orchestrator.state['performance_metrics'] = orchestrator.state['performance_metrics'][-500:]

                # Clean up old agent status history
                if 'agent_status_history' in orchestrator.state:
                    if len(orchestrator.state['agent_status_history']) > 100:
                        orchestrator.state['agent_status_history'] = orchestrator.state['agent_status_history'][-100:]

    except KeyboardInterrupt:
        print("🛑 Shutting down...")
    except Exception as e:
        print(f"❌ Error in main: {e}")
    finally:
        if orchestrator:
            await orchestrator.stop()


if __name__ == "__main__":
    asyncio.run(main())
