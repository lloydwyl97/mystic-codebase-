"""
AuraNet Channel Interface
Biofeedback/EEG-driven API interface for personal energy tuning and system alignment
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from typing import Any

import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.agents.base_agent import BaseAgent


class AuraNetChannel(BaseAgent):
    """AuraNet Channel Interface - Biofeedback/EEG API for energy tuning and alignment"""

    def __init__(self, agent_id: str = "auranet_channel_001"):
        super().__init__(agent_id, "auranet_channel")
        self.state.update(
            {
                "energy_status": {},
                "biofeedback_history": [],
                "alignment_metrics": {},
                "last_update": None,
                "update_count": 0,
            }
        )
        # Supported input types
        self.input_types = ["manual", "eeg", "bci", "wearable"]
        # Register handlers
        self.register_handler("sync_energy_status", self.handle_sync_energy_status)
        self.register_handler("get_alignment_metrics", self.handle_get_alignment_metrics)
        self.register_handler("update_biofeedback", self.handle_update_biofeedback)
        print(f"ðŸŒ€ AuraNet Channel Interface {agent_id} initialized")

    async def initialize(self):
        try:
            await self.load_auranet_config()
            await self.start_biofeedback_monitoring()
            print(f"âœ… AuraNet Channel Interface {self.agent_id} initialized successfully")
        except Exception as e:
            print(f"âŒ Error initializing AuraNet Channel: {e}")
            self.update_health_status("error")

    async def process_loop(self):
        while self.running:
            try:
                await self.monitor_biofeedback()
                await self.update_alignment_metrics()
                await self.cleanup_old_data()
                await asyncio.sleep(60)
            except Exception as e:
                print(f"âŒ Error in AuraNet processing loop: {e}")
                await asyncio.sleep(120)

    async def load_auranet_config(self):
        try:
            config_data = self.redis_client.get("auranet_config")
            if config_data:
                self.state["energy_status"].update(json.loads(config_data))
            print("AuraNet config loaded")
        except Exception as e:
            print(f"âŒ Error loading AuraNet config: {e}")

    async def start_biofeedback_monitoring(self):
        try:
            pubsub = self.redis_client.pubsub()
            pubsub.subscribe("biofeedback_data")
            asyncio.create_task(self.listen_biofeedback_data(pubsub))
            print("ðŸ“¡ Biofeedback monitoring started")
        except Exception as e:
            print(f"âŒ Error starting biofeedback monitoring: {e}")

    async def listen_biofeedback_data(self, pubsub):
        try:
            for message in pubsub.listen():
                if not self.running:
                    break
                if message["type"] == "message":
                    data = json.loads(message["data"])
                    await self.process_biofeedback_data(data)
        except Exception as e:
            print(f"âŒ Error in biofeedback data listener: {e}")
        finally:
            pubsub.close()

    async def process_biofeedback_data(self, data: dict[str, Any]):
        try:
            input_type = data.get("input_type", "manual")
            if input_type not in self.input_types:
                return
            # Store biofeedback
            entry = {
                "input_type": input_type,
                "energy_level": data.get("energy_level", 0.5),
                "focus": data.get("focus", 0.5),
                "stress": data.get("stress", 0.5),
                "timestamp": data.get("timestamp", datetime.now().isoformat()),
            }
            self.state["biofeedback_history"].append(entry)
            self.state["last_update"] = entry["timestamp"]
            self.state["update_count"] += 1
            # Limit history
            if len(self.state["biofeedback_history"]) > 100:
                self.state["biofeedback_history"] = self.state["biofeedback_history"][-100:]
            # Store in Redis
            self.redis_client.set(
                f"auranet_biofeedback:{entry['timestamp']}",
                json.dumps(entry),
                ex=3600,
            )
        except Exception as e:
            print(f"âŒ Error processing biofeedback data: {e}")

    async def monitor_biofeedback(self):
        try:
            # Analyze last 10 entries
            history = self.state["biofeedback_history"][-10:]
            if not history:
                return
            avg_energy = np.mean([e["energy_level"] for e in history])
            avg_focus = np.mean([e["focus"] for e in history])
            avg_stress = np.mean([e["stress"] for e in history])
            self.state["energy_status"] = {
                "avg_energy": float(avg_energy),
                "avg_focus": float(avg_focus),
                "avg_stress": float(avg_stress),
                "timestamp": datetime.now().isoformat(),
            }
            # Store in Redis
            self.redis_client.set(
                "auranet_energy_status",
                json.dumps(self.state["energy_status"]),
                ex=600,
            )
        except Exception as e:
            print(f"âŒ Error monitoring biofeedback: {e}")

    async def update_alignment_metrics(self):
        try:
            # Calculate alignment metrics
            status = self.state["energy_status"]
            alignment = {
                "alignment_score": float(
                    status.get("avg_energy", 0.5) * 0.5
                    + status.get("avg_focus", 0.5) * 0.4
                    - status.get("avg_stress", 0.5) * 0.3
                ),
                "energy_bias": float(status.get("avg_energy", 0.5)),
                "focus_bias": float(status.get("avg_focus", 0.5)),
                "stress_bias": float(status.get("avg_stress", 0.5)),
                "timestamp": datetime.now().isoformat(),
            }
            self.state["alignment_metrics"] = alignment
            self.redis_client.set("auranet_alignment_metrics", json.dumps(alignment), ex=600)
        except Exception as e:
            print(f"âŒ Error updating alignment metrics: {e}")

    async def cleanup_old_data(self):
        try:
            # Clean up old biofeedback (keep last 100)
            if len(self.state["biofeedback_history"]) > 100:
                self.state["biofeedback_history"] = self.state["biofeedback_history"][-100:]
        except Exception as e:
            print(f"âŒ Error cleaning up old data: {e}")

    async def handle_sync_energy_status(self, message: dict[str, Any]):
        try:
            input_type = message.get("input_type", "manual")
            energy_level = message.get("energy_level", 0.5)
            focus = message.get("focus", 0.5)
            stress = message.get("stress", 0.5)
            print(
                f"ðŸŒ€ Manual energy sync: {input_type}, energy={energy_level}, "
                f"focus={focus}, stress={stress}"
            )
            entry = {
                "input_type": input_type,
                "energy_level": energy_level,
                "focus": focus,
                "stress": stress,
                "timestamp": datetime.now().isoformat(),
            }
            self.state["biofeedback_history"].append(entry)
            self.state["last_update"] = entry["timestamp"]
            self.state["update_count"] += 1
            # Store in Redis
            self.redis_client.set(
                f"auranet_biofeedback:{entry['timestamp']}",
                json.dumps(entry),
                ex=3600,
            )
            # Send response
            response = {
                "type": "energy_sync_complete",
                "entry": entry,
                "timestamp": datetime.now().isoformat(),
            }
            sender = message.get("from_agent")
            if sender:
                await self.send_message(sender, response)
        except Exception as e:
            print(f"âŒ Error handling energy sync: {e}")
            await self.broadcast_error(f"Energy sync error: {e}")

    async def handle_get_alignment_metrics(self, message: dict[str, Any]):
        try:
            print("ðŸŒ€ Alignment metrics requested")
            response = {
                "type": "alignment_metrics_response",
                "alignment_metrics": self.state["alignment_metrics"],
                "timestamp": datetime.now().isoformat(),
            }
            sender = message.get("from_agent")
            if sender:
                await self.send_message(sender, response)
        except Exception as e:
            print(f"âŒ Error handling alignment metrics request: {e}")
            await self.broadcast_error(f"Alignment metrics error: {e}")

    async def handle_update_biofeedback(self, message: dict[str, Any]):
        try:
            print("ðŸŒ€ Biofeedback update received")
            await self.process_biofeedback_data(message)
            response = {
                "type": "biofeedback_update_complete",
                "timestamp": datetime.now().isoformat(),
            }
            sender = message.get("from_agent")
            if sender:
                await self.send_message(sender, response)
        except Exception as e:
            print(f"âŒ Error handling biofeedback update: {e}")
            await self.broadcast_error(f"Biofeedback update error: {e}")


if __name__ == "__main__":
    agent = AuraNetChannel()
    asyncio.run(agent.start())


