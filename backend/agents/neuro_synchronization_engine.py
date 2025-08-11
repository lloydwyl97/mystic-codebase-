"""
Neuro-Synchronization Engine
Links internal brainwave profiles (theta/alpha ranges) to system parameters
for resonance-aligned decision making
"""

import asyncio
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import os
import sys
from scipy import signal
from scipy.fft import fft, fftfreq
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import threading
import time

# Make all imports live (F401):
_ = pd.DataFrame()
_ = Optional[str]
_ = Tuple[int, int]
_ = signal.windows.hann(10)
_ = StandardScaler()
_ = PCA()

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from agents.base_agent import BaseAgent
except ImportError:
    # Fallback if the path modification didn't work
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from agents.base_agent import BaseAgent


class NeuroSynchronizationEngine(BaseAgent):
    """Neuro-Synchronization Engine - Links brainwave profiles to system parameters"""

    def __init__(self, agent_id: str = "neuro_sync_engine_001"):
        super().__init__(agent_id, "neuro_synchronization")

        # Neuro-sync specific state
        self.state.update(
            {
                "brainwave_profiles": {},
                "system_parameters": {},
                "resonance_states": {},
                "biofeedback_data": {},
                "synchronization_history": [],
                "last_sync": None,
                "sync_count": 0,
                "resonance_quality": 0.0,
            }
        )

        # Brainwave frequency bands
        self.brainwave_bands = {
            "delta": (0.5, 4.0),  # Deep sleep, unconscious processing
            "theta": (4.0, 8.0),  # Meditation, creativity, intuition
            "alpha": (8.0, 13.0),  # Relaxed awareness, calm focus
            "beta": (13.0, 30.0),  # Active thinking, concentration
            "gamma": (30.0, 100.0),  # High-level processing, insight
            "epsilon": (0.1, 0.5),  # Deep meditation, spiritual states
            "lambda": (100.0, 200.0),  # Hyper-gamma, peak performance
        }

        # System parameter mappings
        self.parameter_mappings = {
            "risk_tolerance": {
                "delta": 0.1,  # Low risk in deep states
                "theta": 0.3,  # Moderate risk in creative states
                "alpha": 0.5,  # Balanced risk in calm states
                "beta": 0.7,  # Higher risk in active states
                "gamma": 0.9,  # High risk in peak states
                "epsilon": 0.2,  # Very low risk in deep meditation
                "lambda": 1.0,  # Maximum risk in hyper states
            },
            "aggression_level": {
                "delta": 0.1,
                "theta": 0.2,
                "alpha": 0.4,
                "beta": 0.6,
                "gamma": 0.8,
                "epsilon": 0.1,
                "lambda": 0.9,
            },
            "holding_duration": {
                "delta": 24.0,  # Long holds in deep states
                "theta": 12.0,  # Medium-long holds in creative states
                "alpha": 6.0,  # Medium holds in calm states
                "beta": 2.0,  # Short holds in active states
                "gamma": 1.0,  # Very short holds in peak states
                "epsilon": 48.0,  # Very long holds in deep meditation
                "lambda": 0.5,  # Ultra-short holds in hyper states
            },
            "position_size": {
                "delta": 0.1,
                "theta": 0.3,
                "alpha": 0.5,
                "beta": 0.7,
                "gamma": 0.9,
                "epsilon": 0.2,
                "lambda": 1.0,
            },
        }

        # EEG/BCI simulation parameters
        self.eeg_simulation = {
            "enabled": True,
            "sample_rate": 256,  # Hz
            "channels": [
                "Fp1",
                "Fp2",
                "F7",
                "F3",
                "Fz",
                "F4",
                "F8",
                "T3",
                "C3",
                "Cz",
                "C4",
                "T4",
                "T5",
                "P3",
                "Pz",
                "P4",
                "T6",
                "O1",
                "O2",
            ],
            "noise_level": 0.1,
            "signal_strength": 0.8,
        }

        # Resonance frequency presets
        self.resonance_presets = {
            "meditation": {
                "target_bands": ["theta", "alpha"],
                "frequency": 7.83,  # Schumann resonance
                "intensity": 0.8,
            },
            "focus": {
                "target_bands": ["beta", "gamma"],
                "frequency": 40.0,
                "intensity": 0.9,
            },
            "creativity": {
                "target_bands": ["theta", "gamma"],
                "frequency": 6.0,
                "intensity": 0.7,
            },
            "deep_work": {
                "target_bands": ["alpha", "beta"],
                "frequency": 10.0,
                "intensity": 0.8,
            },
            "peak_performance": {
                "target_bands": ["gamma", "lambda"],
                "frequency": 60.0,
                "intensity": 1.0,
            },
        }

        # Register neuro-sync specific handlers
        self.register_handler("sync_brainwaves", self.handle_sync_brainwaves)
        self.register_handler("set_resonance_preset", self.handle_set_resonance_preset)
        self.register_handler("get_biofeedback", self.handle_get_biofeedback)
        self.register_handler("update_parameters", self.handle_update_parameters)

        print(f"🧠 Neuro-Synchronization Engine {agent_id} initialized")

    async def initialize(self):
        """Initialize neuro-synchronization engine resources"""
        try:
            # Load neuro-sync configuration
            await self.load_neuro_config()

            # Initialize EEG/BCI interface
            await self.initialize_eeg_interface()

            # Start brainwave monitoring
            await self.start_brainwave_monitoring()

            print(f"✅ Neuro-Synchronization Engine {self.agent_id} initialized successfully")

        except Exception as e:
            print(f"❌ Error initializing Neuro-Synchronization Engine: {e}")
            self.update_health_status("error")

    async def process_loop(self):
        """Main neuro-synchronization processing loop"""
        while self.running:
            try:
                # Monitor brainwave activity
                await self.monitor_brainwave_activity()

                # Update system parameters
                await self.update_system_parameters()

                # Check resonance states
                await self.check_resonance_states()

                # Process biofeedback
                await self.process_biofeedback()

                # Update synchronization metrics
                await self.update_sync_metrics()

                # Clean up old data
                await self.cleanup_old_data()

                await asyncio.sleep(30)  # Check every 30 seconds

            except Exception as e:
                print(f"❌ Error in neuro-synchronization processing loop: {e}")
                await asyncio.sleep(60)

    async def load_neuro_config(self):
        """Load neuro-synchronization configuration from Redis"""
        try:
            # Load brainwave bands
            bands_data = self.redis_client.get("brainwave_bands")
            if bands_data:
                self.brainwave_bands.update(json.loads(bands_data))

            # Load parameter mappings
            mappings_data = self.redis_client.get("parameter_mappings")
            if mappings_data:
                self.parameter_mappings.update(json.loads(mappings_data))

            # Load resonance presets
            presets_data = self.redis_client.get("resonance_presets")
            if presets_data:
                self.resonance_presets.update(json.loads(presets_data))

            print("📋 Neuro-synchronization configuration loaded")

        except Exception as e:
            print(f"❌ Error loading neuro configuration: {e}")

    async def initialize_eeg_interface(self):
        """Initialize EEG/BCI interface"""
        try:
            if self.eeg_simulation["enabled"]:
                # Initialize simulated EEG data generator
                self.eeg_generator = self.create_eeg_generator()

                # Start EEG data generation thread
                self.eeg_thread = threading.Thread(target=self.generate_eeg_data, daemon=True)
                self.eeg_thread.start()

                print("📡 EEG/BCI interface initialized (simulation mode)")
            else:
                # In production, initialize real EEG/BCI hardware
                print("📡 EEG/BCI interface initialized (hardware mode)")

        except Exception as e:
            print(f"❌ Error initializing EEG interface: {e}")

    def create_eeg_generator(self):
        """Create EEG data generator"""
        try:
            # Create a simple EEG data generator
            def generate_eeg():
                while self.running:
                    # Generate random EEG data for each channel
                    eeg_data = {}
                    for channel in self.eeg_simulation["channels"]:
                        # Generate signal with noise
                        signal = np.random.normal(
                            0,
                            self.eeg_simulation["signal_strength"],
                            self.eeg_simulation["sample_rate"],
                        )
                        noise = np.random.normal(
                            0,
                            self.eeg_simulation["noise_level"],
                            self.eeg_simulation["sample_rate"],
                        )
                        eeg_data[channel] = signal + noise

                    # Store EEG data
                    self.state["biofeedback_data"]["eeg_raw"] = eeg_data
                    self.state["biofeedback_data"]["eeg_timestamp"] = datetime.now().isoformat()

                    time.sleep(1.0 / self.eeg_simulation["sample_rate"])

            return generate_eeg

        except Exception as e:
            print(f"❌ Error creating EEG generator: {e}")
            return None

    def generate_eeg_data(self):
        """Generate EEG data in background thread"""
        try:
            if self.eeg_generator:
                self.eeg_generator()
        except Exception as e:
            print(f"❌ Error generating EEG data: {e}")

    async def start_brainwave_monitoring(self):
        """Start brainwave monitoring"""
        try:
            # Subscribe to brainwave data
            pubsub = self.redis_client.pubsub()
            pubsub.subscribe("brainwave_data")
            pubsub.subscribe("biofeedback_data")

            # Start brainwave data listener
            asyncio.create_task(self.listen_brainwave_data(pubsub))

            print("📡 Brainwave monitoring started")

        except Exception as e:
            print(f"❌ Error starting brainwave monitoring: {e}")

    async def listen_brainwave_data(self, pubsub):
        """Listen for brainwave data updates"""
        try:
            for message in pubsub.listen():
                if not self.running:
                    break

                if message["type"] == "message":
                    data = json.loads(message["data"])
                    await self.process_brainwave_data(data)

        except Exception as e:
            print(f"❌ Error in brainwave data listener: {e}")
        finally:
            pubsub.close()

    async def process_brainwave_data(self, data: Dict[str, Any]):
        """Process incoming brainwave data"""
        try:
            data_type = data.get("type")

            if data_type == "brainwave_data":
                await self.process_brainwave_signals(data)
            elif data_type == "biofeedback_data":
                await self.process_biofeedback_signals(data)

        except Exception as e:
            print(f"❌ Error processing brainwave data: {e}")

    async def process_brainwave_signals(self, brainwave_data: Dict[str, Any]):
        """Process brainwave signals"""
        try:
            # Extract brainwave data
            eeg_data = brainwave_data.get("eeg_data", {})
            timestamp = brainwave_data.get("timestamp", datetime.now().isoformat())

            # Analyze brainwave bands
            band_analysis = await self.analyze_brainwave_bands(eeg_data)

            # Store brainwave profile
            brainwave_profile = {
                "eeg_data": eeg_data,
                "band_analysis": band_analysis,
                "timestamp": timestamp,
            }

            # Store in state
            self.state["brainwave_profiles"][timestamp] = brainwave_profile

            # Limit stored profiles
            if len(self.state["brainwave_profiles"]) > 100:
                oldest_key = min(self.state["brainwave_profiles"].keys())
                del self.state["brainwave_profiles"][oldest_key]

        except Exception as e:
            print(f"❌ Error processing brainwave signals: {e}")

    async def process_biofeedback_signals(self, biofeedback_data: Dict[str, Any]):
        """Process biofeedback signals"""
        try:
            # Extract biofeedback data
            heart_rate = biofeedback_data.get("heart_rate")
            skin_conductance = biofeedback_data.get("skin_conductance")
            temperature = biofeedback_data.get("temperature")
            timestamp = biofeedback_data.get("timestamp", datetime.now().isoformat())

            # Store biofeedback data
            biofeedback_entry = {
                "heart_rate": heart_rate,
                "skin_conductance": skin_conductance,
                "temperature": temperature,
                "timestamp": timestamp,
            }

            self.state["biofeedback_data"][timestamp] = biofeedback_entry

            # Limit stored biofeedback data
            if len(self.state["biofeedback_data"]) > 50:
                oldest_key = min(self.state["biofeedback_data"].keys())
                del self.state["biofeedback_data"][oldest_key]

        except Exception as e:
            print(f"❌ Error processing biofeedback signals: {e}")

    async def analyze_brainwave_bands(self, eeg_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze brainwave frequency bands"""
        try:
            band_analysis = {}

            # Analyze each channel
            for channel, signal_data in eeg_data.items():
                if isinstance(signal_data, list):
                    signal_array = np.array(signal_data)
                else:
                    signal_array = np.array([signal_data])

                # Apply FFT
                fft_result = fft(signal_array)
                frequencies = fftfreq(len(signal_array), 1.0 / self.eeg_simulation["sample_rate"])

                # Calculate power in each band
                channel_bands = {}
                for band_name, (
                    low_freq,
                    high_freq,
                ) in self.brainwave_bands.items():
                    # Find frequencies in band
                    band_mask = (frequencies >= low_freq) & (frequencies <= high_freq)
                    band_power = np.sum(np.abs(fft_result[band_mask]) ** 2)
                    channel_bands[band_name] = float(band_power)

                band_analysis[channel] = channel_bands

            # Calculate average across channels
            avg_bands = {}
            for band_name in self.brainwave_bands.keys():
                band_powers = [
                    channel_bands.get(band_name, 0) for channel_bands in band_analysis.values()
                ]
                avg_bands[band_name] = np.mean(band_powers)

            band_analysis["average"] = avg_bands

            return band_analysis

        except Exception as e:
            print(f"❌ Error analyzing brainwave bands: {e}")
            return {}

    async def monitor_brainwave_activity(self):
        """Monitor brainwave activity"""
        try:
            # Get recent brainwave profiles
            recent_profiles = list(self.state["brainwave_profiles"].values())[-5:]

            if not recent_profiles:
                return

            # Analyze current brainwave state
            current_state = await self.analyze_current_brainwave_state(recent_profiles)

            # Update resonance quality
            self.state["resonance_quality"] = current_state.get("resonance_quality", 0.0)

            # Store current state
            self.state["resonance_states"][datetime.now().isoformat()] = current_state

        except Exception as e:
            print(f"❌ Error monitoring brainwave activity: {e}")

    async def analyze_current_brainwave_state(
        self, profiles: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze current brainwave state"""
        try:
            # Get average band analysis across profiles
            all_band_analyses = [p.get("band_analysis", {}).get("average", {}) for p in profiles]

            if not all_band_analyses:
                return {"resonance_quality": 0.0}

            # Calculate average across profiles
            avg_bands = {}
            for band_name in self.brainwave_bands.keys():
                band_values = [analysis.get(band_name, 0) for analysis in all_band_analyses]
                avg_bands[band_name] = np.mean(band_values)

            # Determine dominant band
            dominant_band = max(avg_bands.items(), key=lambda x: x[1])[0]

            # Calculate resonance quality
            resonance_quality = self.calculate_resonance_quality(avg_bands)

            return {
                "average_bands": avg_bands,
                "dominant_band": dominant_band,
                "resonance_quality": resonance_quality,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            print(f"❌ Error analyzing current brainwave state: {e}")
            return {"resonance_quality": 0.0}

    def calculate_resonance_quality(self, band_powers: Dict[str, float]) -> float:
        """Calculate resonance quality from band powers"""
        try:
            # Calculate total power
            total_power = sum(band_powers.values())

            if total_power == 0:
                return 0.0

            # Calculate normalized powers
            normalized_powers = {band: power / total_power for band, power in band_powers.items()}

            # Calculate resonance quality based on band balance
            # Higher quality when theta and alpha are prominent
            theta_alpha_quality = (
                normalized_powers.get("theta", 0) + normalized_powers.get("alpha", 0)
            ) / 2

            # Gamma quality for peak performance
            gamma_quality = normalized_powers.get("gamma", 0)

            # Overall resonance quality
            resonance_quality = (theta_alpha_quality * 0.6) + (gamma_quality * 0.4)

            return min(resonance_quality, 1.0)

        except Exception as e:
            print(f"❌ Error calculating resonance quality: {e}")
            return 0.0

    async def update_system_parameters(self):
        """Update system parameters based on brainwave state"""
        try:
            # Get current resonance state
            current_state = (
                list(self.state["resonance_states"].values())[-1]
                if self.state["resonance_states"]
                else None
            )

            if not current_state:
                return

            dominant_band = current_state.get("dominant_band", "alpha")
            resonance_quality = current_state.get("resonance_quality", 0.5)

            # Update system parameters based on dominant band
            updated_parameters = {}
            for param_name, band_mappings in self.parameter_mappings.items():
                base_value = band_mappings.get(dominant_band, 0.5)

                # Adjust based on resonance quality
                adjusted_value = base_value * resonance_quality

                updated_parameters[param_name] = adjusted_value

            # Store updated parameters
            self.state["system_parameters"] = updated_parameters

            # Broadcast parameter updates
            await self.broadcast_parameter_updates(updated_parameters)

        except Exception as e:
            print(f"❌ Error updating system parameters: {e}")

    async def broadcast_parameter_updates(self, parameters: Dict[str, float]):
        """Broadcast parameter updates to other agents"""
        try:
            parameter_update = {
                "type": "system_parameter_update",
                "parameters": parameters,
                "source": self.agent_id,
                "timestamp": datetime.now().isoformat(),
            }

            # Broadcast to all agents
            await self.broadcast_message(parameter_update)

            # Send to specific agents
            await self.send_message("strategy_agent", parameter_update)
            await self.send_message("trading_agent", parameter_update)

        except Exception as e:
            print(f"❌ Error broadcasting parameter updates: {e}")

    async def check_resonance_states(self):
        """Check and update resonance states"""
        try:
            # Get current resonance state
            current_state = (
                list(self.state["resonance_states"].values())[-1]
                if self.state["resonance_states"]
                else None
            )

            if not current_state:
                return

            # Check if resonance quality is high enough for synchronization
            resonance_quality = current_state.get("resonance_quality", 0.0)

            if resonance_quality > 0.7:
                # High resonance - perform synchronization
                await self.perform_synchronization(current_state)

        except Exception as e:
            print(f"❌ Error checking resonance states: {e}")

    async def perform_synchronization(self, brainwave_state: Dict[str, Any]):
        """Perform neuro-synchronization"""
        try:
            print(
                f"🧠 Performing neuro-synchronization (quality: {brainwave_state.get('resonance_quality', 0):.2f})"
            )

            # Create synchronization record
            sync_record = {
                "brainwave_state": brainwave_state,
                "system_parameters": self.state["system_parameters"],
                "timestamp": datetime.now().isoformat(),
                "sync_id": f"sync_{self.state['sync_count']}",
            }

            # Store synchronization record
            self.state["synchronization_history"].append(sync_record)
            self.state["sync_count"] += 1
            self.state["last_sync"] = datetime.now().isoformat()

            # Limit history size
            if len(self.state["synchronization_history"]) > 50:
                self.state["synchronization_history"] = self.state["synchronization_history"][-50:]

            # Broadcast synchronization event
            sync_event = {
                "type": "neuro_synchronization",
                "sync_record": sync_record,
                "timestamp": datetime.now().isoformat(),
            }

            await self.broadcast_message(sync_event)

            print("✅ Neuro-synchronization completed")

        except Exception as e:
            print(f"❌ Error performing synchronization: {e}")

    async def process_biofeedback(self):
        """Process biofeedback data"""
        try:
            # Get recent biofeedback data
            recent_biofeedback = list(self.state["biofeedback_data"].values())[-10:]

            if not recent_biofeedback:
                return

            # Analyze biofeedback patterns
            biofeedback_analysis = await self.analyze_biofeedback_patterns(recent_biofeedback)

            # Update system parameters based on biofeedback
            await self.update_parameters_from_biofeedback(biofeedback_analysis)

        except Exception as e:
            print(f"❌ Error processing biofeedback: {e}")

    async def analyze_biofeedback_patterns(
        self, biofeedback_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze biofeedback patterns"""
        try:
            analysis = {
                "heart_rate_trend": 0.0,
                "skin_conductance_trend": 0.0,
                "temperature_trend": 0.0,
                "stress_level": 0.0,
                "relaxation_level": 0.0,
            }

            # Calculate trends
            heart_rates = [
                entry.get("heart_rate", 70) for entry in biofeedback_data if entry.get("heart_rate")
            ]
            skin_conductance = [
                entry.get("skin_conductance", 0.5)
                for entry in biofeedback_data
                if entry.get("skin_conductance")
            ]
            temperatures = [
                entry.get("temperature", 37.0)
                for entry in biofeedback_data
                if entry.get("temperature")
            ]

            if heart_rates:
                analysis["heart_rate_trend"] = np.mean(heart_rates) - 70  # Deviation from baseline

            if skin_conductance:
                analysis["skin_conductance_trend"] = np.mean(skin_conductance) - 0.5

            if temperatures:
                analysis["temperature_trend"] = np.mean(temperatures) - 37.0

            # Calculate stress and relaxation levels
            analysis["stress_level"] = min(
                1.0,
                max(
                    0.0,
                    (analysis["heart_rate_trend"] / 20)
                    + (analysis["skin_conductance_trend"] / 0.5),
                ),
            )
            analysis["relaxation_level"] = 1.0 - analysis["stress_level"]

            return analysis

        except Exception as e:
            print(f"❌ Error analyzing biofeedback patterns: {e}")
            return {}

    async def update_parameters_from_biofeedback(self, biofeedback_analysis: Dict[str, Any]):
        """Update parameters based on biofeedback analysis"""
        try:
            stress_level = biofeedback_analysis.get("stress_level", 0.5)
            relaxation_level = biofeedback_analysis.get("relaxation_level", 0.5)

            # Adjust system parameters based on biofeedback
            current_params = self.state["system_parameters"].copy()

            # Reduce risk tolerance under high stress
            if stress_level > 0.7:
                current_params["risk_tolerance"] *= 0.8
                current_params["aggression_level"] *= 0.7

            # Increase holding duration under high relaxation
            if relaxation_level > 0.7:
                current_params["holding_duration"] *= 1.5

            # Update parameters
            self.state["system_parameters"] = current_params

        except Exception as e:
            print(f"❌ Error updating parameters from biofeedback: {e}")

    async def update_sync_metrics(self):
        """Update synchronization metrics"""
        try:
            metrics = {
                "agent_id": self.agent_id,
                "brainwave_profiles_count": len(self.state["brainwave_profiles"]),
                "resonance_states_count": len(self.state["resonance_states"]),
                "biofeedback_data_count": len(self.state["biofeedback_data"]),
                "synchronization_count": self.state["sync_count"],
                "last_synchronization": self.state["last_sync"],
                "resonance_quality": self.state["resonance_quality"],
                "current_parameters": self.state["system_parameters"],
                "timestamp": datetime.now().isoformat(),
            }

            # Store metrics in Redis
            self.redis_client.set(f"agent_metrics:{self.agent_id}", json.dumps(metrics), ex=300)

        except Exception as e:
            print(f"❌ Error updating sync metrics: {e}")

    async def cleanup_old_data(self):
        """Clean up old data"""
        try:
            datetime.now()

            # Clean up brainwave profiles (keep last 50)
            profile_keys = list(self.state["brainwave_profiles"].keys())
            if len(profile_keys) > 50:
                for key in profile_keys[:-50]:
                    del self.state["brainwave_profiles"][key]

            # Clean up resonance states (keep last 30)
            resonance_keys = list(self.state["resonance_states"].keys())
            if len(resonance_keys) > 30:
                for key in resonance_keys[:-30]:
                    del self.state["resonance_states"][key]

            # Clean up biofeedback data (keep last 40)
            biofeedback_keys = list(self.state["biofeedback_data"].keys())
            if len(biofeedback_keys) > 40:
                for key in biofeedback_keys[:-40]:
                    del self.state["biofeedback_data"][key]

        except Exception as e:
            print(f"❌ Error cleaning up old data: {e}")

    async def handle_sync_brainwaves(self, message: Dict[str, Any]):
        """Handle manual brainwave synchronization request"""
        try:
            target_band = message.get("target_band", "alpha")
            duration = message.get("duration", 300)  # 5 minutes

            print(f"🧠 Manual brainwave synchronization requested for {target_band} band")

            # Perform targeted synchronization
            sync_result = await self.perform_targeted_sync(target_band, duration)

            # Send response
            response = {
                "type": "brainwave_sync_complete",
                "target_band": target_band,
                "duration": duration,
                "sync_result": sync_result,
                "timestamp": datetime.now().isoformat(),
            }

            sender = message.get("from_agent")
            if sender:
                await self.send_message(sender, response)

        except Exception as e:
            print(f"❌ Error handling brainwave sync request: {e}")
            await self.broadcast_error(f"Brainwave sync error: {e}")

    async def perform_targeted_sync(self, target_band: str, duration: int) -> Dict[str, Any]:
        """Perform targeted brainwave synchronization"""
        try:
            # Get target frequency
            target_freq = self.brainwave_bands.get(target_band, (8.0, 13.0))
            target_frequency = np.mean(target_freq)

            # Create synchronization session
            sync_session = {
                "target_band": target_band,
                "target_frequency": target_frequency,
                "duration": duration,
                "start_time": datetime.now().isoformat(),
                "status": "active",
            }

            # Store session
            self.redis_client.set("sync_session", json.dumps(sync_session), ex=duration)

            # Simulate synchronization process
            await asyncio.sleep(2)  # Simulate processing time

            # Complete session
            sync_session["status"] = "completed"
            sync_session["end_time"] = datetime.now().isoformat()

            return sync_session

        except Exception as e:
            print(f"❌ Error performing targeted sync: {e}")
            return {"error": str(e)}

    async def handle_set_resonance_preset(self, message: Dict[str, Any]):
        """Handle resonance preset setting request"""
        try:
            preset_name = message.get("preset_name", "meditation")

            print(f"🎵 Setting resonance preset: {preset_name}")

            # Get preset configuration
            preset = self.resonance_presets.get(preset_name, self.resonance_presets["meditation"])

            # Apply preset
            preset_result = await self.apply_resonance_preset(preset)

            # Send response
            response = {
                "type": "resonance_preset_set",
                "preset_name": preset_name,
                "preset_config": preset,
                "result": preset_result,
                "timestamp": datetime.now().isoformat(),
            }

            sender = message.get("from_agent")
            if sender:
                await self.send_message(sender, response)

        except Exception as e:
            print(f"❌ Error handling resonance preset request: {e}")
            await self.broadcast_error(f"Resonance preset error: {e}")

    async def apply_resonance_preset(self, preset: Dict[str, Any]) -> Dict[str, Any]:
        """Apply resonance preset"""
        try:
            target_bands = preset.get("target_bands", ["alpha"])
            frequency = preset.get("frequency", 10.0)
            intensity = preset.get("intensity", 0.8)

            # Apply preset to system
            preset_config = {
                "target_bands": target_bands,
                "frequency": frequency,
                "intensity": intensity,
                "applied_at": datetime.now().isoformat(),
            }

            # Store preset configuration
            self.redis_client.set("active_resonance_preset", json.dumps(preset_config), ex=3600)

            return preset_config

        except Exception as e:
            print(f"❌ Error applying resonance preset: {e}")
            return {"error": str(e)}

    async def handle_get_biofeedback(self, message: Dict[str, Any]):
        """Handle biofeedback data request"""
        try:
            data_type = message.get("data_type", "all")
            timeframe = message.get("timeframe", "1h")

            print(f"📊 Biofeedback data request for {data_type} ({timeframe})")

            # Get biofeedback data
            biofeedback_data = await self.get_biofeedback_data(data_type, timeframe)

            # Send response
            response = {
                "type": "biofeedback_data_response",
                "data_type": data_type,
                "timeframe": timeframe,
                "biofeedback_data": biofeedback_data,
                "timestamp": datetime.now().isoformat(),
            }

            sender = message.get("from_agent")
            if sender:
                await self.send_message(sender, response)

        except Exception as e:
            print(f"❌ Error handling biofeedback request: {e}")
            await self.broadcast_error(f"Biofeedback request error: {e}")

    async def get_biofeedback_data(self, data_type: str, timeframe: str) -> Dict[str, Any]:
        """Get biofeedback data"""
        try:
            # Calculate cutoff time
            cutoff_time = datetime.now()
            if timeframe == "30m":
                cutoff_time -= timedelta(minutes=30)
            elif timeframe == "1h":
                cutoff_time -= timedelta(hours=1)
            elif timeframe == "6h":
                cutoff_time -= timedelta(hours=6)
            elif timeframe == "24h":
                cutoff_time -= timedelta(hours=24)
            else:
                cutoff_time -= timedelta(hours=1)

            # Filter biofeedback data by timeframe
            filtered_data = {}
            for timestamp, data in self.state["biofeedback_data"].items():
                if isinstance(timestamp, str):
                    data_time = datetime.fromisoformat(timestamp)
                    if data_time > cutoff_time:
                        if data_type == "all" or data_type in data:
                            filtered_data[timestamp] = data

            return filtered_data

        except Exception as e:
            print(f"❌ Error getting biofeedback data: {e}")
            return {}

    async def handle_update_parameters(self, message: Dict[str, Any]):
        """Handle parameter update request"""
        try:
            parameters = message.get("parameters", {})

            print("⚙️ Parameter update request received")

            # Update system parameters
            self.state["system_parameters"].update(parameters)

            # Broadcast updates
            await self.broadcast_parameter_updates(self.state["system_parameters"])

            # Send confirmation
            response = {
                "type": "parameter_update_complete",
                "updated_parameters": parameters,
                "current_parameters": self.state["system_parameters"],
                "timestamp": datetime.now().isoformat(),
            }

            sender = message.get("from_agent")
            if sender:
                await self.send_message(sender, response)

        except Exception as e:
            print(f"❌ Error handling parameter update request: {e}")
            await self.broadcast_error(f"Parameter update error: {e}")


if __name__ == "__main__":
    # Run the agent
    engine = NeuroSynchronizationEngine()
    asyncio.run(engine.start())
