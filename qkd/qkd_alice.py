import asyncio
import os
import time

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    generate_latest,
)

app = FastAPI(
    title="Mystic QKD Alice",
    description="Quantum Key Distribution - Alice node.",
    version="1.0.0",
)

# Prometheus metrics
REQUEST_COUNT = Counter(
    "qkd_alice_requests_total", "Total QKD Alice API Requests", ["endpoint"]
)
KEYS_GENERATED = Counter("qkd_alice_keys_generated", "Total quantum keys generated")
KEY_RATE = Gauge("qkd_alice_key_rate_kbps", "Key generation rate in kbps")
QUANTUM_STATE = Gauge("qkd_alice_quantum_state_quality", "Quantum state quality")


@app.get("/health", tags=["Health"])
def health():
    REQUEST_COUNT.labels(endpoint="/health").inc()
    return {"status": "ok", "service": "qkd-alice", "version": "1.0.0"}


@app.get("/metrics")
def metrics():
    return JSONResponse(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/qkd/alice/generate", tags=["QKD"])
async def generate_key(key_length_bits: int = 256, protocol: str = "BB84"):
    REQUEST_COUNT.labels(endpoint="/qkd/alice/generate").inc()
    start_time = time.time()

    try:
        # Simulate quantum key generation
        generation_delay = (key_length_bits / 256) * 2.0
        await asyncio.sleep(generation_delay)

        # Simulate quantum key
        quantum_key = "".join(
            [str(int(time.time() * 1000) % 2) for _ in range(key_length_bits)]
        )

        KEYS_GENERATED.inc()
        KEY_RATE.set(key_length_bits / generation_delay / 1000)  # kbps
        QUANTUM_STATE.set(95.5)  # High quality quantum state

        return {
            "generated": True,
            "key_length_bits": key_length_bits,
            "protocol": protocol,
            "quantum_key": quantum_key,
            "generation_time": time.time() - start_time,
            "key_rate_kbps": key_length_bits / generation_delay / 1000,
            "quantum_state_quality": 95.5,
            "error_rate_percent": 0.5,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/qkd/alice/status", tags=["QKD"])
def alice_status():
    REQUEST_COUNT.labels(endpoint="/qkd/alice/status").inc()
    return {
        "alice_status": "operational",
        "quantum_state": "entangled",
        "keys_generated": 1250,
        "current_key_rate_kbps": 45.5,
        "quantum_state_quality": 95.5,
        "error_rate_percent": 0.5,
        "bob_connection": "established",
        "eve_detection": "active",
    }


@app.post("/qkd/alice/transmit", tags=["QKD"])
async def transmit_quantum_state(
    state_type: str = "photon", polarization: str = "random"
):
    REQUEST_COUNT.labels(endpoint="/qkd/alice/transmit").inc()
    start_time = time.time()

    try:
        await asyncio.sleep(0.1)  # Simulate quantum transmission

        return {
            "transmitted": True,
            "state_type": state_type,
            "polarization": polarization,
            "transmission_time": time.time() - start_time,
            "success_rate_percent": 98.5,
            "quantum_efficiency": 0.95,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/qkd/alice/analytics", tags=["QKD"])
def alice_analytics():
    REQUEST_COUNT.labels(endpoint="/qkd/alice/analytics").inc()
    return {
        "key_generation_performance": {
            "total_keys_generated": 1250,
            "average_key_rate_kbps": 45.5,
            "peak_key_rate_kbps": 52.3,
            "key_generation_success_rate": 99.5,
        },
        "quantum_performance": {
            "quantum_state_quality_avg": 95.5,
            "entanglement_fidelity": 0.98,
            "quantum_efficiency": 0.95,
            "photon_detection_rate": 0.92,
        },
        "security_metrics": {
            "eve_detection_rate": 99.8,
            "quantum_bit_error_rate": 0.5,
            "privacy_amplification_efficiency": 0.85,
            "final_key_rate_kbps": 38.7,
        },
        "system_health": {
            "laser_stability_percent": 99.9,
            "detector_efficiency": 0.95,
            "temperature_stability_celsius": 0.1,
            "optical_alignment": "optimal",
        },
    }


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8106))
    uvicorn.run(app, host="0.0.0.0", port=port)
