import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from prometheus_client import (
    Counter,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
import time
import os
import asyncio

app = FastAPI(
    title="Mystic Edge Node 2",
    description="Edge computing node for distributed processing.",
    version="1.0.0",
)

# Prometheus metrics
REQUEST_COUNT = Counter(
    "edge_node_2_requests_total",
    "Total Edge Node 2 API Requests",
    ["endpoint"],
)
PROCESSING_TIME = Histogram(
    "edge_node_2_processing_duration_seconds",
    "Edge processing time",
    ["operation"],
)
EDGE_MEMORY_USAGE = Counter("edge_node_2_memory_bytes", "Memory usage in bytes")
EDGE_CPU_USAGE = Counter("edge_node_2_cpu_seconds_total", "CPU usage in seconds")


@app.get("/health", tags=["Health"])
def health():
    REQUEST_COUNT.labels(endpoint="/health").inc()
    return {
        "status": "ok",
        "service": "edge-node-2",
        "version": "1.0.0",
        "node_id": "edge-002",
    }


@app.get("/metrics")
def metrics():
    return JSONResponse(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/edge/process", tags=["Edge Computing"])
async def edge_process(data_size: int = 1024, complexity: str = "medium"):
    REQUEST_COUNT.labels(endpoint="/edge/process").inc()
    start_time = time.time()

    try:
        # Simulate edge processing
        processing_delay = {"low": 0.1, "medium": 0.5, "high": 1.0}.get(complexity, 0.5)

        await asyncio.sleep(processing_delay)

        # Simulate resource usage
        memory_usage = data_size * 100  # bytes
        cpu_usage = processing_delay

        EDGE_MEMORY_USAGE.inc(memory_usage)
        EDGE_CPU_USAGE.inc(cpu_usage)
        PROCESSING_TIME.labels(operation=complexity).observe(time.time() - start_time)

        return {
            "processed": True,
            "data_size": data_size,
            "complexity": complexity,
            "processing_time": time.time() - start_time,
            "node_id": "edge-002",
            "memory_used": memory_usage,
            "cpu_used": cpu_usage,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/edge/status", tags=["Edge Computing"])
def edge_status():
    REQUEST_COUNT.labels(endpoint="/edge/status").inc()
    return {
        "node_id": "edge-002",
        "location": "edge-zone-2",
        "capacity": "medium",
        "load": "low",
        "connected_devices": 8,
        "latency_ms": 3,
    }


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8091))
    uvicorn.run(app, host="0.0.0.0", port=port)
