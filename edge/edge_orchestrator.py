import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
import time
import os
import asyncio

app = FastAPI(
    title="Mystic Edge Orchestrator",
    description="Edge computing orchestrator for load balancing and coordination.",
    version="1.0.0",
)

# Prometheus metrics
REQUEST_COUNT = Counter(
    "edge_orchestrator_requests_total",
    "Total Edge Orchestrator API Requests",
    ["endpoint"],
)
ORCHESTRATION_TIME = Histogram(
    "edge_orchestrator_duration_seconds", "Orchestration time", ["operation"]
)
ACTIVE_NODES = Gauge("edge_orchestrator_active_nodes", "Number of active edge nodes")
TOTAL_LOAD = Gauge("edge_orchestrator_total_load", "Total load across all nodes")


@app.get("/health", tags=["Health"])
def health():
    REQUEST_COUNT.labels(endpoint="/health").inc()
    return {"status": "ok", "service": "edge-orchestrator", "version": "1.0.0"}


@app.get("/metrics")
def metrics():
    return JSONResponse(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/edge/orchestrate", tags=["Edge Computing"])
async def edge_orchestrate(workload_size: int = 1000, priority: str = "normal"):
    REQUEST_COUNT.labels(endpoint="/edge/orchestrate").inc()
    start_time = time.time()

    try:
        # Simulate orchestration logic
        orchestration_delay = {"low": 0.2, "normal": 0.5, "high": 1.0}.get(
            priority, 0.5
        )

        await asyncio.sleep(orchestration_delay)

        # Simulate node selection and load balancing
        available_nodes = [
            {"node_id": "edge-001", "load": 0.6, "capacity": "high"},
            {"node_id": "edge-002", "load": 0.3, "capacity": "medium"},
        ]

        # Select best node based on load and capacity
        selected_node = min(available_nodes, key=lambda x: x["load"])

        ACTIVE_NODES.set(len(available_nodes))
        TOTAL_LOAD.set(sum(node["load"] for node in available_nodes))
        ORCHESTRATION_TIME.labels(operation=priority).observe(time.time() - start_time)

        return {
            "orchestrated": True,
            "selected_node": selected_node["node_id"],
            "workload_size": workload_size,
            "priority": priority,
            "orchestration_time": time.time() - start_time,
            "available_nodes": len(available_nodes),
            "total_load": sum(node["load"] for node in available_nodes),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/edge/nodes", tags=["Edge Computing"])
def get_nodes():
    REQUEST_COUNT.labels(endpoint="/edge/nodes").inc()
    return {
        "nodes": [
            {
                "node_id": "edge-001",
                "status": "active",
                "load": 0.6,
                "capacity": "high",
                "location": "edge-zone-1",
                "connected_devices": 15,
            },
            {
                "node_id": "edge-002",
                "status": "active",
                "load": 0.3,
                "capacity": "medium",
                "location": "edge-zone-2",
                "connected_devices": 8,
            },
        ],
        "total_nodes": 2,
        "total_load": 0.9,
    }


@app.get("/edge/analytics", tags=["Edge Computing"])
def edge_analytics():
    REQUEST_COUNT.labels(endpoint="/edge/analytics").inc()
    return {
        "performance_metrics": {
            "average_latency_ms": 4.5,
            "throughput_mbps": 1250.5,
            "error_rate_percent": 0.02,
            "availability_percent": 99.98,
        },
        "resource_utilization": {
            "cpu_avg_percent": 45.2,
            "memory_avg_percent": 62.8,
            "network_avg_percent": 38.5,
        },
        "workload_distribution": {
            "edge_001_percent": 65,
            "edge_002_percent": 35,
        },
    }


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8092))
    uvicorn.run(app, host="0.0.0.0", port=port)
