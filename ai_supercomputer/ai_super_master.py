import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from prometheus_client import (
    Counter,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
import time
import os
import asyncio

app = FastAPI(
    title="Mystic AI Supercomputer Master",
    description="AI Supercomputing master node for distributed AI training.",
    version="1.0.0",
)

# Prometheus metrics
REQUEST_COUNT = Counter(
    "ai_super_master_requests_total",
    "Total AI Super Master API Requests",
    ["endpoint"],
)
TRAINING_JOBS = Counter(
    "ai_super_master_training_jobs_total", "Total training jobs submitted"
)
ACTIVE_WORKERS = Gauge(
    "ai_super_master_active_workers", "Number of active worker nodes"
)
MASTER_LOAD = Gauge("ai_super_master_load_percent", "Master node load percentage")


@app.get("/health", tags=["Health"])
def health():
    REQUEST_COUNT.labels(endpoint="/health").inc()
    return {"status": "ok", "service": "ai-super-master", "version": "1.0.0"}


@app.get("/metrics")
def metrics():
    return JSONResponse(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/ai_super/master/train", tags=["AI Supercomputing"])
async def submit_training_job(
    model_type: str = "transformer",
    dataset_size_gb: float = 100.0,
    epochs: int = 100,
):
    REQUEST_COUNT.labels(endpoint="/ai_super/master/train").inc()
    start_time = time.time()

    try:
        # Simulate job submission and distribution
        submission_delay = (dataset_size_gb * 0.1) + (epochs * 0.05)
        await asyncio.sleep(submission_delay)

        job_id = f"job_{int(time.time())}_{model_type}"
        TRAINING_JOBS.inc()
        MASTER_LOAD.set(75.5)  # Simulate load

        return {
            "submitted": True,
            "job_id": job_id,
            "model_type": model_type,
            "dataset_size_gb": dataset_size_gb,
            "epochs": epochs,
            "submission_time": time.time() - start_time,
            "estimated_duration_hours": dataset_size_gb * 0.5 + epochs * 0.2,
            "assigned_workers": 3,
            "priority": "high",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ai_super/master/status", tags=["AI Supercomputing"])
def master_status():
    REQUEST_COUNT.labels(endpoint="/ai_super/master/status").inc()
    return {
        "master_status": "operational",
        "active_workers": 3,
        "total_workers": 4,
        "active_jobs": 8,
        "queue_size": 5,
        "total_gpu_memory_tb": 12.0,
        "total_cpu_cores": 256,
        "network_bandwidth_gbps": 100.0,
        "storage_capacity_pb": 2.5,
    }


@app.post("/ai_super/master/distribute", tags=["AI Supercomputing"])
async def distribute_workload(workload_type: str = "training", target_workers: int = 3):
    REQUEST_COUNT.labels(endpoint="/ai_super/master/distribute").inc()
    start_time = time.time()

    try:
        await asyncio.sleep(1.0)  # Simulate workload distribution

        ACTIVE_WORKERS.set(target_workers)

        return {
            "distributed": True,
            "workload_type": workload_type,
            "target_workers": target_workers,
            "distribution_time": time.time() - start_time,
            "load_balanced": True,
            "worker_assignment": {
                "worker-1": "gpu_cluster_1",
                "worker-2": "gpu_cluster_2",
                "worker-3": "gpu_cluster_3",
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ai_super/master/analytics", tags=["AI Supercomputing"])
def master_analytics():
    REQUEST_COUNT.labels(endpoint="/ai_super/master/analytics").inc()
    return {
        "master_performance": {
            "total_jobs_processed": 1250,
            "average_job_duration_hours": 8.5,
            "job_success_rate_percent": 99.2,
            "average_queue_time_minutes": 15.5,
        },
        "worker_coordination": {
            "active_workers": 3,
            "worker_utilization_avg_percent": 85.5,
            "load_balancing_efficiency": 92.8,
            "worker_failure_rate_percent": 0.5,
        },
        "resource_utilization": {
            "gpu_utilization_percent": 88.5,
            "cpu_utilization_percent": 65.2,
            "memory_utilization_percent": 72.8,
            "network_utilization_percent": 45.5,
        },
        "training_metrics": {
            "models_trained": 125,
            "average_training_accuracy": 94.5,
            "fastest_training_time_hours": 2.5,
            "largest_model_parameters": "175B",
        },
        "system_health": {
            "master_uptime_percent": 99.95,
            "average_response_time_ms": 25.5,
            "error_rate_percent": 0.1,
            "backup_systems": "active",
        },
    }


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8102))
    uvicorn.run(app, host="0.0.0.0", port=port)
