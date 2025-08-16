import logging
import time
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse

from enhanced_logging import log_operation_performance

from .metrics_collector import MetricsCollector

logger = logging.getLogger(__name__)

# Global metrics collector instance
metrics_collector = MetricsCollector()

# Set up endpoints in the config
metrics_collector.config["endpoints"] = {
    "/metrics": metrics_collector.get_metrics,
    "/metrics/summary": metrics_collector.get_metrics_summary,
    "/metrics/detailed": metrics_collector.get_detailed_metrics,
}


@log_operation_performance("metrics_middleware")
async def metrics_middleware(request: Request, call_next: Any):
    """
    Middleware to collect request metrics.
    """
    start_time = time.time()

    try:
        # Check if it's a metrics endpoint
        if request.url.path in metrics_collector.config["endpoints"]:
            metrics_data = metrics_collector.config["endpoints"][request.url.path]()
            return JSONResponse(content=metrics_data)

        # Process request
        response = await call_next(request)

        # Track metrics
        metrics_collector.track_request(request, start_time)
        if isinstance(response, JSONResponse):
            metrics_collector.track_response(request, response)

        return response

    except Exception as e:
        # Update error metrics
        metrics_collector.metrics["errors"][f"{request.method} {request.url.path}"] += 1
        logger.error(f"Metrics error: {str(e)}")
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})


