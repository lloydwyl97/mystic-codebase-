import logging
import threading
import time
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any

import psutil

logger = logging.getLogger(__name__)


class MetricsCollector:
    def __init__(self):
        # Metrics storage
        self.metrics: dict[str, Any] = {
            "requests": defaultdict(int),
            "responses": defaultdict(int),
            "errors": defaultdict(int),
            "latency": defaultdict(list),
            "bandwidth": defaultdict(int),
            "cpu_usage": [],
            "memory_usage": [],
            "disk_io": [],
            "network_io": [],
        }

        # Metrics configuration
        self.config: dict[str, Any] = {
            "retention_period": 3600,  # 1 hour
            "sampling_interval": 60,  # 1 minute
            "max_samples": 1000,  # Maximum samples to keep
            "endpoints": {},  # Endpoints will be set in metrics.py
        }

        # Start metrics collection thread
        self.collecting = True
        self.collector_thread = threading.Thread(target=self._collect_metrics)
        self.collector_thread.daemon = True
        self.collector_thread.start()

    def _collect_metrics(self):
        """Collect system metrics periodically"""
        while self.collecting:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                self.metrics["cpu_usage"].append(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "value": cpu_percent,
                    }
                )

                # Memory usage
                memory = psutil.virtual_memory()
                self.metrics["memory_usage"].append(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "value": memory.percent,
                    }
                )

                # Disk I/O
                disk_io = psutil.disk_io_counters()
                if disk_io is not None:
                    self.metrics["disk_io"].append(
                        {
                            "timestamp": datetime.now().isoformat(),
                            "read_bytes": getattr(disk_io, "read_bytes", 0),
                            "write_bytes": getattr(disk_io, "write_bytes", 0),
                        }
                    )

                # Network I/O
                net_io = psutil.net_io_counters()
                self.metrics["network_io"].append(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "bytes_sent": net_io.bytes_sent,
                        "bytes_recv": net_io.bytes_recv,
                    }
                )

                # Cleanup old metrics
                self._cleanup_old_metrics()

                time.sleep(self.config["sampling_interval"])

            except Exception as e:
                logger.error(f"Error collecting metrics: {str(e)}")
                time.sleep(self.config["sampling_interval"])

    def _cleanup_old_metrics(self):
        """Remove old metrics data"""
        try:
            cutoff_time = datetime.now() - timedelta(seconds=self.config["retention_period"])

            # Cleanup time series metrics
            for metric in [
                "cpu_usage",
                "memory_usage",
                "disk_io",
                "network_io",
            ]:
                self.metrics[metric] = [
                    m
                    for m in self.metrics[metric]
                    if datetime.fromisoformat(m["timestamp"]) > cutoff_time
                ][-self.config["max_samples"] :]

        except Exception as e:
            logger.error(f"Error cleaning up metrics: {str(e)}")

    def get_metrics(self) -> dict[str, Any]:
        """Get all metrics"""
        try:
            return {
                "timestamp": datetime.now().isoformat(),
                "requests": dict(self.metrics["requests"]),
                "responses": dict(self.metrics["responses"]),
                "errors": dict(self.metrics["errors"]),
                "latency": {
                    path: {
                        "avg": sum(times) / len(times) if times else 0,
                        "min": min(times) if times else 0,
                        "max": max(times) if times else 0,
                        "count": len(times),
                    }
                    for path, times in self.metrics["latency"].items()
                },
                "bandwidth": dict(self.metrics["bandwidth"]),
                "system": {
                    "cpu": (self.metrics["cpu_usage"][-1] if self.metrics["cpu_usage"] else None),
                    "memory": (
                        self.metrics["memory_usage"][-1] if self.metrics["memory_usage"] else None
                    ),
                    "disk_io": (self.metrics["disk_io"][-1] if self.metrics["disk_io"] else None),
                    "network_io": (
                        self.metrics["network_io"][-1] if self.metrics["network_io"] else None
                    ),
                },
            }

        except Exception as e:
            logger.error(f"Error getting metrics: {str(e)}")
            return {"error": str(e)}

    def get_metrics_summary(self) -> dict[str, Any]:
        """Get metrics summary"""
        try:
            total_requests = sum(self.metrics["requests"].values())
            total_errors = sum(self.metrics["errors"].values())

            return {
                "timestamp": datetime.now().isoformat(),
                "total_requests": total_requests,
                "total_errors": total_errors,
                "error_rate": ((total_errors / total_requests * 100) if total_requests > 0 else 0),
                "avg_latency": {
                    path: sum(times) / len(times) if times else 0
                    for path, times in self.metrics["latency"].items()
                },
                "system_health": {
                    "cpu": (
                        self.metrics["cpu_usage"][-1]["value"] if self.metrics["cpu_usage"] else 0
                    ),
                    "memory": (
                        self.metrics["memory_usage"][-1]["value"]
                        if self.metrics["memory_usage"]
                        else 0
                    ),
                },
            }

        except Exception as e:
            logger.error(f"Error getting metrics summary: {str(e)}")
            return {"error": str(e)}

    def get_detailed_metrics(self) -> dict[str, Any]:
        """Get detailed metrics with time series data"""
        try:
            return {
                "timestamp": datetime.now().isoformat(),
                "requests": {
                    "total": sum(self.metrics["requests"].values()),
                    "by_endpoint": dict(self.metrics["requests"]),
                },
                "errors": {
                    "total": sum(self.metrics["errors"].values()),
                    "by_endpoint": dict(self.metrics["errors"]),
                },
                "latency": {
                    path: {
                        "avg": sum(times) / len(times) if times else 0,
                        "min": min(times) if times else 0,
                        "max": max(times) if times else 0,
                        "p95": (sorted(times)[int(len(times) * 0.95)] if times else 0),
                        "p99": (sorted(times)[int(len(times) * 0.99)] if times else 0),
                        "count": len(times),
                    }
                    for path, times in self.metrics["latency"].items()
                },
                "bandwidth": {
                    "total": sum(self.metrics["bandwidth"].values()),
                    "by_endpoint": dict(self.metrics["bandwidth"]),
                },
                "system": {
                    "cpu": self.metrics["cpu_usage"],
                    "memory": self.metrics["memory_usage"],
                    "disk_io": self.metrics["disk_io"],
                    "network_io": self.metrics["network_io"],
                },
            }

        except Exception as e:
            logger.error(f"Error getting detailed metrics: {str(e)}")
            return {"error": str(e)}

    def track_request(self, request: Any, start_time: float) -> None:
        """Track request metrics"""
        try:
            path = request.url.path
            method = request.method

            # Track request count
            self.metrics["requests"][f"{method} {path}"] += 1

            # Track latency
            latency = time.time() - start_time
            self.metrics["latency"][path].append(latency)

            # Track bandwidth
            content_length = request.headers.get("content-length")
            if content_length:
                self.metrics["bandwidth"][path] += int(content_length)

        except Exception as e:
            logger.error(f"Error tracking request: {str(e)}")

    def track_response(self, request: Any, response: Any) -> None:
        """Track response metrics"""
        try:
            path = request.url.path
            method = request.method

            # Track response count
            self.metrics["responses"][f"{method} {path}"] += 1

            # Track errors
            if response.status_code >= 400:
                self.metrics["errors"][f"{method} {path}"] += 1

            # Track bandwidth
            if hasattr(response, "body") and response.body is not None:
                self.metrics["bandwidth"][path] += len(response.body)

        except Exception as e:
            logger.error(f"Error tracking response: {str(e)}")


