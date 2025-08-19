"""
Performance Monitor for Mystic Trading Platform

Comprehensive performance monitoring with:
- Database performance tracking
- Cache performance monitoring
- AI model performance tracking
- System resource monitoring
- Performance alerts
- Optimization recommendations
"""

import asyncio
import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any

try:
    import psutil
except ImportError:
    psutil = None


logger = logging.getLogger(__name__)

# Performance thresholds
DB_QUERY_THRESHOLD = 1.0  # 1 second
CACHE_HIT_RATE_THRESHOLD = 0.8  # 80%
MEMORY_USAGE_THRESHOLD = 0.9  # 90%
CPU_USAGE_THRESHOLD = 0.8  # 80%


@dataclass
class PerformanceMetric:
    """Performance metric data"""
    name: str
    value: float
    timestamp: float
    threshold: float
    status: str  # 'normal', 'warning', 'critical'


class PerformanceAlert:
    """Performance alert system"""

    def __init__(self):
        self.alerts: deque = deque(maxlen=100)
        self.alert_counts = defaultdict(int)

    def add_alert(self, metric_name: str, value: float, threshold: float, severity: str):
        """Add a performance alert"""
        alert = {
            'metric': metric_name,
            'value': value,
            'threshold': threshold,
            'severity': severity,
            'timestamp': time.time()
        }

        self.alerts.append(alert)
        self.alert_counts[metric_name] += 1

        logger.warning(f"Performance alert: {metric_name} = {value} (threshold: {threshold})")

    def get_recent_alerts(self, minutes: int = 10) -> list[dict[str, Any]]:
        """Get recent alerts"""
        cutoff_time = time.time() - (minutes * 60)
        return [alert for alert in self.alerts if alert['timestamp'] > cutoff_time]

    def get_alert_stats(self) -> dict[str, Any]:
        """Get alert statistics"""
        return {
            'total_alerts': len(self.alerts),
            'alert_counts': dict(self.alert_counts),
            'recent_alerts': len(self.get_recent_alerts())
        }


class PerformanceMonitor:
    """Comprehensive performance monitoring system"""

    def __init__(self):
        self.metrics: dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.alerts = PerformanceAlert()
        self.running = False
        self.monitoring_task = None
        self.lock = threading.Lock()

        # Performance tracking
        self.db_queries = defaultdict(list)
        self.cache_operations = defaultdict(list)
        self.model_operations = defaultdict(list)
        self.system_metrics = defaultdict(list)

    async def start(self):
        """Start performance monitoring"""
        self.running = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Performance monitor started")

    async def stop(self):
        """Stop performance monitoring"""
        self.running = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("Performance monitor stopped")

    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                # Collect system metrics
                await self._collect_system_metrics()

                # Check performance thresholds
                await self._check_performance_thresholds()

                # Generate optimization recommendations
                await self._generate_recommendations()

                # Wait for next monitoring cycle
                await asyncio.sleep(30)  # Every 30 seconds

            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
                await asyncio.sleep(60)

    async def _collect_system_metrics(self):
        """Collect system performance metrics"""
        if psutil is None:
            return

        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self._add_metric('cpu_usage', cpu_percent, CPU_USAGE_THRESHOLD)

            # Memory usage
            memory = psutil.virtual_memory()
            self._add_metric('memory_usage', memory.percent / 100, MEMORY_USAGE_THRESHOLD)

            # Disk usage
            disk = psutil.disk_usage('/')
            self._add_metric('disk_usage', disk.percent / 100, 0.9)

        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")

    def _add_metric(self, name: str, value: float, threshold: float):
        """Add a performance metric"""
        status = 'normal'
        if value > threshold:
            status = 'critical'
        elif value > threshold * 0.8:
            status = 'warning'

        metric = PerformanceMetric(
            name=name,
            value=value,
            timestamp=time.time(),
            threshold=threshold,
            status=status
        )

        with self.lock:
            self.metrics[name].append(metric)

    async def _check_performance_thresholds(self):
        """Check performance thresholds and generate alerts"""
        with self.lock:
            for metric_name, metrics in self.metrics.items():
                if not metrics:
                    continue

                latest = metrics[-1]
                if latest.status == 'critical':
                    self.alerts.add_alert(metric_name, latest.value, latest.threshold, 'critical')
                elif latest.status == 'warning':
                    self.alerts.add_alert(metric_name, latest.value, latest.threshold, 'warning')

    async def _generate_recommendations(self):
        """Generate performance optimization recommendations"""
        recommendations = []

        # Check database performance
        avg_query_time = self._get_average_metric('db_query_time')
        if avg_query_time and avg_query_time > DB_QUERY_THRESHOLD:
            recommendations.append({
                'type': 'database',
                'issue': 'Slow database queries',
                'recommendation': 'Consider query optimization or database indexing',
                'severity': 'high'
            })

        # Check cache performance
        cache_hit_rate = self._get_average_metric('cache_hit_rate')
        if cache_hit_rate and cache_hit_rate < CACHE_HIT_RATE_THRESHOLD:
            recommendations.append({
                'type': 'cache',
                'issue': 'Low cache hit rate',
                'recommendation': 'Consider expanding cache size or improving cache strategy',
                'severity': 'medium'
            })

        # Check system resources
        cpu_usage = self._get_average_metric('cpu_usage')
        if cpu_usage and cpu_usage > CPU_USAGE_THRESHOLD:
            recommendations.append({
                'type': 'system',
                'issue': 'High CPU usage',
                'recommendation': 'Consider scaling up or optimizing resource-intensive operations',
                'severity': 'high'
            })

        memory_usage = self._get_average_metric('memory_usage')
        if memory_usage and memory_usage > MEMORY_USAGE_THRESHOLD:
            recommendations.append({
                'type': 'system',
                'issue': 'High memory usage',
                'recommendation': 'Consider memory optimization or increasing available memory',
                'severity': 'critical'
            })

        # Store recommendations
        if recommendations:
            self._store_recommendations(recommendations)
            logger.info(f"Generated {len(recommendations)} performance recommendations")

    def _get_average_metric(self, metric_name: str, window_minutes: int = 5) -> float | None:
        """Get average metric value over a time window"""
        if metric_name not in self.metrics:
            return None

        cutoff_time = time.time() - (window_minutes * 60)
        recent_metrics = [
            metric.value for metric in self.metrics[metric_name]
            if metric.timestamp > cutoff_time
        ]

        return sum(recent_metrics) / len(recent_metrics) if recent_metrics else None

    def _get_latest_metric(self, metric_name: str) -> PerformanceMetric | None:
        """Get the latest metric for a given name"""
        if metric_name not in self.metrics or not self.metrics[metric_name]:
            return None
        return self.metrics[metric_name][-1]

    def _store_recommendations(self, recommendations: list[dict[str, Any]]):
        """Store performance recommendations"""
        # In a real implementation, this would store to database or cache
        # For now, we'll just log them
        for rec in recommendations:
            logger.info(f"Performance recommendation: {rec['issue']} - {rec['recommendation']}")

    def track_db_query(self, query: str, execution_time: float):
        """Track database query performance"""
        self._add_metric('db_query_time', execution_time, DB_QUERY_THRESHOLD)

        # Store detailed query info
        self.db_queries[query].append({
            'execution_time': execution_time,
            'timestamp': time.time()
        })

        if execution_time > DB_QUERY_THRESHOLD:
            logger.warning(f"Slow database query: {query} took {execution_time:.2f}s")

    def track_cache_operation(self, operation: str, hit: bool, response_time: float):
        """Track cache operation performance"""
        # Track hit rate
        if operation not in self.cache_operations:
            self.cache_operations[operation] = {'hits': 0, 'misses': 0}

        if hit:
            self.cache_operations[operation]['hits'] += 1
        else:
            self.cache_operations[operation]['misses'] += 1

        # Calculate hit rate
        total = self.cache_operations[operation]['hits'] + self.cache_operations[operation]['misses']
        if total > 0:
            hit_rate = self.cache_operations[operation]['hits'] / total
            self._add_metric('cache_hit_rate', hit_rate, CACHE_HIT_RATE_THRESHOLD)

        # Track response time
        self._add_metric('cache_response_time', response_time, 0.1)  # 100ms threshold

    def track_model_operation(self, model_name: str, operation: str, duration: float):
        """Track AI model operation performance"""
        metric_name = f"model_{operation}_{model_name}"
        self._add_metric(metric_name, duration, 5.0)  # 5 second threshold

        # Store detailed model info
        if model_name not in self.model_operations:
            self.model_operations[model_name] = defaultdict(list)

        self.model_operations[model_name][operation].append({
            'duration': duration,
            'timestamp': time.time()
        })

    def get_performance_summary(self) -> dict[str, Any]:
        """Get comprehensive performance summary"""
        with self.lock:
            summary = {
                'system_metrics': {},
                'database_metrics': {},
                'cache_metrics': {},
                'model_metrics': {},
                'alerts': self.alerts.get_alert_stats(),
                'recommendations': []
            }

            # System metrics
            for metric_name in ['cpu_usage', 'memory_usage', 'disk_usage']:
                latest = self._get_latest_metric(metric_name)
                if latest:
                    summary['system_metrics'][metric_name] = {
                        'value': latest.value,
                        'status': latest.status,
                        'timestamp': latest.timestamp
                    }

            # Database metrics
            avg_query_time = self._get_average_metric('db_query_time')
            if avg_query_time:
                summary['database_metrics']['avg_query_time'] = avg_query_time
                summary['database_metrics']['total_queries'] = len(self.db_queries)

            # Cache metrics
            cache_hit_rate = self._get_average_metric('cache_hit_rate')
            if cache_hit_rate:
                summary['cache_metrics']['hit_rate'] = cache_hit_rate
                summary['cache_metrics']['total_operations'] = sum(
                    ops['hits'] + ops['misses']
                    for ops in self.cache_operations.values()
                )

            # Model metrics
            for model_name, operations in self.model_operations.items():
                summary['model_metrics'][model_name] = {}
                for operation, times in operations.items():
                    if times:
                        avg_duration = sum(t['duration'] for t in times) / len(times)
                        summary['model_metrics'][model_name][operation] = {
                            'avg_duration': avg_duration,
                            'total_operations': len(times)
                        }

            return summary

    def get_detailed_metrics(self, metric_name: str, minutes: int = 60) -> list[dict[str, Any]]:
        """Get detailed metrics for a specific metric"""
        if metric_name not in self.metrics:
            return []

        cutoff_time = time.time() - (minutes * 60)
        recent_metrics = [
            {
                'value': metric.value,
                'timestamp': metric.timestamp,
                'status': metric.status
            }
            for metric in self.metrics[metric_name]
            if metric.timestamp > cutoff_time
        ]

        return recent_metrics

    def clear_old_metrics(self, days: int = 7):
        """Clear old metrics data"""
        cutoff_time = time.time() - (days * 24 * 60 * 60)

        with self.lock:
            for metric_name in list(self.metrics.keys()):
                # Remove old metrics
                self.metrics[metric_name] = deque(
                    (metric for metric in self.metrics[metric_name]
                     if metric.timestamp > cutoff_time),
                    maxlen=1000
                )

        logger.info(f"Cleared metrics older than {days} days")


# Global performance monitor instance
performance_monitor = PerformanceMonitor()


