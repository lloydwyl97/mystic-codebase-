"""
Analytics Engine for AI Services
Provides analytics and metrics tracking functionality.
"""

class AnalyticsEngine:
    def __init__(self):
        self.metrics = {}
        self.events = []

    def track_event(self, name, data):
        """Track an analytics event"""
        event = {
            "name": name,
            "data": data,
            "timestamp": self._get_timestamp()
        }
        self.events.append(event)
        return event

    def get_metrics(self):
        """Get all metrics"""
        return self.metrics.copy()

    def set_metric(self, key, value):
        """Set a metric value"""
        self.metrics[key] = value

    def get_metric(self, key, default=None):
        """Get a metric value"""
        return self.metrics.get(key, default)

    def _get_timestamp(self):
        """Get current timestamp"""
        import time
        return time.time()

    def get_events(self, limit=None):
        """Get recent events"""
        if limit:
            return self.events[-limit:]
        return self.events.copy()

    def clear_events(self):
        """Clear all events"""
        self.events.clear()

    def clear_metrics(self):
        """Clear all metrics"""
        self.metrics.clear()


