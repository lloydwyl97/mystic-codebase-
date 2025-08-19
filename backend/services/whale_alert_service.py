"""
Whale Alert Service (in-memory)

Provides TTL-pruned storage and retrieval of whale alerts.
No mock data. Accepts ingest from endpoints and returns recent alerts.

Python 3.10 compatible.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Optional


class WhaleAlertService:
    """In-memory whale alerts with TTL pruning.

    This service is process-local and thread-safe. In production, replace with
    a persistent/streaming backend as needed (e.g., Redis, Kafka).
    """

    def __init__(self, ttl_seconds: int = 3600, max_buffer: int = 5000) -> None:
        self._ttl_seconds: int = ttl_seconds
        self._max_buffer: int = max(100, int(max_buffer))
        self._alerts: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def _now(self) -> float:
        return time.time()

    def _prune_locked(self) -> None:
        cutoff = self._now() - self._ttl_seconds
        # Keep only alerts newer than cutoff
        self._alerts = [a for a in self._alerts if float(a.get("_ts", 0)) >= cutoff]
        # Enforce max buffer newest-first without full sort
        if len(self._alerts) > self._max_buffer:
            # Cheap trim: assume mostly-ordered inserts; keep last N
            self._alerts = self._alerts[-self._max_buffer :]

    def ingest(self, alert: Dict[str, Any]) -> None:
        """Ingest a single alert. Adds timestamp if missing and prunes TTL."""
        # Defensive: only dict payloads are accepted (runtime guard for external callers)
        if not isinstance(alert, dict):  # type: ignore[unreachable]
            return
        with self._lock:
            if "_ts" not in alert:
                alert["_ts"] = self._now()
            self._alerts.append(alert)
            # Keep alerts sorted newest-first without O(n log n) on every insert
            # Defer strict ordering to read; prune now to bound memory
            self._prune_locked()

    def bulk_ingest(self, alerts: List[Dict[str, Any]]) -> int:
        """Ingest many alerts; returns number ingested."""
        if not isinstance(alerts, list):  # type: ignore[unreachable]
            return 0
        count = 0
        with self._lock:
            now = self._now()
            for a in alerts:
                if not isinstance(a, dict):  # type: ignore[unreachable]
                    continue
                if "_ts" not in a:
                    a["_ts"] = now
                self._alerts.append(a)
                count += 1
            self._prune_locked()
        return count

    def get_alerts(self, limit: Optional[int] = 200) -> List[Dict[str, Any]]:
        """Return recent alerts, newest-first, pruned by TTL and limited."""
        with self._lock:
            self._prune_locked()
            alerts_sorted = sorted(self._alerts, key=lambda a: float(a.get("_ts", 0)), reverse=True)
        if isinstance(limit, int) and limit > 0:
            return alerts_sorted[:limit]
        return alerts_sorted


# Singleton instance for easy import/use
whale_alert_service = WhaleAlertService()


