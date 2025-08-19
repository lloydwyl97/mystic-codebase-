"""
Security Enhancements for Mystic Trading Platform
Comprehensive security features including encryption, audit logging, and access control
"""

import base64
import hashlib
import hmac
import json
import os
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import wraps
from typing import Any

import structlog
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

logger = structlog.get_logger()


@dataclass
class SecurityEvent:
    event_id: str
    event_type: str
    user_id: str | None
    ip_address: str
    endpoint: str
    method: str
    timestamp: datetime
    details: dict[str, Any]
    severity: str  # low, medium, high, critical
    status: str  # success, failure, blocked


@dataclass
class APIKey:
    key_id: str
    user_id: str
    key_hash: str
    permissions: list[str]
    created_at: datetime
    expires_at: datetime | None
    last_used: datetime | None
    is_active: bool
    rate_limit: int  # requests per minute


class EncryptionManager:
    """Manages encryption and decryption of sensitive data"""

    def __init__(self, master_key: str | None = None):
        if master_key:
            self.master_key = master_key.encode()
        else:
            self.master_key = os.getenv("MASTER_ENCRYPTION_KEY", Fernet.generate_key()).encode()

        self.fernet = Fernet(self.master_key)
        self.key_derivation_salt = os.getenv("KEY_DERIVATION_SALT", secrets.token_hex(16)).encode()

    def encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data"""
        try:
            encrypted_data = self.fernet.encrypt(data.encode())
            return base64.b64encode(encrypted_data).decode()
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise

    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        try:
            encrypted_bytes = base64.b64decode(encrypted_data.encode())
            decrypted_data = self.fernet.decrypt(encrypted_bytes)
            return decrypted_data.decode()
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise

    def derive_key(self, password: str, salt: bytes | None = None) -> bytes:
        """Derive encryption key from password"""
        if salt is None:
            salt = self.key_derivation_salt

        kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=100000)
        return base64.b64encode(kdf.derive(password.encode()))

    def hash_api_key(self, api_key: str) -> str:
        """Hash API key for storage"""
        return hashlib.sha256(api_key.encode()).hexdigest()

    def verify_api_key(self, api_key: str, stored_hash: str) -> bool:
        """Verify API key against stored hash"""
        return hmac.compare_digest(hashlib.sha256(api_key.encode()).hexdigest(), stored_hash)


class RateLimiter:
    """Rate limiting for API endpoints"""

    def __init__(self):
        self.rate_limits: dict[str, dict[str, Any]] = {}
        self.request_counts: dict[str, list[datetime]] = {}

    def add_rate_limit(self, endpoint: str, max_requests: int, window_seconds: int = 60):
        """Add rate limit for endpoint"""
        self.rate_limits[endpoint] = {
            "max_requests": max_requests,
            "window_seconds": window_seconds,
        }

    def is_rate_limited(self, endpoint: str, identifier: str) -> bool:
        """Check if request is rate limited"""
        if endpoint not in self.rate_limits:
            return False

        key = f"{endpoint}:{identifier}"
        now = datetime.now(timezone.utc)

        if key not in self.request_counts:
            self.request_counts[key] = []

        # Remove old requests outside window
        window = self.rate_limits[endpoint]["window_seconds"]
        cutoff_time = now - timedelta(seconds=window)
        self.request_counts[key] = [
            req_time for req_time in self.request_counts[key] if req_time > cutoff_time
        ]

        # Check if limit exceeded
        max_requests = self.rate_limits[endpoint]["max_requests"]
        if len(self.request_counts[key]) >= max_requests:
            return True

        # Add current request
        self.request_counts[key].append(now)
        return False

    def get_remaining_requests(self, endpoint: str, identifier: str) -> int:
        """Get remaining requests for endpoint"""
        if endpoint not in self.rate_limits:
            return float("inf")

        key = f"{endpoint}:{identifier}"
        if key not in self.request_counts:
            return self.rate_limits[endpoint]["max_requests"]

        now = datetime.now(timezone.utc)
        window = self.rate_limits[endpoint]["window_seconds"]
        cutoff_time = now - timedelta(seconds=window)

        recent_requests = [
            req_time for req_time in self.request_counts[key] if req_time > cutoff_time
        ]

        return max(
            0,
            self.rate_limits[endpoint]["max_requests"] - len(recent_requests),
        )


class AuditLogger:
    """Comprehensive audit logging system"""

    def __init__(self, log_file: str = "audit.log"):
        self.log_file = log_file
        self.events: list[SecurityEvent] = []
        self.max_events = 10000  # Keep last 10k events in memory

    def log_event(
        self,
        event_type: str,
        user_id: str | None,
        ip_address: str,
        endpoint: str,
        method: str,
        details: dict[str, Any],
        severity: str = "low",
        status: str = "success",
    ):
        """Log security event"""
        event = SecurityEvent(
            event_id=secrets.token_hex(8),
            event_type=event_type,
            user_id=user_id,
            ip_address=ip_address,
            endpoint=endpoint,
            method=method,
            timestamp=datetime.now(timezone.utc),
            details=details,
            severity=severity,
            status=status,
        )

        # Add to memory
        self.events.append(event)
        if len(self.events) > self.max_events:
            self.events.pop(0)

        # Log to file
        self._write_to_file(event)

        # Log to structured logger
        logger.info(
            "Security event",
            event_id=event.event_id,
            event_type=event.event_type,
            user_id=event.user_id,
            ip_address=event.ip_address,
            endpoint=event.endpoint,
            severity=event.severity,
            status=event.status,
        )

    def _write_to_file(self, event: SecurityEvent):
        """Write event to audit log file"""
        try:
            with open(self.log_file, "a") as f:
                log_entry = {
                    "event_id": event.event_id,
                    "event_type": event.event_type,
                    "user_id": event.user_id,
                    "ip_address": event.ip_address,
                    "endpoint": event.endpoint,
                    "method": event.method,
                    "timestamp": event.timestamp.isoformat(),
                    "details": event.details,
                    "severity": event.severity,
                    "status": event.status,
                }
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")

    def get_events(
        self,
        event_type: str | None = None,
        user_id: str | None = None,
        severity: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> list[SecurityEvent]:
        """Get filtered audit events"""
        filtered_events = self.events

        if event_type:
            filtered_events = [e for e in filtered_events if e.event_type == event_type]

        if user_id:
            filtered_events = [e for e in filtered_events if e.user_id == user_id]

        if severity:
            filtered_events = [e for e in filtered_events if e.severity == severity]

        if start_time:
            filtered_events = [e for e in filtered_events if e.timestamp >= start_time]

        if end_time:
            filtered_events = [e for e in filtered_events if e.timestamp <= end_time]

        return filtered_events

    def get_security_summary(self) -> dict[str, Any]:
        """Get security summary statistics"""
        now = datetime.now(timezone.utc)
        last_24h = now - timedelta(hours=24)
        now - timedelta(days=7)

        recent_events = [e for e in self.events if e.timestamp >= last_24h]

        return {
            "total_events_24h": len(recent_events),
            "failed_attempts_24h": len([e for e in recent_events if e.status == "failure"]),
            "blocked_attempts_24h": len([e for e in recent_events if e.status == "blocked"]),
            "high_severity_events_24h": len(
                [e for e in recent_events if e.severity in ["high", "critical"]]
            ),
            "unique_ips_24h": len(set(e.ip_address for e in recent_events)),
            "unique_users_24h": len(set(e.user_id for e in recent_events if e.user_id)),
            "most_active_endpoints": self._get_most_active_endpoints(recent_events),
            "security_score": self._calculate_security_score(recent_events),
        }

    def _get_most_active_endpoints(self, events: list[SecurityEvent]) -> list[dict[str, Any]]:
        """Get most active endpoints"""
        endpoint_counts = {}
        for event in events:
            endpoint_counts[event.endpoint] = endpoint_counts.get(event.endpoint, 0) + 1

        return [
            {"endpoint": endpoint, "count": count}
            for endpoint, count in sorted(
                endpoint_counts.items(), key=lambda x: x[1], reverse=True
            )[:10]
        ]

    def _calculate_security_score(self, events: list[SecurityEvent]) -> float:
        """Calculate security score (0-100)"""
        if not events:
            return 100.0

        total_events = len(events)
        failed_events = len([e for e in events if e.status == "failure"])
        blocked_events = len([e for e in events if e.status == "blocked"])
        high_severity = len([e for e in events if e.severity in ["high", "critical"]])

        # Calculate score based on failure rates and severity
        failure_rate = failed_events / total_events if total_events > 0 else 0
        severity_score = 1.0 - (high_severity / total_events if total_events > 0 else 0)

        base_score = 100.0 * (1.0 - failure_rate) * severity_score

        # Bonus for blocked attempts (shows security is working)
        block_bonus = min(10.0, blocked_events * 0.5)

        return min(100.0, base_score + block_bonus)


class APIKeyManager:
    """Manages API keys and permissions"""

    def __init__(self, encryption_manager: EncryptionManager):
        self.encryption_manager = encryption_manager
        self.api_keys: dict[str, APIKey] = {}
        self.user_permissions: dict[str, list[str]] = {}

    def generate_api_key(
        self,
        user_id: str,
        permissions: list[str],
        expires_in_days: int | None = None,
    ) -> str:
        """Generate new API key"""
        key_id = secrets.token_hex(16)
        api_key = secrets.token_urlsafe(32)

        expires_at = None
        if expires_in_days:
            expires_at = datetime.now(timezone.utc) + timedelta(days=expires_in_days)

        api_key_obj = APIKey(
            key_id=key_id,
            user_id=user_id,
            key_hash=self.encryption_manager.hash_api_key(api_key),
            permissions=permissions,
            created_at=datetime.now(timezone.utc),
            expires_at=expires_at,
            last_used=None,
            is_active=True,
            rate_limit=100,  # Default rate limit
        )

        self.api_keys[key_id] = api_key_obj

        return f"{key_id}.{api_key}"

    def validate_api_key(self, api_key: str) -> APIKey | None:
        """Validate API key and return key object"""
        try:
            key_id, key_value = api_key.split(".", 1)

            if key_id not in self.api_keys:
                return None

            api_key_obj = self.api_keys[key_id]

            # Check if key is active
            if not api_key_obj.is_active:
                return None

            # Check if key is expired
            if api_key_obj.expires_at and api_key_obj.expires_at < datetime.now(timezone.utc):
                return None

            # Verify key hash
            if not self.encryption_manager.verify_api_key(key_value, api_key_obj.key_hash):
                return None

            # Update last used
            api_key_obj.last_used = datetime.now(timezone.utc)

            return api_key_obj

        except Exception as e:
            logger.error(f"API key validation error: {e}")
            return None

    def has_permission(self, api_key: str, permission: str) -> bool:
        """Check if API key has specific permission"""
        key_obj = self.validate_api_key(api_key)
        if not key_obj:
            return False

        return permission in key_obj.permissions

    def revoke_api_key(self, key_id: str):
        """Revoke API key"""
        if key_id in self.api_keys:
            self.api_keys[key_id].is_active = False

    def get_user_keys(self, user_id: str) -> list[APIKey]:
        """Get all API keys for user"""
        return [key for key in self.api_keys.values() if key.user_id == user_id]


class SecurityMiddleware:
    """Security middleware for FastAPI"""

    def __init__(self):
        self.encryption_manager = EncryptionManager()
        self.rate_limiter = RateLimiter()
        self.audit_logger = AuditLogger()
        self.api_key_manager = APIKeyManager(self.encryption_manager)

        # Set up default rate limits
        self.rate_limiter.add_rate_limit("/api/v1/trading", 60, 60)  # 60 requests per minute
        self.rate_limiter.add_rate_limit("/api/v1/portfolio", 120, 60)  # 120 requests per minute
        self.rate_limiter.add_rate_limit("/api/v1/market-data", 300, 60)  # 300 requests per minute

    def require_api_key(self, required_permissions: list[str] = None):
        """Decorator to require API key authentication"""

        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Extract API key from request
                request = kwargs.get("request")
                if not request:
                    raise ValueError("Request object not found")

                api_key = request.headers.get("X-API-Key")
                if not api_key:
                    self.audit_logger.log_event(
                        event_type="api_key_missing",
                        user_id=None,
                        ip_address=request.client.host,
                        endpoint=request.url.path,
                        method=request.method,
                        details={"error": "API key required"},
                        severity="medium",
                        status="failure",
                    )
                    raise ValueError("API key required")

                # Validate API key
                key_obj = self.api_key_manager.validate_api_key(api_key)
                if not key_obj:
                    self.audit_logger.log_event(
                        event_type="api_key_invalid",
                        user_id=None,
                        ip_address=request.client.host,
                        endpoint=request.url.path,
                        method=request.method,
                        details={"error": "Invalid API key"},
                        severity="medium",
                        status="failure",
                    )
                    raise ValueError("Invalid API key")

                # Check permissions
                if required_permissions:
                    for permission in required_permissions:
                        if not self.api_key_manager.has_permission(api_key, permission):
                            self.audit_logger.log_event(
                                event_type="permission_denied",
                                user_id=key_obj.user_id,
                                ip_address=request.client.host,
                                endpoint=request.url.path,
                                method=request.method,
                                details={"required_permissions": (required_permissions)},
                                severity="high",
                                status="failure",
                            )
                            raise ValueError(f"Permission denied: {permission}")

                # Check rate limiting
                if self.rate_limiter.is_rate_limited(request.url.path, key_obj.user_id):
                    self.audit_logger.log_event(
                        event_type="rate_limit_exceeded",
                        user_id=key_obj.user_id,
                        ip_address=request.client.host,
                        endpoint=request.url.path,
                        method=request.method,
                        details={"rate_limit": "exceeded"},
                        severity="medium",
                        status="blocked",
                    )
                    raise ValueError("Rate limit exceeded")

                # Log successful access
                self.audit_logger.log_event(
                    event_type="api_access",
                    user_id=key_obj.user_id,
                    ip_address=request.client.host,
                    endpoint=request.url.path,
                    method=request.method,
                    details={"permissions": key_obj.permissions},
                    severity="low",
                    status="success",
                )

                return await func(*args, **kwargs)

            return wrapper

        return decorator

    def encrypt_sensitive_data(self, data: dict[str, Any]) -> dict[str, Any]:
        """Encrypt sensitive data in response"""
        encrypted_data = data.copy()

        sensitive_fields = ["api_key", "password", "secret", "private_key"]
        for field in sensitive_fields:
            if field in encrypted_data:
                encrypted_data[field] = self.encryption_manager.encrypt_data(
                    str(encrypted_data[field])
                )

        return encrypted_data

    def get_security_status(self) -> dict[str, Any]:
        """Get security status and statistics"""
        return {
            "encryption": {
                "master_key_configured": bool(self.encryption_manager.master_key),
                "key_derivation_salt_configured": bool(self.encryption_manager.key_derivation_salt),
            },
            "rate_limiting": {
                "endpoints_protected": len(self.rate_limiter.rate_limits),
                "active_limits": list(self.rate_limiter.rate_limits.keys()),
            },
            "audit_logging": {
                "total_events": len(self.audit_logger.events),
                "log_file": self.audit_logger.log_file,
            },
            "api_keys": {
                "total_keys": len(self.api_key_manager.api_keys),
                "active_keys": len(
                    [k for k in self.api_key_manager.api_keys.values() if k.is_active]
                ),
            },
            "security_summary": self.audit_logger.get_security_summary(),
        }


# Global security instance
security_middleware = SecurityMiddleware()

