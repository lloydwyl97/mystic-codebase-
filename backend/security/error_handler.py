"""
Secure Error Handler for Mystic Trading Platform

Provides secure error handling with:
- Information disclosure prevention
- Error logging and monitoring
- Security incident detection
- Custom error responses
- Error sanitization
"""

import hashlib
import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any

from backend.utils.exceptions import MysticException

logger = logging.getLogger(__name__)

# Security configuration
SENSITIVE_PATTERNS = [
    'password', 'token', 'key', 'secret', 'auth', 'credential',
    'api_key', 'private_key', 'access_token', 'refresh_token'
]
MAX_ERROR_LOG_SIZE = 1000
ERROR_RETENTION_HOURS = 24
SECURITY_INCIDENT_THRESHOLD = 10  # errors per minute


@dataclass
class SecurityIncident:
    """Security incident record"""
    incident_id: str
    incident_type: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    description: str
    timestamp: float
    client_id: str | None = None
    endpoint: str | None = None
    error_details: dict[str, Any] | None = None


class ErrorSanitizer:
    """Sanitizes error messages to prevent information disclosure"""

    def __init__(self):
        self.sensitive_patterns = SENSITIVE_PATTERNS
        self.replacement_patterns = {
            'password': '***PASSWORD***',
            'token': '***TOKEN***',
            'key': '***KEY***',
            'secret': '***SECRET***',
            'auth': '***AUTH***',
            'credential': '***CREDENTIAL***',
            'api_key': '***API_KEY***',
            'private_key': '***PRIVATE_KEY***',
            'access_token': '***ACCESS_TOKEN***',
            'refresh_token': '***REFRESH_TOKEN***'
        }

    def sanitize_message(self, message: str) -> str:
        """Sanitize error message to remove sensitive information"""
        if not message:
            return "Internal server error"

        sanitized = message.lower()

        # Replace sensitive patterns
        for pattern, replacement in self.replacement_patterns.items():
            if pattern in sanitized:
                sanitized = sanitized.replace(pattern, replacement)

        # Remove stack traces in production
        if 'traceback' in sanitized or 'stack trace' in sanitized:
            return "Internal server error"

        # Remove file paths
        if '/' in sanitized or '\\' in sanitized:
            sanitized = "Internal server error"

        return sanitized if sanitized != message.lower() else "Internal server error"

    def sanitize_dict(self, data: dict[str, Any]) -> dict[str, Any]:
        """Sanitize dictionary to remove sensitive information"""
        sanitized = {}

        for key, value in data.items():
            if isinstance(value, dict):
                sanitized[key] = self.sanitize_dict(value)
            elif isinstance(value, str):
                # Check if key contains sensitive patterns
                if any(pattern in key.lower() for pattern in self.sensitive_patterns):
                    sanitized[key] = "***SENSITIVE***"
                else:
                    sanitized[key] = self.sanitize_message(value)
            else:
                sanitized[key] = value

        return sanitized


class SecurityIncidentDetector:
    """Detects security incidents from error patterns"""

    def __init__(self):
        self.incidents: deque = deque(maxlen=100)
        self.error_counts: dict[str, int] = defaultdict(int)
        self.client_error_counts: dict[str, int] = defaultdict(int)
        self.lock = threading.Lock()

    def detect_incident(self, error: Exception, client_id: str | None = None,
                       endpoint: str | None = None) -> SecurityIncident | None:
        """Detect security incidents from errors"""
        error_type = type(error).__name__
        current_time = time.time()

        with self.lock:
            # Track error counts
            self.error_counts[error_type] += 1
            if client_id:
                self.client_error_counts[client_id] += 1

            # Check for suspicious patterns
            incident = self._check_suspicious_patterns(error, client_id, endpoint, current_time)

            if incident:
                self.incidents.append(incident)
                logger.warning(f"Security incident detected: {incident.description}")

            return incident

    def _check_suspicious_patterns(self, error: Exception, client_id: str | None,
                                 endpoint: str | None, timestamp: float) -> SecurityIncident | None:
        """Check for suspicious error patterns"""
        error_type = type(error).__name__
        error_message = str(error).lower()

        # Check for authentication/authorization errors
        if 'auth' in error_message or 'unauthorized' in error_message:
            return SecurityIncident(
                incident_id=self._generate_incident_id(),
                incident_type='authentication_error',
                severity='medium',
                description=f'Authentication error: {error_type}',
                timestamp=timestamp,
                client_id=client_id,
                endpoint=endpoint,
                error_details={'error_type': error_type, 'message': error_message}
            )

        # Check for SQL injection attempts
        if any(pattern in error_message for pattern in ['sql', 'select', 'insert', 'update', 'delete']):
            return SecurityIncident(
                incident_id=self._generate_incident_id(),
                incident_type='sql_injection_attempt',
                severity='high',
                description=f'Potential SQL injection attempt: {error_type}',
                timestamp=timestamp,
                client_id=client_id,
                endpoint=endpoint,
                error_details={'error_type': error_type, 'message': error_message}
            )

        # Check for path traversal attempts
        if any(pattern in error_message for pattern in ['../', '..\\', '/etc/', 'c:\\']):
            return SecurityIncident(
                incident_id=self._generate_incident_id(),
                incident_type='path_traversal_attempt',
                severity='high',
                description=f'Potential path traversal attempt: {error_type}',
                timestamp=timestamp,
                client_id=client_id,
                endpoint=endpoint,
                error_details={'error_type': error_type, 'message': error_message}
            )

        # Check for excessive errors from same client
        if client_id and self.client_error_counts[client_id] > SECURITY_INCIDENT_THRESHOLD:
            return SecurityIncident(
                incident_id=self._generate_incident_id(),
                incident_type='excessive_errors',
                severity='medium',
                description=f'Excessive errors from client: {client_id}',
                timestamp=timestamp,
                client_id=client_id,
                endpoint=endpoint,
                error_details={'error_count': self.client_error_counts[client_id]}
            )

        return None

    def _generate_incident_id(self) -> str:
        """Generate unique incident ID"""
        return hashlib.md5(f"{time.time()}".encode()).hexdigest()[:8]

    def get_security_stats(self) -> dict[str, Any]:
        """Get security incident statistics"""
        return {
            'total_incidents': len(self.incidents),
            'error_counts': dict(self.error_counts),
            'client_error_counts': dict(self.client_error_counts),
            'recent_incidents': len([i for i in self.incidents if time.time() - i.timestamp < 3600])
        }


class SecureErrorHandler:
    """Secure error handling system"""

    def __init__(self):
        self.sanitizer = ErrorSanitizer()
        self.incident_detector = SecurityIncidentDetector()
        self.error_log: deque = deque(maxlen=MAX_ERROR_LOG_SIZE)
        self.lock = threading.Lock()

    def handle_error(self, error: Exception, client_id: str | None = None,
                    endpoint: str | None = None, include_details: bool = False) -> dict[str, Any]:
        """Handle error securely"""
        # Detect security incidents
        incident = self.incident_detector.detect_incident(error, client_id, endpoint)

        # Log error
        self._log_error(error, client_id, endpoint, incident)

        # Handle specific error types
        if isinstance(error, MysticException):
            return self._handle_mystic_exception(error, include_details)
        elif 'validation' in str(error).lower() or 'invalid' in str(error).lower():
            return self._handle_validation_error(error, include_details)
        elif 'connection' in str(error).lower() or 'timeout' in str(error).lower():
            return self._handle_connection_error(error, include_details)
        else:
            return self._handle_generic_error(error, include_details)

    def _handle_mystic_exception(self, error: MysticException, include_details: bool) -> dict[str, Any]:
        """Handle Mystic-specific exceptions"""
        response = {
            'error': True,
            'error_code': error.error_code,
            'message': self.sanitizer.sanitize_message(str(error)),
            'timestamp': time.time()
        }

        if include_details:
            response['details'] = {
                'error_type': type(error).__name__,
                'error_code': error.error_code
            }

        return response, error.status_code

    def _handle_validation_error(self, error: Exception, include_details: bool) -> dict[str, Any]:
        """Handle validation errors"""
        response = {
            'error': True,
            'error_code': 'VALIDATION_ERROR',
            'message': 'Invalid request parameters',
            'timestamp': time.time()
        }

        if include_details:
            response['details'] = {
                'error_type': type(error).__name__,
                'message': self.sanitizer.sanitize_message(str(error))
            }

        return response, 400

    def _handle_connection_error(self, error: Exception, include_details: bool) -> dict[str, Any]:
        """Handle connection errors"""
        response = {
            'error': True,
            'error_code': 'CONNECTION_ERROR',
            'message': 'Service temporarily unavailable',
            'timestamp': time.time()
        }

        if include_details:
            response['details'] = {
                'error_type': type(error).__name__,
                'message': self.sanitizer.sanitize_message(str(error))
            }

        return response, 503

    def _handle_generic_error(self, error: Exception, include_details: bool) -> dict[str, Any]:
        """Handle generic errors"""
        response = {
            'error': True,
            'error_code': 'INTERNAL_ERROR',
            'message': 'Internal server error',
            'timestamp': time.time()
        }

        if include_details:
            response['details'] = {
                'error_type': type(error).__name__,
                'message': self.sanitizer.sanitize_message(str(error))
            }

        return response, 500

    def _log_error(self, error: Exception, client_id: str | None,
                  endpoint: str | None, incident: SecurityIncident | None):
        """Log error securely"""
        error_info = {
            'error_type': type(error).__name__,
            'error_message': self.sanitizer.sanitize_message(str(error)),
            'client_id': client_id,
            'endpoint': endpoint,
            'timestamp': time.time(),
            'has_incident': incident is not None
        }

        with self.lock:
            self.error_log.append(error_info)

        # Log to system logger
        log_level = logging.ERROR if incident else logging.WARNING
        logger.log(log_level, f"Error handled: {error_info['error_type']} - {error_info['error_message']}")

    def get_error_stats(self) -> dict[str, Any]:
        """Get error handling statistics"""
        with self.lock:
            # Count error types
            error_types = defaultdict(int)
            for error in self.error_log:
                error_types[error['error_type']] += 1

            return {
                'total_errors': len(self.error_log),
                'error_types': dict(error_types),
                'security_incidents': len(self.incident_detector.incidents),
                'recent_errors': len([e for e in self.error_log
                                    if time.time() - e['timestamp'] < 3600])
            }

    def clear_old_errors(self, max_age_hours: int = ERROR_RETENTION_HOURS):
        """Clear old error logs"""
        cutoff_time = time.time() - (max_age_hours * 3600)

        with self.lock:
            # Remove old errors
            self.error_log = deque(
                (error for error in self.error_log if error['timestamp'] > cutoff_time),
                maxlen=MAX_ERROR_LOG_SIZE
            )

        logger.info("Cleared old error logs")


# Global error handler instance
error_handler = SecureErrorHandler()


