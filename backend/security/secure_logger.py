"""
Secure Logger for Mystic Trading Platform

Provides secure logging with:
- Sensitive data filtering
- Log encryption and rotation
- Security event monitoring
- Audit trail management
- Compliance logging
"""

import logging
import logging.handlers
import os
import time
import json
from typing import Any, Dict, Optional
from collections import defaultdict, deque
import threading
import re


logger = logging.getLogger(__name__)

# Security configuration
SENSITIVE_FIELDS = [
    'password', 'token', 'key', 'secret', 'auth', 'credential',
    'api_key', 'private_key', 'access_token', 'refresh_token',
    'ssn', 'credit_card', 'account_number', 'pin'
]
LOG_ENCRYPTION_ENABLED = True
LOG_RETENTION_DAYS = 30
SECURITY_LOG_LEVEL = logging.WARNING
AUDIT_LOG_LEVEL = logging.INFO


class LogSanitizer:
    """Sanitizes log messages to remove sensitive information"""

    def __init__(self):
        self.sensitive_patterns = SENSITIVE_FIELDS
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
            'refresh_token': '***REFRESH_TOKEN***',
            'ssn': '***SSN***',
            'credit_card': '***CREDIT_CARD***',
            'account_number': '***ACCOUNT_NUMBER***',
            'pin': '***PIN***'
        }

        # Compile regex patterns for better performance
        self.sensitive_regex = re.compile(
            r'\b(' + '|'.join(self.sensitive_patterns) + r')\b',
            re.IGNORECASE
        )

    def sanitize_message(self, message: str) -> str:
        """Sanitize log message to remove sensitive information"""
        if not message:
            return message

        # Replace sensitive patterns
        sanitized = message
        for pattern, replacement in self.replacement_patterns.items():
            # Use case-insensitive replacement
            sanitized = re.sub(
                rf'\b{re.escape(pattern)}\b',
                replacement,
                sanitized,
                flags=re.IGNORECASE
            )

        # Remove potential sensitive data patterns
        # Credit card numbers
        sanitized = re.sub(r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b', '***CREDIT_CARD***', sanitized)

        # Social security numbers
        sanitized = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '***SSN***', sanitized)

        # API keys (common patterns)
        sanitized = re.sub(r'\b[a-zA-Z0-9]{32,}\b', '***API_KEY***', sanitized)

        # Email addresses (keep domain for debugging)
        sanitized = re.sub(r'\b([a-zA-Z0-9._%+-]+)@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b',
                          r'***@\2', sanitized)
        # IP addresses (keep first octet for debugging)
        sanitized = re.sub(r'\b(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})\b',
                          r'\1.***.***.***', sanitized)

        return sanitized

    def sanitize_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize dictionary to remove sensitive information"""
        if not isinstance(data, dict):
            return data

        sanitized = {}
        for key, value in data.items():
            if isinstance(value, dict):
                sanitized[key] = self.sanitize_dict(value)
            elif isinstance(value, list):
                sanitized[key] = [self.sanitize_dict(item) if isinstance(item, dict)
                                else self.sanitize_message(str(item)) for item in value]
            elif isinstance(value, str):
                # Check if key contains sensitive patterns
                if any(pattern in key.lower() for pattern in self.sensitive_patterns):
                    sanitized[key] = "***SENSITIVE***"
                else:
                    sanitized[key] = self.sanitize_message(value)
            else:
                sanitized[key] = value

        return sanitized


class SecurityLogFilter(logging.Filter):
    """Log filter for security events"""

    def __init__(self, name: str = ""):
        super().__init__(name)
        self.security_events: deque = deque(maxlen=1000)
        self.suspicious_patterns: Dict[str, int] = defaultdict(int)
        self.lock = threading.Lock()

    def filter(self, record: logging.LogRecord) -> bool:
        """Filter log records for security events"""
        message = str(record.getMessage())

        # Check for security-related keywords
        security_keywords = [
            'authentication', 'authorization', 'login', 'logout', 'password',
            'token', 'session', 'access', 'permission', 'security', 'audit'
        ]

        if any(keyword in message.lower() for keyword in security_keywords):
            self._record_security_event(record)

        # Check for suspicious patterns
        if self._is_suspicious_message(message):
            self._record_suspicious_activity(record)

        return True

    def _record_security_event(self, record: logging.LogRecord):
        """Record security event"""
        with self.lock:
            event = {
                'timestamp': time.time(),
                'level': record.levelname,
                'message': record.getMessage(),
                'module': record.module,
                'function': record.funcName
            }
            self.security_events.append(event)

    def _is_suspicious_message(self, message: str) -> bool:
        """Check if message contains suspicious patterns"""
        suspicious_patterns = [
            'failed login', 'invalid password', 'unauthorized access',
            'suspicious activity', 'security violation', 'multiple attempts'
        ]

        return any(pattern in message.lower() for pattern in suspicious_patterns)

    def _record_suspicious_activity(self, record: logging.LogRecord):
        """Record suspicious activity"""
        with self.lock:
            pattern = record.getMessage()[:50]  # First 50 chars as pattern
            self.suspicious_patterns[pattern] += 1

    def get_security_stats(self) -> Dict[str, Any]:
        """Get security logging statistics"""
        with self.lock:
            return {
                'total_security_events': len(self.security_events),
                'suspicious_patterns': dict(self.suspicious_patterns),
                'recent_events': list(self.security_events)[-10:] if self.security_events else []
            }


class AuditLogger:
    """Audit logger for compliance and security tracking"""

    def __init__(self, log_file: str = "logs/audit.log"):
        self.log_file = log_file
        self.audit_events: deque = deque(maxlen=10000)
        self.access_logs: deque = deque(maxlen=5000)
        self.auth_logs: deque = deque(maxlen=5000)
        self.data_access_logs: deque = deque(maxlen=5000)
        self.security_events: deque = deque(maxlen=2000)
        self.lock = threading.Lock()

        # Ensure log directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        # Setup audit file handler
        self.audit_handler = logging.FileHandler(log_file)
        self.audit_handler.setLevel(logging.INFO)
        self.audit_logger = logging.getLogger('audit')
        self.audit_logger.addHandler(self.audit_handler)
        self.audit_logger.setLevel(logging.INFO)

    def log_access(self, user_id: str, action: str, resource: str,
                  success: bool, client_ip: Optional[str] = None):
        """Log access attempt"""
        event = {
            'timestamp': time.time(),
            'event_type': 'access',
            'user_id': user_id,
            'action': action,
            'resource': resource,
            'success': success,
            'client_ip': client_ip
        }

        with self.lock:
            self.access_logs.append(event)
            self.audit_events.append(event)

        # Log to file
        status = "SUCCESS" if success else "FAILED"
        log_message = f"ACCESS: {user_id} - {action} - {resource} - {status}"
        if client_ip:
            log_message += f" - IP: {client_ip}"
        self.audit_logger.info(log_message)

    def log_authentication(self, user_id: str, method: str, success: bool,
                         client_ip: Optional[str] = None):
        """Log authentication attempt"""
        event = {
            'timestamp': time.time(),
            'event_type': 'authentication',
            'user_id': user_id,
            'method': method,
            'success': success,
            'client_ip': client_ip
        }

        with self.lock:
            self.auth_logs.append(event)
            self.audit_events.append(event)

        # Log to file
        status = "SUCCESS" if success else "FAILED"
        log_message = f"AUTH: {user_id} - {method} - {status}"
        if client_ip:
            log_message += f" - IP: {client_ip}"
        self.audit_logger.info(log_message)

    def log_data_access(self, user_id: str, data_type: str, operation: str,
                       success: bool, record_count: Optional[int] = None):
        """Log data access attempt"""
        event = {
            'timestamp': time.time(),
            'event_type': 'data_access',
            'user_id': user_id,
            'data_type': data_type,
            'operation': operation,
            'success': success,
            'record_count': record_count
        }

        with self.lock:
            self.data_access_logs.append(event)
            self.audit_events.append(event)

        # Log to file
        status = "SUCCESS" if success else "FAILED"
        log_message = f"DATA_ACCESS: {user_id} - {operation} - {data_type} - {status}"
        if record_count is not None:
            log_message += f" - Records: {record_count}"
        self.audit_logger.info(log_message)

    def log_security_event(self, event_type: str, severity: str, description: str,
                          user_id: Optional[str] = None, client_ip: Optional[str] = None):
        """Log security event"""
        event = {
            'timestamp': time.time(),
            'event_type': 'security',
            'security_type': event_type,
            'severity': severity,
            'description': description,
            'user_id': user_id,
            'client_ip': client_ip
        }

        with self.lock:
            self.security_events.append(event)
            self.audit_events.append(event)

        # Log to file
        log_message = f"SECURITY: {event_type} - {severity} - {description}"
        if user_id:
            log_message += f" - User: {user_id}"
        if client_ip:
            log_message += f" - IP: {client_ip}"
        self.audit_logger.warning(log_message)

    def _log_audit_event(self, event: Dict[str, Any]):
        """Internal method to log audit event"""
        with self.lock:
            self.audit_events.append(event)

        # Log to file
        event_type = event.get('event_type', 'unknown')
        log_message = f"AUDIT: {event_type} - {json.dumps(event)}"
        self.audit_logger.info(log_message)

    def get_audit_stats(self) -> Dict[str, Any]:
        """Get audit logging statistics"""
        with self.lock:
            return {
                'total_audit_events': len(self.audit_events),
                'access_logs': len(self.access_logs),
                'auth_logs': len(self.auth_logs),
                'data_access_logs': len(self.data_access_logs),
                'security_events': len(self.security_events),
                'recent_events': list(self.audit_events)[-20:] if self.audit_events else []
            }


class SecureLogger:
    """Main secure logger class"""

    def __init__(self):
        self.sanitizer = LogSanitizer()
        self.security_filter = SecurityLogFilter()
        self.audit_logger = AuditLogger()
        self.log_encryption_enabled = LOG_ENCRYPTION_ENABLED

        # Setup secure logging
        self._setup_secure_logging()

    def _setup_secure_logging(self):
        """Setup secure logging configuration"""
        # Add security filter to root logger
        logging.getLogger().addFilter(self.security_filter)

        # Configure logging format
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # Setup file handler with rotation
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)

        file_handler = logging.handlers.RotatingFileHandler(
            os.path.join(log_dir, "secure.log"),
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)

        # Add handler to root logger
        logging.getLogger().addHandler(file_handler)

    def log_info(self, message: str, **kwargs):
        """Log info message securely"""
        sanitized_message = self.sanitizer.sanitize_message(message)
        sanitized_kwargs = self.sanitizer.sanitize_dict(kwargs)

        if sanitized_kwargs:
            log_message = f"{sanitized_message} - {json.dumps(sanitized_kwargs)}"
        else:
            log_message = sanitized_message

        logger.info(log_message)

    def log_warning(self, message: str, **kwargs):
        """Log warning message securely"""
        sanitized_message = self.sanitizer.sanitize_message(message)
        sanitized_kwargs = self.sanitizer.sanitize_dict(kwargs)

        if sanitized_kwargs:
            log_message = f"{sanitized_message} - {json.dumps(sanitized_kwargs)}"
        else:
            log_message = sanitized_message

        logger.warning(log_message)

    def log_error(self, message: str, error: Optional[Exception] = None, **kwargs):
        """Log error message securely"""
        sanitized_message = self.sanitizer.sanitize_message(message)
        sanitized_kwargs = self.sanitizer.sanitize_dict(kwargs)

        if error:
            error_message = self.sanitizer.sanitize_message(str(error))
            log_message = f"{sanitized_message} - Error: {error_message}"
        else:
            log_message = sanitized_message

        if sanitized_kwargs:
            log_message += f" - {json.dumps(sanitized_kwargs)}"

        logger.error(log_message)

    def log_security(self, message: str, severity: str = "medium", **kwargs):
        """Log security event"""
        sanitized_message = self.sanitizer.sanitize_message(message)
        sanitized_kwargs = self.sanitizer.sanitize_dict(kwargs)

        log_message = f"[SECURITY-{severity.upper()}] {sanitized_message}"
        if sanitized_kwargs:
            log_message += f" - {json.dumps(sanitized_kwargs)}"

        logger.warning(log_message)

        # Also log to audit
        self.audit_logger.log_security_event(
            event_type="security_log",
            severity=severity,
            description=sanitized_message,
            **sanitized_kwargs
        )

    def log_access_attempt(self, user_id: str, action: str, resource: str,
                          success: bool, client_ip: Optional[str] = None):
        """Log access attempt"""
        self.audit_logger.log_access(user_id, action, resource, success, client_ip)

        # Also log to general log
        status = "SUCCESS" if success else "FAILED"
        log_message = f"Access attempt: {user_id} - {action} - {resource} - {status}"
        if client_ip:
            log_message += f" - IP: {client_ip}"

        logger.info(log_message)

    def log_authentication_attempt(self, user_id: str, method: str, success: bool,
                                 client_ip: Optional[str] = None):
        """Log authentication attempt"""
        self.audit_logger.log_authentication(user_id, method, success, client_ip)

        # Also log to general log
        status = "SUCCESS" if success else "FAILED"
        log_message = f"Authentication: {user_id} - {method} - {status}"
        if client_ip:
            log_message += f" - IP: {client_ip}"

        logger.info(log_message)

    def get_logging_stats(self) -> Dict[str, Any]:
        """Get comprehensive logging statistics"""
        return {
            'security_stats': self.security_filter.get_security_stats(),
            'audit_stats': self.audit_logger.get_audit_stats(),
            'log_encryption_enabled': self.log_encryption_enabled
        }

    def cleanup_old_logs(self, max_age_days: int = LOG_RETENTION_DAYS):
        """Clean up old log files"""
        cutoff_time = time.time() - (max_age_days * 24 * 3600)

        # Clean up old log files
        log_dir = "logs"
        if os.path.exists(log_dir):
            for filename in os.listdir(log_dir):
                filepath = os.path.join(log_dir, filename)
                if os.path.isfile(filepath):
                    file_time = os.path.getmtime(filepath)
                    if file_time < cutoff_time:
                        os.remove(filepath)
                        logger.info(f"Removed old log file: {filename}")

        logger.info("Cleaned up old log files")


# Global secure logger instance
secure_logger = SecureLogger()


