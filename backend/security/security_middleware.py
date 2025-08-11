"""
Security Middleware for Mystic Trading Platform

Integrates all security components:
- Rate limiting
- Authentication and authorization
- Error handling
- Secure logging
- Security monitoring
"""

import time
import logging
from typing import Any, Dict, List, Optional
from fastapi import Request, Response
from fastapi.responses import JSONResponse

from security.rate_limiter import rate_limiter
from security.authentication import auth_manager
from security.error_handler import error_handler
from security.secure_logger import secure_logger

logger = logging.getLogger(__name__)


class SecurityMiddleware:
    """Comprehensive security middleware for FastAPI"""

    def __init__(self):
        self.secure_logger = secure_logger
        self.error_handler = error_handler
        self.rate_limiter = rate_limiter
        self.auth_manager = auth_manager

    async def process_request(self, request: Request) -> Optional[Response]:
        """Process incoming request with security checks"""
        start_time = time.time()
        client_ip = self._get_client_ip(request)
        endpoint = request.url.path

        try:
            # 1. Rate limiting check
            rate_limit_result = self.rate_limiter.is_rate_limited(request, endpoint)
            if rate_limit_result[0]:  # Rate limited
                return self._create_rate_limit_response(rate_limit_result[1])

            # 2. Authentication check (for protected endpoints)
            if self._is_protected_endpoint(endpoint):
                auth_result = await self._authenticate_request(request)
                if not auth_result['authenticated']:
                    return self._create_unauthorized_response(auth_result['reason'])

                # 3. Authorization check
                if not self._authorize_request(auth_result['user_info'], endpoint):
                    return self._create_forbidden_response()

                # Add user info to request state
                request.state.user = auth_result['user_info']

            # 4. Log access attempt
            self.secure_logger.log_access_attempt(
                user_id=getattr(request.state, 'user', {}).get('user_id', 'anonymous'),
                action=request.method,
                resource=endpoint,
                success=True,
                client_ip=client_ip
            )

            # 5. Log request processing time
            processing_time = time.time() - start_time
            self.secure_logger.log_info(
                f"Request processed: {request.method} {endpoint}",
                processing_time=processing_time,
                client_ip=client_ip
            )

            return None  # Continue with request processing

        except Exception as e:
            # Handle security-related errors
            error_response = self.error_handler.handle_error(
                e,
                client_id=self._get_client_id(request),
                endpoint=endpoint,
                include_details=False
            )

            self.secure_logger.log_error(
                f"Security middleware error: {str(e)}",
                error=e,
                endpoint=endpoint,
                client_ip=client_ip
            )

            return JSONResponse(
                status_code=error_response[1],
                content=error_response[0]
            )

    async def process_response(self, request: Request, response: Response) -> Response:
        """Process outgoing response with security headers"""
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = "default-src 'self'"

        # Add rate limit headers if available
        client_id = self._get_client_id(request)
        endpoint = request.url.path
        rate_info = self.rate_limiter.get_rate_limit_info(client_id, endpoint)

        if rate_info:
            response.headers["X-RateLimit-Limit"] = str(rate_info['limit_per_minute'])
            response.headers["X-RateLimit-Remaining"] = str(rate_info['remaining_requests'])
            response.headers["X-RateLimit-Reset"] = str(int(rate_info.get('reset_time', 0)))

        return response

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address"""
        # Check for forwarded headers
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        # Check for real IP header
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        # Fallback to direct client IP
        return request.client.host if request.client else "unknown"

    def _get_client_id(self, request: Request) -> str:
        """Extract client identifier"""
        # Try to get from headers first
        client_id = request.headers.get('X-Client-ID')
        if client_id:
            return client_id

        # Try to get from user authentication
        if hasattr(request.state, 'user') and request.state.user:
            return f"user_{request.state.user.get('user_id', 'unknown')}"

        # Fallback to IP address
        return f"ip_{self._get_client_ip(request)}"

    def _is_protected_endpoint(self, endpoint: str) -> bool:
        """Check if endpoint requires authentication"""
        protected_patterns = [
            '/api/auth/', '/api/admin/', '/api/user/',
            '/api/trading/', '/api/portfolio/', '/api/strategies/'
        ]

        return any(pattern in endpoint for pattern in protected_patterns)

    async def _authenticate_request(self, request: Request) -> Dict[str, Any]:
        """Authenticate the request"""
        try:
            # Get authorization header
            auth_header = request.headers.get("Authorization")
            if not auth_header:
                return {
                    'authenticated': False,
                    'reason': 'No authorization header provided'
                }

            # Extract token
            if not auth_header.startswith("Bearer "):
                return {
                    'authenticated': False,
                    'reason': 'Invalid authorization header format'
                }

            token = auth_header[7:]  # Remove "Bearer " prefix

            # Verify token
            auth_result = await self.auth_manager.verify_token(token)
            if not auth_result['valid']:
                return {
                    'authenticated': False,
                    'reason': auth_result.get('reason', 'Invalid token')
                }

            return {
                'authenticated': True,
                'user_info': auth_result['user_info']
            }

        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return {
                'authenticated': False,
                'reason': 'Authentication failed'
            }

    def _authorize_request(self, user_info: Dict[str, Any], endpoint: str) -> bool:
        """Authorize the request based on user permissions"""
        # For now, implement basic authorization
        # In a real system, you would check user roles and permissions
        user_permissions = user_info.get('permissions', [])

        # Check if user has required permissions for the endpoint
        required_permissions = self._get_required_permissions(endpoint)

        return all(perm in user_permissions for perm in required_permissions)

    def _get_required_permissions(self, endpoint: str) -> List[str]:
        """Get required permissions for an endpoint"""
        # This is a simplified implementation
        # In a real system, you would have a more sophisticated permission mapping
        if '/api/admin/' in endpoint:
            return ['admin']
        elif '/api/trading/' in endpoint:
            return ['trading']
        elif '/api/portfolio/' in endpoint:
            return ['portfolio']
        else:
            return []

    def _create_rate_limit_response(self, rate_limit_info: Dict[str, Any]) -> JSONResponse:
        """Create rate limit exceeded response"""
        return JSONResponse(
            status_code=429,
            content={
                'error': True,
                'error_code': 'RATE_LIMIT_EXCEEDED',
                'message': rate_limit_info.get('reason', 'Rate limit exceeded'),
                'retry_after': rate_limit_info.get('retry_after', 60),
                'remaining_requests': rate_limit_info.get('remaining_requests', 0)
            },
            headers={
                'Retry-After': str(rate_limit_info.get('retry_after', 60)),
                'X-RateLimit-Remaining': str(rate_limit_info.get('remaining_requests', 0))
            }
        )

    def _create_unauthorized_response(self, reason: str) -> JSONResponse:
        """Create unauthorized response"""
        return JSONResponse(
            status_code=401,
            content={
                'error': True,
                'error_code': 'UNAUTHORIZED',
                'message': reason
            }
        )

    def _create_forbidden_response(self) -> JSONResponse:
        """Create forbidden response"""
        return JSONResponse(
            status_code=403,
            content={
                'error': True,
                'error_code': 'FORBIDDEN',
                'message': 'Insufficient permissions to access this resource'
            }
        )


class SecurityDecorator:
    """Security decorator for FastAPI endpoints"""

    def __init__(self, security_middleware: SecurityMiddleware):
        self.middleware = security_middleware

    def require_auth(self, required_permissions: Optional[List[str]] = None):
        """Decorator to require authentication and optional permissions"""
        def decorator(func):
            async def wrapper(*args, **kwargs):
                # This would be implemented in the actual endpoint
                # For now, we'll just return the function
                return await func(*args, **kwargs)
            return wrapper
        return decorator

    def rate_limited(self, requests_per_minute: int = 100):
        """Decorator to apply rate limiting"""
        def decorator(func):
            async def wrapper(*args, **kwargs):
                # This would be implemented in the actual endpoint
                # For now, we'll just return the function
                return await func(*args, **kwargs)
            return wrapper
        return decorator


class SecurityMonitor:
    """Security monitoring and alerting"""

    def __init__(self, security_middleware: SecurityMiddleware):
        self.middleware = security_middleware
        self.security_events: List[Dict[str, Any]] = []

    def log_security_event(self, event_type: str, severity: str,
                          description: str, **kwargs):
        """Log security event"""
        event = {
            'event_type': event_type,
            'severity': severity,
            'description': description,
            'timestamp': time.time(),
            **kwargs
        }

        self.security_events.append(event)
        self.middleware.secure_logger.log_security(
            description, severity=severity, **kwargs
        )

    def get_security_stats(self) -> Dict[str, Any]:
        """Get comprehensive security statistics"""
        return {
            'rate_limiting': self.middleware.rate_limiter.get_rate_limit_stats(),
            'authentication': self.middleware.auth_manager.get_auth_stats(),
            'error_handling': self.middleware.error_handler.get_error_stats(),
            'logging': self.middleware.secure_logger.get_logging_stats(),
            'security_events': len(self.security_events)
        }


# Global security middleware instance
security_middleware = SecurityMiddleware()
security_decorator = SecurityDecorator(security_middleware)
security_monitor = SecurityMonitor(security_middleware)
