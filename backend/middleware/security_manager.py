"""
Security Manager

Handles security headers configuration and management.
"""

import logging

from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


class SecurityHeaders:
    """Security headers configuration and management"""

    def __init__(self):
        # Security headers configuration
        self.security_headers = {
            # Prevent clickjacking
            "X-Frame-Options": "DENY",
            # Enable XSS protection
            "X-XSS-Protection": "1; mode=block",
            # Prevent MIME type sniffing
            "X-Content-Type-Options": "nosniff",
            # Strict Transport Security
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            # Content Security Policy
            "Content-Security-Policy": (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
                "style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data: https:; "
                "font-src 'self' data:; "
                "connect-src 'self' https:; "
                "frame-ancestors 'none'; "
                "form-action 'self'"
            ),
            # Referrer Policy
            "Referrer-Policy": "strict-origin-when-cross-origin",
            # Permissions Policy
            "Permissions-Policy": (
                "accelerometer=(), "
                "camera=(), "
                "geolocation=(), "
                "gyroscope=(), "
                "magnetometer=(), "
                "microphone=(), "
                "payment=(), "
                "usb=()"
            ),
            # Cross-Origin Resource Policy
            "Cross-Origin-Resource-Policy": "same-site",
            # Cross-Origin Embedder Policy
            "Cross-Origin-Embedder-Policy": "require-corp",
            # Cross-Origin Opener Policy
            "Cross-Origin-Opener-Policy": "same-origin",
        }

        # Headers to remove
        self.headers_to_remove = {
            "Server",
            "X-Powered-By",
            "X-AspNet-Version",
            "X-AspNetMvc-Version",
        }

        # Path-specific security headers
        self.path_specific_headers = {
            "/api/auth": {
                "Content-Security-Policy": (
                    "default-src 'self'; "
                    "script-src 'self'; "
                    "style-src 'self' 'unsafe-inline'; "
                    "img-src 'self' data:; "
                    "font-src 'self' data:; "
                    "connect-src 'self'; "
                    "frame-ancestors 'none'; "
                    "form-action 'self'"
                )
            },
            "/api/coinbase": {
                "Content-Security-Policy": (
                    "default-src 'self' https://api.coinbase.us; "
                    "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
                    "style-src 'self' 'unsafe-inline'; "
                    "img-src 'self' data: https:; "
                    "font-src 'self' data:; "
                    "connect-src 'self' https://api.coinbase.us; "
                    "frame-ancestors 'none'; "
                    "form-action 'self'"
                )
            },
        }

    def get_path_specific_headers(self, path: str) -> dict[str, str]:
        """Get security headers specific to the request path"""
        for prefix, headers in self.path_specific_headers.items():
            if path.startswith(prefix):
                return headers
        return {}

    def add_security_headers(self, response: JSONResponse, path: str) -> JSONResponse:
        """Add security headers to the response"""
        try:
            # Get path-specific headers
            path_headers = self.get_path_specific_headers(path)

            # Add security headers
            for header, value in self.security_headers.items():
                # Override with path-specific header if exists
                if header in path_headers:
                    response.headers[header] = path_headers[header]
                else:
                    response.headers[header] = value

            # Remove unnecessary headers
            for header in self.headers_to_remove:
                if header in response.headers:
                    del response.headers[header]

            return response

        except Exception as e:
            logger.error(f"Error adding security headers: {str(e)}")
            return response


