"""
Response Sanitizer Manager

Handles response sanitization and data cleaning.
"""

import json
import logging
import re
from typing import Any, cast

from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


class ResponseSanitizer:
    """Response sanitization and formatting"""

    def __init__(self):
        # Fields to remove from responses
        self.sensitive_fields: set[str] = {
            "password",
            "api_key",
            "secret",
            "token",
            "authorization",
        }

        # Fields to mask in responses
        self.mask_fields: dict[str, str] = {
            "email": r"(?<=.{3}).(?=.*@)",
            "phone": r"(?<=.{3}).(?=.{4}$)",
            "credit_card": r"(?<=.{4}).(?=.{4}$)",
        }

        # Response size limits
        self.max_response_size = 1024 * 1024  # 1MB

        # Response format configurations
        self.response_formats: dict[str, dict[str, Any]] = {
            "default": {
                "success": bool,
                "data": (dict, list),
                "error": str,
                "timestamp": str,
            },
            "error": {
                "success": bool,
                "error": str,
                "code": int,
                "timestamp": str,
            },
        }

    def sanitize_value(self, value: Any) -> Any:
        """Sanitize a single value"""
        if isinstance(value, str):
            # Remove control characters
            value = re.sub(r"[\x00-\x1F\x7F-\x9F]", "", value)
            # Escape HTML
            value = value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        elif isinstance(value, dict):
            value = self.sanitize_dict(cast(dict[str, Any], value))
        elif isinstance(value, list):
            value = self.sanitize_list(cast(list[Any], value))
        return value

    def sanitize_dict(self, data: dict[str, Any]) -> dict[str, Any]:
        """Sanitize dictionary values"""
        sanitized: dict[str, Any] = {}
        for key, value in data.items():
            # Remove sensitive fields
            if key.lower() in self.sensitive_fields:
                continue

            # Mask sensitive data
            if key.lower() in self.mask_fields:
                if isinstance(value, str):
                    pattern = self.mask_fields[key.lower()]
                    value = re.sub(pattern, "*", value)

            sanitized[key] = self.sanitize_value(value)
        return sanitized

    def sanitize_list(self, data: list[Any]) -> list[Any]:
        """Sanitize list values"""
        return [self.sanitize_value(item) for item in data]

    def format_response(self, data: dict[str, Any], status_code: int) -> dict[str, Any]:
        """Format response according to configuration"""
        if status_code >= 400:
            format_config = self.response_formats["error"]
            formatted: dict[str, Any] = {
                "success": False,
                "error": data.get("detail", "Unknown error"),
                "code": status_code,
                "timestamp": data.get("timestamp", ""),
            }
        else:
            format_config = self.response_formats["default"]
            formatted = {
                "success": True,
                "data": data.get("data", {}),
                "error": None,
                "timestamp": data.get("timestamp", ""),
            }

        # Validate types
        for key, expected_type in format_config.items():
            if key in formatted and not isinstance(formatted[key], expected_type):
                if expected_type == (dict, list):
                    # Handle tuple of types - no conversion needed
                    pass
                else:
                    try:
                        formatted[key] = expected_type(formatted[key])
                    except (ValueError, TypeError):
                        # Keep original value if conversion fails
                        pass

        return formatted

    def check_response_size(self, data: dict[str, Any]) -> dict[str, Any]:
        """Check if response size exceeds limit"""
        try:
            response_size = len(json.dumps(data).encode())
            if response_size > self.max_response_size:
                logger.warning(
                    f"Response size {response_size} exceeds limit {self.max_response_size}"
                )
                return {
                    "success": False,
                    "error": "Response too large",
                    "code": 500,
                    "timestamp": data.get("timestamp", ""),
                }
        except Exception as e:
            logger.error(f"Error checking response size: {str(e)}")
        return data

    def sanitize_response(self, response: JSONResponse) -> JSONResponse:
        """Sanitize and format API response"""
        try:
            # Get response data
            response_body = response.body
            if isinstance(response_body, bytes):
                data_str = response_body.decode("utf-8")
            else:
                data_str = str(response_body)

            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                return response

            # Sanitize data
            if isinstance(data, dict):
                data = self.sanitize_dict(cast(dict[str, Any], data))
            elif isinstance(data, list):
                data = self.sanitize_list(cast(list[Any], data))
                # Convert list to dict for format_response
                data = {"data": data}

            # Format response
            if isinstance(data, dict):
                data = self.format_response(data, response.status_code)
            else:
                data = {
                    "success": True,
                    "data": data,
                    "error": None,
                    "timestamp": "",
                }

            # Check size
            data = self.check_response_size(data)

            # Create new response
            return JSONResponse(
                content=data,
                status_code=response.status_code,
                headers=response.headers,
            )

        except Exception as e:
            logger.error(f"Error sanitizing response: {str(e)}")
            return response


