"""
Request Validator Manager

Handles request validation and field checking.
"""

import json
import logging
import re
from typing import Any, Dict, Optional, Set

from fastapi import HTTPException, Request

logger = logging.getLogger(__name__)


class RequestValidator:
    """Request validation and field checking"""

    def __init__(self):
        # Maximum request size (1MB)
        self.max_request_size = 1024 * 1024

        # Allowed content types
        self.allowed_content_types: Set[str] = {
            "application/json",
            "application/x-www-form-urlencoded",
            "multipart/form-data",
        }

        # Path-specific validations
        self.path_validations: Dict[str, Dict[str, Any]] = {
            "/api/auth/login": {
                "methods": {"POST"},
                "required_fields": {"username", "password"},
                "max_fields": 2,
            },
            "/api/auth/register": {
                "methods": {"POST"},
                "required_fields": {"username", "password", "email"},
                "max_fields": 3,
            },
            "/api/coinbase": {
                "methods": {"GET", "POST"},
                "required_fields": {"symbol"},
                "max_fields": 5,
            },
        }

        # Field validation patterns
        self.field_patterns: Dict[str, str] = {
            "username": r"^[a-zA-Z0-9_-]{3,32}$",
            "password": r"^.{8,}$",
            "email": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
            "symbol": r"^[A-Z0-9-]{1,10}$",
        }

    def validate_content_type(self, request: Request) -> None:
        """Validate request content type"""
        content_type = request.headers.get("content-type", "")
        if not any(content_type.startswith(ct) for ct in self.allowed_content_types):
            raise HTTPException(status_code=415, detail="Unsupported media type")

    def validate_request_size(self, request: Request) -> None:
        """Validate request size"""
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.max_request_size:
            raise HTTPException(status_code=413, detail="Request entity too large")

    def validate_method(self, request: Request, path: str) -> None:
        """Validate request method"""
        if path in self.path_validations:
            allowed_methods: Set[str] = self.path_validations[path]["methods"]
            if request.method not in allowed_methods:
                raise HTTPException(
                    status_code=405,
                    detail=f"Method {request.method} not allowed",
                )

    async def validate_json_body(self, request: Request, path: str) -> Optional[Dict[str, Any]]:
        """Validate JSON request body"""
        try:
            body = await request.json()
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON")

        if path in self.path_validations:
            validation = self.path_validations[path]

            # Check required fields
            required_fields: Set[str] = validation["required_fields"]
            missing_fields: Set[str] = required_fields - set(body.keys())
            if missing_fields:
                raise HTTPException(
                    status_code=400,
                    detail=f"Missing required fields: {', '.join(missing_fields)}",
                )

            # Check maximum fields
            max_fields: int = validation["max_fields"]
            if len(body) > max_fields:
                raise HTTPException(
                    status_code=400,
                    detail=f"Too many fields. Maximum allowed: {max_fields}",
                )

            # Validate field patterns
            for field, pattern in self.field_patterns.items():
                if field in body and not re.match(pattern, str(body[field])):
                    raise HTTPException(status_code=400, detail=f"Invalid {field} format")

        return body

    async def validate_request(self, request: Request) -> None:
        """Validate incoming request"""
        try:
            path = request.url.path

            # Validate content type
            self.validate_content_type(request)

            # Validate request size
            self.validate_request_size(request)

            # Validate method
            self.validate_method(request, path)

            # Validate JSON body for POST/PUT requests
            if request.method in {"POST", "PUT"}:
                await self.validate_json_body(request, path)

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error in request validation: {str(e)}")
            raise HTTPException(status_code=400, detail="Invalid request")
