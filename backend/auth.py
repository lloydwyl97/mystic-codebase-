import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

import jwt
from fastapi import HTTPException, WebSocket, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

# Security scheme for HTTP authentication
security = HTTPBearer()

# JWT configuration
JWT_SECRET = os.getenv("JWT_SECRET", "your-secret-key")  # In production, use a strong secret
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30


class AuthenticationException(Exception):
    """Custom exception for authentication errors"""

    pass


def create_access_token(data: Dict[str, Any]) -> str:
    """Create a new JWT access token"""
    to_encode = data.copy()
    expire = datetime.now(timezone.timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    # The exp field will be a datetime
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return encoded_jwt


def verify_token(token: str) -> Dict[str, Any]:
    """Verify a JWT token and return the payload"""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
        )
    except (jwt.InvalidTokenError, jwt.DecodeError):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")


async def get_current_user(
    credentials: HTTPAuthorizationCredentials,
) -> Dict[str, Any]:
    """Get the current user from the JWT token"""
    token = credentials.credentials
    return verify_token(token)


async def verify_websocket_token(
    websocket: WebSocket,
) -> Optional[Dict[str, Any]]:
    """Verify the token from WebSocket connection"""
    try:
        # Get token from query parameters
        token = websocket.query_params.get("token")
        if not token:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return None

        # Verify token
        payload = verify_token(token)
        return payload
    except Exception:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return None


async def authenticate_user(username: str, password: str) -> Dict[str, Any]:
    """Authenticate a user with username and password"""
    try:
        # Real implementation using database authentication
        from database import get_user_by_credentials
        from backend.utils.password_utils import verify_password

        user = await get_user_by_credentials(username)
        if not user or not verify_password(password, user.get("password_hash", "")):
            raise AuthenticationException("Invalid credentials")

        token = create_access_token({"sub": username, "user_id": user["id"]})
        return {"token": token, "user_id": user["id"], "username": username}
    except Exception as e:
        raise AuthenticationException(f"Authentication failed: {str(e)}")


async def register_user(username: str, password: str, email: str) -> Dict[str, Any]:
    """Register a new user"""
    try:
        # Real implementation using database registration
        from database import create_user, get_user_by_credentials
        from backend.utils.password_utils import hash_password

        # Check if user already exists
        existing_user = await get_user_by_credentials(username)
        if existing_user:
            raise AuthenticationException("User already exists")

        # Hash password and create user
        password_hash = hash_password(password)
        user_data = {
            "username": username,
            "email": email,
            "password_hash": password_hash,
        }

        user = await create_user(user_data)
        token = create_access_token({"sub": username, "user_id": user["id"]})
        return {"user_id": user["id"], "username": username, "token": token}
    except Exception as e:
        raise AuthenticationException(f"Registration failed: {str(e)}")


