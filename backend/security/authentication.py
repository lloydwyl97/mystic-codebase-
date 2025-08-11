"""
API Authentication System for Mystic Trading Platform

Provides secure authentication with:
- JWT token management
- Role-based access control (RBAC)
- API key authentication
- Session management
- Security monitoring
"""

import hashlib
import jwt
import time
import secrets
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from collections import defaultdict
import threading
import json
from dataclasses import dataclass

try:
    import redis
    from redis import Redis
except ImportError:
    redis = None
    Redis = None

from config import Settings

logger = logging.getLogger(__name__)

# Get settings instance
settings = Settings()

# Authentication configuration
JWT_SECRET_KEY = settings.security.secret_key
JWT_ALGORITHM = settings.security.algorithm
JWT_EXPIRATION_HOURS = settings.security.access_token_expire_minutes // 60
API_KEY_LENGTH = 32
SESSION_TIMEOUT = 3600  # 1 hour
MAX_LOGIN_ATTEMPTS = 5
ACCOUNT_LOCKOUT_DURATION = 900  # 15 minutes


@dataclass
class User:
    """User information"""
    user_id: str
    username: str
    email: str
    roles: List[str]
    permissions: List[str]
    is_active: bool = True
    created_at: float = 0.0
    last_login: float = 0.0


@dataclass
class APIToken:
    """API token information"""
    token_id: str
    user_id: str
    token_hash: str
    permissions: List[str]
    is_active: bool = True
    created_at: float = 0.0
    last_used: float = 0.0


class TokenManager:
    """Manages JWT tokens and API keys"""

    def __init__(self):
        self.secret_key = JWT_SECRET_KEY
        self.algorithm = JWT_ALGORITHM
        self.expiration_hours = JWT_EXPIRATION_HOURS
        self.blacklisted_tokens: set[str] = set()
        self.lock = threading.Lock()

    def generate_jwt_token(self, user_id: str, roles: List[str],
                          permissions: List[str]) -> str:
        """Generate JWT token for user"""
        payload = {
            'user_id': user_id,
            'roles': roles,
            'permissions': permissions,
            'iat': datetime.now(timezone.utc),
            'exp': datetime.now(timezone.utc) + timedelta(hours=self.expiration_hours)
        }

        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

    def verify_jwt_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify JWT token and return payload"""
        try:
            # Check if token is blacklisted
            with self.lock:
                if token in self.blacklisted_tokens:
                    return None

            # Decode and verify token
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])

            # Check if token is expired
            if datetime.now(timezone.utc).timestamp() > payload['exp']:
                return None

            return payload

        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
        except Exception as e:
            logger.error(f"Token verification error: {e}")
            return None

    def blacklist_token(self, token: str):
        """Blacklist a token"""
        with self.lock:
            self.blacklisted_tokens.add(token)

    def generate_api_key(self) -> str:
        """Generate a new API key"""
        return secrets.token_urlsafe(API_KEY_LENGTH)

    def hash_api_key(self, api_key: str) -> str:
        """Hash API key for storage"""
        return hashlib.sha256(api_key.encode()).hexdigest()


class RoleBasedAccessControl:
    """Role-based access control system"""

    def __init__(self):
        self.roles_permissions: Dict[str, List[str]] = {
            'admin': [
                'read:all', 'write:all', 'delete:all', 'admin:all',
                'trading:all', 'analytics:all', 'system:all'
            ],
            'trader': [
                'read:trading', 'write:trading', 'read:portfolio',
                'write:portfolio', 'read:analytics'
            ],
            'analyst': [
                'read:analytics', 'write:analytics', 'read:market_data',
                'read:reports'
            ],
            'viewer': [
                'read:portfolio', 'read:market_data', 'read:reports'
            ],
            'api_user': [
                'read:market_data', 'read:portfolio', 'read:analytics'
            ]
        }

        self.resource_permissions: Dict[str, List[str]] = {
            '/api/market-data': ['read:market_data'],
            '/api/trading': ['write:trading'],
            '/api/portfolio': ['read:portfolio', 'write:portfolio'],
            '/api/analytics': ['read:analytics', 'write:analytics'],
            '/api/admin': ['admin:all'],
            '/api/system': ['system:all']
        }

    def check_permission(self, user_permissions: List[str],
                        required_permissions: List[str]) -> bool:
        """Check if user has required permissions"""
        if not user_permissions or not required_permissions:
            return False

        # Check if user has any of the required permissions
        return any(perm in user_permissions for perm in required_permissions)

    def get_user_permissions(self, roles: List[str]) -> List[str]:
        """Get permissions for user roles"""
        permissions = []
        for role in roles:
            if role in self.roles_permissions:
                permissions.extend(self.roles_permissions[role])
        return list(set(permissions))

    def get_resource_permissions(self, resource: str) -> List[str]:
        """Get required permissions for a resource"""
        return self.resource_permissions.get(resource, [])

    def add_role_permissions(self, role: str, permissions: List[str]):
        """Add permissions to a role"""
        if role not in self.roles_permissions:
            self.roles_permissions[role] = []
        self.roles_permissions[role].extend(permissions)


class AuthenticationManager:
    """Main authentication manager"""

    def __init__(self):
        self.token_manager = TokenManager()
        self.rbac = RoleBasedAccessControl()
        self.redis_client = None
        self.failed_attempts = defaultdict(int)
        self.lockout_times = defaultdict(float)
        self._initialize_redis()
        self._load_default_users()

    def _initialize_redis(self):
        """Initialize Redis connection"""
        try:
            if redis and Redis:
                self.redis_client = Redis(
                    host=settings.redis.url.split('://')[1].split(':')[0],
                    port=int(settings.redis.url.split(':')[-1].split('/')[0]),
                    db=settings.redis.db,
                    decode_responses=True
                )
                # Test connection
                self.redis_client.ping()
                logger.info("✅ Redis connection established")
            else:
                logger.warning("⚠️ Redis not available, using in-memory storage")
        except Exception as e:
            logger.error(f"❌ Redis connection failed: {e}")
            self.redis_client = None

    def _load_default_users(self):
        """Load default users if Redis is available"""
        if not self.redis_client:
            return

        default_users = {
            'admin': {
                'password_hash': hashlib.sha256('admin123'.encode()).hexdigest(),
                'roles': ['admin'],
                'permissions': self.rbac.get_user_permissions(['admin'])
            },
            'trader': {
                'password_hash': hashlib.sha256('trader123'.encode()).hexdigest(),
                'roles': ['trader'],
                'permissions': self.rbac.get_user_permissions(['trader'])
            },
            'analyst': {
                'password_hash': hashlib.sha256('analyst123'.encode()).hexdigest(),
                'roles': ['analyst'],
                'permissions': self.rbac.get_user_permissions(['analyst'])
            }
        }

        for username, user_data in default_users.items():
            user_key = f"user:{username}"
            if not self.redis_client.exists(user_key):
                self.redis_client.hset(user_key, mapping=user_data)
                logger.info(f"✅ Created default user: {username}")

    def authenticate_user(self, username: str, password: str,
                         client_ip: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Authenticate user with username and password"""
        try:
            # Check if account is locked
            if self._is_account_locked(username):
                logger.warning(f"Account locked for user: {username}")
                return None

            # Get user data
            if self.redis_client:
                user_key = f"user:{username}"
                user_data = self.redis_client.hgetall(user_key)
                if not user_data:
                    self._record_failed_login(username, client_ip)
                    return None

                stored_password_hash = user_data.get('password_hash')
                roles = json.loads(user_data.get('roles', '[]'))
                permissions = json.loads(user_data.get('permissions', '[]'))
            else:
                # Fallback to hardcoded users for testing
                if username == 'admin' and password == 'admin123':
                    roles = ['admin']
                    permissions = self.rbac.get_user_permissions(roles)
                elif username == 'trader' and password == 'trader123':
                    roles = ['trader']
                    permissions = self.rbac.get_user_permissions(roles)
                elif username == 'analyst' and password == 'analyst123':
                    roles = ['analyst']
                    permissions = self.rbac.get_user_permissions(roles)
                else:
                    self._record_failed_login(username, client_ip)
                    return None

            # Verify password
            password_hash = hashlib.sha256(password.encode()).hexdigest()
            if self.redis_client and password_hash != stored_password_hash:
                self._record_failed_login(username, client_ip)
                return None

            # Generate token
            token = self.token_manager.generate_jwt_token(username, roles, permissions)

            # Create session
            session_data = {
                'user_id': username,
                'roles': roles,
                'permissions': permissions,
                'client_ip': client_ip,
                'created_at': time.time()
            }
            self._create_session(token, session_data)

            # Log success
            self._log_authentication_success(username, client_ip)

            return {
                'token': token,
                'user_id': username,
                'roles': roles,
                'permissions': permissions,
                'expires_in': JWT_EXPIRATION_HOURS * 3600
            }

        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return None

    def authenticate_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Authenticate user with API key"""
        try:
            if not self.redis_client:
                return None

            api_key_hash = self.token_manager.hash_api_key(api_key)
            token_key = f"api_token:{api_key_hash}"

            token_data = self.redis_client.hgetall(token_key)
            if not token_data or not token_data.get('is_active', 'false') == 'true':
                return None

            user_id = token_data.get('user_id')
            permissions = json.loads(token_data.get('permissions', '[]'))

            # Update last used time
            self.redis_client.hset(token_key, 'last_used', time.time())

            return {
                'user_id': user_id,
                'permissions': permissions,
                'auth_type': 'api_key'
            }

        except Exception as e:
            logger.error(f"API key authentication error: {e}")
            return None

    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify JWT token and return user data"""
        try:
            payload = self.token_manager.verify_jwt_token(token)
            if not payload:
                return None

            # Get session data
            session_data = self._get_session(token)
            if not session_data:
                return None

            return {
                'user_id': payload['user_id'],
                'roles': payload['roles'],
                'permissions': payload['permissions'],
                'session_data': session_data
            }

        except Exception as e:
            logger.error(f"Token verification error: {e}")
            return None

    def check_access(self, user_permissions: List[str], resource: str) -> bool:
        """Check if user has access to resource"""
        required_permissions = self.rbac.get_resource_permissions(resource)
        return self.rbac.check_permission(user_permissions, required_permissions)

    def create_api_key(self, user_id: str, permissions: List[str]) -> str:
        """Create new API key for user"""
        try:
            api_key = self.token_manager.generate_api_key()
            api_key_hash = self.token_manager.hash_api_key(api_key)

            token_data = {
                'user_id': user_id,
                'token_hash': api_key_hash,
                'permissions': json.dumps(permissions),
                'is_active': 'true',
                'created_at': str(time.time()),
                'last_used': str(time.time())
            }

            if self.redis_client:
                token_id = f"api_token:{api_key_hash}"
                self.redis_client.hset(token_id, mapping=token_data)

            return api_key

        except Exception as e:
            logger.error(f"API key creation error: {e}")
            return ""

    def revoke_api_key(self, token_id: str) -> bool:
        """Revoke API key"""
        try:
            if self.redis_client:
                self.redis_client.hset(token_id, 'is_active', 'false')
                return True
            return False
        except Exception as e:
            logger.error(f"API key revocation error: {e}")
            return False

    def logout(self, token: str):
        """Logout user and invalidate token"""
        try:
            self.token_manager.blacklist_token(token)
            self._remove_session(token)
            logger.info("User logged out successfully")
        except Exception as e:
            logger.error(f"Logout error: {e}")

    def _is_account_locked(self, username: str) -> bool:
        """Check if account is locked due to failed attempts"""
        if username not in self.lockout_times:
            return False

        lockout_time = self.lockout_times[username]
        if time.time() - lockout_time < ACCOUNT_LOCKOUT_DURATION:
            return True

        # Clear lockout if expired
        del self.lockout_times[username]
        self.failed_attempts[username] = 0
        return False

    def _record_failed_login(self, username: str, client_ip: Optional[str]):
        """Record failed login attempt"""
        self.failed_attempts[username] += 1

        if self.failed_attempts[username] >= MAX_LOGIN_ATTEMPTS:
            self.lockout_times[username] = time.time()
            logger.warning(f"Account locked for user: {username}")

        logger.warning(f"Failed login attempt for user: {username} from IP: {client_ip}")

    def _create_session(self, token: str, session_data: Dict[str, Any]):
        """Create user session"""
        try:
            if self.redis_client:
                session_key = f"session:{token}"
                self.redis_client.hset(session_key, mapping=session_data)
                self.redis_client.expire(session_key, SESSION_TIMEOUT)
        except Exception as e:
            logger.error(f"Session creation error: {e}")

    def _get_session(self, token: str) -> Optional[Dict[str, Any]]:
        """Get user session data"""
        try:
            if self.redis_client:
                session_key = f"session:{token}"
                session_data = self.redis_client.hgetall(session_key)
                if session_data:
                    # Convert string values back to proper types
                    session_data['created_at'] = float(session_data.get('created_at', 0))
                    session_data['roles'] = json.loads(session_data.get('roles', '[]'))
                    session_data['permissions'] = json.loads(session_data.get('permissions', '[]'))
                    return session_data
            return None
        except Exception as e:
            logger.error(f"Session retrieval error: {e}")
            return None

    def _remove_session(self, token: str):
        """Remove user session"""
        try:
            if self.redis_client:
                session_key = f"session:{token}"
                self.redis_client.delete(session_key)
        except Exception as e:
            logger.error(f"Session removal error: {e}")

    def _log_authentication_success(self, user_id: str, client_ip: Optional[str]):
        """Log successful authentication"""
        logger.info(f"Successful authentication for user: {user_id} from IP: {client_ip}")

    def get_auth_stats(self) -> Dict[str, Any]:
        """Get authentication statistics"""
        return {
            'failed_attempts': dict(self.failed_attempts),
            'lockout_times': dict(self.lockout_times),
            'redis_connected': self.redis_client is not None
        }
