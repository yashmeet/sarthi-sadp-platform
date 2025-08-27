"""
Authentication and Authorization System
Production-ready multi-tenant authentication with role-based access control
"""

from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, OAuth2PasswordBearer
from pydantic import BaseModel, EmailStr
import os
import secrets
from google.cloud import firestore
from google.oauth2 import id_token
from google.auth.transport import requests
import structlog

logger = structlog.get_logger()

# Configuration
SECRET_KEY = os.environ.get("JWT_SECRET_KEY", secrets.token_urlsafe(32))
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")
security = HTTPBearer()

# Firestore client
db = firestore.Client(project=os.environ.get("GCP_PROJECT_ID", "sarthi-patient-experience-hub"))

# User roles
class UserRole:
    SUPER_ADMIN = "super_admin"
    ORG_ADMIN = "org_admin"
    DEVELOPER = "developer"
    VIEWER = "viewer"
    
    @classmethod
    def get_permissions(cls, role: str) -> List[str]:
        """Get permissions for a role"""
        permissions = {
            cls.SUPER_ADMIN: ["*"],  # All permissions
            cls.ORG_ADMIN: [
                "org:read", "org:write", "org:delete",
                "user:read", "user:write", "user:delete",
                "agent:read", "agent:write", "agent:delete", "agent:execute",
                "template:read", "template:write", "template:delete", "template:execute"
            ],
            cls.DEVELOPER: [
                "org:read",
                "user:read",
                "agent:read", "agent:write", "agent:execute",
                "template:read", "template:write", "template:execute"
            ],
            cls.VIEWER: [
                "org:read",
                "user:read",
                "agent:read",
                "template:read"
            ]
        }
        return permissions.get(role, [])

class Organization(BaseModel):
    """Organization model"""
    id: str
    name: str
    domain: Optional[str] = None
    settings: Dict[str, Any] = {}
    subscription_tier: str = "free"
    api_keys: List[str] = []
    created_at: datetime
    updated_at: datetime
    is_active: bool = True
    
class User(BaseModel):
    """User model"""
    id: str
    email: EmailStr
    full_name: str
    organization_id: str
    role: str = UserRole.VIEWER
    is_active: bool = True
    is_verified: bool = False
    created_at: datetime
    updated_at: datetime
    last_login: Optional[datetime] = None
    preferences: Dict[str, Any] = {}
    
class TokenData(BaseModel):
    """Token data model"""
    user_id: str
    email: str
    organization_id: str
    role: str
    permissions: List[str]
    
class AuthService:
    """Authentication service"""
    
    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """Verify password"""
        return pwd_context.verify(plain_password, hashed_password)
    
    @staticmethod
    def get_password_hash(password: str) -> str:
        """Hash password"""
        return pwd_context.hash(password)
    
    @staticmethod
    def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
        """Create JWT access token"""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        
        to_encode.update({"exp": expire, "type": "access"})
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt
    
    @staticmethod
    def create_refresh_token(data: dict):
        """Create JWT refresh token"""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
        to_encode.update({"exp": expire, "type": "refresh"})
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt
    
    @staticmethod
    async def create_user(
        email: str,
        password: str,
        full_name: str,
        organization_id: str,
        role: str = UserRole.VIEWER
    ) -> User:
        """Create a new user"""
        try:
            # Check if user exists
            users_ref = db.collection("users")
            existing = users_ref.where("email", "==", email).limit(1).get()
            if existing:
                raise ValueError("User already exists")
            
            # Create user
            user_id = secrets.token_urlsafe(16)
            user_data = {
                "id": user_id,
                "email": email,
                "password_hash": AuthService.get_password_hash(password),
                "full_name": full_name,
                "organization_id": organization_id,
                "role": role,
                "is_active": True,
                "is_verified": False,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
                "preferences": {}
            }
            
            users_ref.document(user_id).set(user_data)
            
            # Return user without password
            user_data.pop("password_hash")
            return User(**user_data)
            
        except Exception as e:
            logger.error(f"Failed to create user: {e}")
            raise
    
    @staticmethod
    async def create_organization(
        name: str,
        domain: Optional[str] = None,
        admin_email: str = None,
        admin_password: str = None
    ) -> Organization:
        """Create a new organization with admin user"""
        try:
            # Create organization
            org_id = secrets.token_urlsafe(16)
            org_data = {
                "id": org_id,
                "name": name,
                "domain": domain,
                "settings": {},
                "subscription_tier": "free",
                "api_keys": [],
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
                "is_active": True
            }
            
            orgs_ref = db.collection("organizations")
            orgs_ref.document(org_id).set(org_data)
            
            # Create admin user if provided
            if admin_email and admin_password:
                await AuthService.create_user(
                    email=admin_email,
                    password=admin_password,
                    full_name=f"{name} Admin",
                    organization_id=org_id,
                    role=UserRole.ORG_ADMIN
                )
            
            return Organization(**org_data)
            
        except Exception as e:
            logger.error(f"Failed to create organization: {e}")
            raise
    
    @staticmethod
    async def authenticate_user(email: str, password: str) -> Optional[User]:
        """Authenticate user with email and password"""
        try:
            users_ref = db.collection("users")
            user_docs = users_ref.where("email", "==", email).limit(1).get()
            
            if not user_docs:
                return None
            
            user_data = user_docs[0].to_dict()
            
            if not AuthService.verify_password(password, user_data.get("password_hash", "")):
                return None
            
            # Update last login
            users_ref.document(user_data["id"]).update({
                "last_login": datetime.utcnow()
            })
            
            # Return user without password
            user_data.pop("password_hash", None)
            return User(**user_data)
            
        except Exception as e:
            logger.error(f"Failed to authenticate user: {e}")
            return None
    
    @staticmethod
    async def verify_google_token(token: str) -> Optional[Dict[str, Any]]:
        """Verify Google OAuth token"""
        try:
            # Verify the token
            idinfo = id_token.verify_oauth2_token(
                token, 
                requests.Request(),
                os.environ.get("GOOGLE_OAUTH_CLIENT_ID")
            )
            
            if idinfo['iss'] not in ['accounts.google.com', 'https://accounts.google.com']:
                raise ValueError('Invalid issuer')
            
            return {
                "email": idinfo['email'],
                "name": idinfo.get('name', ''),
                "picture": idinfo.get('picture', ''),
                "email_verified": idinfo.get('email_verified', False)
            }
            
        except Exception as e:
            logger.error(f"Failed to verify Google token: {e}")
            return None
    
    @staticmethod
    async def create_api_key(organization_id: str, name: str) -> str:
        """Create API key for organization"""
        try:
            api_key = f"sk_{secrets.token_urlsafe(32)}"
            
            # Store API key
            api_keys_ref = db.collection("api_keys")
            api_keys_ref.document(api_key).set({
                "key": api_key,
                "name": name,
                "organization_id": organization_id,
                "created_at": datetime.utcnow(),
                "last_used": None,
                "is_active": True
            })
            
            # Add to organization
            orgs_ref = db.collection("organizations")
            orgs_ref.document(organization_id).update({
                "api_keys": firestore.ArrayUnion([api_key])
            })
            
            return api_key
            
        except Exception as e:
            logger.error(f"Failed to create API key: {e}")
            raise

# Dependency injection
async def get_current_user(credentials: HTTPAuthorizationCredentials = Security(security)) -> TokenData:
    """Get current user from JWT token"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        
        user_id: str = payload.get("user_id")
        if user_id is None:
            raise credentials_exception
        
        return TokenData(
            user_id=payload.get("user_id"),
            email=payload.get("email"),
            organization_id=payload.get("organization_id"),
            role=payload.get("role"),
            permissions=payload.get("permissions", [])
        )
        
    except JWTError:
        raise credentials_exception

def require_permission(permission: str):
    """Decorator to require specific permission"""
    async def permission_checker(current_user: TokenData = Depends(get_current_user)):
        if "*" in current_user.permissions or permission in current_user.permissions:
            return current_user
        
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Permission denied. Required: {permission}"
        )
    
    return permission_checker

async def get_current_organization(current_user: TokenData = Depends(get_current_user)) -> Organization:
    """Get current user's organization"""
    try:
        orgs_ref = db.collection("organizations")
        org_doc = orgs_ref.document(current_user.organization_id).get()
        
        if not org_doc.exists:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Organization not found"
            )
        
        return Organization(**org_doc.to_dict())
        
    except Exception as e:
        logger.error(f"Failed to get organization: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get organization"
        )

# Rate limiting
class RateLimiter:
    """Rate limiter for API endpoints"""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = {}
    
    async def check_rate_limit(self, key: str) -> bool:
        """Check if rate limit is exceeded"""
        now = datetime.utcnow()
        window_start = now - timedelta(seconds=self.window_seconds)
        
        # Clean old requests
        if key in self.requests:
            self.requests[key] = [
                req_time for req_time in self.requests[key]
                if req_time > window_start
            ]
        else:
            self.requests[key] = []
        
        # Check limit
        if len(self.requests[key]) >= self.max_requests:
            return False
        
        # Add current request
        self.requests[key].append(now)
        return True

# Initialize rate limiter
rate_limiter = RateLimiter()

async def check_rate_limit(current_user: TokenData = Depends(get_current_user)):
    """Check rate limit for current user"""
    key = f"{current_user.organization_id}:{current_user.user_id}"
    
    if not await rate_limiter.check_rate_limit(key):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded"
        )
    
    return current_user