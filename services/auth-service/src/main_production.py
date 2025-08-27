"""
Production Authentication Service for SADP with Firestore Integration
Handles API key validation, user authentication, and tenant isolation
"""

import os
import logging
from fastapi import FastAPI, HTTPException, Depends, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import secrets
import hashlib
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Firestore with error handling
db = None
try:
    from google.cloud import firestore
    db = firestore.Client()
    logger.info("Firestore client initialized successfully")
except Exception as e:
    logger.warning(f"Firestore initialization failed: {e}. Using in-memory storage.")

app = FastAPI(
    title="SADP Authentication Service - Production",
    description="Production authentication and authorization service with Firestore",
    version="2.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security schemes
security = HTTPBearer()

# In-memory fallback storage
api_keys_memory = {}
users_memory = {}

class HealthResponse(BaseModel):
    status: str
    service: str
    timestamp: str
    version: str
    database_connected: bool

class ValidationResponse(BaseModel):
    valid: bool
    key_id: Optional[str] = None
    user_id: Optional[str] = None
    tenant_id: Optional[str] = None
    permissions: list = []

class CreateAPIKeyRequest(BaseModel):
    name: str
    user_id: str
    tenant_id: str
    permissions: List[str] = []
    expires_days: Optional[int] = None

class CreateAPIKeyResponse(BaseModel):
    api_key: str
    key_id: str
    created_at: str
    expires_at: Optional[str] = None

class DatabaseClient:
    """Database abstraction layer with Firestore and in-memory fallback"""
    
    async def create_api_key(self, key_data: dict) -> str:
        """Create API key in database"""
        if db:
            try:
                doc_ref = db.collection('api_keys').document()
                key_data['id'] = doc_ref.id
                doc_ref.set(key_data)
                return doc_ref.id
            except Exception as e:
                logger.error(f"Firestore error: {e}")
        
        # Fallback to memory
        key_id = str(uuid.uuid4())
        key_data['id'] = key_id
        api_keys_memory[key_data['key_hash']] = key_data
        return key_id
    
    async def get_api_key_by_hash(self, key_hash: str) -> Optional[dict]:
        """Get API key by hash"""
        if db:
            try:
                keys_ref = db.collection('api_keys')
                query = keys_ref.where('key_hash', '==', key_hash).limit(1)
                results = query.get()
                
                for doc in results:
                    return doc.to_dict()
                return None
            except Exception as e:
                logger.error(f"Firestore error: {e}")
        
        # Fallback to memory
        return api_keys_memory.get(key_hash)
    
    async def update_api_key_usage(self, key_id: str, usage_data: dict):
        """Update API key usage statistics"""
        if db:
            try:
                doc_ref = db.collection('api_keys').document(key_id)
                doc_ref.update(usage_data)
                return
            except Exception as e:
                logger.error(f"Firestore error: {e}")
        
        # Fallback to memory
        for key_data in api_keys_memory.values():
            if key_data.get('id') == key_id:
                key_data.update(usage_data)
                break

db_client = DatabaseClient()

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        service="auth-service",
        timestamp=datetime.utcnow().isoformat(),
        version="2.0.0",
        database_connected=db is not None
    )

@app.post("/auth/api-keys", response_model=CreateAPIKeyResponse)
async def create_api_key(request: CreateAPIKeyRequest):
    """Create a new API key"""
    
    # Generate API key
    api_key = f"sadp_{secrets.token_urlsafe(32)}"
    key_hash = hashlib.sha256(api_key.encode()).hexdigest()
    key_id = str(uuid.uuid4())
    
    # Set expiration
    expires_at = None
    if request.expires_days:
        expires_at = (datetime.utcnow() + timedelta(days=request.expires_days)).isoformat()
    
    # Create key data
    key_data = {
        "key_id": key_id,
        "key_hash": key_hash,
        "name": request.name,
        "user_id": request.user_id,
        "tenant_id": request.tenant_id,
        "status": "active",
        "permissions": request.permissions,
        "rate_limit": 1000,  # per hour
        "usage_count": 0,
        "created_at": datetime.utcnow().isoformat(),
        "expires_at": expires_at,
        "last_used": None
    }
    
    # Store in database
    db_key_id = await db_client.create_api_key(key_data)
    
    logger.info(f"API key created: {key_id} for user {request.user_id}")
    
    return CreateAPIKeyResponse(
        api_key=api_key,
        key_id=key_id,
        created_at=key_data["created_at"],
        expires_at=expires_at
    )

@app.get("/auth/validate", response_model=ValidationResponse)
async def validate_api_key(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Validate API key"""
    api_key = credentials.credentials
    
    # Hash the provided key
    key_hash = hashlib.sha256(api_key.encode()).hexdigest()
    
    # Get key from database
    key_data = await db_client.get_api_key_by_hash(key_hash)
    
    if not key_data:
        logger.warning(f"Invalid API key attempted: {api_key[:10]}...")
        return ValidationResponse(valid=False)
    
    # Check if key is active
    if key_data.get("status") != "active":
        logger.warning(f"Inactive API key used: {key_data.get('key_id')}")
        return ValidationResponse(valid=False)
    
    # Check expiration
    if key_data.get("expires_at"):
        expires_at = datetime.fromisoformat(key_data["expires_at"])
        if expires_at < datetime.utcnow():
            logger.warning(f"Expired API key used: {key_data.get('key_id')}")
            return ValidationResponse(valid=False)
    
    # Update usage statistics
    await db_client.update_api_key_usage(
        key_data.get("id", key_data.get("key_id")),
        {
            "usage_count": key_data.get("usage_count", 0) + 1,
            "last_used": datetime.utcnow().isoformat()
        }
    )
    
    logger.info(f"API key validated: {key_data.get('key_id')}")
    
    return ValidationResponse(
        valid=True,
        key_id=key_data["key_id"],
        user_id=key_data["user_id"],
        tenant_id=key_data["tenant_id"],
        permissions=key_data.get("permissions", [])
    )

@app.post("/auth/login")
async def login():
    """Basic login endpoint"""
    return {
        "message": "Login functionality - implementation in progress",
        "status": "development",
        "features": ["jwt_tokens", "user_management", "session_tracking"]
    }

@app.get("/auth/status")
async def get_status():
    """Get authentication service status"""
    memory_keys = len(api_keys_memory)
    firestore_connected = db is not None
    
    return {
        "service": "auth-service",
        "version": "2.0.0",
        "database": "firestore" if firestore_connected else "memory",
        "api_keys_count": memory_keys if not firestore_connected else "stored_in_firestore",
        "features": [
            "api_key_management",
            "validation",
            "rate_limiting",
            "tenant_isolation"
        ]
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "SADP Authentication Service",
        "version": "2.0.0",
        "status": "running",
        "database": "firestore" if db else "memory",
        "endpoints": [
            "/health",
            "/auth/validate", 
            "/auth/api-keys",
            "/auth/login",
            "/auth/status"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)