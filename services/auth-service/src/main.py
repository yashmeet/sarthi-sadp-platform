"""
Simplified Authentication Service for SADP - Production Ready
Handles basic API key validation and health checks
"""

import os
import logging
from fastapi import FastAPI, HTTPException, Depends, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import secrets
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="SADP Authentication Service",
    description="Production authentication and authorization service",
    version="1.0.0"
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

# In-memory storage for MVP (will be replaced with Firestore)
api_keys_db = {
    "sadp_test_key_12345": {
        "key_id": "test_key_1",
        "name": "Test API Key",
        "user_id": "test_user",
        "tenant_id": "test_tenant",
        "status": "active",
        "permissions": ["read", "write"],
        "rate_limit": 1000,
        "usage_count": 0,
        "created_at": datetime.utcnow().isoformat()
    }
}

class HealthResponse(BaseModel):
    status: str
    service: str
    timestamp: str
    version: str

class ValidationResponse(BaseModel):
    valid: bool
    key_id: Optional[str] = None
    user_id: Optional[str] = None
    tenant_id: Optional[str] = None
    permissions: list = []

class ApiKeyCreateRequest(BaseModel):
    name: str
    user_id: str
    tenant_id: str
    permissions: list = ["read"]
    expires_days: Optional[int] = 30

class ApiKeyCreateResponse(BaseModel):
    success: bool
    key_id: str
    api_key: str
    name: str
    user_id: str
    tenant_id: str
    permissions: list
    expires_at: str

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        service="auth-service",
        timestamp=datetime.utcnow().isoformat(),
        version="1.0.0"
    )

@app.get("/auth/validate", response_model=ValidationResponse)
async def validate_api_key(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Validate API key"""
    api_key = credentials.credentials
    
    # Check if key exists and is valid
    if api_key in api_keys_db:
        key_data = api_keys_db[api_key]
        if key_data["status"] == "active":
            # Update usage count
            key_data["usage_count"] += 1
            
            return ValidationResponse(
                valid=True,
                key_id=key_data["key_id"],
                user_id=key_data["user_id"],
                tenant_id=key_data["tenant_id"],
                permissions=key_data["permissions"]
            )
    
    # Invalid key
    return ValidationResponse(valid=False)

@app.post("/auth/api-keys", response_model=ApiKeyCreateResponse)
async def create_api_key(request: ApiKeyCreateRequest):
    """Create a new API key"""
    try:
        # Generate unique API key and key ID
        api_key = f"sadp_{secrets.token_urlsafe(32)}"
        key_id = str(uuid.uuid4())
        
        # Calculate expiry date
        expires_at = datetime.utcnow() + timedelta(days=request.expires_days)
        
        # Store in database
        api_keys_db[api_key] = {
            "key_id": key_id,
            "name": request.name,
            "user_id": request.user_id,
            "tenant_id": request.tenant_id,
            "status": "active",
            "permissions": request.permissions,
            "rate_limit": 1000,
            "usage_count": 0,
            "created_at": datetime.utcnow().isoformat(),
            "expires_at": expires_at.isoformat()
        }
        
        logger.info(f"Created API key {key_id} for user {request.user_id}")
        
        return ApiKeyCreateResponse(
            success=True,
            key_id=key_id,
            api_key=api_key,
            name=request.name,
            user_id=request.user_id,
            tenant_id=request.tenant_id,
            permissions=request.permissions,
            expires_at=expires_at.isoformat()
        )
        
    except Exception as e:
        logger.error(f"Failed to create API key: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create API key: {str(e)}")

@app.post("/auth/login")
async def login():
    """Basic login endpoint"""
    return {
        "message": "Login endpoint - implementation coming soon",
        "status": "development"
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "SADP Authentication Service",
        "version": "1.0.0",
        "status": "running",
        "endpoints": ["/health", "/auth/validate", "/auth/api-keys", "/auth/login"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)