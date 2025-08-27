"""
Authentication API Endpoints
Production-ready authentication endpoints with OAuth support
"""

from fastapi import APIRouter, Depends, HTTPException, status, Response, BackgroundTasks
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import secrets
import structlog

from auth import (
    AuthService,
    User,
    Organization,
    TokenData,
    UserRole,
    get_current_user,
    get_current_organization,
    require_permission,
    check_rate_limit
)

logger = structlog.get_logger()
router = APIRouter(prefix="/auth", tags=["Authentication"])

# Request/Response models
class UserRegistration(BaseModel):
    email: EmailStr
    password: str
    full_name: str
    organization_name: Optional[str] = None
    organization_id: Optional[str] = None
    
class OrganizationRegistration(BaseModel):
    name: str
    domain: Optional[str] = None
    admin_email: EmailStr
    admin_password: str
    
class LoginResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    user: User
    organization: Organization
    
class GoogleAuthRequest(BaseModel):
    token: str
    
class PasswordResetRequest(BaseModel):
    email: EmailStr
    
class PasswordResetConfirm(BaseModel):
    token: str
    new_password: str
    
class ApiKeyRequest(BaseModel):
    name: str
    permissions: Optional[List[str]] = None

@router.post("/register", response_model=User)
async def register_user(
    registration: UserRegistration,
    background_tasks: BackgroundTasks
):
    """Register a new user"""
    try:
        # If organization_id not provided, create new organization
        if not registration.organization_id:
            if not registration.organization_name:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Either organization_id or organization_name is required"
                )
            
            # Create new organization
            org = await AuthService.create_organization(
                name=registration.organization_name,
                admin_email=registration.email,
                admin_password=registration.password
            )
            organization_id = org.id
            role = UserRole.ORG_ADMIN
        else:
            organization_id = registration.organization_id
            role = UserRole.VIEWER  # Default role for new users
        
        # Create user
        user = await AuthService.create_user(
            email=registration.email,
            password=registration.password,
            full_name=registration.full_name,
            organization_id=organization_id,
            role=role
        )
        
        # Send verification email in background
        background_tasks.add_task(send_verification_email, user.email)
        
        return user
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Registration failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed"
        )

@router.post("/login", response_model=LoginResponse)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """Login with email and password"""
    user = await AuthService.authenticate_user(form_data.username, form_data.password)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive"
        )
    
    # Get user's organization
    from google.cloud import firestore
    db = firestore.Client()
    org_doc = db.collection("organizations").document(user.organization_id).get()
    
    if not org_doc.exists:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Organization not found"
        )
    
    organization = Organization(**org_doc.to_dict())
    
    # Create tokens
    token_data = {
        "user_id": user.id,
        "email": user.email,
        "organization_id": user.organization_id,
        "role": user.role,
        "permissions": UserRole.get_permissions(user.role)
    }
    
    access_token = AuthService.create_access_token(token_data)
    refresh_token = AuthService.create_refresh_token(token_data)
    
    return LoginResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        user=user,
        organization=organization
    )

@router.post("/google", response_model=LoginResponse)
async def google_auth(request: GoogleAuthRequest):
    """Authenticate with Google OAuth"""
    try:
        # Verify Google token
        google_user = await AuthService.verify_google_token(request.token)
        
        if not google_user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid Google token"
            )
        
        # Check if user exists
        from google.cloud import firestore
        db = firestore.Client()
        users_ref = db.collection("users")
        user_docs = users_ref.where("email", "==", google_user["email"]).limit(1).get()
        
        if user_docs:
            # Existing user - login
            user_data = user_docs[0].to_dict()
            user = User(**user_data)
            
            # Update last login
            users_ref.document(user.id).update({
                "last_login": datetime.utcnow()
            })
        else:
            # New user - create account
            # Extract domain from email
            domain = google_user["email"].split("@")[1]
            
            # Check if organization exists for this domain
            orgs_ref = db.collection("organizations")
            org_docs = orgs_ref.where("domain", "==", domain).limit(1).get()
            
            if org_docs:
                # Join existing organization
                org_data = org_docs[0].to_dict()
                organization_id = org_data["id"]
            else:
                # Create new organization
                org = await AuthService.create_organization(
                    name=f"{domain} Organization",
                    domain=domain
                )
                organization_id = org.id
            
            # Create user (no password for OAuth users)
            user_id = secrets.token_urlsafe(16)
            user_data = {
                "id": user_id,
                "email": google_user["email"],
                "full_name": google_user["name"],
                "organization_id": organization_id,
                "role": UserRole.VIEWER,
                "is_active": True,
                "is_verified": google_user["email_verified"],
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
                "last_login": datetime.utcnow(),
                "preferences": {},
                "auth_provider": "google"
            }
            
            users_ref.document(user_id).set(user_data)
            user = User(**user_data)
        
        # Get organization
        org_doc = db.collection("organizations").document(user.organization_id).get()
        organization = Organization(**org_doc.to_dict())
        
        # Create tokens
        token_data = {
            "user_id": user.id,
            "email": user.email,
            "organization_id": user.organization_id,
            "role": user.role,
            "permissions": UserRole.get_permissions(user.role)
        }
        
        access_token = AuthService.create_access_token(token_data)
        refresh_token = AuthService.create_refresh_token(token_data)
        
        return LoginResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            user=user,
            organization=organization
        )
        
    except Exception as e:
        logger.error(f"Google auth failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication failed"
        )

@router.post("/refresh")
async def refresh_token(refresh_token: str):
    """Refresh access token"""
    try:
        from jose import jwt, JWTError
        from auth import SECRET_KEY, ALGORITHM
        
        payload = jwt.decode(refresh_token, SECRET_KEY, algorithms=[ALGORITHM])
        
        if payload.get("type") != "refresh":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token type"
            )
        
        # Create new access token
        token_data = {
            "user_id": payload.get("user_id"),
            "email": payload.get("email"),
            "organization_id": payload.get("organization_id"),
            "role": payload.get("role"),
            "permissions": payload.get("permissions")
        }
        
        new_access_token = AuthService.create_access_token(token_data)
        
        return {
            "access_token": new_access_token,
            "token_type": "bearer"
        }
        
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )

@router.post("/logout")
async def logout(current_user: TokenData = Depends(get_current_user)):
    """Logout user (client should delete tokens)"""
    # In production, you might want to blacklist the token
    return {"message": "Successfully logged out"}

@router.get("/me", response_model=User)
async def get_current_user_info(current_user: TokenData = Depends(get_current_user)):
    """Get current user information"""
    from google.cloud import firestore
    db = firestore.Client()
    
    user_doc = db.collection("users").document(current_user.user_id).get()
    
    if not user_doc.exists:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    user_data = user_doc.to_dict()
    user_data.pop("password_hash", None)
    return User(**user_data)

@router.put("/me")
async def update_current_user(
    updates: Dict[str, Any],
    current_user: TokenData = Depends(get_current_user)
):
    """Update current user information"""
    from google.cloud import firestore
    db = firestore.Client()
    
    # Prevent updating sensitive fields
    protected_fields = ["id", "email", "organization_id", "role", "password_hash"]
    for field in protected_fields:
        updates.pop(field, None)
    
    updates["updated_at"] = datetime.utcnow()
    
    db.collection("users").document(current_user.user_id).update(updates)
    
    return {"message": "User updated successfully"}

@router.post("/password-reset")
async def request_password_reset(
    request: PasswordResetRequest,
    background_tasks: BackgroundTasks
):
    """Request password reset"""
    # Generate reset token
    reset_token = secrets.token_urlsafe(32)
    
    # Store reset token with expiration
    from google.cloud import firestore
    db = firestore.Client()
    
    db.collection("password_resets").document(reset_token).set({
        "email": request.email,
        "created_at": datetime.utcnow(),
        "expires_at": datetime.utcnow() + timedelta(hours=1),
        "used": False
    })
    
    # Send reset email in background
    background_tasks.add_task(send_password_reset_email, request.email, reset_token)
    
    return {"message": "Password reset email sent if account exists"}

@router.post("/password-reset/confirm")
async def confirm_password_reset(request: PasswordResetConfirm):
    """Confirm password reset"""
    from google.cloud import firestore
    db = firestore.Client()
    
    # Verify reset token
    reset_doc = db.collection("password_resets").document(request.token).get()
    
    if not reset_doc.exists:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid reset token"
        )
    
    reset_data = reset_doc.to_dict()
    
    if reset_data["used"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Reset token already used"
        )
    
    if datetime.utcnow() > reset_data["expires_at"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Reset token expired"
        )
    
    # Update password
    users_ref = db.collection("users")
    user_docs = users_ref.where("email", "==", reset_data["email"]).limit(1).get()
    
    if not user_docs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    user_data = user_docs[0].to_dict()
    
    users_ref.document(user_data["id"]).update({
        "password_hash": AuthService.get_password_hash(request.new_password),
        "updated_at": datetime.utcnow()
    })
    
    # Mark token as used
    db.collection("password_resets").document(request.token).update({"used": True})
    
    return {"message": "Password reset successful"}

@router.post("/organization", response_model=Organization)
async def create_organization(
    registration: OrganizationRegistration,
    current_user: TokenData = Depends(require_permission("org:write"))
):
    """Create a new organization (super admin only)"""
    try:
        org = await AuthService.create_organization(
            name=registration.name,
            domain=registration.domain,
            admin_email=registration.admin_email,
            admin_password=registration.admin_password
        )
        return org
    except Exception as e:
        logger.error(f"Failed to create organization: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create organization"
        )

@router.post("/api-keys", dependencies=[Depends(check_rate_limit)])
async def create_api_key(
    request: ApiKeyRequest,
    current_user: TokenData = Depends(require_permission("org:write"))
):
    """Create API key for organization"""
    try:
        api_key = await AuthService.create_api_key(
            organization_id=current_user.organization_id,
            name=request.name
        )
        
        return {
            "api_key": api_key,
            "message": "API key created successfully. Store it securely as it won't be shown again."
        }
    except Exception as e:
        logger.error(f"Failed to create API key: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create API key"
        )

@router.delete("/api-keys/{api_key}")
async def revoke_api_key(
    api_key: str,
    current_user: TokenData = Depends(require_permission("org:write"))
):
    """Revoke API key"""
    from google.cloud import firestore
    db = firestore.Client()
    
    # Verify API key belongs to organization
    api_key_doc = db.collection("api_keys").document(api_key).get()
    
    if not api_key_doc.exists:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found"
        )
    
    api_key_data = api_key_doc.to_dict()
    
    if api_key_data["organization_id"] != current_user.organization_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Permission denied"
        )
    
    # Deactivate API key
    db.collection("api_keys").document(api_key).update({"is_active": False})
    
    # Remove from organization
    db.collection("organizations").document(current_user.organization_id).update({
        "api_keys": firestore.ArrayRemove([api_key])
    })
    
    return {"message": "API key revoked successfully"}

# Email sending functions (to be implemented with actual email service)
async def send_verification_email(email: str):
    """Send verification email"""
    logger.info(f"Sending verification email to {email}")
    # Implement with SendGrid, AWS SES, or other email service
    pass

async def send_password_reset_email(email: str, reset_token: str):
    """Send password reset email"""
    logger.info(f"Sending password reset email to {email}")
    # Implement with SendGrid, AWS SES, or other email service
    pass