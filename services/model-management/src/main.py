"""
AI Model Management Service - SADP Self-Tuning System
Manages multiple AI model configurations and API keys
"""

from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, SecretStr
from typing import Dict, Any, Optional, List, Literal
from datetime import datetime
from enum import Enum
import os
import json
import structlog
import uuid
import httpx
import asyncio
from cryptography.fernet import Fernet
import base64

# Setup structured logging
logger = structlog.get_logger()

# Initialize FastAPI app
app = FastAPI(
    title="AI Model Management Service",
    description="Manage AI model configurations and API keys",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer(auto_error=False)

# Encryption key for API keys (in production, this should be in Secret Manager)
ENCRYPTION_KEY = os.getenv('ENCRYPTION_KEY', Fernet.generate_key().decode())
fernet = Fernet(ENCRYPTION_KEY.encode() if isinstance(ENCRYPTION_KEY, str) else ENCRYPTION_KEY)

# Database (in production, use Cloud SQL or Firestore)
models_db: Dict[str, 'AIModel'] = {}

class ModelProvider(str, Enum):
    """Supported AI model providers"""
    OPENAI = "openai"
    GOOGLE_GEMINI = "google_gemini"
    ANTHROPIC = "anthropic"
    AZURE_OPENAI = "azure_openai"
    HUGGINGFACE = "huggingface"
    COHERE = "cohere"
    MISTRAL = "mistral"
    AWS_BEDROCK = "aws_bedrock"
    CUSTOM = "custom"

class ModelCapability(str, Enum):
    """Model capabilities"""
    TEXT_GENERATION = "text_generation"
    CHAT = "chat"
    EMBEDDINGS = "embeddings"
    CODE_GENERATION = "code_generation"
    MEDICAL_ANALYSIS = "medical_analysis"
    VISION = "vision"
    FUNCTION_CALLING = "function_calling"

class AIModelConfig(BaseModel):
    """AI Model configuration"""
    name: str = Field(..., description="Model name (e.g., gpt-4, gemini-pro)")
    provider: ModelProvider = Field(..., description="Model provider")
    api_endpoint: Optional[str] = Field(None, description="Custom API endpoint")
    api_key: SecretStr = Field(..., description="API key for the model")
    api_key_secondary: Optional[SecretStr] = Field(None, description="Secondary API key for fallback")
    
    # Model settings
    max_tokens: int = Field(4096, description="Maximum tokens for the model")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Temperature setting")
    top_p: float = Field(1.0, ge=0.0, le=1.0, description="Top-p sampling")
    
    # Capabilities
    capabilities: List[ModelCapability] = Field(..., description="Model capabilities")
    supports_streaming: bool = Field(False, description="Whether model supports streaming")
    supports_json_mode: bool = Field(False, description="Whether model supports JSON mode")
    
    # Rate limits
    requests_per_minute: Optional[int] = Field(None, description="Rate limit (requests/min)")
    tokens_per_minute: Optional[int] = Field(None, description="Token limit (tokens/min)")
    
    # Medical-specific settings
    medical_certified: bool = Field(False, description="Whether model is certified for medical use")
    hipaa_compliant: bool = Field(False, description="Whether model is HIPAA compliant")
    
    # Additional settings
    custom_headers: Dict[str, str] = Field(default_factory=dict, description="Custom headers")
    timeout_seconds: int = Field(30, description="Request timeout in seconds")
    retry_attempts: int = Field(3, description="Number of retry attempts")

class AIModel(BaseModel):
    """AI Model with metadata"""
    id: str = Field(..., description="Unique model ID")
    config: AIModelConfig = Field(..., description="Model configuration")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: str = Field("system", description="User who created the model")
    
    # Status
    enabled: bool = Field(True, description="Whether model is enabled")
    test_status: Optional[str] = Field(None, description="Last test status")
    test_timestamp: Optional[datetime] = Field(None, description="Last test timestamp")
    
    # Usage stats
    total_requests: int = Field(0, description="Total requests made")
    total_tokens: int = Field(0, description="Total tokens used")
    total_errors: int = Field(0, description="Total errors encountered")
    avg_latency_ms: float = Field(0.0, description="Average latency in milliseconds")

class ModelTestRequest(BaseModel):
    """Request to test a model"""
    prompt: str = Field("Hello, can you respond to this test message?", description="Test prompt")
    max_tokens: Optional[int] = Field(50, description="Max tokens for test")

class ModelTestResponse(BaseModel):
    """Response from model test"""
    success: bool
    response: Optional[str] = None
    error: Optional[str] = None
    latency_ms: float
    model_info: Dict[str, Any] = Field(default_factory=dict)

def encrypt_api_key(api_key: str) -> str:
    """Encrypt an API key"""
    return fernet.encrypt(api_key.encode()).decode()

def decrypt_api_key(encrypted_key: str) -> str:
    """Decrypt an API key"""
    try:
        # If the key is already decrypted (not base64), return as-is
        if not encrypted_key.startswith('gAAAAA'):
            return encrypted_key
        return fernet.decrypt(encrypted_key.encode()).decode()
    except Exception:
        # If decryption fails, assume it's already decrypted
        return encrypted_key

async def test_openai_model(config: AIModelConfig, prompt: str, max_tokens: int) -> ModelTestResponse:
    """Test OpenAI model"""
    start_time = datetime.utcnow()
    
    try:
        import openai
        
        # Decrypt API key for use
        decrypted_key = decrypt_api_key(config.api_key.get_secret_value())
        client = openai.OpenAI(api_key=decrypted_key)
        
        response = client.chat.completions.create(
            model=config.name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=config.temperature
        )
        
        latency_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        return ModelTestResponse(
            success=True,
            response=response.choices[0].message.content,
            latency_ms=latency_ms,
            model_info={
                "model": response.model,
                "usage": response.usage.dict() if response.usage else {}
            }
        )
        
    except Exception as e:
        latency_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        return ModelTestResponse(
            success=False,
            error=str(e),
            latency_ms=latency_ms
        )

async def test_google_gemini_model(config: AIModelConfig, prompt: str, max_tokens: int) -> ModelTestResponse:
    """Test Google Gemini model"""
    start_time = datetime.utcnow()
    
    try:
        import google.generativeai as genai
        
        # Decrypt API key for use
        decrypted_key = decrypt_api_key(config.api_key.get_secret_value())
        genai.configure(api_key=decrypted_key)
        
        # Handle different Gemini model names
        model_name = config.name.lower()
        if 'flash' in model_name:
            model_name = 'gemini-1.5-flash'
        elif 'pro' in model_name and '1.5' in model_name:
            model_name = 'gemini-1.5-pro'
        elif 'pro' in model_name:
            model_name = 'gemini-pro'
        else:
            model_name = config.name  # Use as-is if not recognized
        
        model = genai.GenerativeModel(model_name)
        
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=config.temperature,
                top_p=config.top_p
            )
        )
        
        latency_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        return ModelTestResponse(
            success=True,
            response=response.text,
            latency_ms=latency_ms,
            model_info={
                "model": config.name,
                "finish_reason": response.candidates[0].finish_reason.name if response.candidates else None
            }
        )
        
    except Exception as e:
        latency_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        return ModelTestResponse(
            success=False,
            error=str(e),
            latency_ms=latency_ms
        )

async def test_anthropic_model(config: AIModelConfig, prompt: str, max_tokens: int) -> ModelTestResponse:
    """Test Anthropic Claude model"""
    start_time = datetime.utcnow()
    
    try:
        import anthropic
        
        # Decrypt API key for use
        decrypted_key = decrypt_api_key(config.api_key.get_secret_value())
        client = anthropic.Anthropic(api_key=decrypted_key)
        
        response = client.messages.create(
            model=config.name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=config.temperature
        )
        
        latency_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        return ModelTestResponse(
            success=True,
            response=response.content[0].text if response.content else None,
            latency_ms=latency_ms,
            model_info={
                "model": response.model,
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens
                }
            }
        )
        
    except Exception as e:
        latency_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        return ModelTestResponse(
            success=False,
            error=str(e),
            latency_ms=latency_ms
        )

async def test_model(config: AIModelConfig, prompt: str, max_tokens: int) -> ModelTestResponse:
    """Test any model based on provider"""
    
    if config.provider == ModelProvider.OPENAI:
        return await test_openai_model(config, prompt, max_tokens)
    elif config.provider == ModelProvider.GOOGLE_GEMINI:
        return await test_google_gemini_model(config, prompt, max_tokens)
    elif config.provider == ModelProvider.ANTHROPIC:
        return await test_anthropic_model(config, prompt, max_tokens)
    else:
        # For other providers, return a generic response
        return ModelTestResponse(
            success=False,
            error=f"Provider {config.provider} testing not yet implemented",
            latency_ms=0
        )

@app.post("/models", response_model=AIModel)
async def create_model(config: AIModelConfig):
    """
    Create a new AI model configuration
    """
    model_id = f"model_{uuid.uuid4().hex[:12]}"
    
    # Encrypt the API key before storing
    encrypted_config = config.dict()
    encrypted_config['api_key'] = encrypt_api_key(config.api_key.get_secret_value())
    if config.api_key_secondary:
        encrypted_config['api_key_secondary'] = encrypt_api_key(config.api_key_secondary.get_secret_value())
    
    model = AIModel(
        id=model_id,
        config=AIModelConfig(**encrypted_config)
    )
    
    models_db[model_id] = model
    
    logger.info("Model created", model_id=model_id, provider=config.provider, name=config.name)
    
    # Return model without encrypted keys
    return_model = model.dict()
    return_model['config']['api_key'] = "***ENCRYPTED***"
    if return_model['config'].get('api_key_secondary'):
        return_model['config']['api_key_secondary'] = "***ENCRYPTED***"
    
    return AIModel(**return_model)

@app.get("/models", response_model=List[AIModel])
async def list_models(
    provider: Optional[ModelProvider] = None,
    enabled: Optional[bool] = None,
    medical_certified: Optional[bool] = None
):
    """
    List all AI models with optional filtering
    """
    models = list(models_db.values())
    
    # Filter by provider
    if provider:
        models = [m for m in models if m.config.provider == provider]
    
    # Filter by enabled status
    if enabled is not None:
        models = [m for m in models if m.enabled == enabled]
    
    # Filter by medical certification
    if medical_certified is not None:
        models = [m for m in models if m.config.medical_certified == medical_certified]
    
    # Hide API keys in response
    return_models = []
    for model in models:
        return_model = model.dict()
        return_model['config']['api_key'] = "***ENCRYPTED***"
        if return_model['config'].get('api_key_secondary'):
            return_model['config']['api_key_secondary'] = "***ENCRYPTED***"
        return_models.append(AIModel(**return_model))
    
    return return_models

@app.get("/models/{model_id}", response_model=AIModel)
async def get_model(model_id: str):
    """
    Get a specific AI model configuration
    """
    if model_id not in models_db:
        raise HTTPException(status_code=404, detail="Model not found")
    
    model = models_db[model_id]
    
    # Hide API keys in response
    return_model = model.dict()
    return_model['config']['api_key'] = "***ENCRYPTED***"
    if return_model['config'].get('api_key_secondary'):
        return_model['config']['api_key_secondary'] = "***ENCRYPTED***"
    
    return AIModel(**return_model)

@app.put("/models/{model_id}", response_model=AIModel)
async def update_model(model_id: str, config: AIModelConfig):
    """
    Update an AI model configuration
    """
    if model_id not in models_db:
        raise HTTPException(status_code=404, detail="Model not found")
    
    model = models_db[model_id]
    
    # Encrypt the API key before storing
    encrypted_config = config.dict()
    encrypted_config['api_key'] = encrypt_api_key(config.api_key.get_secret_value())
    if config.api_key_secondary:
        encrypted_config['api_key_secondary'] = encrypt_api_key(config.api_key_secondary.get_secret_value())
    
    model.config = AIModelConfig(**encrypted_config)
    model.updated_at = datetime.utcnow()
    
    logger.info("Model updated", model_id=model_id)
    
    # Return model without encrypted keys
    return_model = model.dict()
    return_model['config']['api_key'] = "***ENCRYPTED***"
    if return_model['config'].get('api_key_secondary'):
        return_model['config']['api_key_secondary'] = "***ENCRYPTED***"
    
    return AIModel(**return_model)

@app.delete("/models/{model_id}")
async def delete_model(model_id: str):
    """
    Delete an AI model configuration
    """
    if model_id not in models_db:
        raise HTTPException(status_code=404, detail="Model not found")
    
    del models_db[model_id]
    
    logger.info("Model deleted", model_id=model_id)
    
    return {"message": "Model deleted successfully"}

@app.post("/models/{model_id}/test", response_model=ModelTestResponse)
async def test_model_endpoint(model_id: str, request: ModelTestRequest):
    """
    Test an AI model with a sample prompt
    """
    if model_id not in models_db:
        raise HTTPException(status_code=404, detail="Model not found")
    
    model = models_db[model_id]
    
    # Decrypt API key for testing
    decrypted_config = model.config.dict()
    decrypted_config['api_key'] = SecretStr(decrypt_api_key(model.config.api_key.get_secret_value()))
    if model.config.api_key_secondary:
        decrypted_config['api_key_secondary'] = SecretStr(decrypt_api_key(model.config.api_key_secondary.get_secret_value()))
    
    config = AIModelConfig(**decrypted_config)
    
    # Test the model
    result = await test_model(config, request.prompt, request.max_tokens or 50)
    
    # Update model status
    model.test_status = "success" if result.success else "failed"
    model.test_timestamp = datetime.utcnow()
    
    if result.success:
        model.total_requests += 1
        model.avg_latency_ms = ((model.avg_latency_ms * (model.total_requests - 1)) + result.latency_ms) / model.total_requests
    else:
        model.total_errors += 1
    
    logger.info("Model tested", model_id=model_id, success=result.success)
    
    return result

@app.post("/models/{model_id}/enable")
async def enable_model(model_id: str):
    """
    Enable an AI model
    """
    if model_id not in models_db:
        raise HTTPException(status_code=404, detail="Model not found")
    
    models_db[model_id].enabled = True
    models_db[model_id].updated_at = datetime.utcnow()
    
    logger.info("Model enabled", model_id=model_id)
    
    return {"message": "Model enabled successfully"}

@app.post("/models/{model_id}/disable")
async def disable_model(model_id: str):
    """
    Disable an AI model
    """
    if model_id not in models_db:
        raise HTTPException(status_code=404, detail="Model not found")
    
    models_db[model_id].enabled = False
    models_db[model_id].updated_at = datetime.utcnow()
    
    logger.info("Model disabled", model_id=model_id)
    
    return {"message": "Model disabled successfully"}

@app.get("/models/providers/list")
async def list_providers():
    """
    List all supported AI model providers
    """
    return {
        "providers": [
            {
                "id": provider.value,
                "name": provider.value.replace("_", " ").title(),
                "supported": provider in [ModelProvider.OPENAI, ModelProvider.GOOGLE_GEMINI, ModelProvider.ANTHROPIC]
            }
            for provider in ModelProvider
        ]
    }

@app.get("/models/capabilities/list")
async def list_capabilities():
    """
    List all model capabilities
    """
    return {
        "capabilities": [
            {
                "id": capability.value,
                "name": capability.value.replace("_", " ").title()
            }
            for capability in ModelCapability
        ]
    }

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {
        "status": "healthy",
        "service": "model-management",
        "timestamp": datetime.utcnow().isoformat(),
        "total_models": len(models_db),
        "enabled_models": len([m for m in models_db.values() if m.enabled]),
        "version": "1.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)