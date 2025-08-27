from pydantic_settings import BaseSettings
from typing import List, Optional
import os

class Settings(BaseSettings):
    # Service Info
    SERVICE_NAME: str = "agent-runtime"
    VERSION: str = "1.0.0"
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    PORT: int = int(os.getenv("PORT", "8000"))
    
    # GCP Configuration
    PROJECT_ID: str = os.getenv("GCP_PROJECT_ID", "sarthi-patient-experience-hub")
    REGION: str = os.getenv("GCP_REGION", "us-central1")
    
    # Database
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "postgresql+asyncpg://postgres:password@localhost/sarthi"
    )
    
    # Pub/Sub
    PUBSUB_TOPIC: str = os.getenv("PUBSUB_TOPIC", "agent-runtime-topic")
    PUBSUB_SUBSCRIPTION: str = os.getenv("PUBSUB_SUBSCRIPTION", "agent-runtime-sub")
    
    # Storage
    STORAGE_BUCKET: str = os.getenv("STORAGE_BUCKET", "sarthi-app-data-bucket")
    POML_BUCKET: str = os.getenv("POML_BUCKET", "sarthi-poml-library-bucket")
    
    # AI/ML Services
    VERTEX_AI_LOCATION: str = os.getenv("VERTEX_AI_LOCATION", "us-central1")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-pro")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    
    # Secret Manager
    SQL_PASSWORD_SECRET: str = os.getenv("SQL_PASSWORD_SECRET", "sql-password")
    POML_API_KEY_SECRET: str = os.getenv("POML_API_KEY_SECRET", "poml-api-key")
    GEMINI_API_KEY_SECRET: str = os.getenv("GEMINI_API_KEY_SECRET", "gemini-api-key")
    
    # Security
    JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "your-secret-key-here")
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # CORS
    ALLOWED_ORIGINS: List[str] = os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:3000,http://localhost:8080"
    ).split(",")
    
    # Supported Agents
    SUPPORTED_AGENTS: List[str] = [
        "document_processor",
        "clinical",
        "billing",
        "voice",
        "health_assistant",
        "medication_entry",
        "referral_processor",
        "lab_result_entry"
    ]
    
    # Performance
    MAX_CONCURRENT_REQUESTS: int = int(os.getenv("MAX_CONCURRENT_REQUESTS", "100"))
    REQUEST_TIMEOUT_SECONDS: int = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "300"))
    
    # Monitoring
    ENABLE_METRICS: bool = os.getenv("ENABLE_METRICS", "True").lower() == "true"
    ENABLE_TRACING: bool = os.getenv("ENABLE_TRACING", "True").lower() == "true"
    
    class Config:
        env_file = ".env"
        case_sensitive = True