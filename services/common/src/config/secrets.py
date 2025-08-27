"""
Secret Manager Integration for SADP
Centralized secret management with fallback to environment variables
"""

import os
from typing import Optional
from google.cloud import secretmanager
import structlog

logger = structlog.get_logger()

class SecretManager:
    """Centralized secret management service"""
    
    def __init__(self, project_id: str = None):
        self.project_id = project_id or os.environ.get("GCP_PROJECT_ID", "sarthi-patient-experience-hub")
        self.client = None
        self._cache = {}
        
        try:
            self.client = secretmanager.SecretManagerServiceClient()
            logger.info("Secret Manager client initialized", project_id=self.project_id)
        except Exception as e:
            logger.warning("Failed to initialize Secret Manager, falling back to env vars", error=str(e))
    
    def get_secret(self, secret_name: str, version: str = "latest") -> Optional[str]:
        """
        Get secret from Secret Manager with fallback to environment variables
        """
        # Check cache first
        cache_key = f"{secret_name}:{version}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Try Secret Manager first
        if self.client:
            try:
                secret_path = f"projects/{self.project_id}/secrets/{secret_name}/versions/{version}"
                response = self.client.access_secret_version(request={"name": secret_path})
                secret_value = response.payload.data.decode("UTF-8")
                
                # Cache the secret
                self._cache[cache_key] = secret_value
                logger.info("Secret retrieved from Secret Manager", secret_name=secret_name)
                return secret_value
                
            except Exception as e:
                logger.warning("Failed to get secret from Secret Manager", 
                             secret_name=secret_name, error=str(e))
        
        # Fallback to environment variable
        env_value = os.environ.get(secret_name)
        if env_value:
            self._cache[cache_key] = env_value
            logger.info("Secret retrieved from environment", secret_name=secret_name)
            return env_value
        
        logger.error("Secret not found in Secret Manager or environment", secret_name=secret_name)
        return None
    
    def get_gemini_api_key(self) -> Optional[str]:
        """Get Gemini API key"""
        return self.get_secret("GEMINI_API_KEY")
    
    def get_openai_api_key(self) -> Optional[str]:
        """Get OpenAI API key (fallback)"""
        return self.get_secret("OPENAI_API_KEY")
    
    def get_db_connection_string(self) -> Optional[str]:
        """Get database connection string"""
        return self.get_secret("DB_CONNECTION_STRING")
    
    def get_api_encryption_key(self) -> Optional[str]:
        """Get API encryption key for PHI protection"""
        return self.get_secret("API_ENCRYPTION_KEY")
    
    def get_jwt_secret(self) -> Optional[str]:
        """Get JWT signing secret"""
        return self.get_secret("JWT_SECRET")

# Global instance
secret_manager = SecretManager()