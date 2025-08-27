"""
Centralized AI Client for SADP Production
Handles Gemini API, Vertex AI, and OpenAI with retry logic and fallbacks
"""

import asyncio
import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum
import structlog
import httpx
import google.generativeai as genai
from google.cloud import aiplatform
import openai

from config.secrets import secret_manager

logger = structlog.get_logger()

class AIProvider(Enum):
    GEMINI = "gemini"
    VERTEX_AI = "vertex_ai"
    OPENAI = "openai"

@dataclass
class AIResponse:
    """Standardized AI response format"""
    text: str
    tokens_used: int
    confidence: float
    provider: AIProvider
    model: str
    execution_time_ms: int
    metadata: Dict[str, Any] = None

class AIClientError(Exception):
    """AI Client specific exceptions"""
    pass

class AIClient:
    """
    Production-ready AI client with retry logic, fallbacks, and monitoring
    """
    
    def __init__(self):
        self.gemini_model = None
        self.vertex_client = None
        self.openai_client = None
        self.initialized = False
        self.retry_attempts = 3
        self.retry_delay = 1.0  # seconds
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_tokens": 0,
            "provider_usage": {provider.value: 0 for provider in AIProvider}
        }
    
    async def initialize(self) -> bool:
        """Initialize AI providers"""
        try:
            # Initialize Gemini
            await self._init_gemini()
            
            # Initialize Vertex AI
            await self._init_vertex_ai()
            
            # Initialize OpenAI as fallback
            await self._init_openai()
            
            self.initialized = True
            logger.info("AI Client initialized successfully")
            return True
            
        except Exception as e:
            logger.error("Failed to initialize AI Client", error=str(e))
            return False
    
    async def _init_gemini(self):
        """Initialize Gemini API"""
        try:
            api_key = secret_manager.get_gemini_api_key()
            if not api_key:
                logger.warning("No Gemini API key found")
                return
                
            genai.configure(api_key=api_key)
            self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
            logger.info("Gemini API initialized")
            
        except Exception as e:
            logger.error("Failed to initialize Gemini", error=str(e))
    
    async def _init_vertex_ai(self):
        """Initialize Vertex AI"""
        try:
            project_id = secret_manager.project_id
            aiplatform.init(project=project_id, location="us-central1")
            logger.info("Vertex AI initialized", project_id=project_id)
            
        except Exception as e:
            logger.error("Failed to initialize Vertex AI", error=str(e))
    
    async def _init_openai(self):
        """Initialize OpenAI as fallback"""
        try:
            api_key = secret_manager.get_openai_api_key()
            if not api_key:
                logger.warning("No OpenAI API key found")
                return
                
            self.openai_client = openai.AsyncOpenAI(api_key=api_key)
            logger.info("OpenAI API initialized")
            
        except Exception as e:
            logger.error("Failed to initialize OpenAI", error=str(e))
    
    async def generate_response(
        self, 
        prompt: str, 
        preferred_provider: AIProvider = AIProvider.GEMINI,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        agent_type: str = "general"
    ) -> AIResponse:
        """
        Generate AI response with retry logic and fallbacks
        """
        if not self.initialized:
            await self.initialize()
        
        self.metrics["total_requests"] += 1
        start_time = time.time()
        
        # Try providers in order of preference
        providers = [preferred_provider]
        if preferred_provider != AIProvider.GEMINI and self.gemini_model:
            providers.append(AIProvider.GEMINI)
        if preferred_provider != AIProvider.VERTEX_AI:
            providers.append(AIProvider.VERTEX_AI)
        if preferred_provider != AIProvider.OPENAI and self.openai_client:
            providers.append(AIProvider.OPENAI)
        
        last_error = None
        
        for provider in providers:
            try:
                response = await self._generate_with_provider(
                    provider, prompt, max_tokens, temperature, agent_type
                )
                
                execution_time = int((time.time() - start_time) * 1000)
                response.execution_time_ms = execution_time
                
                # Update metrics
                self.metrics["successful_requests"] += 1
                self.metrics["total_tokens"] += response.tokens_used
                self.metrics["provider_usage"][provider.value] += 1
                
                logger.info("AI response generated successfully",
                           provider=provider.value,
                           tokens=response.tokens_used,
                           time_ms=execution_time,
                           agent_type=agent_type)
                
                return response
                
            except Exception as e:
                last_error = e
                logger.warning("Provider failed, trying next",
                              provider=provider.value,
                              error=str(e))
                continue
        
        # All providers failed
        self.metrics["failed_requests"] += 1
        error_msg = f"All AI providers failed. Last error: {str(last_error)}"
        logger.error("AI generation failed", error=error_msg)
        raise AIClientError(error_msg)
    
    async def _generate_with_provider(
        self,
        provider: AIProvider,
        prompt: str,
        max_tokens: int,
        temperature: float,
        agent_type: str
    ) -> AIResponse:
        """Generate response using specific provider"""
        
        for attempt in range(self.retry_attempts):
            try:
                if provider == AIProvider.GEMINI and self.gemini_model:
                    return await self._generate_gemini(prompt, max_tokens, temperature)
                
                elif provider == AIProvider.VERTEX_AI:
                    return await self._generate_vertex_ai(prompt, max_tokens, temperature)
                
                elif provider == AIProvider.OPENAI and self.openai_client:
                    return await self._generate_openai(prompt, max_tokens, temperature)
                
                else:
                    raise AIClientError(f"Provider {provider.value} not available")
                    
            except Exception as e:
                if attempt < self.retry_attempts - 1:
                    wait_time = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning("Retrying AI request",
                                  provider=provider.value,
                                  attempt=attempt + 1,
                                  wait_time=wait_time,
                                  error=str(e))
                    await asyncio.sleep(wait_time)
                else:
                    raise e
    
    async def _generate_gemini(self, prompt: str, max_tokens: int, temperature: float) -> AIResponse:
        """Generate using Gemini"""
        generation_config = genai.types.GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=temperature
        )
        
        response = await self.gemini_model.generate_content_async(
            prompt,
            generation_config=generation_config
        )
        
        tokens_used = response.usage_metadata.total_token_count if hasattr(response, 'usage_metadata') else 0
        
        return AIResponse(
            text=response.text,
            tokens_used=tokens_used,
            confidence=0.9,  # Gemini doesn't provide confidence scores
            provider=AIProvider.GEMINI,
            model="gemini-1.5-flash",
            execution_time_ms=0,  # Will be set by caller
            metadata={"generation_config": generation_config}
        )
    
    async def _generate_vertex_ai(self, prompt: str, max_tokens: int, temperature: float) -> AIResponse:
        """Generate using Vertex AI"""
        # Implementation for Vertex AI
        # This is a placeholder - actual implementation would use Vertex AI SDK
        raise AIClientError("Vertex AI implementation pending")
    
    async def _generate_openai(self, prompt: str, max_tokens: int, temperature: float) -> AIResponse:
        """Generate using OpenAI"""
        response = await self.openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        return AIResponse(
            text=response.choices[0].message.content,
            tokens_used=response.usage.total_tokens,
            confidence=0.85,  # OpenAI doesn't provide confidence scores
            provider=AIProvider.OPENAI,
            model="gpt-3.5-turbo",
            execution_time_ms=0,  # Will be set by caller
            metadata={"finish_reason": response.choices[0].finish_reason}
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get AI client metrics"""
        return {
            **self.metrics,
            "success_rate": self.metrics["successful_requests"] / max(self.metrics["total_requests"], 1),
            "average_tokens_per_request": self.metrics["total_tokens"] / max(self.metrics["successful_requests"], 1)
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for AI services"""
        health = {
            "gemini": False,
            "vertex_ai": False,
            "openai": False,
            "overall": False
        }
        
        # Test Gemini
        if self.gemini_model:
            try:
                test_response = await self._generate_gemini("Hello", 10, 0.1)
                health["gemini"] = True
            except Exception:
                pass
        
        # Test OpenAI
        if self.openai_client:
            try:
                test_response = await self._generate_openai("Hello", 10, 0.1)
                health["openai"] = True
            except Exception:
                pass
        
        health["overall"] = any([health["gemini"], health["vertex_ai"], health["openai"]])
        return health

# Global AI client instance
ai_client = AIClient()