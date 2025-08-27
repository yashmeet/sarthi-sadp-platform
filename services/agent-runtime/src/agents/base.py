from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import structlog
import google.generativeai as genai
from datetime import datetime

from ..models import AgentRequest
from ..config import Settings

logger = structlog.get_logger()

class BaseAgent(ABC):
    """Base class for all AI agents"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.version = "1.0.0"
        self.model_name = settings.GEMINI_MODEL
        self.model = None
        self.metrics = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "total_tokens_used": 0
        }
    
    async def initialize(self):
        """Initialize the agent"""
        try:
            # Initialize Gemini model
            self.model = genai.GenerativeModel(self.model_name)
            logger.info(f"Initialized {self.__class__.__name__} with model {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize agent", 
                        agent=self.__class__.__name__,
                        error=str(e))
            raise
    
    @abstractmethod
    async def execute(self, request: AgentRequest) -> Dict[str, Any]:
        """Execute the agent logic"""
        pass
    
    @abstractmethod
    async def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input data"""
        pass
    
    async def preprocess(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess input data"""
        return input_data
    
    async def postprocess(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Postprocess results"""
        return result
    
    async def generate_prompt(self, input_data: Dict[str, Any], template: str) -> str:
        """Generate prompt from template and input data"""
        try:
            # Simple template replacement
            prompt = template
            for key, value in input_data.items():
                placeholder = f"{{{key}}}"
                if placeholder in prompt:
                    prompt = prompt.replace(placeholder, str(value))
            return prompt
        except Exception as e:
            logger.error(f"Error generating prompt", error=str(e))
            raise
    
    async def call_gemini(self, prompt: str, **kwargs) -> str:
        """Call Gemini model"""
        try:
            response = self.model.generate_content(prompt, **kwargs)
            self.metrics["total_tokens_used"] += response.usage_metadata.total_token_count if hasattr(response, 'usage_metadata') else 0
            return response.text
        except Exception as e:
            logger.error(f"Error calling Gemini", error=str(e))
            raise
    
    async def log_execution(self, request_id: str, status: str, execution_time_ms: int):
        """Log execution details"""
        logger.info(
            "Agent execution completed",
            agent=self.__class__.__name__,
            request_id=request_id,
            status=status,
            execution_time_ms=execution_time_ms
        )
        
        if status == "success":
            self.metrics["successful_executions"] += 1
        else:
            self.metrics["failed_executions"] += 1
        self.metrics["total_executions"] += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get agent metrics"""
        return {
            **self.metrics,
            "success_rate": self.metrics["successful_executions"] / max(self.metrics["total_executions"], 1),
            "average_tokens_per_execution": self.metrics["total_tokens_used"] / max(self.metrics["total_executions"], 1)
        }
    
    async def cleanup(self):
        """Cleanup resources"""
        pass