"""
Execution Record Models for Agent and POML executions
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum

class ExecutionStatus(Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"

class AIProvider(Enum):
    GEMINI = "gemini"
    VERTEX_AI = "vertex_ai"
    OPENAI = "openai"

@dataclass
class ExecutionRecord:
    """Base execution record"""
    execution_id: str
    request_id: str
    user_id: str
    tenant_id: str
    status: ExecutionStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[int] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class AgentExecution(ExecutionRecord):
    """Agent-specific execution record"""
    agent_type: str
    agent_version: str
    input_data: Dict[str, Any]
    output_data: Dict[str, Any] = field(default_factory=dict)
    confidence_score: Optional[float] = None
    tokens_used: Optional[int] = None
    ai_provider: Optional[AIProvider] = None
    model_used: Optional[str] = None
    prompt_template_id: Optional[str] = None
    prompt_version: Optional[str] = None
    
    # PHI tracking
    phi_detected: bool = False
    phi_removed_fields: List[str] = field(default_factory=list)
    
    # Performance metrics
    inference_time_ms: Optional[int] = None
    preprocessing_time_ms: Optional[int] = None
    postprocessing_time_ms: Optional[int] = None

@dataclass
class POMLExecution(ExecutionRecord):
    """POML template execution record"""
    template_id: str
    template_version: str
    template_name: str
    agent_type: str
    medical_domain: str
    variables_used: Dict[str, Any] = field(default_factory=dict)
    compiled_prompt: str = ""
    execution_result: Dict[str, Any] = field(default_factory=dict)
    optimization_strategy: Optional[str] = None
    
    # AI provider details
    ai_provider: Optional[AIProvider] = None
    model_used: Optional[str] = None
    tokens_used: Optional[int] = None
    
    # Performance tracking
    compilation_time_ms: Optional[int] = None
    execution_time_ms: Optional[int] = None
    total_time_ms: Optional[int] = None
    
    # Quality metrics
    confidence_score: Optional[float] = None
    accuracy_score: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Firestore"""
        return {
            'execution_id': self.execution_id,
            'request_id': self.request_id,
            'user_id': self.user_id,
            'tenant_id': self.tenant_id,
            'status': self.status.value,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration_ms': self.duration_ms,
            'error_message': self.error_message,
            'metadata': self.metadata,
            'template_id': self.template_id,
            'template_version': self.template_version,
            'template_name': self.template_name,
            'agent_type': self.agent_type,
            'medical_domain': self.medical_domain,
            'variables_used': self.variables_used,
            'compiled_prompt': self.compiled_prompt,
            'execution_result': self.execution_result,
            'optimization_strategy': self.optimization_strategy,
            'ai_provider': self.ai_provider.value if self.ai_provider else None,
            'model_used': self.model_used,
            'tokens_used': self.tokens_used,
            'compilation_time_ms': self.compilation_time_ms,
            'execution_time_ms': self.execution_time_ms,
            'total_time_ms': self.total_time_ms,
            'confidence_score': self.confidence_score,
            'accuracy_score': self.accuracy_score,
            'created_at': self.created_at,
            'updated_at': self.updated_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'POMLExecution':
        """Create from dictionary"""
        # Convert status enum
        if isinstance(data.get('status'), str):
            data['status'] = ExecutionStatus(data['status'])
        
        # Convert AI provider enum
        if data.get('ai_provider'):
            data['ai_provider'] = AIProvider(data['ai_provider'])
        
        return cls(**data)