from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum

class AgentStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class AgentType(str, Enum):
    DOCUMENT_PROCESSOR = "document_processor"
    CLINICAL = "clinical"
    BILLING = "billing"
    VOICE = "voice"
    HEALTH_ASSISTANT = "health_assistant"
    MEDICATION_ENTRY = "medication_entry"
    REFERRAL_PROCESSOR = "referral_processor"
    LAB_RESULT_ENTRY = "lab_result_entry"

class AgentRequest(BaseModel):
    request_id: str = Field(..., description="Unique request identifier")
    input_data: Dict[str, Any] = Field(..., description="Input data for the agent")
    context: Optional[Dict[str, Any]] = Field(default={}, description="Additional context")
    parameters: Optional[Dict[str, Any]] = Field(default={}, description="Agent-specific parameters")
    user_id: Optional[str] = Field(None, description="User identifier")
    session_id: Optional[str] = Field(None, description="Session identifier")
    
    class Config:
        json_schema_extra = {
            "example": {
                "request_id": "req_123456",
                "input_data": {
                    "document_url": "gs://bucket/document.pdf",
                    "document_type": "lab_report"
                },
                "context": {
                    "patient_id": "PAT001",
                    "provider_id": "DOC001"
                },
                "parameters": {
                    "confidence_threshold": 0.85,
                    "language": "en"
                }
            }
        }

class AgentResponse(BaseModel):
    request_id: str
    agent_type: str
    status: AgentStatus
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    execution_time_ms: Optional[int] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_schema_extra = {
            "example": {
                "request_id": "req_123456",
                "agent_type": "document_processor",
                "status": "completed",
                "result": {
                    "extracted_text": "Patient Name: John Doe...",
                    "entities": ["patient_name", "date", "diagnosis"],
                    "confidence": 0.92
                },
                "metadata": {
                    "model_version": "1.0.0",
                    "processing_steps": ["ocr", "entity_extraction", "validation"]
                },
                "execution_time_ms": 2345,
                "timestamp": "2024-01-20T10:30:00Z"
            }
        }

class WorkflowStep(BaseModel):
    step_id: str
    agent_type: AgentType
    input_mapping: Dict[str, str] = Field(default_factory=dict)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    depends_on: List[str] = Field(default_factory=list)
    retry_config: Optional[Dict[str, Any]] = None

class WorkflowRequest(BaseModel):
    workflow_id: str = Field(..., description="Unique workflow identifier")
    name: str = Field(..., description="Workflow name")
    steps: List[WorkflowStep] = Field(..., description="Workflow steps")
    initial_input: Dict[str, Any] = Field(..., description="Initial input data")
    context: Optional[Dict[str, Any]] = Field(default={}, description="Workflow context")
    
    class Config:
        json_schema_extra = {
            "example": {
                "workflow_id": "wf_123456",
                "name": "Document Processing Pipeline",
                "steps": [
                    {
                        "step_id": "step1",
                        "agent_type": "document_processor",
                        "parameters": {"ocr_enabled": True},
                        "depends_on": []
                    },
                    {
                        "step_id": "step2",
                        "agent_type": "clinical",
                        "input_mapping": {"text": "step1.result.extracted_text"},
                        "depends_on": ["step1"]
                    }
                ],
                "initial_input": {
                    "document_url": "gs://bucket/document.pdf"
                }
            }
        }

class ExecutionMetrics(BaseModel):
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_execution_time_ms: float
    active_requests: int
    agent_metrics: Dict[str, Dict[str, Any]]
    timestamp: datetime = Field(default_factory=datetime.utcnow)