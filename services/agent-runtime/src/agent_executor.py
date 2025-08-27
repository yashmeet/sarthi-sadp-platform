import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime
import structlog
import json
from google.cloud import aiplatform
import google.generativeai as genai

from models import AgentRequest, AgentResponse, AgentStatus, WorkflowRequest
from config import Settings
from agents import (
    DocumentProcessorAgent,
    ClinicalAgent,
    BillingAgent,
    VoiceAgent,
    HealthAssistantAgent,
    MedicationEntryAgent,
    ReferralProcessorAgent,
    LabResultEntryAgent
)

logger = structlog.get_logger()

class AgentExecutor:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.agents = {}
        self.active_requests = {}
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "execution_times": []
        }
        
    async def initialize(self):
        """Initialize AI services and agent instances"""
        try:
            # Initialize Vertex AI
            aiplatform.init(
                project=self.settings.PROJECT_ID,
                location=self.settings.VERTEX_AI_LOCATION
            )
            
            # Initialize Gemini
            api_key = await self._get_gemini_api_key()
            if api_key:
                genai.configure(api_key=api_key)
                logger.info("Gemini API configured successfully")
            else:
                logger.warning("Gemini API key not available - AI features will be limited")
            
            # Initialize agent instances
            self.agents = {
                "document_processor": DocumentProcessorAgent(self.settings),
                "clinical": ClinicalAgent(self.settings),
                "billing": BillingAgent(self.settings),
                "voice": VoiceAgent(self.settings),
                "health_assistant": HealthAssistantAgent(self.settings),
                "medication_entry": MedicationEntryAgent(self.settings),
                "referral_processor": ReferralProcessorAgent(self.settings),
                "lab_result_entry": LabResultEntryAgent(self.settings)
            }
            
            # Initialize each agent
            for agent_name, agent in self.agents.items():
                await agent.initialize()
                logger.info(f"Initialized agent: {agent_name}")
                
        except Exception as e:
            logger.error(f"Failed to initialize AgentExecutor", error=str(e))
            raise
    
    async def execute_agent(
        self,
        agent_type: str,
        request: AgentRequest
    ) -> AgentResponse:
        """Execute a specific AI agent"""
        start_time = datetime.utcnow()
        self.metrics["total_requests"] += 1
        
        # Track active request
        self.active_requests[request.request_id] = {
            "agent_type": agent_type,
            "status": AgentStatus.RUNNING,
            "start_time": start_time
        }
        
        try:
            # Get the appropriate agent
            agent = self.agents.get(agent_type)
            if not agent:
                raise ValueError(f"Unknown agent type: {agent_type}")
            
            # Execute the agent
            logger.info(f"Executing {agent_type} agent", request_id=request.request_id)
            result = await agent.execute(request)
            
            # Calculate execution time
            execution_time_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            self.metrics["execution_times"].append(execution_time_ms)
            self.metrics["successful_requests"] += 1
            
            # Create response
            response = AgentResponse(
                request_id=request.request_id,
                agent_type=agent_type,
                status=AgentStatus.COMPLETED,
                result=result,
                execution_time_ms=execution_time_ms,
                metadata={
                    "agent_version": agent.version,
                    "model_used": agent.model_name
                }
            )
            
            # Update active request status
            self.active_requests[request.request_id]["status"] = AgentStatus.COMPLETED
            
            return response
            
        except Exception as e:
            logger.error(f"Agent execution failed", 
                        agent_type=agent_type,
                        request_id=request.request_id,
                        error=str(e))
            
            self.metrics["failed_requests"] += 1
            self.active_requests[request.request_id]["status"] = AgentStatus.FAILED
            
            return AgentResponse(
                request_id=request.request_id,
                agent_type=agent_type,
                status=AgentStatus.FAILED,
                error=str(e),
                execution_time_ms=int((datetime.utcnow() - start_time).total_seconds() * 1000)
            )
        finally:
            # Clean up completed request after a delay
            asyncio.create_task(self._cleanup_request(request.request_id))
    
    async def execute_workflow(self, request: WorkflowRequest) -> List[Dict[str, Any]]:
        """Execute a workflow of multiple agents"""
        results = {}
        step_status = {}
        
        try:
            # Initialize workflow context
            context = {
                "workflow_id": request.workflow_id,
                "initial_input": request.initial_input,
                **request.context
            }
            
            # Sort steps by dependencies
            sorted_steps = self._topological_sort(request.steps)
            
            # Execute steps in order
            for step in sorted_steps:
                logger.info(f"Executing workflow step", 
                           workflow_id=request.workflow_id,
                           step_id=step.step_id)
                
                # Prepare input for this step
                step_input = self._prepare_step_input(
                    step,
                    request.initial_input,
                    results
                )
                
                # Create agent request
                agent_request = AgentRequest(
                    request_id=f"{request.workflow_id}_{step.step_id}",
                    input_data=step_input,
                    context=context,
                    parameters=step.parameters
                )
                
                # Execute agent
                response = await self.execute_agent(
                    agent_type=step.agent_type.value,
                    request=agent_request
                )
                
                # Store result
                results[step.step_id] = {
                    "status": response.status,
                    "result": response.result,
                    "execution_time_ms": response.execution_time_ms
                }
                
                # Check if step failed
                if response.status == AgentStatus.FAILED:
                    raise Exception(f"Step {step.step_id} failed: {response.error}")
            
            return list(results.values())
            
        except Exception as e:
            logger.error(f"Workflow execution failed",
                        workflow_id=request.workflow_id,
                        error=str(e))
            raise
    
    async def get_status(self, agent_type: str, request_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of an agent execution"""
        if request_id in self.active_requests:
            return self.active_requests[request_id]
        return None
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get execution metrics"""
        avg_execution_time = 0
        if self.metrics["execution_times"]:
            avg_execution_time = sum(self.metrics["execution_times"]) / len(self.metrics["execution_times"])
        
        return {
            "total_requests": self.metrics["total_requests"],
            "successful_requests": self.metrics["successful_requests"],
            "failed_requests": self.metrics["failed_requests"],
            "average_execution_time_ms": avg_execution_time,
            "active_requests": len(self.active_requests),
            "agent_metrics": self._get_agent_metrics()
        }
    
    async def cleanup(self):
        """Cleanup resources"""
        for agent in self.agents.values():
            await agent.cleanup()
    
    async def _get_gemini_api_key(self) -> Optional[str]:
        """Get Gemini API key from environment variable or Secret Manager"""
        # First try environment variable
        if self.settings.GEMINI_API_KEY:
            return self.settings.GEMINI_API_KEY
            
        # Then try Secret Manager
        try:
            from google.cloud import secretmanager
            client = secretmanager.SecretManagerServiceClient()
            name = f"projects/{self.settings.PROJECT_ID}/secrets/{self.settings.GEMINI_API_KEY_SECRET}/versions/latest"
            response = client.access_secret_version(request={"name": name})
            api_key = response.payload.data.decode("UTF-8")
            if api_key and api_key.strip():
                return api_key.strip()
        except Exception as e:
            logger.warning("Failed to get Gemini API key from Secret Manager", error=str(e))
            
        return None
    
    async def _cleanup_request(self, request_id: str, delay: int = 300):
        """Clean up completed request after delay"""
        await asyncio.sleep(delay)
        if request_id in self.active_requests:
            del self.active_requests[request_id]
    
    def _topological_sort(self, steps) -> List:
        """Sort workflow steps based on dependencies"""
        # Simple topological sort implementation
        sorted_steps = []
        visited = set()
        
        def visit(step):
            if step.step_id in visited:
                return
            visited.add(step.step_id)
            
            for dep_id in step.depends_on:
                dep_step = next((s for s in steps if s.step_id == dep_id), None)
                if dep_step:
                    visit(dep_step)
            
            sorted_steps.append(step)
        
        for step in steps:
            visit(step)
        
        return sorted_steps
    
    def _prepare_step_input(self, step, initial_input, results) -> Dict[str, Any]:
        """Prepare input for a workflow step"""
        step_input = {}
        
        # Apply input mapping
        for target_key, source_path in step.input_mapping.items():
            value = self._get_nested_value(source_path, initial_input, results)
            if value is not None:
                step_input[target_key] = value
        
        # If no mapping, use initial input
        if not step_input:
            step_input = initial_input
        
        return step_input
    
    def _get_nested_value(self, path: str, initial_input: Dict, results: Dict):
        """Get nested value from path like 'step1.result.text'"""
        parts = path.split(".")
        
        if parts[0] in results:
            value = results[parts[0]]
            for part in parts[1:]:
                if isinstance(value, dict):
                    value = value.get(part)
                else:
                    return None
            return value
        
        # Check in initial input
        value = initial_input
        for part in parts:
            if isinstance(value, dict):
                value = value.get(part)
            else:
                return None
        return value
    
    def _get_agent_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics for each agent"""
        metrics = {}
        for agent_name, agent in self.agents.items():
            metrics[agent_name] = agent.get_metrics()
        return metrics