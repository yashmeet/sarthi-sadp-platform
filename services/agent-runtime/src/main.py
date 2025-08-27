from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional, List
import structlog
import os
from datetime import datetime
import json

from models import AgentRequest, AgentResponse, AgentStatus, WorkflowRequest
from agent_executor import AgentExecutor
from pubsub_client import PubSubClient
from config import Settings
from middleware import setup_monitoring

logger = structlog.get_logger()
settings = Settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting Agent Runtime Service", 
                version=settings.VERSION,
                environment=settings.ENVIRONMENT)
    
    # Initialize services
    app.state.executor = AgentExecutor(settings)
    app.state.pubsub = PubSubClient(settings)
    await app.state.executor.initialize()
    await app.state.pubsub.connect()
    
    yield
    
    # Shutdown
    logger.info("Shutting down Agent Runtime Service")
    await app.state.executor.cleanup()
    await app.state.pubsub.disconnect()

app = FastAPI(
    title="Agent Runtime Service",
    description="Executes AI agents and workflows for SADP",
    version=settings.VERSION,
    lifespan=lifespan
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
setup_monitoring(app)

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "agent-runtime",
        "version": settings.VERSION,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/agents/{agent_type}/execute", response_model=AgentResponse)
async def execute_agent(
    agent_type: str,
    request: AgentRequest,
    background_tasks: BackgroundTasks
):
    """Execute a specific AI agent"""
    try:
        logger.info(f"Executing agent", agent_type=agent_type, request_id=request.request_id)
        
        # Validate agent type
        if agent_type not in settings.SUPPORTED_AGENTS:
            raise HTTPException(status_code=400, detail=f"Unsupported agent type: {agent_type}")
        
        # Execute agent
        result = await app.state.executor.execute_agent(
            agent_type=agent_type,
            request=request
        )
        
        # Publish result to Pub/Sub for monitoring
        background_tasks.add_task(
            app.state.pubsub.publish_message,
            topic="agent-runtime-topic",
            message={
                "event": "agent_executed",
                "agent_type": agent_type,
                "request_id": request.request_id,
                "status": result.status,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error executing agent", 
                    agent_type=agent_type, 
                    error=str(e),
                    request_id=request.request_id)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/workflows/execute")
async def execute_workflow(
    request: WorkflowRequest,
    background_tasks: BackgroundTasks
):
    """Execute a workflow of multiple agents"""
    try:
        logger.info(f"Executing workflow", workflow_id=request.workflow_id)
        
        # Execute workflow
        results = await app.state.executor.execute_workflow(request)
        
        # Publish workflow completion
        background_tasks.add_task(
            app.state.pubsub.publish_message,
            topic="agent-runtime-topic",
            message={
                "event": "workflow_completed",
                "workflow_id": request.workflow_id,
                "steps_completed": len(results),
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
        return {
            "workflow_id": request.workflow_id,
            "status": "completed",
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Error executing workflow", 
                    workflow_id=request.workflow_id,
                    error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/agents/{agent_type}/status/{request_id}")
async def get_agent_status(agent_type: str, request_id: str):
    """Get the status of an agent execution"""
    try:
        status = await app.state.executor.get_status(agent_type, request_id)
        if not status:
            raise HTTPException(status_code=404, detail="Request not found")
        return status
    except Exception as e:
        logger.error(f"Error getting status", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/agents/supported")
async def get_supported_agents():
    """Get list of supported AI agents"""
    return {
        "agents": settings.SUPPORTED_AGENTS,
        "descriptions": {
            "document_processor": "OCR, handwriting recognition, form extraction",
            "clinical": "Treatment plan generation, clinical note synthesis",
            "billing": "Claim generation, prior authorization, denial management",
            "voice": "Appointment scheduling, medication reminders, symptom triage",
            "health_assistant": "24/7 patient portal support and Q&A",
            "medication_entry": "Medication reconciliation via photo analysis",
            "referral_processor": "Referral intake and urgency assessment",
            "lab_result_entry": "Digitizing and analyzing paper lab results"
        }
    }

@app.get("/metrics")
async def get_metrics():
    """Get service metrics"""
    return await app.state.executor.get_metrics()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
    )