"""
SADP Agent Runtime Service with Real AI Integration
FastAPI application with Gemini AI capabilities
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
from datetime import datetime
import json
import asyncio
import os

# Import our Gemini integration
from gemini_integration_simple import get_gemini_service

app = FastAPI(
    title="SADP Agent Runtime API with AI",
    version="2.0.0",
    description="Healthcare AI Agent Platform with Real Gemini Integration"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get Gemini service
gemini_service = get_gemini_service()

# Store for agents (in production, use Firestore)
agents_store = {}
audit_trail = []

class AgentRegistration(BaseModel):
    name: str
    type: str
    description: str
    capabilities: List[str]
    implementation_type: str
    poml_template: Optional[str] = None
    custom_code: Optional[str] = None
    api_endpoint: Optional[str] = None
    model_id: Optional[str] = None
    configuration: Dict[str, Any] = {}

class AgentExecutionRequest(BaseModel):
    input_data: str
    parameters: Dict[str, Any] = {}
    use_real_ai: bool = True
    custom_prompt: Optional[str] = None

class POMLTestRequest(BaseModel):
    agent_type: str
    input_data: str
    prompt_a: str
    prompt_b: str
    iterations: int = 5

@app.get("/")
async def root():
    """Health check and API info"""
    return {
        "service": "SADP Agent Runtime with AI",
        "version": "2.0.0",
        "status": "healthy",
        "ai_enabled": True,
        "model": "gemini-1.5-flash",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/agents/register")
async def register_agent(agent: AgentRegistration):
    """Register a new AI agent"""
    agent_id = f"agent_{len(agents_store) + 1}"
    
    agent_data = {
        "id": agent_id,
        "name": agent.name,
        "type": agent.type,
        "description": agent.description,
        "capabilities": agent.capabilities,
        "implementation_type": agent.implementation_type,
        "poml_template": agent.poml_template,
        "custom_code": agent.custom_code,
        "api_endpoint": agent.api_endpoint,
        "model_id": agent.model_id or "gemini-1.5-flash",
        "configuration": agent.configuration,
        "created_at": datetime.utcnow().isoformat(),
        "status": "active"
    }
    
    agents_store[agent_id] = agent_data
    
    # Log to audit trail
    audit_trail.append({
        "event": "agent_registered",
        "agent_id": agent_id,
        "agent_name": agent.name,
        "timestamp": datetime.utcnow().isoformat(),
        "user": "system"
    })
    
    return {"agent_id": agent_id, "status": "registered", "agent": agent_data}

@app.get("/agents/list")
async def list_agents(agent_type: Optional[str] = None):
    """List all registered agents"""
    if agent_type:
        filtered = {k: v for k, v in agents_store.items() if v["type"] == agent_type}
        return {"agents": list(filtered.values()), "count": len(filtered)}
    return {"agents": list(agents_store.values()), "count": len(agents_store)}

@app.get("/agents/{agent_id}")
async def get_agent(agent_id: str):
    """Get specific agent details"""
    if agent_id not in agents_store:
        raise HTTPException(status_code=404, detail="Agent not found")
    return agents_store[agent_id]

@app.post("/agents/{agent_id}/execute")
async def execute_agent(agent_id: str, request: AgentExecutionRequest):
    """Execute an agent with real AI processing"""
    
    if agent_id not in agents_store:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    agent = agents_store[agent_id]
    
    # Log execution start
    audit_trail.append({
        "event": "agent_execution_started",
        "agent_id": agent_id,
        "agent_name": agent["name"],
        "input_length": len(request.input_data),
        "use_real_ai": request.use_real_ai,
        "timestamp": datetime.utcnow().isoformat()
    })
    
    try:
        if request.use_real_ai:
            # Use real Gemini AI
            result = await gemini_service.process_with_agent(
                agent_type=agent["type"],
                input_data=request.input_data,
                poml_template=agent.get("poml_template"),
                custom_prompt=request.custom_prompt
            )
        else:
            # Use mock response for testing
            result = {
                "status": "success",
                "agent_id": agent_id,
                "agent_name": agent["name"],
                "result": f"Mock processing of {len(request.input_data)} characters",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        # Log successful execution
        audit_trail.append({
            "event": "agent_execution_completed",
            "agent_id": agent_id,
            "agent_name": agent["name"],
            "status": "success",
            "timestamp": datetime.utcnow().isoformat()
        })
        
        return result
        
    except Exception as e:
        # Log error
        audit_trail.append({
            "event": "agent_execution_failed",
            "agent_id": agent_id,
            "agent_name": agent["name"],
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        })
        
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/agents/{agent_id}/execute-file")
async def execute_agent_with_file(
    agent_id: str,
    file: UploadFile = File(...),
    use_real_ai: bool = True
):
    """Execute agent with file upload"""
    
    if agent_id not in agents_store:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    agent = agents_store[agent_id]
    
    # Read file content
    content = await file.read()
    
    # Convert to text (simplified - in production, handle different file types)
    try:
        text_content = content.decode('utf-8')
    except:
        text_content = f"Binary file: {file.filename}, Size: {len(content)} bytes"
    
    # Process with AI
    if use_real_ai:
        result = await gemini_service.process_with_agent(
            agent_type=agent["type"],
            input_data=text_content,
            poml_template=agent.get("poml_template")
        )
    else:
        result = {
            "status": "success",
            "file_name": file.filename,
            "file_size": len(content),
            "agent_id": agent_id,
            "result": "File processed (mock)"
        }
    
    return result

@app.post("/poml/test")
async def test_poml_prompts(request: POMLTestRequest):
    """Run A/B test on POML prompts"""
    
    result = await gemini_service.run_ab_test(
        agent_type=request.agent_type,
        input_data=request.input_data,
        prompt_a=request.prompt_a,
        prompt_b=request.prompt_b,
        num_iterations=request.iterations
    )
    
    # Log A/B test
    audit_trail.append({
        "event": "poml_ab_test",
        "agent_type": request.agent_type,
        "iterations": request.iterations,
        "winner": result["recommendation"],
        "timestamp": datetime.utcnow().isoformat()
    })
    
    return result

@app.get("/audit/trail")
async def get_audit_trail(limit: int = 100):
    """Get audit trail for compliance"""
    return {
        "events": audit_trail[-limit:],
        "total_events": len(audit_trail),
        "retrieved": min(limit, len(audit_trail))
    }

@app.post("/agents/bulk-register")
async def bulk_register_agents():
    """Register default healthcare agents"""
    
    default_agents = [
        {
            "name": "Clinical Assistant",
            "type": "clinical",
            "description": "AI assistant for clinical decision support",
            "capabilities": ["diagnosis_support", "treatment_recommendations", "drug_interactions"],
            "implementation_type": "built_in_model"
        },
        {
            "name": "Billing Processor",
            "type": "billing",
            "description": "Automated medical billing and coding",
            "capabilities": ["code_extraction", "claim_validation", "denial_management"],
            "implementation_type": "built_in_model"
        },
        {
            "name": "Document Extractor",
            "type": "document",
            "description": "Extract structured data from medical documents",
            "capabilities": ["ocr", "field_extraction", "document_classification"],
            "implementation_type": "built_in_model"
        },
        {
            "name": "Voice Transcriber",
            "type": "voice",
            "description": "Medical voice transcription and analysis",
            "capabilities": ["transcription", "medical_terminology", "summary_generation"],
            "implementation_type": "built_in_model"
        },
        {
            "name": "Referral Manager",
            "type": "referral",
            "description": "Process and route medical referrals",
            "capabilities": ["referral_extraction", "provider_matching", "urgency_detection"],
            "implementation_type": "built_in_model"
        },
        {
            "name": "Lab Result Analyzer",
            "type": "lab_result",
            "description": "Interpret and flag lab results",
            "capabilities": ["result_interpretation", "abnormal_detection", "trend_analysis"],
            "implementation_type": "built_in_model"
        }
    ]
    
    registered = []
    for agent_data in default_agents:
        agent = AgentRegistration(**agent_data)
        result = await register_agent(agent)
        registered.append(result)
    
    return {
        "message": "Default agents registered",
        "count": len(registered),
        "agents": registered
    }

@app.get("/ai/status")
async def get_ai_status():
    """Check AI service status"""
    
    # Test AI connectivity
    try:
        test_result = await gemini_service.process_with_agent(
            agent_type="clinical",
            input_data="Test connection",
            custom_prompt="Reply with 'AI service operational'"
        )
        
        ai_operational = test_result.get("status") == "success"
    except:
        ai_operational = False
    
    return {
        "ai_service": "Gemini",
        "model": "gemini-1.5-flash",
        "operational": ai_operational,
        "vertex_ai_enabled": gemini_service.vertex_initialized,
        "gemini_api_enabled": gemini_service.gemini_api_available,
        "project_id": gemini_service.project_id,
        "location": gemini_service.location,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/metrics")
async def get_metrics():
    """Get Prometheus-style metrics"""
    
    total_agents = len(agents_store)
    total_executions = len([e for e in audit_trail if "execution" in e["event"]])
    successful_executions = len([e for e in audit_trail if e["event"] == "agent_execution_completed"])
    failed_executions = len([e for e in audit_trail if e["event"] == "agent_execution_failed"])
    
    metrics = f"""# HELP sadp_agents_total Total number of registered agents
# TYPE sadp_agents_total gauge
sadp_agents_total {total_agents}

# HELP sadp_executions_total Total number of agent executions
# TYPE sadp_executions_total counter
sadp_executions_total {total_executions}

# HELP sadp_executions_successful Successful agent executions
# TYPE sadp_executions_successful counter
sadp_executions_successful {successful_executions}

# HELP sadp_executions_failed Failed agent executions
# TYPE sadp_executions_failed counter
sadp_executions_failed {failed_executions}

# HELP sadp_ai_service_up AI service availability
# TYPE sadp_ai_service_up gauge
sadp_ai_service_up {1 if (gemini_service.vertex_initialized or gemini_service.gemini_api_available) else 0}
"""
    
    return metrics

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)