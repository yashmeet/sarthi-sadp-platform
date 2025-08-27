"""
SADP Production API - Simplified Version
Working production API without complex dependencies
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
from datetime import datetime
import os
import json
import google.generativeai as genai

# Import POML router
from poml_simple_api import router as poml_router, init_sample_templates
import asyncio

# Initialize Gemini
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
else:
    gemini_model = None

app = FastAPI(
    title="SADP Production API",
    description="Sarthi AI Agent Development Platform - Production",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include POML router
app.include_router(poml_router)

# Initialize sample templates on startup
@app.on_event("startup")
async def startup_event():
    await init_sample_templates()

# Models
class AgentExecutionRequest(BaseModel):
    agent_type: str
    input_data: Dict[str, Any]

class TemplateTestRequest(BaseModel):
    template: str
    variables: Dict[str, Any]
    test_data: str

# Storage for demonstration
agents = []
templates = []
executions = []

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "service": "SADP Production API",
        "version": "1.0.0",
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "gemini_configured": gemini_model is not None
    }

@app.get("/health")
async def health():
    """Detailed health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "api": "healthy",
            "gemini": "configured" if gemini_model else "not_configured"
        }
    }

@app.post("/agents/execute")
async def execute_agent(request: AgentExecutionRequest):
    """Execute an AI agent with real Gemini processing"""
    if not gemini_model:
        raise HTTPException(status_code=503, detail="Gemini API not configured")
    
    try:
        # Map agent types to prompts
        prompts = {
            "clinical_document": """Analyze this clinical document and extract key medical information:
                                   {input}
                                   Extract: diagnoses, medications, procedures, and follow-up recommendations.""",
            "medication_analysis": """Analyze these medications for interactions and safety:
                                     {input}
                                     Check for: drug interactions, dosage appropriateness, and contraindications.""",
            "lab_results": """Interpret these lab results:
                            {input}
                            Provide: normal/abnormal classification, clinical significance, and recommendations.""",
            "health_assistant": """Answer this health question:
                                 {input}
                                 Provide helpful medical information while noting this doesn't replace professional advice."""
        }
        
        agent_type = request.agent_type
        if agent_type not in prompts:
            agent_type = "health_assistant"
        
        prompt = prompts[agent_type].format(input=json.dumps(request.input_data))
        
        # Execute with Gemini
        response = gemini_model.generate_content(prompt)
        
        execution = {
            "id": f"exec_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            "agent_type": request.agent_type,
            "status": "completed",
            "result": response.text,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        executions.append(execution)
        
        return execution
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/templates/test")
async def test_template(request: TemplateTestRequest):
    """Test a POML template with real Gemini"""
    if not gemini_model:
        raise HTTPException(status_code=503, detail="Gemini API not configured")
    
    try:
        # Render template with variables
        rendered = request.template
        for key, value in request.variables.items():
            rendered = rendered.replace(f"{{{{{key}}}}}", str(value))
        
        # Add test data
        full_prompt = f"{rendered}\n\nInput: {request.test_data}"
        
        # Execute with Gemini
        response = gemini_model.generate_content(full_prompt)
        
        return {
            "success": True,
            "response": response.text,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/agents/list")
async def list_agents():
    """List available agents"""
    return {
        "agents": [
            {
                "id": "clinical_document",
                "name": "Clinical Document Processor",
                "description": "Analyzes clinical documents and extracts medical information",
                "status": "active"
            },
            {
                "id": "medication_analysis",
                "name": "Medication Analyzer",
                "description": "Checks drug interactions and medication safety",
                "status": "active"
            },
            {
                "id": "lab_results",
                "name": "Lab Result Interpreter",
                "description": "Interprets laboratory test results",
                "status": "active"
            },
            {
                "id": "health_assistant",
                "name": "Health Assistant",
                "description": "Answers health-related questions",
                "status": "active"
            }
        ]
    }

@app.get("/monitoring/metrics")
async def get_metrics():
    """Get basic metrics"""
    return {
        "total_executions": len(executions),
        "successful_executions": len([e for e in executions if e.get("status") == "completed"]),
        "agents_available": 4,
        "gemini_status": "active" if gemini_model else "not_configured",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/executions/recent")
async def get_recent_executions():
    """Get recent executions"""
    return {
        "executions": executions[-10:],
        "total": len(executions)
    }

# Simple authentication endpoint for demo
@app.post("/auth/demo-login")
async def demo_login():
    """Demo login endpoint"""
    return {
        "access_token": "demo_token_" + datetime.utcnow().strftime('%Y%m%d%H%M%S'),
        "token_type": "bearer",
        "user": {
            "id": "demo_user",
            "email": "demo@sadp.ai",
            "organization": "Demo Organization"
        }
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)