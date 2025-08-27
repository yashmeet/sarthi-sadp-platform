from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any
from datetime import datetime
import os
import json

app = FastAPI(
    title="SADP Agent Runtime Service",
    description="AI Agent Runtime for Healthcare Applications",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration from environment
PROJECT_ID = os.getenv("GCP_PROJECT_ID", "sarthi-patient-experience-hub")
REGION = os.getenv("GCP_REGION", "us-central1")
ENVIRONMENT = os.getenv("ENVIRONMENT", "demo")

# Supported agents
SUPPORTED_AGENTS = [
    "document_processor",
    "clinical", 
    "billing",
    "voice",
    "health_assistant",
    "medication_entry",
    "referral_processor",
    "lab_result_entry"
]

@app.get("/")
async def root():
    return {
        "service": "SADP Agent Runtime",
        "version": "1.0.0",
        "status": "running",
        "environment": ENVIRONMENT,
        "project": PROJECT_ID
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "agent-runtime", 
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat(),
        "environment": ENVIRONMENT,
        "uptime": "running"
    }

@app.get("/agents/supported")
async def get_supported_agents():
    """Get list of supported AI agents"""
    return {
        "agents": SUPPORTED_AGENTS,
        "count": len(SUPPORTED_AGENTS),
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

@app.post("/agents/{agent_type}/execute")
async def execute_agent(agent_type: str, request: dict):
    """Execute a specific AI agent (demo mode)"""
    if agent_type not in SUPPORTED_AGENTS:
        raise HTTPException(status_code=400, detail=f"Unsupported agent type: {agent_type}")
    
    # Mock execution for demo
    return {
        "request_id": f"req_{agent_type}_{datetime.now().timestamp()}",
        "agent_type": agent_type,
        "status": "completed",
        "results": {
            "processed": True,
            "confidence": 0.95,
            "processing_time": 1.23,
            "demo_mode": True,
            "message": f"Successfully processed request using {agent_type} agent"
        },
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/agents/marketplace/search")
async def marketplace_search(q: str = "", category: str = ""):
    """Search agent marketplace"""
    agents = []
    for i, agent_type in enumerate(SUPPORTED_AGENTS):
        if not q or q.lower() in agent_type.lower():
            agents.append({
                "id": agent_type,
                "name": agent_type.replace("_", " ").title(),
                "version": "1.0.0",
                "category": "healthcare",
                "author": "SADP Team",
                "rating": 4.5 + (i * 0.1),
                "downloads": 1000 + (i * 100),
                "description": f"Advanced {agent_type.replace('_', ' ')} capabilities",
                "auto_load": True
            })
    
    return {
        "agents": agents,
        "total": len(agents),
        "query": q,
        "category": category
    }

@app.get("/agents/marketplace/categories")
async def marketplace_categories():
    """Get marketplace categories"""
    return {
        "categories": [
            {"id": "healthcare", "name": "Healthcare", "count": len(SUPPORTED_AGENTS)},
            {"id": "document", "name": "Document Processing", "count": 3},
            {"id": "clinical", "name": "Clinical", "count": 2},
            {"id": "administrative", "name": "Administrative", "count": 3}
        ]
    }

@app.get("/agents/marketplace/featured")
async def marketplace_featured():
    """Get featured agents"""
    return {
        "featured": [
            {
                "id": "document_processor",
                "name": "Document Processor",
                "description": "Advanced OCR and form extraction",
                "rating": 4.9,
                "downloads": 5000,
                "featured_reason": "Most popular"
            },
            {
                "id": "clinical",
                "name": "Clinical Assistant", 
                "description": "AI-powered clinical analysis",
                "rating": 4.8,
                "downloads": 3500,
                "featured_reason": "Editor's choice"
            }
        ]
    }

@app.post("/agents/marketplace/load")
async def marketplace_load(request: dict):
    """Load agent from marketplace"""
    agent_id = request.get("agent_id", "")
    version = request.get("version", "1.0.0")
    
    if agent_id not in SUPPORTED_AGENTS:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
    
    return {
        "status": "loaded",
        "agent_id": agent_id,
        "version": version,
        "loaded_at": datetime.utcnow().isoformat()
    }

@app.get("/poml/templates")
async def get_poml_templates():
    """Get POML templates"""
    return {
        "templates": [
            {
                "id": "clinical_analysis",
                "name": "Clinical Analysis Template",
                "version": "1.0.0",
                "description": "Template for clinical data analysis"
            },
            {
                "id": "document_extraction", 
                "name": "Document Extraction Template",
                "version": "1.0.0",
                "description": "Template for document processing"
            }
        ]
    }

@app.get("/poml/versions")
async def get_poml_versions():
    """Get POML template versions"""
    return {
        "versions": [
            {"template_id": "clinical_analysis", "version": "1.0.0", "active": True},
            {"template_id": "document_extraction", "version": "1.0.0", "active": True}
        ]
    }

@app.post("/poml/ab-tests")
async def create_ab_test(request: dict):
    """Create A/B test"""
    test_id = f"test_{datetime.now().timestamp()}"
    return {
        "test_id": test_id,
        "name": request.get("name", "Untitled Test"),
        "status": "created",
        "created_at": datetime.utcnow().isoformat()
    }

@app.get("/metrics")
async def get_metrics():
    """Get service metrics in Prometheus format"""
    metrics = f"""# HELP sadp_requests_total Total number of requests
# TYPE sadp_requests_total counter
sadp_requests_total{{service="agent-runtime"}} 100

# HELP sadp_request_duration_seconds Request duration in seconds
# TYPE sadp_request_duration_seconds histogram
sadp_request_duration_seconds_bucket{{service="agent-runtime",le="0.1"}} 50
sadp_request_duration_seconds_bucket{{service="agent-runtime",le="0.5"}} 80
sadp_request_duration_seconds_bucket{{service="agent-runtime",le="1.0"}} 95
sadp_request_duration_seconds_bucket{{service="agent-runtime",le="+Inf"}} 100
sadp_request_duration_seconds_sum{{service="agent-runtime"}} 25.5
sadp_request_duration_seconds_count{{service="agent-runtime"}} 100

# HELP sadp_agents_available Number of available agents
# TYPE sadp_agents_available gauge
sadp_agents_available{{service="agent-runtime"}} {len(SUPPORTED_AGENTS)}
"""
    
    from fastapi import Response
    return Response(content=metrics, media_type="text/plain")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main_simple:app", host="0.0.0.0", port=port, reload=False)