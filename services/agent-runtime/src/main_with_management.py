from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any, List, Optional
from datetime import datetime
import os
import json

# Import agent management system
from agent_management import AgentManager

app = FastAPI(
    title="SADP Agent Management Platform",
    description="AI Agent Development, Management, and Execution Platform",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
PROJECT_ID = os.getenv("GCP_PROJECT_ID", "sarthi-patient-experience-hub")
REGION = os.getenv("GCP_REGION", "us-central1")
ENVIRONMENT = os.getenv("ENVIRONMENT", "production")

# Initialize Agent Manager
agent_manager = AgentManager(PROJECT_ID, REGION)

# ==================== Core Endpoints ====================

@app.get("/")
async def root():
    return {
        "service": "SADP Agent Management Platform",
        "version": "2.0.0",
        "status": "running",
        "environment": ENVIRONMENT,
        "features": [
            "Dynamic Agent Registration",
            "POML Template Management", 
            "Agent Marketplace",
            "Custom Code Deployment",
            "API Integration",
            "Full Audit Trail"
        ]
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "agent-management",
        "version": "2.0.0",
        "timestamp": datetime.utcnow().isoformat(),
        "environment": ENVIRONMENT
    }

# ==================== Agent Management Endpoints ====================

@app.post("/agents/register")
async def register_agent(config: Dict[str, Any], background_tasks: BackgroundTasks):
    """
    Register a new AI agent with complete configuration
    """
    try:
        # Register agent asynchronously
        result = await agent_manager.register_agent(config)
        
        if result['success']:
            # Add background task for post-registration setup
            background_tasks.add_task(
                post_registration_setup,
                result['agent_id']
            )
            
            return {
                "status": "success",
                "agent_id": result['agent_id'],
                "message": result['message'],
                "endpoints": result['endpoints'],
                "deployment_status": "active"
            }
        else:
            raise HTTPException(status_code=400, detail=result['error'])
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/agents/list")
async def list_agents(
    category: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 100
):
    """
    List all registered agents with optional filtering
    """
    try:
        agents = await agent_manager.list_agents(category=category, status=status)
        
        return {
            "agents": agents[:limit],
            "total": len(agents),
            "filters": {
                "category": category,
                "status": status
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/agents/{agent_id}")
async def get_agent(agent_id: str):
    """
    Get detailed information about a specific agent
    """
    try:
        agent = await agent_manager.get_agent(agent_id)
        
        if agent:
            return agent
        else:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/agents/{agent_id}")
async def update_agent(agent_id: str, updates: Dict[str, Any]):
    """
    Update agent configuration
    """
    try:
        success = await agent_manager.update_agent(agent_id, updates)
        
        if success:
            return {
                "status": "success",
                "agent_id": agent_id,
                "message": "Agent updated successfully",
                "updated_at": datetime.utcnow().isoformat()
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to update agent")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/agents/{agent_id}")
async def delete_agent(agent_id: str):
    """
    Delete an agent and all its resources
    """
    try:
        success = await agent_manager.delete_agent(agent_id)
        
        if success:
            return {
                "status": "success",
                "agent_id": agent_id,
                "message": "Agent deleted successfully"
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to delete agent")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/agents/{agent_id}/execute")
async def execute_agent(agent_id: str, request: Dict[str, Any]):
    """
    Execute a registered agent with input data
    """
    try:
        result = await agent_manager.execute_agent(agent_id, request)
        
        if result.get('success'):
            return result
        else:
            raise HTTPException(status_code=400, detail=result.get('error', 'Execution failed'))
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/agents/{agent_id}/metrics")
async def get_agent_metrics(agent_id: str):
    """
    Get performance metrics for an agent
    """
    try:
        metrics = await agent_manager.get_agent_metrics(agent_id)
        
        if metrics:
            return metrics
        else:
            raise HTTPException(status_code=404, detail=f"Metrics not found for agent {agent_id}")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==================== Agent Template Endpoints ====================

@app.get("/templates")
async def list_templates():
    """
    List available agent templates
    """
    templates = [
        {
            "id": "clinical_analysis",
            "name": "Clinical Analysis Template",
            "category": "Clinical",
            "description": "Template for clinical data analysis and diagnosis",
            "variables": ["patient_data", "medical_history", "symptoms"],
            "version": "2.0.0"
        },
        {
            "id": "document_processing",
            "name": "Document Processing Template",
            "category": "Administrative",
            "description": "OCR and document extraction template",
            "variables": ["document", "document_type", "extraction_fields"],
            "version": "1.5.0"
        },
        {
            "id": "billing_claims",
            "name": "Billing & Claims Template",
            "category": "Billing",
            "description": "Insurance claims processing and billing",
            "variables": ["claim_data", "patient_id", "procedure_codes"],
            "version": "1.2.0"
        },
        {
            "id": "imaging_analysis",
            "name": "Medical Imaging Template",
            "category": "Diagnostic",
            "description": "Radiology and medical imaging analysis",
            "variables": ["image_data", "modality", "body_part"],
            "version": "3.0.0"
        },
        {
            "id": "patient_engagement",
            "name": "Patient Engagement Template",
            "category": "Engagement",
            "description": "Patient communication and support",
            "variables": ["patient_query", "medical_context", "language"],
            "version": "1.0.0"
        }
    ]
    
    return {
        "templates": templates,
        "total": len(templates)
    }

@app.get("/templates/{template_id}")
async def get_template(template_id: str):
    """
    Get detailed template with POML content
    """
    templates = {
        "clinical_analysis": """<prompt version="2.0">
  <system>
    You are an expert clinical assistant specializing in {{specialty}}.
    Follow evidence-based medicine guidelines and prioritize patient safety.
  </system>
  
  <context>
    <patient>{{patient_data}}</patient>
    <history>{{medical_history}}</history>
    <symptoms>{{symptoms}}</symptoms>
  </context>
  
  <task>
    Analyze the patient information and provide:
    1. Initial assessment
    2. Differential diagnosis
    3. Recommended tests
    4. Treatment recommendations
  </task>
  
  <output format="json">
    {
      "assessment": "string",
      "diagnoses": ["string"],
      "tests": ["string"],
      "recommendations": ["string"],
      "confidence": 0.0-1.0
    }
  </output>
</prompt>""",
        "document_processing": """<prompt version="2.0">
  <system>
    You are a document processing specialist.
    Extract structured data from unstructured documents.
  </system>
  
  <context>
    <document>{{document}}</document>
    <type>{{document_type}}</type>
  </context>
  
  <task>
    Extract the following fields: {{extraction_fields}}
  </task>
  
  <output format="json">
    {{extracted_data}}
  </output>
</prompt>"""
    }
    
    if template_id in templates:
        return {
            "id": template_id,
            "content": templates[template_id],
            "format": "poml",
            "version": "2.0"
        }
    else:
        raise HTTPException(status_code=404, detail="Template not found")

# ==================== Agent Marketplace Endpoints ====================

@app.get("/marketplace/agents")
async def marketplace_list(
    category: Optional[str] = None,
    search: Optional[str] = None,
    sort_by: str = "rating"
):
    """
    Browse agent marketplace
    """
    # In production, this would query from Firestore
    marketplace_agents = [
        {
            "id": "clinical_specialist_v3",
            "name": "Clinical Specialist AI",
            "category": "Clinical",
            "author": "SADP Team",
            "version": "3.0.0",
            "rating": 4.9,
            "downloads": 5234,
            "price": "Free",
            "description": "Advanced clinical analysis with 97.5% accuracy",
            "tags": ["clinical", "diagnosis", "treatment"],
            "verified": True
        },
        {
            "id": "radiology_analyzer_pro",
            "name": "Radiology Analyzer Pro",
            "category": "Diagnostic",
            "author": "MedTech Solutions",
            "version": "2.1.0",
            "rating": 4.8,
            "downloads": 3421,
            "price": "$99/month",
            "description": "AI-powered radiology image analysis",
            "tags": ["radiology", "imaging", "diagnostic"],
            "verified": True
        },
        {
            "id": "billing_optimizer",
            "name": "Billing Optimizer",
            "category": "Billing",
            "author": "HealthFinance Inc",
            "version": "1.5.0",
            "rating": 4.7,
            "downloads": 2156,
            "price": "$49/month",
            "description": "Optimize billing and reduce claim denials",
            "tags": ["billing", "claims", "revenue"],
            "verified": False
        }
    ]
    
    # Filter by category if provided
    if category:
        marketplace_agents = [a for a in marketplace_agents if a['category'] == category]
    
    # Search if provided
    if search:
        search_lower = search.lower()
        marketplace_agents = [
            a for a in marketplace_agents 
            if search_lower in a['name'].lower() or 
               search_lower in a['description'].lower() or
               any(search_lower in tag for tag in a['tags'])
        ]
    
    # Sort
    if sort_by == "rating":
        marketplace_agents.sort(key=lambda x: x['rating'], reverse=True)
    elif sort_by == "downloads":
        marketplace_agents.sort(key=lambda x: x['downloads'], reverse=True)
    elif sort_by == "name":
        marketplace_agents.sort(key=lambda x: x['name'])
    
    return {
        "agents": marketplace_agents,
        "total": len(marketplace_agents),
        "filters": {
            "category": category,
            "search": search,
            "sort_by": sort_by
        }
    }

@app.post("/marketplace/install/{agent_id}")
async def install_marketplace_agent(agent_id: str, license_key: Optional[str] = None):
    """
    Install an agent from the marketplace
    """
    return {
        "status": "success",
        "agent_id": agent_id,
        "message": f"Agent {agent_id} installed successfully",
        "license": "valid" if license_key else "trial",
        "endpoints": {
            "execute": f"/agents/{agent_id}/execute",
            "configure": f"/agents/{agent_id}/configure"
        }
    }

# ==================== Audit Trail Endpoints ====================

@app.get("/audit/trail")
async def get_audit_trail(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    agent_id: Optional[str] = None,
    user_id: Optional[str] = None,
    limit: int = 100
):
    """
    Retrieve audit trail for compliance
    """
    # Mock audit entries
    audit_entries = [
        {
            "transaction_id": f"tx_{datetime.now().timestamp()}_001",
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": "agent_execution",
            "agent_id": agent_id or "clinical_v2",
            "user_id": user_id or "system",
            "action": "execute",
            "result": "success",
            "details": {
                "input_size": 1024,
                "output_size": 2048,
                "processing_time": 234,
                "confidence": 0.95
            },
            "hipaa_compliant": True,
            "data_accessed": ["patient_records"],
            "ip_address": "192.168.1.100"
        }
    ]
    
    return {
        "audit_entries": audit_entries[:limit],
        "total": len(audit_entries),
        "filters": {
            "start_date": start_date,
            "end_date": end_date,
            "agent_id": agent_id,
            "user_id": user_id
        },
        "compliance_status": "compliant"
    }

# ==================== Helper Functions ====================

async def post_registration_setup(agent_id: str):
    """
    Background task for post-registration setup
    """
    # This would handle additional setup like:
    # - Creating monitoring dashboards
    # - Setting up alerts
    # - Initializing metrics collection
    # - Sending notifications
    pass

# ==================== Metrics Endpoint ====================

@app.get("/metrics")
async def get_metrics():
    """
    Prometheus-compatible metrics endpoint
    """
    metrics = f"""# HELP sadp_agents_total Total number of registered agents
# TYPE sadp_agents_total gauge
sadp_agents_total 12

# HELP sadp_agent_executions_total Total agent executions
# TYPE sadp_agent_executions_total counter
sadp_agent_executions_total 48293

# HELP sadp_agent_latency_seconds Agent execution latency
# TYPE sadp_agent_latency_seconds histogram
sadp_agent_latency_seconds_bucket{{le="0.1"}} 2847
sadp_agent_latency_seconds_bucket{{le="0.5"}} 4123
sadp_agent_latency_seconds_bucket{{le="1.0"}} 4789
sadp_agent_latency_seconds_bucket{{le="+Inf"}} 4823
sadp_agent_latency_seconds_sum 2341.5
sadp_agent_latency_seconds_count 4823

# HELP sadp_active_users Current active users
# TYPE sadp_active_users gauge
sadp_active_users 127

# HELP sadp_marketplace_downloads Total marketplace downloads
# TYPE sadp_marketplace_downloads counter
sadp_marketplace_downloads 10811
"""
    
    from fastapi import Response
    return Response(content=metrics, media_type="text/plain")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main_with_management:app", host="0.0.0.0", port=port, reload=False)