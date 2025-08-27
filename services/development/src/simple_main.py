"""
SADP Development Service - Simple Working Version
"""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List, Optional

app = FastAPI(
    title="SADP Development Service",
    description="Agent development and template management",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AgentTemplate(BaseModel):
    template_id: Optional[str] = None
    name: str
    description: str
    category: str
    template_data: Dict[str, Any]

class AgentDefinition(BaseModel):
    name: str
    template_id: str
    configuration: Dict[str, Any]

class BuildResult(BaseModel):
    build_id: str
    agent_id: str
    success: bool
    artifacts: List[str]
    timestamp: str

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "development"}

@app.get("/")
async def root():
    return {"message": "SADP Development Service is running"}

@app.get("/templates")
async def list_templates(category: Optional[str] = None):
    # Mock templates
    templates = [
        {
            "template_id": "clinical-agent-v1",
            "name": "Clinical Agent Template",
            "description": "Template for clinical decision support agents",
            "category": "healthcare"
        },
        {
            "template_id": "billing-agent-v1", 
            "name": "Billing Agent Template",
            "description": "Template for medical billing automation",
            "category": "finance"
        }
    ]
    
    if category:
        templates = [t for t in templates if t.get("category") == category]
    
    return templates

@app.post("/templates", response_model=AgentTemplate)
async def create_template(template: AgentTemplate):
    import uuid
    template.template_id = str(uuid.uuid4())
    return template

@app.post("/agents/build", response_model=BuildResult)
async def build_agent(agent_definition: AgentDefinition):
    import uuid
    from datetime import datetime
    
    build_id = str(uuid.uuid4())
    agent_id = f"agent_{agent_definition.name.lower().replace(' ', '_')}"
    
    # Mock build process
    artifacts = [
        f"{agent_id}.py",
        f"{agent_id}_config.json",
        f"{agent_id}_requirements.txt"
    ]
    
    return BuildResult(
        build_id=build_id,
        agent_id=agent_id,
        success=True,
        artifacts=artifacts,
        timestamp=datetime.utcnow().isoformat()
    )

@app.get("/agents/{agent_id}/status")
async def get_agent_status(agent_id: str):
    return {
        "agent_id": agent_id,
        "status": "ready",
        "version": "1.0.0",
        "health_score": 0.95
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)