"""
SADP Development Service
Agent development, template management, and deployment pipeline
"""

import asyncio
import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, status, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer
from pydantic import BaseModel
import uvicorn

from agent_builder import AgentBuilder
from template_manager import TemplateManager
from version_control import VersionController
from deployment import DeploymentManager
from models import (
    AgentTemplate, AgentDefinition, DeploymentRequest, 
    BuildResult, TestResult, VersionInfo
)
from config import Settings
from auth import get_current_user, require_permission, TokenData
from telemetry import setup_telemetry
from pubsub_client import PubSubClient

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Global instances
settings = Settings()
security = HTTPBearer()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    logger.info("Starting SADP Development Service", version="1.0.0")
    
    # Initialize services
    app.state.agent_builder = AgentBuilder(settings)
    app.state.template_manager = TemplateManager(settings)
    app.state.version_controller = VersionController(settings)
    app.state.deployment_manager = DeploymentManager(settings)
    app.state.pubsub_client = PubSubClient(settings)
    
    # Setup telemetry
    setup_telemetry(settings)
    
    logger.info("Development service started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down development service")
    await app.state.pubsub_client.close()
    logger.info("Development service shutdown complete")

# Create FastAPI application
app = FastAPI(
    title="SADP Development Service",
    description="Agent development, template management, and deployment service",
    version="1.0.0",
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Health check endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint for load balancer"""
    return {
        "status": "healthy",
        "service": "development",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/ready")
async def readiness_check():
    """Readiness check for Kubernetes"""
    try:
        # Check services availability
        await app.state.template_manager.health_check()
        return {"status": "ready"}
    except Exception as e:
        logger.error("Readiness check failed", error=str(e))
        raise HTTPException(status_code=503, detail="Service not ready")

# Template Management
@app.get("/templates", response_model=List[AgentTemplate])
async def list_templates(
    category: Optional[str] = None,
    current_user: TokenData = Depends(require_permission("template:read"))
):
    """List available agent templates"""
    try:
        templates = await app.state.template_manager.list_templates(
            organization_id=current_user.organization_id,
            category=category
        )
        return templates
    except Exception as e:
        logger.error("Failed to list templates", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve templates")

@app.get("/templates/{template_id}", response_model=AgentTemplate)
async def get_template(
    template_id: str,
    current_user: TokenData = Depends(require_permission("template:read"))
):
    """Get specific agent template"""
    try:
        template = await app.state.template_manager.get_template(
            template_id, current_user.organization_id
        )
        if not template:
            raise HTTPException(status_code=404, detail="Template not found")
        return template
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get template", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve template")

@app.post("/templates", response_model=AgentTemplate)
async def create_template(
    template: AgentTemplate,
    current_user: TokenData = Depends(require_permission("template:write"))
):
    """Create new agent template"""
    try:
        template.organization_id = current_user.organization_id
        template.created_by = current_user.user_id
        
        created_template = await app.state.template_manager.create_template(template)
        
        logger.info(
            "Template created",
            template_id=created_template.template_id,
            user_id=current_user.user_id
        )
        
        return created_template
    except Exception as e:
        logger.error("Failed to create template", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to create template")

@app.put("/templates/{template_id}", response_model=AgentTemplate)
async def update_template(
    template_id: str,
    template: AgentTemplate,
    current_user: TokenData = Depends(require_permission("template:write"))
):
    """Update existing template"""
    try:
        template.template_id = template_id
        template.organization_id = current_user.organization_id
        template.updated_by = current_user.user_id
        
        updated_template = await app.state.template_manager.update_template(template)
        
        if not updated_template:
            raise HTTPException(status_code=404, detail="Template not found")
        
        logger.info(
            "Template updated",
            template_id=template_id,
            user_id=current_user.user_id
        )
        
        return updated_template
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to update template", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to update template")

@app.delete("/templates/{template_id}")
async def delete_template(
    template_id: str,
    current_user: TokenData = Depends(require_permission("template:delete"))
):
    """Delete template"""
    try:
        success = await app.state.template_manager.delete_template(
            template_id, current_user.organization_id
        )
        
        if not success:
            raise HTTPException(status_code=404, detail="Template not found")
        
        logger.info(
            "Template deleted",
            template_id=template_id,
            user_id=current_user.user_id
        )
        
        return {"message": "Template deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete template", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to delete template")

# Agent Development
@app.post("/agents/build", response_model=BuildResult)
async def build_agent(
    agent_definition: AgentDefinition,
    background_tasks: BackgroundTasks,
    current_user: TokenData = Depends(require_permission("agent:write"))
):
    """Build agent from definition"""
    try:
        logger.info(
            "Starting agent build",
            agent_name=agent_definition.name,
            user_id=current_user.user_id
        )
        
        agent_definition.organization_id = current_user.organization_id
        agent_definition.created_by = current_user.user_id
        
        # Start build process
        build_result = await app.state.agent_builder.build_agent(agent_definition)
        
        # Schedule background testing
        background_tasks.add_task(
            app.state.agent_builder.run_automated_tests,
            build_result.build_id
        )
        
        logger.info(
            "Agent build completed",
            build_id=build_result.build_id,
            success=build_result.success
        )
        
        return build_result
    except Exception as e:
        logger.error("Failed to build agent", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to build agent")

@app.get("/builds/{build_id}", response_model=BuildResult)
async def get_build_result(
    build_id: str,
    current_user: TokenData = Depends(require_permission("agent:read"))
):
    """Get build result"""
    try:
        build_result = await app.state.agent_builder.get_build_result(
            build_id, current_user.organization_id
        )
        
        if not build_result:
            raise HTTPException(status_code=404, detail="Build not found")
        
        return build_result
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get build result", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve build result")

@app.post("/agents/{agent_id}/test", response_model=TestResult)
async def test_agent(
    agent_id: str,
    test_config: Dict[str, Any],
    current_user: TokenData = Depends(require_permission("agent:execute"))
):
    """Test agent with given configuration"""
    try:
        test_result = await app.state.agent_builder.test_agent(
            agent_id=agent_id,
            config=test_config,
            organization_id=current_user.organization_id
        )
        
        logger.info(
            "Agent test completed",
            agent_id=agent_id,
            success=test_result.success,
            test_count=len(test_result.test_cases)
        )
        
        return test_result
    except Exception as e:
        logger.error("Failed to test agent", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to test agent")

# Version Control
@app.get("/agents/{agent_id}/versions", response_model=List[VersionInfo])
async def list_agent_versions(
    agent_id: str,
    current_user: TokenData = Depends(require_permission("agent:read"))
):
    """List all versions of an agent"""
    try:
        versions = await app.state.version_controller.list_versions(
            agent_id, current_user.organization_id
        )
        return versions
    except Exception as e:
        logger.error("Failed to list agent versions", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve versions")

@app.post("/agents/{agent_id}/versions", response_model=VersionInfo)
async def create_agent_version(
    agent_id: str,
    version_data: Dict[str, Any],
    current_user: TokenData = Depends(require_permission("agent:write"))
):
    """Create new version of an agent"""
    try:
        version_info = await app.state.version_controller.create_version(
            agent_id=agent_id,
            version_data=version_data,
            organization_id=current_user.organization_id,
            created_by=current_user.user_id
        )
        
        logger.info(
            "Agent version created",
            agent_id=agent_id,
            version=version_info.version,
            user_id=current_user.user_id
        )
        
        return version_info
    except Exception as e:
        logger.error("Failed to create agent version", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to create version")

@app.post("/agents/{agent_id}/versions/{version}/promote")
async def promote_version(
    agent_id: str,
    version: str,
    current_user: TokenData = Depends(require_permission("agent:write"))
):
    """Promote version to production"""
    try:
        success = await app.state.version_controller.promote_version(
            agent_id=agent_id,
            version=version,
            organization_id=current_user.organization_id,
            promoted_by=current_user.user_id
        )
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to promote version")
        
        logger.info(
            "Agent version promoted",
            agent_id=agent_id,
            version=version,
            user_id=current_user.user_id
        )
        
        return {"message": "Version promoted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to promote version", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to promote version")

# Deployment
@app.post("/deploy")
async def deploy_agent(
    deployment_request: DeploymentRequest,
    background_tasks: BackgroundTasks,
    current_user: TokenData = Depends(require_permission("agent:deploy"))
):
    """Deploy agent to specified environment"""
    try:
        logger.info(
            "Starting deployment",
            agent_id=deployment_request.agent_id,
            environment=deployment_request.environment,
            user_id=current_user.user_id
        )
        
        deployment_request.organization_id = current_user.organization_id
        deployment_request.deployed_by = current_user.user_id
        
        # Start deployment process
        deployment_id = await app.state.deployment_manager.deploy(deployment_request)
        
        # Schedule monitoring
        background_tasks.add_task(
            app.state.deployment_manager.monitor_deployment,
            deployment_id
        )
        
        return {
            "deployment_id": deployment_id,
            "message": "Deployment started successfully"
        }
    except Exception as e:
        logger.error("Failed to deploy agent", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to deploy agent")

@app.get("/deployments/{deployment_id}")
async def get_deployment_status(
    deployment_id: str,
    current_user: TokenData = Depends(require_permission("agent:read"))
):
    """Get deployment status"""
    try:
        status = await app.state.deployment_manager.get_deployment_status(
            deployment_id, current_user.organization_id
        )
        
        if not status:
            raise HTTPException(status_code=404, detail="Deployment not found")
        
        return status
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get deployment status", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve deployment status")

# File Upload for Templates/Configs
@app.post("/upload/template")
async def upload_template_file(
    file: UploadFile = File(...),
    current_user: TokenData = Depends(require_permission("template:write"))
):
    """Upload template file"""
    try:
        # Validate file type
        if not file.filename.endswith(('.json', '.yaml', '.yml')):
            raise HTTPException(
                status_code=400,
                detail="Only JSON and YAML files are supported"
            )
        
        # Read and validate content
        content = await file.read()
        
        # Store file and return reference
        file_id = await app.state.template_manager.store_template_file(
            filename=file.filename,
            content=content,
            organization_id=current_user.organization_id,
            uploaded_by=current_user.user_id
        )
        
        logger.info(
            "Template file uploaded",
            file_id=file_id,
            filename=file.filename,
            user_id=current_user.user_id
        )
        
        return {
            "file_id": file_id,
            "filename": file.filename,
            "message": "File uploaded successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to upload template file", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to upload file")

# Development Environment Management
@app.post("/environments/{env_name}/provision")
async def provision_development_environment(
    env_name: str,
    config: Dict[str, Any],
    current_user: TokenData = Depends(require_permission("org:write"))
):
    """Provision development environment"""
    try:
        env_id = await app.state.deployment_manager.provision_environment(
            name=env_name,
            config=config,
            organization_id=current_user.organization_id,
            created_by=current_user.user_id
        )
        
        logger.info(
            "Development environment provisioned",
            env_id=env_id,
            env_name=env_name,
            user_id=current_user.user_id
        )
        
        return {
            "environment_id": env_id,
            "message": "Environment provisioned successfully"
        }
    except Exception as e:
        logger.error("Failed to provision environment", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to provision environment")

# Analytics and Insights
@app.get("/analytics/build-metrics")
async def get_build_metrics(
    days: int = 30,
    current_user: TokenData = Depends(require_permission("agent:read"))
):
    """Get build and deployment metrics"""
    try:
        metrics = await app.state.agent_builder.get_build_metrics(
            organization_id=current_user.organization_id,
            days=days
        )
        return metrics
    except Exception as e:
        logger.error("Failed to get build metrics", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve metrics")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.PORT,
        log_config=None,  # Use structlog instead
        access_log=False
    )