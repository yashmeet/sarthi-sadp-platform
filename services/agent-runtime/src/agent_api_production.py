"""
Agent Management Production API
Real agent deployment and execution without mock data
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, UploadFile, File
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
from datetime import datetime
import json
import uuid
import os
import tempfile
import subprocess
import asyncio
import base64
import structlog
from google.cloud import firestore, storage, run_v2, secretmanager
import google.generativeai as genai
import docker
import yaml

from auth import get_current_user, TokenData, require_permission, get_current_organization
from auth import Organization

logger = structlog.get_logger()

router = APIRouter(prefix="/agents", tags=["Agent Management"])

# Initialize clients
db = firestore.Client(project=os.environ.get("GCP_PROJECT_ID", "sarthi-patient-experience-hub"))
storage_client = storage.Client(project=os.environ.get("GCP_PROJECT_ID"))
run_client = run_v2.ServicesClient()
secret_client = secretmanager.SecretManagerServiceClient()
docker_client = docker.from_env()

# Initialize Gemini
if os.environ.get("GEMINI_API_KEY"):
    genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
else:
    gemini_model = None

# Models
class AgentConfig(BaseModel):
    name: str
    description: str
    category: str
    implementation_type: str  # builtin, custom, container, api
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    poml_template_id: Optional[str] = None
    model_config: Optional[Dict[str, Any]] = None
    api_config: Optional[Dict[str, Any]] = None
    container_config: Optional[Dict[str, Any]] = None
    resource_limits: Optional[Dict[str, Any]] = None
    environment_vars: Optional[Dict[str, str]] = None
    
class AgentExecutionRequest(BaseModel):
    input_data: Dict[str, Any]
    execution_mode: str = "sync"  # sync, async, stream
    timeout_seconds: int = 300
    
class AgentDeploymentConfig(BaseModel):
    min_instances: int = 0
    max_instances: int = 10
    cpu: str = "1"
    memory: str = "512Mi"
    environment: str = "production"  # development, staging, production
    region: str = "us-central1"

class AgentExecutor:
    """Handles real agent execution"""
    
    @staticmethod
    async def execute_builtin_agent(
        agent: Dict[str, Any],
        input_data: Dict[str, Any],
        poml_template: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute built-in AI agent using Gemini"""
        if not gemini_model:
            raise HTTPException(status_code=503, detail="Gemini API not configured")
        
        try:
            # Prepare prompt
            if poml_template:
                # Load and render POML template
                from poml_api_production import render_poml_template
                prompt = render_poml_template(poml_template, input_data)
            else:
                # Use default prompt structure
                prompt = f"""
                Agent: {agent['name']}
                Task: {agent['description']}
                
                Input:
                {json.dumps(input_data, indent=2)}
                
                Please process this input according to the agent's purpose and return a structured response.
                """
            
            # Configure generation
            generation_config = genai.GenerationConfig(
                temperature=agent.get('model_config', {}).get('temperature', 0.7),
                max_output_tokens=agent.get('model_config', {}).get('max_tokens', 2000),
                top_p=0.95,
                top_k=40
            )
            
            # Execute
            start_time = datetime.utcnow()
            response = gemini_model.generate_content(prompt, generation_config=generation_config)
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Parse response
            try:
                # Try to extract JSON from response
                import re
                json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
                if json_match:
                    result_data = json.loads(json_match.group())
                else:
                    result_data = {"response": response.text}
            except:
                result_data = {"response": response.text}
            
            return {
                "success": True,
                "agent_id": agent['id'],
                "execution_time": execution_time,
                "result": result_data,
                "usage": {
                    "prompt_tokens": response.usage_metadata.prompt_token_count,
                    "completion_tokens": response.usage_metadata.candidates_token_count,
                    "total_tokens": response.usage_metadata.total_token_count
                }
            }
            
        except Exception as e:
            logger.error(f"Builtin agent execution failed: {e}")
            raise
    
    @staticmethod
    async def execute_container_agent(
        agent: Dict[str, Any],
        input_data: Dict[str, Any],
        service_url: str
    ) -> Dict[str, Any]:
        """Execute containerized agent via Cloud Run"""
        import httpx
        
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                start_time = datetime.utcnow()
                
                response = await client.post(
                    f"{service_url}/execute",
                    json=input_data,
                    headers={"Authorization": f"Bearer {await get_id_token(service_url)}"}
                )
                
                execution_time = (datetime.utcnow() - start_time).total_seconds()
                
                if response.status_code != 200:
                    raise Exception(f"Agent returned status {response.status_code}: {response.text}")
                
                return {
                    "success": True,
                    "agent_id": agent['id'],
                    "execution_time": execution_time,
                    "result": response.json()
                }
                
        except Exception as e:
            logger.error(f"Container agent execution failed: {e}")
            raise
    
    @staticmethod
    async def execute_api_agent(
        agent: Dict[str, Any],
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute external API agent"""
        import httpx
        
        try:
            api_config = agent.get('api_config', {})
            
            # Prepare headers
            headers = api_config.get('headers', {})
            if api_config.get('auth_type') == 'bearer':
                # Get token from secret manager
                if api_config.get('auth_secret'):
                    token = await get_secret(api_config['auth_secret'])
                    headers['Authorization'] = f"Bearer {token}"
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                start_time = datetime.utcnow()
                
                response = await client.request(
                    method=api_config.get('method', 'POST'),
                    url=api_config['endpoint'],
                    json=input_data,
                    headers=headers
                )
                
                execution_time = (datetime.utcnow() - start_time).total_seconds()
                
                if response.status_code not in [200, 201]:
                    raise Exception(f"API returned status {response.status_code}: {response.text}")
                
                return {
                    "success": True,
                    "agent_id": agent['id'],
                    "execution_time": execution_time,
                    "result": response.json()
                }
                
        except Exception as e:
            logger.error(f"API agent execution failed: {e}")
            raise

@router.post("/register")
async def register_agent(
    config: AgentConfig,
    background_tasks: BackgroundTasks,
    current_user: TokenData = Depends(require_permission("agent:write")),
    org: Organization = Depends(get_current_organization)
):
    """Register a new agent with real deployment"""
    try:
        # Generate agent ID
        agent_id = f"agent_{uuid.uuid4().hex[:12]}"
        
        # Validate configuration
        if config.implementation_type not in ['builtin', 'custom', 'container', 'api']:
            raise HTTPException(status_code=400, detail="Invalid implementation type")
        
        # Create agent document
        agent_data = {
            "id": agent_id,
            "name": config.name,
            "description": config.description,
            "category": config.category,
            "implementation_type": config.implementation_type,
            "input_schema": config.input_schema,
            "output_schema": config.output_schema,
            "poml_template_id": config.poml_template_id,
            "model_config": config.model_config,
            "api_config": config.api_config,
            "container_config": config.container_config,
            "resource_limits": config.resource_limits or {
                "cpu": "1",
                "memory": "512Mi",
                "timeout": 300
            },
            "environment_vars": config.environment_vars,
            "organization_id": org.id,
            "created_by": current_user.user_id,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "version": "1.0.0",
            "status": "registering",
            "metrics": {
                "executions": 0,
                "successes": 0,
                "failures": 0,
                "avg_latency": 0
            }
        }
        
        # Store in Firestore
        db.collection("agents").document(agent_id).set(agent_data)
        
        # Deploy based on implementation type
        if config.implementation_type == 'container':
            background_tasks.add_task(
                deploy_container_agent,
                agent_id,
                agent_data,
                org.id
            )
        elif config.implementation_type == 'api':
            # Store API credentials securely
            if config.api_config and config.api_config.get('auth_token'):
                await store_api_credentials(agent_id, config.api_config['auth_token'])
        
        # Update status
        db.collection("agents").document(agent_id).update({"status": "active"})
        
        logger.info(f"Registered agent {agent_id}", 
                   user_id=current_user.user_id,
                   organization_id=org.id)
        
        return {
            "id": agent_id,
            "name": config.name,
            "status": "active",
            "endpoints": {
                "execute": f"/agents/{agent_id}/execute",
                "status": f"/agents/{agent_id}/status",
                "metrics": f"/agents/{agent_id}/metrics"
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to register agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/list")
async def list_agents(
    category: Optional[str] = None,
    status: Optional[str] = None,
    current_user: TokenData = Depends(get_current_user),
    org: Organization = Depends(get_current_organization)
):
    """List organization's agents"""
    try:
        query = db.collection("agents").where("organization_id", "==", org.id)
        
        if category:
            query = query.where("category", "==", category)
        if status:
            query = query.where("status", "==", status)
        
        agents = []
        for doc in query.stream():
            agent_data = doc.to_dict()
            agents.append({
                "id": agent_data['id'],
                "name": agent_data['name'],
                "description": agent_data['description'],
                "category": agent_data['category'],
                "implementation_type": agent_data['implementation_type'],
                "status": agent_data['status'],
                "version": agent_data.get('version'),
                "created_at": agent_data['created_at'],
                "metrics": agent_data.get('metrics', {})
            })
        
        return {"agents": agents, "total": len(agents)}
        
    except Exception as e:
        logger.error(f"Failed to list agents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{agent_id}")
async def get_agent(
    agent_id: str,
    current_user: TokenData = Depends(get_current_user),
    org: Organization = Depends(get_current_organization)
):
    """Get agent details"""
    try:
        agent_doc = db.collection("agents").document(agent_id).get()
        
        if not agent_doc.exists:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        agent_data = agent_doc.to_dict()
        
        # Check permissions
        if agent_data['organization_id'] != org.id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Get deployment status if container
        if agent_data['implementation_type'] == 'container':
            agent_data['deployment'] = await get_deployment_status(agent_id, org.id)
        
        return agent_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{agent_id}/execute")
async def execute_agent(
    agent_id: str,
    request: AgentExecutionRequest,
    background_tasks: BackgroundTasks,
    current_user: TokenData = Depends(require_permission("agent:execute")),
    org: Organization = Depends(get_current_organization)
):
    """Execute agent with real processing"""
    try:
        # Get agent
        agent_doc = db.collection("agents").document(agent_id).get()
        
        if not agent_doc.exists:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        agent = agent_doc.to_dict()
        
        # Check permissions
        if agent['organization_id'] != org.id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        if agent['status'] != 'active':
            raise HTTPException(status_code=400, detail=f"Agent is {agent['status']}")
        
        # Create execution record
        execution_id = f"exec_{uuid.uuid4().hex[:12]}"
        execution_data = {
            "id": execution_id,
            "agent_id": agent_id,
            "user_id": current_user.user_id,
            "organization_id": org.id,
            "input_data": request.input_data,
            "status": "running",
            "created_at": datetime.utcnow()
        }
        
        db.collection("executions").document(execution_id).set(execution_data)
        
        # Execute based on type
        try:
            if agent['implementation_type'] == 'builtin':
                # Load POML template if specified
                poml_template = None
                if agent.get('poml_template_id'):
                    template_doc = db.collection("poml_templates")\
                                    .document(agent['poml_template_id']).get()
                    if template_doc.exists:
                        version_doc = db.collection("poml_templates")\
                                       .document(agent['poml_template_id'])\
                                       .collection("versions")\
                                       .document(template_doc.to_dict()['current_version']).get()
                        poml_template = version_doc.to_dict()['content']
                
                result = await AgentExecutor.execute_builtin_agent(
                    agent,
                    request.input_data,
                    poml_template
                )
                
            elif agent['implementation_type'] == 'container':
                service_url = await get_cloud_run_url(agent_id, org.id)
                result = await AgentExecutor.execute_container_agent(
                    agent,
                    request.input_data,
                    service_url
                )
                
            elif agent['implementation_type'] == 'api':
                result = await AgentExecutor.execute_api_agent(
                    agent,
                    request.input_data
                )
                
            else:
                raise Exception(f"Unknown implementation type: {agent['implementation_type']}")
            
            # Update execution record
            db.collection("executions").document(execution_id).update({
                "status": "completed",
                "result": result,
                "completed_at": datetime.utcnow(),
                "execution_time": result.get('execution_time', 0)
            })
            
            # Update agent metrics in background
            background_tasks.add_task(
                update_agent_metrics,
                agent_id,
                result.get('execution_time', 0),
                True
            )
            
            logger.info(f"Agent execution successful",
                       agent_id=agent_id,
                       execution_id=execution_id,
                       execution_time=result.get('execution_time', 0))
            
            return {
                "execution_id": execution_id,
                **result
            }
            
        except Exception as e:
            # Update execution record with error
            db.collection("executions").document(execution_id).update({
                "status": "failed",
                "error": str(e),
                "completed_at": datetime.utcnow()
            })
            
            # Update failure metrics
            background_tasks.add_task(
                update_agent_metrics,
                agent_id,
                0,
                False
            )
            
            logger.error(f"Agent execution failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to execute agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{agent_id}/deploy")
async def deploy_agent(
    agent_id: str,
    config: AgentDeploymentConfig,
    current_user: TokenData = Depends(require_permission("agent:write")),
    org: Organization = Depends(get_current_organization)
):
    """Deploy or update agent deployment"""
    try:
        # Get agent
        agent_doc = db.collection("agents").document(agent_id).get()
        
        if not agent_doc.exists:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        agent = agent_doc.to_dict()
        
        # Check permissions
        if agent['organization_id'] != org.id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        if agent['implementation_type'] != 'container':
            raise HTTPException(status_code=400, detail="Only container agents can be deployed")
        
        # Deploy to Cloud Run
        service_name = f"{org.id}-{agent_id}".lower()[:63]  # Cloud Run name limit
        
        deployment_result = await deploy_to_cloud_run(
            service_name=service_name,
            agent=agent,
            config=config,
            org_id=org.id
        )
        
        # Update agent with deployment info
        db.collection("agents").document(agent_id).update({
            "deployment": {
                "service_name": service_name,
                "url": deployment_result['url'],
                "region": config.region,
                "environment": config.environment,
                "deployed_at": datetime.utcnow(),
                "deployed_by": current_user.user_id
            },
            "status": "active"
        })
        
        return {
            "agent_id": agent_id,
            "deployment": deployment_result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to deploy agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{agent_id}/upload-code")
async def upload_agent_code(
    agent_id: str,
    file: UploadFile = File(...),
    current_user: TokenData = Depends(require_permission("agent:write")),
    org: Organization = Depends(get_current_organization)
):
    """Upload custom agent code or container image"""
    try:
        # Get agent
        agent_doc = db.collection("agents").document(agent_id).get()
        
        if not agent_doc.exists:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        agent = agent_doc.to_dict()
        
        # Check permissions
        if agent['organization_id'] != org.id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Store code in Cloud Storage
        bucket_name = f"{org.id}-agent-code"
        bucket = storage_client.bucket(bucket_name)
        if not bucket.exists():
            bucket = storage_client.create_bucket(bucket_name, location="us-central1")
        
        # Upload file
        blob_name = f"{agent_id}/{file.filename}"
        blob = bucket.blob(blob_name)
        content = await file.read()
        blob.upload_from_string(content)
        
        # If it's a Dockerfile, build and push image
        if file.filename == "Dockerfile":
            # Build Docker image
            image_tag = await build_and_push_image(
                agent_id=agent_id,
                dockerfile_content=content.decode('utf-8'),
                org_id=org.id
            )
            
            # Update agent configuration
            db.collection("agents").document(agent_id).update({
                "container_config.image": image_tag,
                "updated_at": datetime.utcnow()
            })
            
            return {
                "message": "Container image built and pushed",
                "image": image_tag
            }
        else:
            return {
                "message": "Code uploaded",
                "location": f"gs://{bucket_name}/{blob_name}"
            }
        
    except Exception as e:
        logger.error(f"Failed to upload agent code: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{agent_id}/metrics")
async def get_agent_metrics(
    agent_id: str,
    current_user: TokenData = Depends(get_current_user),
    org: Organization = Depends(get_current_organization)
):
    """Get agent performance metrics"""
    try:
        # Get agent
        agent_doc = db.collection("agents").document(agent_id).get()
        
        if not agent_doc.exists:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        agent = agent_doc.to_dict()
        
        # Check permissions
        if agent['organization_id'] != org.id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Get execution history
        executions = []
        exec_query = db.collection("executions")\
                      .where("agent_id", "==", agent_id)\
                      .order_by("created_at", direction=firestore.Query.DESCENDING)\
                      .limit(100)
        
        for doc in exec_query.stream():
            exec_data = doc.to_dict()
            executions.append({
                "id": exec_data['id'],
                "status": exec_data['status'],
                "created_at": exec_data['created_at'],
                "execution_time": exec_data.get('execution_time', 0)
            })
        
        # Calculate metrics
        total_executions = len(executions)
        successful = len([e for e in executions if e['status'] == 'completed'])
        failed = len([e for e in executions if e['status'] == 'failed'])
        
        avg_latency = 0
        if successful > 0:
            latencies = [e['execution_time'] for e in executions if e['status'] == 'completed' and e.get('execution_time')]
            avg_latency = sum(latencies) / len(latencies) if latencies else 0
        
        return {
            "agent_id": agent_id,
            "metrics": {
                "total_executions": total_executions,
                "successful_executions": successful,
                "failed_executions": failed,
                "success_rate": (successful / total_executions * 100) if total_executions > 0 else 0,
                "average_latency": avg_latency,
                "recent_executions": executions[:10]
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get agent metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{agent_id}")
async def delete_agent(
    agent_id: str,
    current_user: TokenData = Depends(require_permission("agent:delete")),
    org: Organization = Depends(get_current_organization)
):
    """Delete agent and its resources"""
    try:
        # Get agent
        agent_doc = db.collection("agents").document(agent_id).get()
        
        if not agent_doc.exists:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        agent = agent_doc.to_dict()
        
        # Check permissions
        if agent['organization_id'] != org.id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Delete Cloud Run service if exists
        if agent['implementation_type'] == 'container' and agent.get('deployment'):
            await delete_cloud_run_service(
                agent['deployment']['service_name'],
                agent['deployment']['region']
            )
        
        # Delete from storage
        bucket_name = f"{org.id}-agent-code"
        try:
            bucket = storage_client.bucket(bucket_name)
            blobs = bucket.list_blobs(prefix=f"{agent_id}/")
            for blob in blobs:
                blob.delete()
        except:
            pass
        
        # Delete executions
        exec_query = db.collection("executions").where("agent_id", "==", agent_id)
        for doc in exec_query.stream():
            doc.reference.delete()
        
        # Delete agent document
        db.collection("agents").document(agent_id).delete()
        
        logger.info(f"Deleted agent {agent_id}",
                   user_id=current_user.user_id,
                   organization_id=org.id)
        
        return {"message": "Agent deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Helper functions
async def deploy_container_agent(agent_id: str, agent_data: Dict[str, Any], org_id: str):
    """Deploy container agent to Cloud Run"""
    try:
        service_name = f"{org_id}-{agent_id}".lower()[:63]
        
        # Default container if not specified
        if not agent_data.get('container_config', {}).get('image'):
            # Use default agent runtime image
            image = f"gcr.io/{os.environ.get('GCP_PROJECT_ID')}/agent-runtime:latest"
        else:
            image = agent_data['container_config']['image']
        
        # Deploy to Cloud Run
        deployment_result = await deploy_to_cloud_run(
            service_name=service_name,
            agent=agent_data,
            config=AgentDeploymentConfig(
                cpu=agent_data['resource_limits']['cpu'],
                memory=agent_data['resource_limits']['memory']
            ),
            org_id=org_id
        )
        
        # Update agent status
        db.collection("agents").document(agent_id).update({
            "status": "active",
            "deployment": {
                "service_name": service_name,
                "url": deployment_result['url'],
                "deployed_at": datetime.utcnow()
            }
        })
        
    except Exception as e:
        logger.error(f"Failed to deploy container agent: {e}")
        db.collection("agents").document(agent_id).update({
            "status": "failed",
            "error": str(e)
        })

async def deploy_to_cloud_run(
    service_name: str,
    agent: Dict[str, Any],
    config: AgentDeploymentConfig,
    org_id: str
) -> Dict[str, Any]:
    """Deploy service to Cloud Run"""
    try:
        project_id = os.environ.get("GCP_PROJECT_ID")
        
        # Prepare service configuration
        service = run_v2.Service()
        service.name = f"projects/{project_id}/locations/{config.region}/services/{service_name}"
        
        # Configure template
        template = service.template
        template.containers = [run_v2.Container()]
        container = template.containers[0]
        
        # Set image
        container.image = agent.get('container_config', {}).get('image', 
                                    f"gcr.io/{project_id}/agent-runtime:latest")
        
        # Set resources
        container.resources.limits = {
            "cpu": config.cpu,
            "memory": config.memory
        }
        
        # Set environment variables
        container.env = [
            run_v2.EnvVar(name="AGENT_ID", value=agent['id']),
            run_v2.EnvVar(name="ORGANIZATION_ID", value=org_id),
            run_v2.EnvVar(name="ENVIRONMENT", value=config.environment)
        ]
        
        if agent.get('environment_vars'):
            for key, value in agent['environment_vars'].items():
                container.env.append(run_v2.EnvVar(name=key, value=value))
        
        # Set scaling
        template.scaling.min_instance_count = config.min_instances
        template.scaling.max_instance_count = config.max_instances
        
        # Deploy service
        request = run_v2.CreateServiceRequest(
            parent=f"projects/{project_id}/locations/{config.region}",
            service=service,
            service_id=service_name
        )
        
        operation = run_client.create_service(request=request)
        response = operation.result()
        
        return {
            "url": response.uri,
            "name": service_name,
            "region": config.region,
            "status": "deployed"
        }
        
    except Exception as e:
        logger.error(f"Cloud Run deployment failed: {e}")
        raise

async def build_and_push_image(agent_id: str, dockerfile_content: str, org_id: str) -> str:
    """Build and push Docker image to Container Registry"""
    try:
        project_id = os.environ.get("GCP_PROJECT_ID")
        image_tag = f"gcr.io/{project_id}/{org_id}-{agent_id}:latest"
        
        # Create temp directory
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write Dockerfile
            dockerfile_path = os.path.join(tmpdir, "Dockerfile")
            with open(dockerfile_path, 'w') as f:
                f.write(dockerfile_content)
            
            # Build image
            image, logs = docker_client.images.build(
                path=tmpdir,
                tag=image_tag,
                rm=True
            )
            
            # Push to registry
            docker_client.images.push(image_tag)
            
            return image_tag
            
    except Exception as e:
        logger.error(f"Failed to build/push image: {e}")
        raise

async def get_cloud_run_url(agent_id: str, org_id: str) -> str:
    """Get Cloud Run service URL"""
    project_id = os.environ.get("GCP_PROJECT_ID")
    service_name = f"{org_id}-{agent_id}".lower()[:63]
    
    request = run_v2.GetServiceRequest(
        name=f"projects/{project_id}/locations/us-central1/services/{service_name}"
    )
    
    service = run_client.get_service(request=request)
    return service.uri

async def delete_cloud_run_service(service_name: str, region: str):
    """Delete Cloud Run service"""
    try:
        project_id = os.environ.get("GCP_PROJECT_ID")
        
        request = run_v2.DeleteServiceRequest(
            name=f"projects/{project_id}/locations/{region}/services/{service_name}"
        )
        
        operation = run_client.delete_service(request=request)
        operation.result()
        
    except Exception as e:
        logger.error(f"Failed to delete Cloud Run service: {e}")

async def get_deployment_status(agent_id: str, org_id: str) -> Dict[str, Any]:
    """Get deployment status from Cloud Run"""
    try:
        project_id = os.environ.get("GCP_PROJECT_ID")
        service_name = f"{org_id}-{agent_id}".lower()[:63]
        
        request = run_v2.GetServiceRequest(
            name=f"projects/{project_id}/locations/us-central1/services/{service_name}"
        )
        
        service = run_client.get_service(request=request)
        
        return {
            "status": "running" if service.latest_ready_revision else "deploying",
            "url": service.uri,
            "revision": service.latest_ready_revision,
            "traffic": service.traffic
        }
        
    except Exception:
        return {"status": "not_deployed"}

async def update_agent_metrics(agent_id: str, execution_time: float, success: bool):
    """Update agent execution metrics"""
    try:
        agent_ref = db.collection("agents").document(agent_id)
        agent_doc = agent_ref.get()
        
        if agent_doc.exists:
            metrics = agent_doc.to_dict().get('metrics', {})
            
            # Update metrics
            total = metrics.get('executions', 0) + 1
            successes = metrics.get('successes', 0) + (1 if success else 0)
            failures = metrics.get('failures', 0) + (0 if success else 1)
            
            total_latency = metrics.get('avg_latency', 0) * metrics.get('executions', 0) + execution_time
            avg_latency = total_latency / total if total > 0 else 0
            
            agent_ref.update({
                "metrics.executions": total,
                "metrics.successes": successes,
                "metrics.failures": failures,
                "metrics.avg_latency": avg_latency
            })
            
    except Exception as e:
        logger.error(f"Failed to update agent metrics: {e}")

async def store_api_credentials(agent_id: str, auth_token: str):
    """Store API credentials in Secret Manager"""
    try:
        project_id = os.environ.get("GCP_PROJECT_ID")
        secret_id = f"agent-{agent_id}-api-token"
        
        parent = f"projects/{project_id}"
        
        # Create secret
        secret = secret_client.create_secret(
            request={
                "parent": parent,
                "secret_id": secret_id,
                "secret": {"replication": {"automatic": {}}}
            }
        )
        
        # Add secret version
        secret_client.add_secret_version(
            request={
                "parent": secret.name,
                "payload": {"data": auth_token.encode()}
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to store API credentials: {e}")
        raise

async def get_secret(secret_name: str) -> str:
    """Retrieve secret from Secret Manager"""
    try:
        project_id = os.environ.get("GCP_PROJECT_ID")
        name = f"projects/{project_id}/secrets/{secret_name}/versions/latest"
        
        response = secret_client.access_secret_version(request={"name": name})
        return response.payload.data.decode('UTF-8')
        
    except Exception as e:
        logger.error(f"Failed to get secret: {e}")
        raise

async def get_id_token(audience: str) -> str:
    """Get ID token for Cloud Run authentication"""
    import google.auth
    from google.auth.transport.requests import Request
    
    credentials, project = google.auth.default()
    credentials.refresh(Request())
    
    return credentials.token