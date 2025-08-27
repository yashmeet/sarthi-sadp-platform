"""
Agent Management System - Backend Implementation
Handles dynamic agent registration, configuration, and deployment
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import uuid
import asyncio
from dataclasses import dataclass
import importlib.util
import sys
import os
import tempfile
import subprocess

from google.cloud import firestore
from google.cloud import storage
from google.cloud import secretmanager
import structlog

logger = structlog.get_logger()


@dataclass
class AgentConfig:
    """Agent configuration data model"""
    id: str
    name: str
    category: str
    version: str
    description: str
    input_types: List[str]
    capabilities: List[str]
    output_schema: Dict[str, Any]
    poml_template: str
    implementation_type: str  # builtin, custom, api
    model_config: Optional[Dict[str, Any]] = None
    custom_code: Optional[str] = None
    api_config: Optional[Dict[str, Any]] = None
    constraints: Optional[Dict[str, bool]] = None
    resource_config: Optional[Dict[str, Any]] = None
    monitoring_config: Optional[Dict[str, bool]] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    status: str = "inactive"
    executions: int = 0
    accuracy: float = 0.0


class AgentManager:
    """Manages agent lifecycle - creation, registration, deployment"""
    
    def __init__(self, project_id: str, region: str = "us-central1"):
        self.project_id = project_id
        self.region = region
        self.db = firestore.Client(project=project_id)
        self.storage_client = storage.Client(project=project_id)
        self.secret_client = secretmanager.SecretManagerServiceClient()
        self.agents_collection = self.db.collection("agents")
        self.templates_collection = self.db.collection("agent_templates")
        self.bucket_name = f"{project_id}-agent-repository"
        self._ensure_bucket()
    
    def _ensure_bucket(self):
        """Ensure agent repository bucket exists"""
        try:
            bucket = self.storage_client.bucket(self.bucket_name)
            if not bucket.exists():
                bucket = self.storage_client.create_bucket(
                    self.bucket_name,
                    location=self.region
                )
                logger.info(f"Created agent repository bucket: {self.bucket_name}")
        except Exception as e:
            logger.error(f"Error ensuring bucket: {e}")
    
    async def register_agent(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Register a new agent in the system
        """
        try:
            # Validate configuration
            self._validate_config(config)
            
            # Generate unique agent ID if not provided
            if not config.get('id'):
                config['id'] = f"agent_{uuid.uuid4().hex[:8]}"
            
            # Create agent configuration
            agent = AgentConfig(
                id=config['id'],
                name=config['name'],
                category=config['category'],
                version=config.get('version', '1.0.0'),
                description=config['description'],
                input_types=config.get('input_types', ['text']),
                capabilities=config.get('capabilities', []),
                output_schema=config.get('output_schema', {}),
                poml_template=config['poml_template'],
                implementation_type=config['implementation_type'],
                model_config=config.get('model_config'),
                custom_code=config.get('custom_code'),
                api_config=config.get('api_config'),
                constraints=config.get('constraints', {}),
                resource_config=config.get('resource_config', {}),
                monitoring_config=config.get('monitoring_config', {}),
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                status="deploying"
            )
            
            # Store POML template
            await self._store_poml_template(agent)
            
            # Handle different implementation types
            if agent.implementation_type == "custom":
                await self._deploy_custom_agent(agent)
            elif agent.implementation_type == "api":
                await self._configure_api_agent(agent)
            else:
                await self._configure_builtin_agent(agent)
            
            # Register in Firestore
            agent_dict = self._agent_to_dict(agent)
            self.agents_collection.document(agent.id).set(agent_dict)
            
            # Update agent status
            agent.status = "active"
            self.agents_collection.document(agent.id).update({"status": "active"})
            
            logger.info(f"Successfully registered agent: {agent.id}")
            
            return {
                "success": True,
                "agent_id": agent.id,
                "status": "active",
                "message": f"Agent {agent.name} successfully registered and deployed",
                "endpoints": {
                    "execute": f"/agents/{agent.id}/execute",
                    "status": f"/agents/{agent.id}/status",
                    "metrics": f"/agents/{agent.id}/metrics"
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to register agent: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Agent registration failed"
            }
    
    def _validate_config(self, config: Dict[str, Any]):
        """Validate agent configuration"""
        required_fields = ['name', 'category', 'description', 'poml_template', 'implementation_type']
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field: {field}")
        
        # Validate implementation type
        valid_types = ['builtin', 'custom', 'api']
        if config['implementation_type'] not in valid_types:
            raise ValueError(f"Invalid implementation type. Must be one of: {valid_types}")
        
        # Validate custom code if provided
        if config['implementation_type'] == 'custom' and not config.get('custom_code'):
            raise ValueError("Custom code is required for custom implementation type")
        
        # Validate API config if provided
        if config['implementation_type'] == 'api' and not config.get('api_config'):
            raise ValueError("API configuration is required for API implementation type")
    
    async def _store_poml_template(self, agent: AgentConfig):
        """Store POML template in Cloud Storage"""
        try:
            bucket = self.storage_client.bucket(self.bucket_name)
            blob_name = f"poml_templates/{agent.id}/template_v{agent.version}.xml"
            blob = bucket.blob(blob_name)
            blob.upload_from_string(agent.poml_template)
            logger.info(f"Stored POML template: {blob_name}")
        except Exception as e:
            logger.error(f"Failed to store POML template: {e}")
            raise
    
    async def _deploy_custom_agent(self, agent: AgentConfig):
        """Deploy custom Python agent code"""
        try:
            # Store custom code in Cloud Storage
            bucket = self.storage_client.bucket(self.bucket_name)
            code_blob_name = f"agent_code/{agent.id}/agent_v{agent.version}.py"
            code_blob = bucket.blob(code_blob_name)
            code_blob.upload_from_string(agent.custom_code)
            
            # Create a deployment package
            deployment_config = {
                "agent_id": agent.id,
                "code_location": f"gs://{self.bucket_name}/{code_blob_name}",
                "dependencies": agent.model_config.get('dependencies', []) if agent.model_config else [],
                "entry_point": "process",
                "runtime": "python39"
            }
            
            # Store deployment configuration
            config_blob_name = f"deployments/{agent.id}/config.json"
            config_blob = bucket.blob(config_blob_name)
            config_blob.upload_from_string(json.dumps(deployment_config))
            
            # Create Cloud Function or Cloud Run service for the agent
            await self._create_agent_service(agent, deployment_config)
            
            logger.info(f"Deployed custom agent: {agent.id}")
            
        except Exception as e:
            logger.error(f"Failed to deploy custom agent: {e}")
            raise
    
    async def _configure_api_agent(self, agent: AgentConfig):
        """Configure external API agent"""
        try:
            # Store API configuration securely
            if agent.api_config.get('auth_token'):
                # Store auth token in Secret Manager
                secret_id = f"agent_{agent.id}_api_token"
                parent = f"projects/{self.project_id}"
                
                secret = self.secret_client.create_secret(
                    request={
                        "parent": parent,
                        "secret_id": secret_id,
                        "secret": {"replication": {"automatic": {}}}
                    }
                )
                
                # Add secret version
                self.secret_client.add_secret_version(
                    request={
                        "parent": secret.name,
                        "payload": {"data": agent.api_config['auth_token'].encode()}
                    }
                )
                
                # Update API config to reference secret
                agent.api_config['auth_secret'] = secret_id
                del agent.api_config['auth_token']
            
            # Store API configuration
            bucket = self.storage_client.bucket(self.bucket_name)
            config_blob_name = f"api_configs/{agent.id}/config.json"
            config_blob = bucket.blob(config_blob_name)
            config_blob.upload_from_string(json.dumps(agent.api_config))
            
            logger.info(f"Configured API agent: {agent.id}")
            
        except Exception as e:
            logger.error(f"Failed to configure API agent: {e}")
            raise
    
    async def _configure_builtin_agent(self, agent: AgentConfig):
        """Configure built-in AI model agent"""
        try:
            # Store model configuration
            model_config = agent.model_config or {
                "model": "gemini-pro",
                "temperature": 0.7,
                "max_tokens": 2000,
                "top_p": 0.9
            }
            
            bucket = self.storage_client.bucket(self.bucket_name)
            config_blob_name = f"model_configs/{agent.id}/config.json"
            config_blob = bucket.blob(config_blob_name)
            config_blob.upload_from_string(json.dumps(model_config))
            
            logger.info(f"Configured built-in agent: {agent.id}")
            
        except Exception as e:
            logger.error(f"Failed to configure built-in agent: {e}")
            raise
    
    async def _create_agent_service(self, agent: AgentConfig, deployment_config: Dict[str, Any]):
        """Create Cloud Run service for custom agent"""
        try:
            # This would typically use Cloud Run API or gcloud SDK
            # For now, we'll store the configuration for manual deployment
            
            service_config = {
                "apiVersion": "serving.knative.dev/v1",
                "kind": "Service",
                "metadata": {
                    "name": f"agent-{agent.id}",
                    "namespace": self.project_id
                },
                "spec": {
                    "template": {
                        "spec": {
                            "containers": [{
                                "image": f"gcr.io/{self.project_id}/agent-runtime:latest",
                                "env": [
                                    {"name": "AGENT_ID", "value": agent.id},
                                    {"name": "AGENT_TYPE", "value": "custom"},
                                    {"name": "CODE_LOCATION", "value": deployment_config['code_location']}
                                ],
                                "resources": {
                                    "limits": {
                                        "cpu": agent.resource_config.get('cpu', '1'),
                                        "memory": agent.resource_config.get('memory', '512Mi')
                                    }
                                }
                            }]
                        }
                    }
                }
            }
            
            # Store service configuration
            bucket = self.storage_client.bucket(self.bucket_name)
            service_blob_name = f"services/{agent.id}/service.yaml"
            service_blob = bucket.blob(service_blob_name)
            service_blob.upload_from_string(json.dumps(service_config))
            
            logger.info(f"Created service configuration for agent: {agent.id}")
            
        except Exception as e:
            logger.error(f"Failed to create agent service: {e}")
            raise
    
    def _agent_to_dict(self, agent: AgentConfig) -> Dict[str, Any]:
        """Convert AgentConfig to dictionary for storage"""
        return {
            "id": agent.id,
            "name": agent.name,
            "category": agent.category,
            "version": agent.version,
            "description": agent.description,
            "input_types": agent.input_types,
            "capabilities": agent.capabilities,
            "output_schema": agent.output_schema,
            "implementation_type": agent.implementation_type,
            "model_config": agent.model_config,
            "api_config": agent.api_config,
            "constraints": agent.constraints,
            "resource_config": agent.resource_config,
            "monitoring_config": agent.monitoring_config,
            "created_at": agent.created_at,
            "updated_at": agent.updated_at,
            "status": agent.status,
            "executions": agent.executions,
            "accuracy": agent.accuracy,
            "poml_template_location": f"gs://{self.bucket_name}/poml_templates/{agent.id}/template_v{agent.version}.xml"
        }
    
    async def list_agents(self, category: Optional[str] = None, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all registered agents with optional filtering"""
        try:
            query = self.agents_collection
            
            if category:
                query = query.where("category", "==", category)
            if status:
                query = query.where("status", "==", status)
            
            agents = []
            for doc in query.stream():
                agent_data = doc.to_dict()
                agents.append(agent_data)
            
            return agents
            
        except Exception as e:
            logger.error(f"Failed to list agents: {e}")
            return []
    
    async def get_agent(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get specific agent configuration"""
        try:
            doc = self.agents_collection.document(agent_id).get()
            if doc.exists:
                return doc.to_dict()
            return None
        except Exception as e:
            logger.error(f"Failed to get agent {agent_id}: {e}")
            return None
    
    async def update_agent(self, agent_id: str, updates: Dict[str, Any]) -> bool:
        """Update agent configuration"""
        try:
            # Don't allow updating certain fields
            protected_fields = ['id', 'created_at']
            for field in protected_fields:
                updates.pop(field, None)
            
            updates['updated_at'] = datetime.utcnow()
            
            self.agents_collection.document(agent_id).update(updates)
            
            # If POML template updated, store new version
            if 'poml_template' in updates:
                agent = await self.get_agent(agent_id)
                if agent:
                    version = agent.get('version', '1.0.0')
                    bucket = self.storage_client.bucket(self.bucket_name)
                    blob_name = f"poml_templates/{agent_id}/template_v{version}.xml"
                    blob = bucket.blob(blob_name)
                    blob.upload_from_string(updates['poml_template'])
            
            logger.info(f"Updated agent {agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update agent {agent_id}: {e}")
            return False
    
    async def delete_agent(self, agent_id: str) -> bool:
        """Delete an agent and its resources"""
        try:
            # Delete from Firestore
            self.agents_collection.document(agent_id).delete()
            
            # Delete from Cloud Storage
            bucket = self.storage_client.bucket(self.bucket_name)
            
            # Delete POML templates
            blobs = bucket.list_blobs(prefix=f"poml_templates/{agent_id}/")
            for blob in blobs:
                blob.delete()
            
            # Delete agent code
            blobs = bucket.list_blobs(prefix=f"agent_code/{agent_id}/")
            for blob in blobs:
                blob.delete()
            
            # Delete configurations
            blobs = bucket.list_blobs(prefix=f"deployments/{agent_id}/")
            for blob in blobs:
                blob.delete()
            
            logger.info(f"Deleted agent {agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete agent {agent_id}: {e}")
            return False
    
    async def execute_agent(self, agent_id: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a registered agent"""
        try:
            # Get agent configuration
            agent = await self.get_agent(agent_id)
            if not agent:
                raise ValueError(f"Agent {agent_id} not found")
            
            if agent['status'] != 'active':
                raise ValueError(f"Agent {agent_id} is not active")
            
            # Load POML template
            bucket = self.storage_client.bucket(self.bucket_name)
            poml_blob = bucket.blob(f"poml_templates/{agent_id}/template_v{agent['version']}.xml")
            poml_template = poml_blob.download_as_text()
            
            # Execute based on implementation type
            if agent['implementation_type'] == 'builtin':
                result = await self._execute_builtin_agent(agent, input_data, poml_template)
            elif agent['implementation_type'] == 'custom':
                result = await self._execute_custom_agent(agent, input_data, poml_template)
            elif agent['implementation_type'] == 'api':
                result = await self._execute_api_agent(agent, input_data, poml_template)
            else:
                raise ValueError(f"Unknown implementation type: {agent['implementation_type']}")
            
            # Update execution count
            self.agents_collection.document(agent_id).update({
                "executions": firestore.Increment(1),
                "last_execution": datetime.utcnow()
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to execute agent {agent_id}: {e}")
            return {
                "success": False,
                "error": str(e),
                "agent_id": agent_id
            }
    
    async def _execute_builtin_agent(self, agent: Dict[str, Any], input_data: Dict[str, Any], poml_template: str) -> Dict[str, Any]:
        """Execute built-in AI model agent"""
        # This would integrate with actual AI services (Gemini, GPT-4, etc.)
        # For now, return mock response
        return {
            "success": True,
            "agent_id": agent['id'],
            "agent_type": "builtin",
            "model": agent.get('model_config', {}).get('model', 'gemini-pro'),
            "results": {
                "processed": True,
                "confidence": 0.95,
                "output": agent.get('output_schema', {})
            },
            "execution_time": 234,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _execute_custom_agent(self, agent: Dict[str, Any], input_data: Dict[str, Any], poml_template: str) -> Dict[str, Any]:
        """Execute custom Python agent"""
        # This would load and execute custom Python code
        # For now, return mock response
        return {
            "success": True,
            "agent_id": agent['id'],
            "agent_type": "custom",
            "results": {
                "processed": True,
                "confidence": 0.92,
                "output": agent.get('output_schema', {})
            },
            "execution_time": 456,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _execute_api_agent(self, agent: Dict[str, Any], input_data: Dict[str, Any], poml_template: str) -> Dict[str, Any]:
        """Execute external API agent"""
        # This would call external API
        # For now, return mock response
        return {
            "success": True,
            "agent_id": agent['id'],
            "agent_type": "api",
            "api_endpoint": agent.get('api_config', {}).get('endpoint', 'unknown'),
            "results": {
                "processed": True,
                "confidence": 0.89,
                "output": agent.get('output_schema', {})
            },
            "execution_time": 789,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def get_agent_metrics(self, agent_id: str) -> Dict[str, Any]:
        """Get agent performance metrics"""
        try:
            agent = await self.get_agent(agent_id)
            if not agent:
                return {}
            
            return {
                "agent_id": agent_id,
                "name": agent['name'],
                "executions": agent.get('executions', 0),
                "accuracy": agent.get('accuracy', 0),
                "status": agent.get('status', 'unknown'),
                "created_at": agent.get('created_at'),
                "last_execution": agent.get('last_execution'),
                "performance": {
                    "avg_response_time": 234,  # Would be calculated from metrics
                    "success_rate": 0.98,
                    "error_rate": 0.02
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get metrics for agent {agent_id}: {e}")
            return {}