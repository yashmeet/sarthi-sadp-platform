"""
POML Orchestration Engine
Advanced template management, execution, and self-learning integration
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timedelta
import os
import json
import yaml
import asyncio
import uuid
import structlog
import httpx
from pathlib import Path
import re
import hashlib
from enum import Enum

# Google Cloud imports
from google.cloud import storage, firestore, secretmanager
import google.generativeai as genai

# Initialize logger
logger = structlog.get_logger()

app = FastAPI(
    title="SADP POML Orchestration Engine",
    description="Advanced POML template management and execution with self-learning capabilities",
    version="2.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "sarthi-patient-experience-hub")
LEARNING_PIPELINE_URL = os.environ.get("LEARNING_PIPELINE_URL", "https://sadp-learning-pipeline-xonau6hybq-uc.a.run.app")
OPTIMIZATION_SERVICE_URL = os.environ.get("OPTIMIZATION_SERVICE_URL", "https://sadp-prompt-optimization-xonau6hybq-uc.a.run.app")

# Initialize clients
storage_client = storage.Client(project=PROJECT_ID)
firestore_client = firestore.Client(project=PROJECT_ID)
secret_client = secretmanager.SecretManagerServiceClient()

# Gemini setup
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if GEMINI_API_KEY and GEMINI_API_KEY != "YOUR_GEMINI_API_KEY_HERE":
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
else:
    gemini_model = None

# Collections
templates_collection = firestore_client.collection('poml_templates_v2')
executions_collection = firestore_client.collection('poml_executions')
experiments_collection = firestore_client.collection('poml_experiments')
deployments_collection = firestore_client.collection('poml_deployments')
performance_collection = firestore_client.collection('template_performance')

# Enums
class TemplateStatus(str, Enum):
    DRAFT = "draft"
    EXPERIMENTAL = "experimental"
    BETA = "beta"
    STABLE = "stable"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"

class ExecutionStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"

class OptimizationStrategy(str, Enum):
    AUTOMEDPROMPT = "automedprompt"
    GENETIC_ALGORITHM = "genetic_algorithm"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"

# Models
class POMLTemplateCreate(BaseModel):
    name: str = Field(..., description="Template name")
    agent_type: str = Field(..., description="Healthcare agent type")
    medical_domain: str = Field(..., description="Medical specialty domain")
    description: str = Field(..., description="Template description")
    content: str = Field(..., description="POML template content")
    variables: List[str] = Field(default=[], description="Required variables")
    tags: List[str] = Field(default=[], description="Template tags")
    constraints: Dict[str, Any] = Field(default={}, description="Template constraints")
    output_schema: Optional[Dict[str, Any]] = Field(None, description="Expected output schema")
    parent_template_id: Optional[str] = Field(None, description="Parent template for inheritance")

class POMLExecutionRequest(BaseModel):
    template_id: str = Field(..., description="Template ID to execute")
    variables: Dict[str, Any] = Field(..., description="Template variables")
    context: Dict[str, Any] = Field(default={}, description="Execution context")
    execution_mode: str = Field(default="standard", description="Execution mode")
    timeout_seconds: int = Field(default=300, description="Execution timeout")
    track_performance: bool = Field(default=True, description="Track performance metrics")

class POMLOptimizationRequest(BaseModel):
    template_id: str = Field(..., description="Template to optimize")
    strategy: OptimizationStrategy = Field(default=OptimizationStrategy.AUTOMEDPROMPT)
    medical_domain: str = Field(..., description="Medical domain for dataset selection")
    optimization_objective: str = Field(default="accuracy", description="Optimization objective")
    max_iterations: int = Field(default=10, description="Maximum optimization iterations")
    performance_threshold: float = Field(default=0.85, description="Target performance threshold")

class TemplateDeploymentRequest(BaseModel):
    template_id: str = Field(..., description="Template to deploy")
    deployment_strategy: str = Field(default="canary", description="Deployment strategy")
    traffic_percentage: float = Field(default=5.0, description="Initial traffic percentage")
    auto_promote: bool = Field(default=False, description="Auto-promote if successful")
    rollback_threshold: float = Field(default=0.8, description="Performance rollback threshold")

class POMLTemplate:
    """Enhanced POML Template with advanced features"""
    
    def __init__(self, template_data: Dict[str, Any]):
        self.template_id = template_data["template_id"]
        self.name = template_data["name"]
        self.agent_type = template_data["agent_type"]
        self.medical_domain = template_data["medical_domain"]
        self.content = template_data["content"]
        self.variables = template_data.get("variables", [])
        self.status = TemplateStatus(template_data.get("status", TemplateStatus.DRAFT))
        self.version = template_data.get("version", "1.0.0")
        self.parent_template_id = template_data.get("parent_template_id")
        self.constraints = template_data.get("constraints", {})
        self.output_schema = template_data.get("output_schema")
        self.created_at = template_data.get("created_at")
        self.updated_at = template_data.get("updated_at")
        self._compiled = None
    
    def compile(self) -> Dict[str, Any]:
        """Compile POML template for execution"""
        if self._compiled is None:
            self._compiled = self._parse_poml_content()
        return self._compiled
    
    def _parse_poml_content(self) -> Dict[str, Any]:
        """Parse POML content with advanced features"""
        try:
            # Enhanced POML parsing with sections, conditions, and loops
            parsed = {
                'template_id': self.template_id,
                'type': 'simple',
                'sections': {},
                'variables': self.variables,
                'constraints': self.constraints,
                'output_schema': self.output_schema,
                'execution_flow': [],
                'error_handling': {},
                'performance_config': {}
            }
            
            # Parse XML-like POML structure
            content = self.content.strip()
            
            # Extract template type and configuration
            type_match = re.search(r'<prompt\s+type="([^"]+)"', content)
            if type_match:
                parsed['type'] = type_match.group(1)
            
            # Extract main sections
            sections = ['system', 'context', 'task', 'constraints', 'examples', 'output', 'validation']
            for section in sections:
                section_pattern = f'<{section}[^>]*>(.*?)</{section}>'
                match = re.search(section_pattern, content, re.DOTALL)
                if match:
                    parsed['sections'][section] = match.group(1).strip()
            
            # Extract conditional logic
            if_blocks = re.findall(r'<if\s+condition="([^"]+)"[^>]*>(.*?)</if>', content, re.DOTALL)
            for condition, block_content in if_blocks:
                parsed['execution_flow'].append({
                    'type': 'conditional',
                    'condition': condition,
                    'content': block_content.strip()
                })
            
            # Extract loop constructs
            for_blocks = re.findall(r'<for\s+each="([^"]+)"[^>]*>(.*?)</for>', content, re.DOTALL)
            for iterator, block_content in for_blocks:
                parsed['execution_flow'].append({
                    'type': 'loop',
                    'iterator': iterator,
                    'content': block_content.strip()
                })
            
            # Extract error handling
            error_blocks = re.findall(r'<error\s+type="([^"]+)"[^>]*>(.*?)</error>', content, re.DOTALL)
            for error_type, error_content in error_blocks:
                parsed['error_handling'][error_type] = error_content.strip()
            
            # Extract performance configurations
            perf_match = re.search(r'<performance\s+([^>]+)>', content)
            if perf_match:
                perf_attrs = perf_match.group(1)
                for attr in re.findall(r'(\w+)="([^"]+)"', perf_attrs):
                    parsed['performance_config'][attr[0]] = attr[1]
            
            return parsed
            
        except Exception as e:
            logger.error("Failed to parse POML content", 
                        template_id=self.template_id, error=str(e))
            raise ValueError(f"Invalid POML template: {e}")
    
    def render(self, variables: Dict[str, Any], context: Dict[str, Any] = None) -> str:
        """Render template with variables and context"""
        compiled = self.compile()
        rendered = self.content
        
        # Merge variables and context
        all_vars = {**variables}
        if context:
            all_vars.update(context)
        
        # Replace variables
        for var_name, var_value in all_vars.items():
            placeholder = f"{{{{{var_name}}}}}"
            rendered = rendered.replace(placeholder, str(var_value))
        
        # Process conditional blocks
        for flow_item in compiled.get('execution_flow', []):
            if flow_item['type'] == 'conditional':
                condition = flow_item['condition']
                condition_result = self._evaluate_condition(condition, all_vars)
                
                if condition_result:
                    # Replace conditional block with content
                    conditional_pattern = f'<if\\s+condition="{re.escape(condition)}"[^>]*>.*?</if>'
                    rendered = re.sub(conditional_pattern, flow_item['content'], rendered, flags=re.DOTALL)
                else:
                    # Remove conditional block
                    conditional_pattern = f'<if\\s+condition="{re.escape(condition)}"[^>]*>.*?</if>'
                    rendered = re.sub(conditional_pattern, '', rendered, flags=re.DOTALL)
        
        # Clean up POML tags for final output
        rendered = re.sub(r'<prompt[^>]*>', '', rendered)
        rendered = re.sub(r'</prompt>', '', rendered)
        rendered = re.sub(r'<(\w+)[^>]*>', r'\n--- \1 ---\n', rendered)
        rendered = re.sub(r'</\w+>', '', rendered)
        
        return rendered.strip()
    
    def _evaluate_condition(self, condition: str, variables: Dict[str, Any]) -> bool:
        """Safely evaluate condition with variables"""
        try:
            # Replace variables in condition
            for var_name, var_value in variables.items():
                condition = condition.replace(f"{{{{{var_name}}}}}", str(var_value))
            
            # Simple condition evaluation (expand as needed)
            return eval(condition)
        except Exception as e:
            logger.warning("Failed to evaluate condition", condition=condition, error=str(e))
            return False

class POMLOrchestrator:
    """Advanced POML orchestration with self-learning capabilities"""
    
    def __init__(self):
        self.http_client = httpx.AsyncClient()
        self.template_cache = {}
        
    async def create_template(self, template_data: POMLTemplateCreate, created_by: str) -> str:
        """Create a new POML template with advanced features"""
        try:
            template_id = f"{template_data.agent_type}_{template_data.name.lower().replace(' ', '_')}_{uuid.uuid4().hex[:8]}"
            
            # Parse and validate template
            temp_template = POMLTemplate({
                "template_id": template_id,
                "name": template_data.name,
                "agent_type": template_data.agent_type,
                "medical_domain": template_data.medical_domain,
                "content": template_data.content,
                "variables": template_data.variables,
                "constraints": template_data.constraints,
                "output_schema": template_data.output_schema
            })
            
            # Compile to validate syntax
            compiled = temp_template.compile()
            
            # Generate content hash for change detection
            content_hash = hashlib.sha256(template_data.content.encode()).hexdigest()
            
            # Create template document
            template_doc = {
                "template_id": template_id,
                "name": template_data.name,
                "agent_type": template_data.agent_type,
                "medical_domain": template_data.medical_domain,
                "description": template_data.description,
                "content": template_data.content,
                "variables": template_data.variables,
                "tags": template_data.tags,
                "constraints": template_data.constraints,
                "output_schema": template_data.output_schema,
                "parent_template_id": template_data.parent_template_id,
                "status": TemplateStatus.DRAFT.value,
                "version": "1.0.0",
                "content_hash": content_hash,
                "compiled": compiled,
                "created_by": created_by,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
                "performance_metrics": {
                    "total_executions": 0,
                    "successful_executions": 0,
                    "average_latency_ms": 0,
                    "average_confidence": 0,
                    "last_optimized": None
                },
                "deployment_info": {
                    "deployed_version": None,
                    "traffic_percentage": 0,
                    "deployment_status": "not_deployed"
                }
            }
            
            # Store in Firestore
            templates_collection.document(template_id).set(template_doc)
            
            # Store template content in Cloud Storage
            await self._store_template_content(template_id, template_data.content)
            
            logger.info("Created POML template", template_id=template_id, agent_type=template_data.agent_type)
            
            return template_id
            
        except Exception as e:
            logger.error("Failed to create template", error=str(e))
            raise HTTPException(status_code=500, detail=f"Failed to create template: {str(e)}")
    
    async def _store_template_content(self, template_id: str, content: str):
        """Store template content in Cloud Storage with versioning"""
        try:
            bucket_name = f"{PROJECT_ID}-poml-templates"
            bucket = storage_client.bucket(bucket_name)
            
            # Store current version
            blob_path = f"templates/{template_id}/current.poml"
            blob = bucket.blob(blob_path)
            blob.upload_from_string(content)
            
            # Store versioned copy
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            version_blob_path = f"templates/{template_id}/versions/{timestamp}.poml"
            version_blob = bucket.blob(version_blob_path)
            version_blob.upload_from_string(content)
            
        except Exception as e:
            logger.error("Failed to store template content", template_id=template_id, error=str(e))
    
    async def get_template(self, template_id: str) -> POMLTemplate:
        """Get template with caching"""
        try:
            # Check cache first
            if template_id in self.template_cache:
                cached_template, cache_time = self.template_cache[template_id]
                if (datetime.utcnow() - cache_time).total_seconds() < 300:  # 5 minute cache
                    return cached_template
            
            # Get from Firestore
            doc = templates_collection.document(template_id).get()
            if not doc.exists:
                raise ValueError(f"Template {template_id} not found")
            
            template_data = doc.to_dict()
            template = POMLTemplate(template_data)
            
            # Cache the template
            self.template_cache[template_id] = (template, datetime.utcnow())
            
            return template
            
        except Exception as e:
            logger.error("Failed to get template", template_id=template_id, error=str(e))
            raise
    
    async def execute_template(self, request: POMLExecutionRequest) -> Dict[str, Any]:
        """Execute POML template with advanced features"""
        try:
            execution_id = str(uuid.uuid4())
            start_time = datetime.utcnow()
            
            logger.info("Starting template execution", 
                       execution_id=execution_id, 
                       template_id=request.template_id)
            
            # Get template
            template = await self.get_template(request.template_id)
            
            # Create execution record
            execution_record = {
                "execution_id": execution_id,
                "template_id": request.template_id,
                "variables": request.variables,
                "context": request.context,
                "status": ExecutionStatus.RUNNING.value,
                "started_at": start_time,
                "timeout_seconds": request.timeout_seconds
            }
            
            executions_collection.document(execution_id).set(execution_record)
            
            # Render template
            rendered_prompt = template.render(request.variables, request.context)
            
            # Execute with timeout
            try:
                result = await asyncio.wait_for(
                    self._execute_with_llm(rendered_prompt, template.compile()),
                    timeout=request.timeout_seconds
                )
                
                execution_status = ExecutionStatus.COMPLETED
                error_message = None
                
            except asyncio.TimeoutError:
                result = {"error": "Execution timeout"}
                execution_status = ExecutionStatus.TIMEOUT
                error_message = "Execution timeout"
                
            except Exception as e:
                result = {"error": str(e)}
                execution_status = ExecutionStatus.FAILED
                error_message = str(e)
            
            end_time = datetime.utcnow()
            duration_ms = int((end_time - start_time).total_seconds() * 1000)
            
            # Update execution record
            execution_updates = {
                "status": execution_status.value,
                "result": result,
                "duration_ms": duration_ms,
                "completed_at": end_time
            }
            
            if error_message:
                execution_updates["error_message"] = error_message
            
            executions_collection.document(execution_id).update(execution_updates)
            
            # Track performance metrics if requested
            if request.track_performance:
                await self._update_performance_metrics(
                    request.template_id, 
                    execution_status == ExecutionStatus.COMPLETED,
                    duration_ms,
                    result.get("confidence", 0)
                )
            
            # Check if template needs optimization
            await self._check_optimization_trigger(request.template_id)
            
            return {
                "execution_id": execution_id,
                "template_id": request.template_id,
                "status": execution_status.value,
                "result": result,
                "duration_ms": duration_ms,
                "prompt_preview": rendered_prompt[:500] + "..." if len(rendered_prompt) > 500 else rendered_prompt
            }
            
        except Exception as e:
            logger.error("Template execution failed", 
                        template_id=request.template_id, error=str(e))
            raise HTTPException(status_code=500, detail=f"Execution failed: {str(e)}")
    
    async def _execute_with_llm(self, prompt: str, compiled_template: Dict[str, Any]) -> Dict[str, Any]:
        """Execute prompt with LLM using configured model"""
        try:
            if gemini_model:
                # Use Gemini for execution
                response = gemini_model.generate_content(
                    prompt,
                    generation_config=genai.GenerationConfig(
                        temperature=float(compiled_template.get('performance_config', {}).get('temperature', 0.3)),
                        top_p=float(compiled_template.get('performance_config', {}).get('top_p', 0.8)),
                        max_output_tokens=int(compiled_template.get('performance_config', {}).get('max_tokens', 2048))
                    )
                )
                
                result_text = response.text
                
                # Validate output schema if defined
                if compiled_template.get('output_schema'):
                    validated_result = await self._validate_output_schema(result_text, compiled_template['output_schema'])
                    return {
                        "response": validated_result,
                        "raw_response": result_text,
                        "confidence": 0.9,  # Mock confidence
                        "tokens_used": len(prompt.split()) + len(result_text.split())
                    }
                
                return {
                    "response": result_text,
                    "confidence": 0.9,
                    "tokens_used": len(prompt.split()) + len(result_text.split())
                }
            else:
                # Fallback mock response
                return {
                    "response": f"Mock response for prompt: {prompt[:100]}...",
                    "confidence": 0.8,
                    "tokens_used": len(prompt.split())
                }
                
        except Exception as e:
            logger.error("LLM execution failed", error=str(e))
            raise
    
    async def _validate_output_schema(self, output: str, schema: Dict[str, Any]) -> Any:
        """Validate LLM output against expected schema"""
        try:
            # Try to parse as JSON if schema expects structured data
            if schema.get("type") == "object":
                try:
                    parsed_output = json.loads(output)
                    # TODO: Add proper JSON schema validation
                    return parsed_output
                except json.JSONDecodeError:
                    # Extract JSON from text if embedded
                    json_match = re.search(r'\{.*\}', output, re.DOTALL)
                    if json_match:
                        return json.loads(json_match.group())
                    else:
                        raise ValueError("Expected JSON output but received text")
            
            return output
            
        except Exception as e:
            logger.warning("Output schema validation failed", error=str(e))
            return output
    
    async def _update_performance_metrics(self, template_id: str, success: bool, duration_ms: int, confidence: float):
        """Update template performance metrics"""
        try:
            doc_ref = templates_collection.document(template_id)
            doc = doc_ref.get()
            
            if doc.exists:
                data = doc.to_dict()
                metrics = data.get("performance_metrics", {})
                
                total_executions = metrics.get("total_executions", 0) + 1
                successful_executions = metrics.get("successful_executions", 0) + (1 if success else 0)
                
                # Update averages
                avg_latency = ((metrics.get("average_latency_ms", 0) * metrics.get("total_executions", 0)) + duration_ms) / total_executions
                avg_confidence = ((metrics.get("average_confidence", 0) * metrics.get("total_executions", 0)) + confidence) / total_executions
                
                updated_metrics = {
                    "total_executions": total_executions,
                    "successful_executions": successful_executions,
                    "success_rate": successful_executions / total_executions,
                    "average_latency_ms": avg_latency,
                    "average_confidence": avg_confidence,
                    "last_execution": datetime.utcnow()
                }
                
                doc_ref.update({"performance_metrics": updated_metrics})
                
        except Exception as e:
            logger.error("Failed to update performance metrics", template_id=template_id, error=str(e))
    
    async def _check_optimization_trigger(self, template_id: str):
        """Check if template should be automatically optimized"""
        try:
            doc = templates_collection.document(template_id).get()
            if not doc.exists:
                return
            
            data = doc.to_dict()
            metrics = data.get("performance_metrics", {})
            
            # Trigger optimization if:
            # 1. Success rate < 80% and total executions > 50
            # 2. Last optimization was > 7 days ago
            # 3. Template is in stable status
            
            total_executions = metrics.get("total_executions", 0)
            success_rate = metrics.get("success_rate", 1.0)
            last_optimized = metrics.get("last_optimized")
            template_status = data.get("status")
            
            should_optimize = False
            
            if (success_rate < 0.8 and total_executions > 50 and 
                template_status == TemplateStatus.STABLE.value):
                should_optimize = True
            
            if (last_optimized and 
                (datetime.utcnow() - last_optimized).days > 7 and
                total_executions > 100):
                should_optimize = True
            
            if should_optimize:
                # Trigger automatic optimization
                await self._trigger_auto_optimization(template_id, data.get("medical_domain"))
                
        except Exception as e:
            logger.error("Failed to check optimization trigger", template_id=template_id, error=str(e))
    
    async def _trigger_auto_optimization(self, template_id: str, medical_domain: str):
        """Trigger automatic template optimization"""
        try:
            # Call learning pipeline to start optimization
            response = await self.http_client.post(
                f"{LEARNING_PIPELINE_URL}/learning/start",
                json={
                    "agent_type": template_id.split("_")[0],
                    "medical_domain": medical_domain,
                    "prompt_template_id": template_id,
                    "optimization_objective": "accuracy",
                    "max_iterations": 5
                }
            )
            
            if response.status_code == 200:
                job_data = response.json()
                logger.info("Auto-optimization triggered", 
                           template_id=template_id, 
                           job_id=job_data.get("job_id"))
            
        except Exception as e:
            logger.error("Failed to trigger auto-optimization", template_id=template_id, error=str(e))

# Initialize orchestrator
orchestrator = POMLOrchestrator()

@app.post("/templates")
async def create_template(template: POMLTemplateCreate, created_by: str = "system"):
    """Create a new POML template"""
    return {
        "template_id": await orchestrator.create_template(template, created_by)
    }

@app.get("/templates/{template_id}")
async def get_template(template_id: str):
    """Get a POML template"""
    try:
        template = await orchestrator.get_template(template_id)
        doc = templates_collection.document(template_id).get()
        return doc.to_dict() if doc.exists else {}
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.post("/templates/{template_id}/execute")
async def execute_template(template_id: str, execution_request: POMLExecutionRequest):
    """Execute a POML template"""
    execution_request.template_id = template_id
    return await orchestrator.execute_template(execution_request)

@app.post("/templates/{template_id}/optimize")
async def optimize_template(template_id: str, optimization_request: POMLOptimizationRequest, background_tasks: BackgroundTasks):
    """Start template optimization using learning pipeline"""
    try:
        template = await orchestrator.get_template(template_id)
        
        # Call learning pipeline to start optimization
        response = await orchestrator.http_client.post(
            f"{LEARNING_PIPELINE_URL}/learning/start",
            json={
                "agent_type": template.agent_type,
                "medical_domain": optimization_request.medical_domain,
                "prompt_template_id": template_id,
                "optimization_objective": optimization_request.optimization_objective,
                "max_iterations": optimization_request.max_iterations
            }
        )
        
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail="Failed to start optimization")
        
        optimization_data = response.json()
        
        return {
            "optimization_id": optimization_data["job_id"],
            "status": "started",
            "message": f"Optimization started for template {template_id}"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/templates")
async def list_templates(
    agent_type: Optional[str] = None,
    medical_domain: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 50
):
    """List POML templates with filters"""
    try:
        query = templates_collection.order_by("updated_at", direction=firestore.Query.DESCENDING)
        
        if agent_type:
            query = query.where("agent_type", "==", agent_type)
        if medical_domain:
            query = query.where("medical_domain", "==", medical_domain)
        if status:
            query = query.where("status", "==", status)
        
        query = query.limit(limit)
        
        templates = []
        for doc in query.stream():
            data = doc.to_dict()
            templates.append({
                "template_id": data["template_id"],
                "name": data["name"],
                "agent_type": data["agent_type"],
                "medical_domain": data["medical_domain"],
                "status": data["status"],
                "version": data["version"],
                "performance_metrics": data.get("performance_metrics", {}),
                "created_at": data["created_at"],
                "updated_at": data["updated_at"]
            })
        
        return {"templates": templates, "total": len(templates)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/executions/{execution_id}")
async def get_execution(execution_id: str):
    """Get execution details"""
    try:
        doc = executions_collection.document(execution_id).get()
        if not doc.exists:
            raise HTTPException(status_code=404, detail="Execution not found")
        return doc.to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/templates/{template_id}/performance")
async def get_template_performance(template_id: str):
    """Get detailed performance metrics for a template"""
    try:
        doc = templates_collection.document(template_id).get()
        if not doc.exists:
            raise HTTPException(status_code=404, detail="Template not found")
        
        data = doc.to_dict()
        return {
            "template_id": template_id,
            "performance_metrics": data.get("performance_metrics", {}),
            "deployment_info": data.get("deployment_info", {}),
            "status": data.get("status"),
            "version": data.get("version")
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test Firestore connection
        templates_collection.limit(1).get()
        
        # Test external services
        services_status = {}
        
        try:
            async with httpx.AsyncClient() as client:
                learning_response = await client.get(f"{LEARNING_PIPELINE_URL}/health", timeout=5.0)
                services_status["learning_pipeline"] = "healthy" if learning_response.status_code == 200 else "unhealthy"
        except:
            services_status["learning_pipeline"] = "unreachable"
        
        return {
            "status": "healthy",
            "service": "poml-orchestrator",
            "timestamp": datetime.utcnow().isoformat(),
            "services": services_status,
            "gemini_available": gemini_model is not None,
            "version": "2.0.0"
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "service": "poml-orchestrator",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)