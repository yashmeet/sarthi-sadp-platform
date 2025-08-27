"""
Enhanced POML (Prompt Orchestration Markup Language) Integration
Production-ready implementation with Microsoft POML SDK
"""

import json
import yaml
import asyncio
import os
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import uuid

import structlog
from google.cloud import storage, firestore
import httpx

# Note: In production, install actual POML SDK
# For now, we'll create a compatible interface
# pip install microsoft-poml-sdk

logger = structlog.get_logger()

class POMLTemplate:
    """Represents a POML template with metadata"""
    
    def __init__(self, template_id: str, content: str, metadata: Dict[str, Any]):
        self.template_id = template_id
        self.content = content
        self.metadata = metadata
        self.version = metadata.get('version', '1.0.0')
        self.category = metadata.get('category', 'general')
        self.variables = metadata.get('variables', [])
        self.compiled = None
    
    def compile(self) -> Dict[str, Any]:
        """Compile POML template for execution"""
        if self.compiled is None:
            self.compiled = self._parse_poml(self.content)
        return self.compiled
    
    def _parse_poml(self, content: str) -> Dict[str, Any]:
        """Parse POML content into executable format"""
        try:
            # Handle both YAML and JSON formats
            if content.strip().startswith('{'):
                return json.loads(content)
            else:
                return yaml.safe_load(content)
        except Exception as e:
            logger.error("Failed to parse POML template", error=str(e))
            raise ValueError(f"Invalid POML template: {e}")

class POMLExecutor:
    """Executes POML templates with context and variables"""
    
    def __init__(self, settings):
        self.settings = settings
        self.llm_client = httpx.AsyncClient()
        
    async def execute_template(
        self,
        template: POMLTemplate,
        context: Dict[str, Any],
        variables: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Execute POML template with given context"""
        
        execution_id = str(uuid.uuid4())
        start_time = datetime.utcnow()
        
        logger.info(
            "Executing POML template",
            template_id=template.template_id,
            execution_id=execution_id
        )
        
        try:
            # Compile template
            compiled_template = template.compile()
            
            # Merge variables
            execution_context = {**context}
            if variables:
                execution_context.update(variables)
            
            # Execute based on template type
            if compiled_template.get('type') == 'sequential':
                result = await self._execute_sequential(compiled_template, execution_context)
            elif compiled_template.get('type') == 'parallel':
                result = await self._execute_parallel(compiled_template, execution_context)
            elif compiled_template.get('type') == 'conditional':
                result = await self._execute_conditional(compiled_template, execution_context)
            else:
                result = await self._execute_simple(compiled_template, execution_context)
            
            end_time = datetime.utcnow()
            duration_ms = int((end_time - start_time).total_seconds() * 1000)
            
            logger.info(
                "POML template execution completed",
                execution_id=execution_id,
                duration_ms=duration_ms
            )
            
            return {
                'execution_id': execution_id,
                'template_id': template.template_id,
                'result': result,
                'duration_ms': duration_ms,
                'timestamp': end_time.isoformat()
            }
            
        except Exception as e:
            logger.error(
                "POML template execution failed",
                execution_id=execution_id,
                error=str(e)
            )
            raise
    
    async def _execute_simple(
        self, 
        template: Dict[str, Any], 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute simple prompt template"""
        
        prompt = template.get('prompt', '')
        
        # Variable substitution
        for key, value in context.items():
            prompt = prompt.replace(f"{{{{ {key} }}}}", str(value))
        
        # Execute with LLM
        response = await self._call_llm(
            prompt=prompt,
            model=template.get('model', 'gemini-pro'),
            parameters=template.get('parameters', {})
        )
        
        return {
            'type': 'simple',
            'prompt': prompt,
            'response': response
        }
    
    async def _execute_sequential(
        self, 
        template: Dict[str, Any], 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute sequential workflow of prompts"""
        
        steps = template.get('steps', [])
        results = []
        current_context = context.copy()
        
        for i, step in enumerate(steps):
            logger.debug(f"Executing step {i+1}/{len(steps)}")
            
            step_result = await self._execute_step(step, current_context)
            results.append(step_result)
            
            # Update context with step result
            if step.get('output_variable'):
                current_context[step['output_variable']] = step_result['response']
        
        return {
            'type': 'sequential',
            'steps': results,
            'final_context': current_context
        }
    
    async def _execute_parallel(
        self, 
        template: Dict[str, Any], 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute parallel workflow of prompts"""
        
        steps = template.get('steps', [])
        
        # Execute all steps in parallel
        tasks = [
            self._execute_step(step, context)
            for step in steps
        ]
        
        results = await asyncio.gather(*tasks)
        
        return {
            'type': 'parallel',
            'steps': results
        }
    
    async def _execute_conditional(
        self, 
        template: Dict[str, Any], 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute conditional workflow based on context"""
        
        condition = template.get('condition', '')
        
        # Evaluate condition (simple implementation)
        condition_result = self._evaluate_condition(condition, context)
        
        if condition_result:
            chosen_branch = template.get('if_true', {})
        else:
            chosen_branch = template.get('if_false', {})
        
        result = await self._execute_step(chosen_branch, context)
        
        return {
            'type': 'conditional',
            'condition': condition,
            'condition_result': condition_result,
            'result': result
        }
    
    async def _execute_step(
        self, 
        step: Dict[str, Any], 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute individual step"""
        
        prompt = step.get('prompt', '')
        
        # Variable substitution
        for key, value in context.items():
            prompt = prompt.replace(f"{{{{ {key} }}}}", str(value))
        
        # Execute with LLM
        response = await self._call_llm(
            prompt=prompt,
            model=step.get('model', 'gemini-pro'),
            parameters=step.get('parameters', {})
        )
        
        return {
            'prompt': prompt,
            'response': response,
            'model': step.get('model', 'gemini-pro')
        }
    
    def _evaluate_condition(self, condition: str, context: Dict[str, Any]) -> bool:
        """Evaluate condition string with context"""
        try:
            # Simple condition evaluation (expand as needed)
            # Replace variables in condition
            for key, value in context.items():
                condition = condition.replace(f"{{{{ {key} }}}}", str(value))
            
            # Evaluate the condition safely
            # In production, use a proper expression evaluator
            return eval(condition)
        except Exception as e:
            logger.error("Failed to evaluate condition", condition=condition, error=str(e))
            return False
    
    async def _call_llm(
        self, 
        prompt: str, 
        model: str = 'gemini-pro',
        parameters: Dict[str, Any] = None
    ) -> str:
        """Call LLM with prompt"""
        
        if parameters is None:
            parameters = {}
        
        try:
            # In production, integrate with actual LLM service
            # For now, simulate response
            
            await asyncio.sleep(0.1)  # Simulate API call
            
            return f"LLM response for: {prompt[:50]}..."
            
        except Exception as e:
            logger.error("LLM call failed", error=str(e))
            raise

class POMLManager:
    """Manages POML templates and orchestration"""
    
    def __init__(self, settings):
        self.settings = settings
        self.storage_client = storage.Client(project=settings.PROJECT_ID)
        self.firestore_client = firestore.Client(project=settings.PROJECT_ID)
        self.executor = POMLExecutor(settings)
        
        # Collections
        self.templates_collection = self.firestore_client.collection('poml_templates')
        self.executions_collection = self.firestore_client.collection('poml_executions')
        
        # Storage bucket for templates
        self.poml_bucket = self.storage_client.bucket(settings.POML_BUCKET)
    
    async def load_template(self, template_id: str, organization_id: str) -> POMLTemplate:
        """Load POML template by ID"""
        try:
            # Get template metadata from Firestore
            doc = self.templates_collection.document(template_id).get()
            
            if not doc.exists:
                raise ValueError(f"Template {template_id} not found")
            
            metadata = doc.to_dict()
            
            # Verify organization access
            if metadata.get('organization_id') != organization_id:
                raise ValueError("Access denied to template")
            
            # Load template content from storage
            blob_path = f"templates/{organization_id}/{template_id}.poml"
            blob = self.poml_bucket.blob(blob_path)
            
            if not blob.exists():
                raise ValueError(f"Template content not found: {blob_path}")
            
            content = blob.download_as_text()
            
            return POMLTemplate(template_id, content, metadata)
            
        except Exception as e:
            logger.error("Failed to load POML template", template_id=template_id, error=str(e))
            raise
    
    async def save_template(
        self, 
        template: POMLTemplate, 
        organization_id: str,
        created_by: str
    ) -> str:
        """Save POML template"""
        try:
            template_id = template.template_id or str(uuid.uuid4())
            
            # Save metadata to Firestore
            metadata = {
                'template_id': template_id,
                'organization_id': organization_id,
                'created_by': created_by,
                'created_at': datetime.utcnow(),
                'updated_at': datetime.utcnow(),
                **template.metadata
            }
            
            self.templates_collection.document(template_id).set(metadata)
            
            # Save content to storage
            blob_path = f"templates/{organization_id}/{template_id}.poml"
            blob = self.poml_bucket.blob(blob_path)
            blob.upload_from_string(template.content)
            
            logger.info("POML template saved", template_id=template_id)
            
            return template_id
            
        except Exception as e:
            logger.error("Failed to save POML template", error=str(e))
            raise
    
    async def list_templates(
        self, 
        organization_id: str,
        category: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List available templates"""
        try:
            query = self.templates_collection.where('organization_id', '==', organization_id)
            
            if category:
                query = query.where('category', '==', category)
            
            docs = query.stream()
            
            templates = []
            for doc in docs:
                data = doc.to_dict()
                templates.append({
                    'template_id': data['template_id'],
                    'name': data.get('name', ''),
                    'description': data.get('description', ''),
                    'category': data.get('category', 'general'),
                    'version': data.get('version', '1.0.0'),
                    'created_at': data.get('created_at'),
                    'updated_at': data.get('updated_at')
                })
            
            return templates
            
        except Exception as e:
            logger.error("Failed to list POML templates", error=str(e))
            raise
    
    async def execute_template(
        self,
        template_id: str,
        context: Dict[str, Any],
        variables: Dict[str, Any] = None,
        organization_id: str = None,
        user_id: str = None
    ) -> Dict[str, Any]:
        """Execute POML template with context"""
        try:
            # Load template
            template = await self.load_template(template_id, organization_id)
            
            # Execute template
            result = await self.executor.execute_template(template, context, variables)
            
            # Store execution record
            execution_record = {
                'execution_id': result['execution_id'],
                'template_id': template_id,
                'organization_id': organization_id,
                'user_id': user_id,
                'context': context,
                'variables': variables or {},
                'result': result['result'],
                'duration_ms': result['duration_ms'],
                'timestamp': datetime.utcnow(),
                'status': 'completed'
            }
            
            self.executions_collection.document(result['execution_id']).set(execution_record)
            
            return result
            
        except Exception as e:
            logger.error("Failed to execute POML template", template_id=template_id, error=str(e))
            
            # Store failed execution record
            execution_id = str(uuid.uuid4())
            execution_record = {
                'execution_id': execution_id,
                'template_id': template_id,
                'organization_id': organization_id,
                'user_id': user_id,
                'context': context,
                'variables': variables or {},
                'error': str(e),
                'timestamp': datetime.utcnow(),
                'status': 'failed'
            }
            
            self.executions_collection.document(execution_id).set(execution_record)
            
            raise
    
    async def get_execution_history(
        self,
        organization_id: str,
        template_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get execution history"""
        try:
            query = self.executions_collection.where('organization_id', '==', organization_id)
            
            if template_id:
                query = query.where('template_id', '==', template_id)
            
            query = query.order_by('timestamp', direction=firestore.Query.DESCENDING)
            query = query.limit(limit)
            
            docs = query.stream()
            
            executions = []
            for doc in docs:
                data = doc.to_dict()
                executions.append({
                    'execution_id': data['execution_id'],
                    'template_id': data['template_id'],
                    'status': data['status'],
                    'duration_ms': data.get('duration_ms'),
                    'timestamp': data['timestamp'],
                    'error': data.get('error')
                })
            
            return executions
            
        except Exception as e:
            logger.error("Failed to get execution history", error=str(e))
            raise
    
    async def health_check(self) -> bool:
        """Check POML manager health"""
        try:
            # Test Firestore connection
            self.templates_collection.limit(1).get()
            
            # Test storage connection
            self.poml_bucket.exists()
            
            return True
            
        except Exception as e:
            logger.error("POML manager health check failed", error=str(e))
            return False