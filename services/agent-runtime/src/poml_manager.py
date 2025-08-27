"""
POML (Prompt Orchestration Markup Language) Manager
Handles prompt templates, versioning, and A/B testing
"""
import json
import yaml
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import structlog
from google.cloud import storage, firestore
import hashlib
import re
from dataclasses import dataclass
from enum import Enum

logger = structlog.get_logger()

class PromptVersion(Enum):
    STABLE = "stable"
    BETA = "beta"
    EXPERIMENTAL = "experimental"
    ARCHIVED = "archived"

@dataclass
class PromptMetrics:
    """Metrics for prompt performance"""
    prompt_id: str
    version: str
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    average_confidence: float = 0.0
    average_latency_ms: float = 0.0
    average_tokens_used: int = 0
    user_satisfaction_score: float = 0.0
    last_executed: Optional[datetime] = None

class POMLManager:
    """Manages POML templates and prompt engineering"""
    
    def __init__(self, settings):
        self.settings = settings
        self.storage_client = storage.Client(project=settings.PROJECT_ID)
        self.firestore_client = firestore.Client(project=settings.PROJECT_ID)
        self.poml_bucket = self.storage_client.bucket(settings.POML_BUCKET)
        self.prompts_collection = self.firestore_client.collection('poml_prompts')
        self.metrics_collection = self.firestore_client.collection('prompt_metrics')
        self.experiments_collection = self.firestore_client.collection('prompt_experiments')
        
        # Cache for frequently used prompts
        self.prompt_cache: Dict[str, Dict] = {}
        self.metrics_cache: Dict[str, PromptMetrics] = {}
    
    async def initialize(self):
        """Initialize POML manager"""
        try:
            # Load default prompts
            await self.load_default_prompts()
            
            # Load metrics
            await self.load_metrics()
            
            logger.info("POML Manager initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize POML Manager", error=str(e))
            raise
    
    async def load_default_prompts(self):
        """Load default prompt templates"""
        try:
            # Load stable prompts into cache
            docs = self.prompts_collection.where('version', '==', PromptVersion.STABLE.value).stream()
            
            for doc in docs:
                prompt_data = doc.to_dict()
                self.prompt_cache[doc.id] = prompt_data
            
            logger.info(f"Loaded {len(self.prompt_cache)} default prompts")
            
        except Exception as e:
            logger.error(f"Failed to load default prompts", error=str(e))
    
    async def create_prompt(self, prompt_config: Dict[str, Any]) -> str:
        """Create a new POML prompt template"""
        try:
            # Validate prompt configuration
            required_fields = ['name', 'agent_type', 'template', 'variables', 'description']
            for field in required_fields:
                if field not in prompt_config:
                    raise ValueError(f"Missing required field: {field}")
            
            # Generate prompt ID
            prompt_id = f"{prompt_config['agent_type']}_{prompt_config['name']}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
            
            # Parse and validate POML template
            parsed_template = self.parse_poml_template(prompt_config['template'])
            
            # Add metadata
            prompt_data = {
                **prompt_config,
                'id': prompt_id,
                'version': PromptVersion.EXPERIMENTAL.value,
                'created_at': datetime.utcnow(),
                'updated_at': datetime.utcnow(),
                'hash': hashlib.sha256(prompt_config['template'].encode()).hexdigest(),
                'parsed_template': parsed_template,
                'active': True
            }
            
            # Store in Firestore
            self.prompts_collection.document(prompt_id).set(prompt_data)
            
            # Upload to Cloud Storage
            await self.upload_prompt_template(prompt_id, prompt_config['template'])
            
            # Initialize metrics
            await self.initialize_prompt_metrics(prompt_id)
            
            logger.info(f"Created prompt: {prompt_id}")
            return prompt_id
            
        except Exception as e:
            logger.error(f"Failed to create prompt", error=str(e))
            raise
    
    def parse_poml_template(self, template: str) -> Dict[str, Any]:
        """Parse POML template and extract structure"""
        try:
            # POML format example:
            # <prompt version="1.0">
            #   <system>You are a medical assistant...</system>
            #   <context>{{patient_data}}</context>
            #   <task>{{task_description}}</task>
            #   <output format="json">{{expected_output}}</output>
            # </prompt>
            
            parsed = {
                'sections': [],
                'variables': [],
                'constraints': [],
                'output_format': None
            }
            
            # Extract sections
            sections = re.findall(r'<(\w+)>(.*?)</\1>', template, re.DOTALL)
            for section_name, content in sections:
                parsed['sections'].append({
                    'name': section_name,
                    'content': content.strip()
                })
            
            # Extract variables ({{variable_name}})
            variables = re.findall(r'\{\{(\w+)\}\}', template)
            parsed['variables'] = list(set(variables))
            
            # Extract output format
            output_match = re.search(r'<output\s+format="(\w+)">', template)
            if output_match:
                parsed['output_format'] = output_match.group(1)
            
            return parsed
            
        except Exception as e:
            logger.error(f"Failed to parse POML template", error=str(e))
            raise
    
    async def upload_prompt_template(self, prompt_id: str, template: str):
        """Upload prompt template to Cloud Storage"""
        try:
            blob_name = f"prompts/{prompt_id}/template.poml"
            blob = self.poml_bucket.blob(blob_name)
            blob.upload_from_string(template)
            
            # Update location in Firestore
            self.prompts_collection.document(prompt_id).update({
                'template_location': f"gs://{self.settings.POML_BUCKET}/{blob_name}"
            })
            
        except Exception as e:
            logger.error(f"Failed to upload prompt template", error=str(e))
            raise
    
    async def get_prompt(self, prompt_id: str, version: Optional[str] = None) -> Dict[str, Any]:
        """Get a prompt template"""
        try:
            # Check cache first
            if prompt_id in self.prompt_cache and not version:
                return self.prompt_cache[prompt_id]
            
            # Query Firestore
            query = self.prompts_collection.document(prompt_id)
            
            if version:
                # Get specific version
                query = self.prompts_collection.where('name', '==', prompt_id).where('version', '==', version).limit(1)
                docs = query.stream()
                doc = next(docs, None)
                if doc:
                    return doc.to_dict()
            else:
                doc = query.get()
                if doc.exists:
                    return doc.to_dict()
            
            raise ValueError(f"Prompt {prompt_id} not found")
            
        except Exception as e:
            logger.error(f"Failed to get prompt", prompt_id=prompt_id, error=str(e))
            raise
    
    async def update_prompt(self, prompt_id: str, updates: Dict[str, Any], create_version: bool = True):
        """Update a prompt template"""
        try:
            # Get current prompt
            current = await self.get_prompt(prompt_id)
            
            if create_version:
                # Create a new version
                new_version_id = f"{prompt_id}_v{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
                
                # Copy current to new version
                versioned_prompt = {
                    **current,
                    'id': new_version_id,
                    'parent_id': prompt_id,
                    'version': PromptVersion.EXPERIMENTAL.value,
                    'created_at': datetime.utcnow()
                }
                
                # Apply updates
                versioned_prompt.update(updates)
                
                # Save new version
                self.prompts_collection.document(new_version_id).set(versioned_prompt)
                
                return new_version_id
            else:
                # Update in place
                updates['updated_at'] = datetime.utcnow()
                self.prompts_collection.document(prompt_id).update(updates)
                
                # Clear cache
                if prompt_id in self.prompt_cache:
                    del self.prompt_cache[prompt_id]
                
                return prompt_id
                
        except Exception as e:
            logger.error(f"Failed to update prompt", prompt_id=prompt_id, error=str(e))
            raise
    
    async def render_prompt(self, prompt_id: str, variables: Dict[str, Any]) -> str:
        """Render a prompt template with variables"""
        try:
            # Get prompt template
            prompt_data = await self.get_prompt(prompt_id)
            template = prompt_data['template']
            
            # Replace variables
            rendered = template
            for var_name, var_value in variables.items():
                placeholder = f"{{{{{var_name}}}}}"
                rendered = rendered.replace(placeholder, str(var_value))
            
            # Check for missing variables
            missing_vars = re.findall(r'\{\{(\w+)\}\}', rendered)
            if missing_vars:
                logger.warning(f"Missing variables in prompt", 
                             prompt_id=prompt_id, 
                             missing=missing_vars)
            
            return rendered
            
        except Exception as e:
            logger.error(f"Failed to render prompt", prompt_id=prompt_id, error=str(e))
            raise
    
    async def initialize_prompt_metrics(self, prompt_id: str):
        """Initialize metrics for a new prompt"""
        try:
            metrics = PromptMetrics(
                prompt_id=prompt_id,
                version=PromptVersion.EXPERIMENTAL.value
            )
            
            self.metrics_collection.document(prompt_id).set({
                'prompt_id': metrics.prompt_id,
                'version': metrics.version,
                'total_executions': 0,
                'successful_executions': 0,
                'failed_executions': 0,
                'average_confidence': 0.0,
                'average_latency_ms': 0.0,
                'average_tokens_used': 0,
                'user_satisfaction_score': 0.0,
                'last_executed': None
            })
            
            self.metrics_cache[prompt_id] = metrics
            
        except Exception as e:
            logger.error(f"Failed to initialize prompt metrics", error=str(e))
    
    async def track_execution(self, prompt_id: str, execution_data: Dict[str, Any]):
        """Track prompt execution metrics"""
        try:
            # Get current metrics
            metrics = self.metrics_cache.get(prompt_id)
            if not metrics:
                doc = self.metrics_collection.document(prompt_id).get()
                if doc.exists:
                    data = doc.to_dict()
                    metrics = PromptMetrics(**data)
                else:
                    await self.initialize_prompt_metrics(prompt_id)
                    metrics = self.metrics_cache[prompt_id]
            
            # Update metrics
            metrics.total_executions += 1
            
            if execution_data.get('success', False):
                metrics.successful_executions += 1
            else:
                metrics.failed_executions += 1
            
            # Update averages
            if 'confidence' in execution_data:
                metrics.average_confidence = (
                    (metrics.average_confidence * (metrics.total_executions - 1) + 
                     execution_data['confidence']) / metrics.total_executions
                )
            
            if 'latency_ms' in execution_data:
                metrics.average_latency_ms = (
                    (metrics.average_latency_ms * (metrics.total_executions - 1) + 
                     execution_data['latency_ms']) / metrics.total_executions
                )
            
            if 'tokens_used' in execution_data:
                metrics.average_tokens_used = int(
                    (metrics.average_tokens_used * (metrics.total_executions - 1) + 
                     execution_data['tokens_used']) / metrics.total_executions
                )
            
            metrics.last_executed = datetime.utcnow()
            
            # Update Firestore
            self.metrics_collection.document(prompt_id).update({
                'total_executions': metrics.total_executions,
                'successful_executions': metrics.successful_executions,
                'failed_executions': metrics.failed_executions,
                'average_confidence': metrics.average_confidence,
                'average_latency_ms': metrics.average_latency_ms,
                'average_tokens_used': metrics.average_tokens_used,
                'last_executed': metrics.last_executed
            })
            
            # Update cache
            self.metrics_cache[prompt_id] = metrics
            
        except Exception as e:
            logger.error(f"Failed to track execution", prompt_id=prompt_id, error=str(e))
    
    async def get_metrics(self, prompt_id: str) -> PromptMetrics:
        """Get metrics for a prompt"""
        try:
            if prompt_id in self.metrics_cache:
                return self.metrics_cache[prompt_id]
            
            doc = self.metrics_collection.document(prompt_id).get()
            if doc.exists:
                data = doc.to_dict()
                metrics = PromptMetrics(**data)
                self.metrics_cache[prompt_id] = metrics
                return metrics
            
            return PromptMetrics(prompt_id=prompt_id, version="unknown")
            
        except Exception as e:
            logger.error(f"Failed to get metrics", prompt_id=prompt_id, error=str(e))
            return PromptMetrics(prompt_id=prompt_id, version="unknown")
    
    async def create_ab_test(self, test_config: Dict[str, Any]) -> str:
        """Create an A/B test for prompts"""
        try:
            # Validate test configuration
            required_fields = ['name', 'prompt_a', 'prompt_b', 'traffic_split', 'success_metric']
            for field in required_fields:
                if field not in test_config:
                    raise ValueError(f"Missing required field: {field}")
            
            # Generate test ID
            test_id = f"ab_test_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
            
            # Create test document
            test_data = {
                **test_config,
                'id': test_id,
                'status': 'active',
                'created_at': datetime.utcnow(),
                'results': {
                    'prompt_a': {'executions': 0, 'successes': 0, 'metric_sum': 0},
                    'prompt_b': {'executions': 0, 'successes': 0, 'metric_sum': 0}
                }
            }
            
            self.experiments_collection.document(test_id).set(test_data)
            
            logger.info(f"Created A/B test: {test_id}")
            return test_id
            
        except Exception as e:
            logger.error(f"Failed to create A/B test", error=str(e))
            raise
    
    async def get_ab_test_prompt(self, test_id: str) -> Tuple[str, str]:
        """Get prompt for A/B test based on traffic split"""
        try:
            doc = self.experiments_collection.document(test_id).get()
            if not doc.exists:
                raise ValueError(f"A/B test {test_id} not found")
            
            test_data = doc.to_dict()
            
            # Determine which prompt to use based on traffic split
            import random
            if random.random() < test_data['traffic_split']:
                return test_data['prompt_a'], 'a'
            else:
                return test_data['prompt_b'], 'b'
                
        except Exception as e:
            logger.error(f"Failed to get A/B test prompt", test_id=test_id, error=str(e))
            raise
    
    async def update_ab_test_results(self, test_id: str, variant: str, result: Dict[str, Any]):
        """Update A/B test results"""
        try:
            doc_ref = self.experiments_collection.document(test_id)
            doc = doc_ref.get()
            
            if not doc.exists:
                return
            
            test_data = doc.to_dict()
            results = test_data['results']
            
            # Update variant results
            variant_key = f"prompt_{variant}"
            results[variant_key]['executions'] += 1
            
            if result.get('success', False):
                results[variant_key]['successes'] += 1
            
            if 'metric_value' in result:
                results[variant_key]['metric_sum'] += result['metric_value']
            
            # Update document
            doc_ref.update({'results': results})
            
        except Exception as e:
            logger.error(f"Failed to update A/B test results", test_id=test_id, error=str(e))
    
    async def get_ab_test_results(self, test_id: str) -> Dict[str, Any]:
        """Get A/B test results with statistical analysis"""
        try:
            doc = self.experiments_collection.document(test_id).get()
            if not doc.exists:
                raise ValueError(f"A/B test {test_id} not found")
            
            test_data = doc.to_dict()
            results = test_data['results']
            
            # Calculate statistics
            analysis = {
                'test_id': test_id,
                'test_name': test_data['name'],
                'status': test_data['status'],
                'created_at': test_data['created_at'],
                'variants': {}
            }
            
            for variant in ['prompt_a', 'prompt_b']:
                variant_data = results[variant]
                executions = variant_data['executions']
                
                if executions > 0:
                    success_rate = variant_data['successes'] / executions
                    avg_metric = variant_data['metric_sum'] / executions
                else:
                    success_rate = 0
                    avg_metric = 0
                
                analysis['variants'][variant] = {
                    'executions': executions,
                    'success_rate': success_rate,
                    'average_metric': avg_metric
                }
            
            # Determine winner if enough data
            if (results['prompt_a']['executions'] >= 100 and 
                results['prompt_b']['executions'] >= 100):
                
                a_success = analysis['variants']['prompt_a']['success_rate']
                b_success = analysis['variants']['prompt_b']['success_rate']
                
                if abs(a_success - b_success) > 0.05:  # 5% difference threshold
                    analysis['winner'] = 'prompt_a' if a_success > b_success else 'prompt_b'
                    analysis['confidence'] = abs(a_success - b_success)
                else:
                    analysis['winner'] = 'no_significant_difference'
                    analysis['confidence'] = 0
            else:
                analysis['winner'] = 'insufficient_data'
                analysis['confidence'] = 0
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to get A/B test results", test_id=test_id, error=str(e))
            raise
    
    async def load_metrics(self):
        """Load all metrics into cache"""
        try:
            docs = self.metrics_collection.stream()
            for doc in docs:
                data = doc.to_dict()
                self.metrics_cache[doc.id] = PromptMetrics(**data)
            
            logger.info(f"Loaded {len(self.metrics_cache)} prompt metrics")
            
        except Exception as e:
            logger.error(f"Failed to load metrics", error=str(e))
    
    async def promote_prompt(self, prompt_id: str, new_version: PromptVersion):
        """Promote a prompt to a new version (e.g., experimental to stable)"""
        try:
            self.prompts_collection.document(prompt_id).update({
                'version': new_version.value,
                'promoted_at': datetime.utcnow()
            })
            
            # Clear cache
            if prompt_id in self.prompt_cache:
                del self.prompt_cache[prompt_id]
            
            logger.info(f"Promoted prompt {prompt_id} to {new_version.value}")
            
        except Exception as e:
            logger.error(f"Failed to promote prompt", prompt_id=prompt_id, error=str(e))
            raise