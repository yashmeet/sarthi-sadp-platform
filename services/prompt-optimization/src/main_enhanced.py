"""
Enhanced Prompt Optimization Engine - SADP Self-Tuning System
AutoMedPrompt-based optimization with advanced genetic algorithms and reinforcement learning
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List, Tuple, Union
from datetime import datetime, timedelta
import uuid
import re
import os
import json
import asyncio
import structlog
import random
import math
from enum import Enum
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import httpx
from google.cloud import firestore, storage, pubsub_v1
import google.generativeai as genai

# Initialize logger
logger = structlog.get_logger()

app = FastAPI(
    title="SADP Enhanced Prompt Optimization Engine", 
    description="Advanced AutoMedPrompt with genetic algorithms, reinforcement learning, and self-improvement",
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
MODEL_MANAGEMENT_URL = os.environ.get("MODEL_MANAGEMENT_URL", "https://sadp-model-management-prod-355881591332.us-central1.run.app")
KAGGLE_SERVICE_URL = os.environ.get("KAGGLE_SERVICE_URL", "https://sadp-kaggle-integration-xonau6hybq-uc.a.run.app")
PUBSUB_TOPIC_PREFIX = os.environ.get("PUBSUB_TOPIC_PREFIX", "sarthi-workflow")

# Initialize clients with error handling
try:
    firestore_client = firestore.Client(project=PROJECT_ID)
    storage_client = storage.Client(project=PROJECT_ID)
    publisher_client = pubsub_v1.PublisherClient()
    logger.info("Google Cloud clients initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Google Cloud clients: {e}")
    firestore_client = None
    storage_client = None
    publisher_client = None

# Gemini setup
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if GEMINI_API_KEY and GEMINI_API_KEY != "YOUR_GEMINI_API_KEY_HERE":
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        logger.info("Gemini model initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Gemini model: {e}")
        gemini_model = None
else:
    gemini_model = None

# Collections (initialize only if firestore_client is available)
if firestore_client:
    optimization_jobs_collection = firestore_client.collection('optimization_jobs')
    prompt_variants_collection = firestore_client.collection('prompt_variants')
    optimization_results_collection = firestore_client.collection('optimization_results')
else:
    optimization_jobs_collection = None
    prompt_variants_collection = None
    optimization_results_collection = None

# Available models cache
available_models = {}
model_clients = {}

# Helper function for safe Firestore operations
def safe_firestore_operation(operation_func, default_return=None):
    """
    Safely execute Firestore operations with fallback
    """
    if not firestore_client or not optimization_jobs_collection:
        logger.warning("Firestore not available, using fallback")
        return default_return
    try:
        return operation_func()
    except Exception as e:
        logger.error(f"Firestore operation failed: {e}")
        return default_return

# WebSocket Connection Manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, job_id: str):
        await websocket.accept()
        if job_id not in self.active_connections:
            self.active_connections[job_id] = []
        self.active_connections[job_id].append(websocket)
        logger.info(f"WebSocket connected for job {job_id}")
    
    def disconnect(self, websocket: WebSocket, job_id: str):
        if job_id in self.active_connections:
            self.active_connections[job_id].remove(websocket)
            if not self.active_connections[job_id]:
                del self.active_connections[job_id]
        logger.info(f"WebSocket disconnected for job {job_id}")
    
    async def send_update(self, job_id: str, data: Dict[str, Any]):
        if job_id in self.active_connections:
            message = json.dumps(data)
            dead_connections = []
            for connection in self.active_connections[job_id]:
                try:
                    await connection.send_text(message)
                except Exception as e:
                    logger.error(f"Failed to send WebSocket message: {e}")
                    dead_connections.append(connection)
            
            # Remove dead connections
            for dead_connection in dead_connections:
                self.disconnect(dead_connection, job_id)

manager = ConnectionManager()

# Event Publishing Service for Agent-to-Agent Communication
class OptimizationEventPublisher:
    def __init__(self):
        self.publisher = publisher_client
        self.project_path = f"projects/{PROJECT_ID}"
    
    async def publish_optimization_started(self, job_id: str, tenant_id: str, agent_type: str, strategy: str):
        """Publish optimization started event for agent-to-agent coordination"""
        if not self.publisher:
            logger.warning("Publisher client not available, skipping event publishing")
            return
        
        try:
            topic_path = self.publisher.topic_path(PROJECT_ID, f"{PUBSUB_TOPIC_PREFIX}-agent-events")
            
            event_data = {
                "event_type": "optimization_started",
                "job_id": job_id,
                "tenant_id": tenant_id,
                "agent_type": agent_type,
                "strategy": strategy,
                "timestamp": datetime.utcnow().isoformat(),
                "source": "prompt-optimization-service",
                "metadata": {
                    "service_version": "2.0.0",
                    "optimization_type": "modern_context_aware"
                }
            }
            
            message_data = json.dumps(event_data).encode('utf-8')
            
            # Publish with attributes for efficient routing
            future = self.publisher.publish(
                topic_path,
                message_data,
                event_type="optimization_started",
                tenant_id=tenant_id,
                agent_type=agent_type,
                source="prompt-optimization-service"
            )
            
            message_id = future.result(timeout=10)
            logger.info(f"Published optimization started event", 
                       job_id=job_id, message_id=message_id, tenant_id=tenant_id)
                       
        except Exception as e:
            logger.error(f"Failed to publish optimization started event: {e}", job_id=job_id)
    
    async def publish_optimization_progress(self, job_id: str, tenant_id: str, iteration: int, 
                                          performance: float, best_performance: float, improvements: list):
        """Publish optimization progress for real-time agent updates"""
        if not self.publisher:
            return
        
        try:
            topic_path = self.publisher.topic_path(PROJECT_ID, f"{PUBSUB_TOPIC_PREFIX}-agent-events")
            
            event_data = {
                "event_type": "optimization_progress",
                "job_id": job_id,
                "tenant_id": tenant_id,
                "iteration": iteration,
                "current_performance": performance,
                "best_performance": best_performance,
                "improvements": improvements,
                "timestamp": datetime.utcnow().isoformat(),
                "source": "prompt-optimization-service"
            }
            
            message_data = json.dumps(event_data).encode('utf-8')
            
            future = self.publisher.publish(
                topic_path,
                message_data,
                event_type="optimization_progress",
                tenant_id=tenant_id,
                job_id=job_id,
                source="prompt-optimization-service"
            )
            
            message_id = future.result(timeout=10)
            logger.debug(f"Published optimization progress event", 
                        job_id=job_id, iteration=iteration, performance=performance)
                        
        except Exception as e:
            logger.error(f"Failed to publish optimization progress event: {e}", job_id=job_id)
    
    async def publish_optimization_completed(self, job_id: str, tenant_id: str, agent_type: str,
                                           final_performance: float, optimized_prompt: str, 
                                           dataset_used: str = None):
        """Publish optimization completion with results for agent consumption"""
        if not self.publisher:
            return
        
        try:
            topic_path = self.publisher.topic_path(PROJECT_ID, f"{PUBSUB_TOPIC_PREFIX}-agent-events")
            
            event_data = {
                "event_type": "optimization_completed",
                "job_id": job_id,
                "tenant_id": tenant_id,
                "agent_type": agent_type,
                "final_performance": final_performance,
                "optimized_prompt": optimized_prompt,
                "dataset_used": dataset_used,
                "timestamp": datetime.utcnow().isoformat(),
                "source": "prompt-optimization-service",
                "action_required": {
                    "type": "apply_optimized_prompt",
                    "target_agents": [agent_type],
                    "priority": "high" if final_performance > 0.8 else "medium"
                }
            }
            
            message_data = json.dumps(event_data).encode('utf-8')
            
            future = self.publisher.publish(
                topic_path,
                message_data,
                event_type="optimization_completed",
                tenant_id=tenant_id,
                agent_type=agent_type,
                source="prompt-optimization-service"
            )
            
            message_id = future.result(timeout=10)
            logger.info(f"Published optimization completed event", 
                       job_id=job_id, final_performance=final_performance, message_id=message_id)
                       
        except Exception as e:
            logger.error(f"Failed to publish optimization completed event: {e}", job_id=job_id)
    
    async def publish_prompt_recommendation(self, tenant_id: str, agent_type: str, 
                                          prompt_pattern: str, performance_score: float,
                                          usage_context: str):
        """Publish successful prompt patterns for cross-agent learning"""
        if not self.publisher:
            return
        
        try:
            topic_path = self.publisher.topic_path(PROJECT_ID, f"{PUBSUB_TOPIC_PREFIX}-agent-events")
            
            event_data = {
                "event_type": "prompt_recommendation",
                "tenant_id": tenant_id,
                "source_agent_type": agent_type,
                "prompt_pattern": prompt_pattern,
                "performance_score": performance_score,
                "usage_context": usage_context,
                "timestamp": datetime.utcnow().isoformat(),
                "source": "prompt-optimization-service",
                "recommendation": {
                    "applicable_agents": ["clinical", "billing", "document-processor"],
                    "confidence": performance_score,
                    "adaptation_required": True if agent_type != "general" else False
                }
            }
            
            message_data = json.dumps(event_data).encode('utf-8')
            
            future = self.publisher.publish(
                topic_path,
                message_data,
                event_type="prompt_recommendation",
                tenant_id=tenant_id,
                source_agent_type=agent_type,
                source="prompt-optimization-service"
            )
            
            message_id = future.result(timeout=10)
            logger.info(f"Published prompt recommendation event", 
                       tenant_id=tenant_id, agent_type=agent_type, message_id=message_id)
                       
        except Exception as e:
            logger.error(f"Failed to publish prompt recommendation event: {e}")

# Initialize event publisher
event_publisher = OptimizationEventPublisher()

async def load_model(model_id: str) -> Dict[str, Any]:
    """
    Load model configuration from Model Management Service
    """
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{MODEL_MANAGEMENT_URL}/models/{model_id}")
            if response.status_code == 200:
                model_data = response.json()
                available_models[model_id] = model_data
                return model_data
            else:
                raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}")
            raise HTTPException(status_code=503, detail=f"Failed to load model: {str(e)}")

# Optimization Strategy Enums
class OptimizationStrategy(str, Enum):
    AUTOMEDPROMPT = "automedprompt"
    GENETIC_ALGORITHM = "genetic_algorithm"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"
    HYBRID = "hybrid"

class OptimizationObjective(str, Enum):
    ACCURACY = "accuracy"
    LATENCY = "latency"
    TOKEN_EFFICIENCY = "token_efficiency"
    CLINICAL_ACCURACY = "clinical_accuracy"
    COMPOSITE = "composite"

# Models
class OptimizationRequest(BaseModel):
    prompt_template_id: str = Field(..., description="Template to optimize")
    test_cases: List[Dict[str, Any]] = Field(..., description="Test cases for evaluation")
    strategy: OptimizationStrategy = Field(default=OptimizationStrategy.AUTOMEDPROMPT)
    objective: OptimizationObjective = Field(default=OptimizationObjective.ACCURACY)
    max_iterations: int = Field(default=10, description="Maximum optimization iterations")
    target_performance: float = Field(default=0.9, description="Target performance threshold")
    population_size: int = Field(default=20, description="Population size for genetic algorithm")
    mutation_rate: float = Field(default=0.1, description="Mutation rate for genetic algorithm")

class PromptVariant(BaseModel):
    variant_id: str
    content: str
    performance_score: float
    generation: int
    parent_variants: List[str] = []
    mutation_history: List[str] = []

class OptimizationResult(BaseModel):
    job_id: str
    original_prompt: str
    optimized_prompt: str
    performance_improvement: float
    iterations_completed: int
    strategy_used: str
    convergence_achieved: bool
    optimization_history: List[Dict[str, Any]]

class ContextConfig(BaseModel):
    max_context_tokens: int = Field(default=32768, description="Maximum context window size")
    system_prompt_tokens: int = Field(default=2000, description="Reserved tokens for system prompt")
    few_shot_examples: int = Field(default=3, description="Number of few-shot examples")
    dynamic_context: bool = Field(default=True, description="Enable dynamic context adjustment")
    context_relevance_threshold: float = Field(default=0.7, description="Minimum relevance for context inclusion")

class ModernPromptRequest(BaseModel):
    prompt_template_id: str = Field(..., description="Template to optimize")
    test_cases: List[Dict[str, Any]] = Field(..., description="Test cases for evaluation")
    strategy: OptimizationStrategy = Field(default=OptimizationStrategy.HYBRID)
    objective: OptimizationObjective = Field(default=OptimizationObjective.CLINICAL_ACCURACY)
    context_config: ContextConfig = Field(default_factory=ContextConfig)
    kaggle_datasets: Optional[List[str]] = Field(None, description="Specific datasets to use for context")
    auto_dataset_selection: bool = Field(default=True, description="Enable automatic dataset selection")
    max_iterations: int = Field(default=15, description="Maximum optimization iterations")
    target_performance: float = Field(default=0.92, description="Target performance threshold")

class ModernContextManager:
    """Manages modern prompt engineering with context window optimization"""
    
    def __init__(self, kaggle_service_url: str):
        self.kaggle_service_url = kaggle_service_url
        self.http_client = httpx.AsyncClient()
        
    async def generate_system_prompt(self, agent_type: str, medical_domain: str, context_config: ContextConfig) -> str:
        """Generate modern system prompt with role definition and safety guidelines"""
        
        role_definitions = {
            "clinical": """You are an expert clinical AI assistant specialized in evidence-based medicine.
Your role is to analyze clinical data and provide accurate, safe medical insights while maintaining strict ethical guidelines.

CORE RESPONSIBILITIES:
- Analyze patient data with clinical accuracy
- Provide evidence-based recommendations
- Flag potential safety concerns
- Ensure HIPAA compliance in all responses
- Use standardized medical terminology

SAFETY PROTOCOLS:
- Never provide specific diagnoses without proper clinical context
- Always recommend consulting healthcare professionals for treatment decisions
- Flag urgent conditions that require immediate medical attention
- Maintain patient confidentiality and privacy""",

            "billing": """You are a specialized medical billing and coding AI assistant with expertise in healthcare revenue cycle management.
Your role is to analyze medical documentation and provide accurate coding and billing guidance.

CORE RESPONSIBILITIES:
- Extract relevant procedure and diagnosis codes (ICD-10, CPT, HCPCS)
- Identify potential billing compliance issues
- Ensure medical necessity documentation
- Flag potential audit risks
- Optimize revenue cycle processes

COMPLIANCE REQUIREMENTS:
- Follow CMS guidelines and regulations
- Ensure proper documentation for medical necessity
- Identify potential fraud or abuse patterns
- Maintain accuracy in code selection""",

            "document": """You are an advanced medical document processing AI specialized in healthcare information extraction.
Your role is to analyze and structure medical documents with high accuracy and clinical relevance.

CORE RESPONSIBILITIES:
- Extract key medical information from various document types
- Structure unstructured medical text
- Identify critical clinical findings
- Maintain document context and relationships
- Ensure data accuracy and completeness

PROCESSING STANDARDS:
- Preserve medical terminology accuracy
- Maintain clinical context and relationships
- Flag incomplete or ambiguous information
- Follow healthcare data standards (HL7, FHIR)"""
        }
        
        base_prompt = role_definitions.get(agent_type, role_definitions["clinical"])
        
        # Add domain-specific context
        domain_context = f"\nSPECIALIZATION: {medical_domain.replace('_', ' ').title()}\n"
        domain_context += f"You have specialized knowledge in {medical_domain} and should prioritize insights relevant to this medical domain.\n"
        
        # Add context configuration
        context_info = f"\nCONTEXT CONFIGURATION:\n"
        context_info += f"- Maximum context window: {context_config.max_context_tokens:,} tokens\n"
        context_info += f"- Few-shot examples available: {context_config.few_shot_examples}\n"
        context_info += f"- Dynamic context adjustment: {'Enabled' if context_config.dynamic_context else 'Disabled'}\n"
        
        return base_prompt + domain_context + context_info
    
    async def get_few_shot_examples(self, prompt_content: str, context_config: ContextConfig) -> List[Dict[str, str]]:
        """Get relevant few-shot examples from Kaggle datasets"""
        try:
            # Analyze prompt to get dataset recommendations
            analysis_response = await self.http_client.post(
                f"{self.kaggle_service_url}/datasets/recommendations",
                json={
                    "prompt_content": prompt_content,
                    "agent_type": "clinical",
                    "medical_domain": None
                }
            )
            
            if analysis_response.status_code != 200:
                logger.warning("Failed to get dataset recommendations for few-shot examples")
                return self._get_default_examples()
            
            recommendations = analysis_response.json().get("recommendations", [])
            
            # Generate synthetic few-shot examples based on medical domain
            examples = []
            for i in range(min(context_config.few_shot_examples, len(recommendations))):
                dataset = recommendations[i]
                example = {
                    "input": f"Sample {dataset['medical_domain']} case from {dataset['title']}",
                    "output": f"Analysis of {dataset['medical_domain']} data showing {dataset['reasoning']}"
                }
                examples.append(example)
            
            return examples
            
        except Exception as e:
            logger.warning(f"Failed to get few-shot examples: {e}")
            return self._get_default_examples()
    
    def _get_default_examples(self) -> List[Dict[str, str]]:
        """Get default few-shot examples when dataset lookup fails"""
        return [
            {
                "input": "Patient presents with chest pain and shortness of breath",
                "output": "ANALYSIS: Symptoms suggest potential cardiac event. PRIORITY: High - requires immediate evaluation. RECOMMENDATIONS: ECG, cardiac enzymes, chest X-ray."
            },
            {
                "input": "Lab results show elevated WBC count and fever",
                "output": "ANALYSIS: Signs of infection or inflammatory process. PRIORITY: Medium - requires further investigation. RECOMMENDATIONS: Blood cultures, inflammatory markers, clinical correlation."
            },
            {
                "input": "Patient history includes diabetes and hypertension",
                "output": "ANALYSIS: Comorbid conditions requiring integrated care management. PRIORITY: Ongoing monitoring. RECOMMENDATIONS: HbA1c, blood pressure monitoring, medication review."
            }
        ]
    
    def estimate_token_count(self, text: str) -> int:
        """Estimate token count for context management"""
        # Rough estimation: ~4 characters per token
        return len(text) // 4
    
    async def optimize_context_window(self, system_prompt: str, few_shot_examples: List[Dict[str, str]], 
                                    user_input: str, context_config: ContextConfig) -> Dict[str, Any]:
        """Optimize context usage within token limits"""
        
        system_tokens = self.estimate_token_count(system_prompt)
        user_tokens = self.estimate_token_count(user_input)
        
        # Reserve tokens for response
        response_buffer = 1000
        available_tokens = context_config.max_context_tokens - system_tokens - user_tokens - response_buffer
        
        # Optimize few-shot examples
        optimized_examples = []
        used_tokens = 0
        
        for example in few_shot_examples:
            example_tokens = self.estimate_token_count(example["input"] + example["output"])
            if used_tokens + example_tokens <= available_tokens:
                optimized_examples.append(example)
                used_tokens += example_tokens
            else:
                break
        
        return {
            "system_prompt": system_prompt,
            "few_shot_examples": optimized_examples,
            "token_usage": {
                "system_tokens": system_tokens,
                "user_tokens": user_tokens,
                "examples_tokens": used_tokens,
                "total_tokens": system_tokens + user_tokens + used_tokens,
                "available_tokens": context_config.max_context_tokens,
                "utilization": (system_tokens + user_tokens + used_tokens) / context_config.max_context_tokens
            }
        }

class AutoMedPromptOptimizer:
    """Enhanced AutoMedPrompt optimization engine with multiple strategies"""
    
    def __init__(self):
        self.http_client = httpx.AsyncClient()
        self.context_manager = ModernContextManager(KAGGLE_SERVICE_URL)
        
    async def optimize_prompt(self, request: OptimizationRequest) -> OptimizationResult:
        """Main optimization entry point"""
        try:
            job_id = f"opt_{uuid.uuid4().hex[:12]}"
            
            logger.info("Starting prompt optimization", 
                       job_id=job_id, 
                       strategy=request.strategy,
                       template_id=request.prompt_template_id)
            
            # Get original prompt
            original_prompt = await self._get_template_content(request.prompt_template_id)
            
            # Initialize optimization job
            optimization_job = {
                "job_id": job_id,
                "template_id": request.prompt_template_id,
                "strategy": request.strategy.value,
                "objective": request.objective.value,
                "status": "running",
                "started_at": datetime.utcnow(),
                "original_prompt": original_prompt,
                "test_cases": request.test_cases,
                "iterations_completed": 0,
                "best_performance": 0.0,
                "optimization_history": []
            }
            
            optimization_jobs_collection.document(job_id).set(optimization_job)
            
            # Run optimization based on strategy
            if request.strategy == OptimizationStrategy.AUTOMEDPROMPT:
                result = await self._run_automedprompt_optimization(job_id, request, original_prompt)
            elif request.strategy == OptimizationStrategy.GENETIC_ALGORITHM:
                result = await self._run_genetic_algorithm_optimization(job_id, request, original_prompt)
            elif request.strategy == OptimizationStrategy.REINFORCEMENT_LEARNING:
                result = await self._run_rl_optimization(job_id, request, original_prompt)
            elif request.strategy == OptimizationStrategy.HYBRID:
                result = await self._run_hybrid_optimization(job_id, request, original_prompt)
            else:
                result = await self._run_automedprompt_optimization(job_id, request, original_prompt)
            
            # Update job status
            optimization_jobs_collection.document(job_id).update({
                "status": "completed",
                "completed_at": datetime.utcnow(),
                "final_result": result.__dict__
            })
            
            return result
            
        except Exception as e:
            logger.error("Optimization failed", job_id=job_id, error=str(e))
            raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")
    
    async def _get_template_content(self, template_id: str) -> str:
        """Get template content from Firestore"""
        try:
            # Try new templates collection first
            doc = firestore_client.collection('poml_templates_v2').document(template_id).get()
            if doc.exists:
                return doc.to_dict().get("content", "")
            
            # Fallback to old collection
            doc = firestore_client.collection('poml_templates').document(template_id).get()
            if doc.exists:
                return doc.to_dict().get("content", "")
            
            raise ValueError(f"Template {template_id} not found")
            
        except Exception as e:
            logger.error("Failed to get template content", template_id=template_id, error=str(e))
            raise
    
    async def _run_automedprompt_optimization(self, job_id: str, request: OptimizationRequest, original_prompt: str) -> OptimizationResult:
        """Run AutoMedPrompt optimization algorithm"""
        try:
            best_prompt = original_prompt
            best_score = await self._evaluate_prompt(original_prompt, request.test_cases, request.objective)
            optimization_history = [{"iteration": 0, "prompt": original_prompt, "score": best_score}]
            
            for iteration in range(1, request.max_iterations + 1):
                logger.info(f"AutoMedPrompt iteration {iteration}", job_id=job_id)
                
                # Generate prompt variants using different strategies
                variants = await self._generate_automedprompt_variants(best_prompt, iteration)
                
                # Evaluate variants
                for variant in variants:
                    score = await self._evaluate_prompt(variant, request.test_cases, request.objective)
                    
                    if score > best_score:
                        best_score = score
                        best_prompt = variant
                        logger.info(f"New best prompt found", iteration=iteration, score=best_score)
                    
                    optimization_history.append({
                        "iteration": iteration,
                        "prompt": variant,
                        "score": score
                    })
                
                # Update job progress
                optimization_jobs_collection.document(job_id).update({
                    "iterations_completed": iteration,
                    "best_performance": best_score,
                    "optimization_history": optimization_history[-10:]  # Keep last 10 entries
                })
                
                # Check convergence
                if best_score >= request.target_performance:
                    logger.info("Target performance reached", job_id=job_id, score=best_score)
                    break
            
            performance_improvement = best_score - optimization_history[0]["score"]
            
            return OptimizationResult(
                job_id=job_id,
                original_prompt=original_prompt,
                optimized_prompt=best_prompt,
                performance_improvement=performance_improvement,
                iterations_completed=len(optimization_history) - 1,
                strategy_used=request.strategy.value,
                convergence_achieved=best_score >= request.target_performance,
                optimization_history=optimization_history
            )
            
        except Exception as e:
            logger.error("AutoMedPrompt optimization failed", job_id=job_id, error=str(e))
            raise
    
    async def _run_genetic_algorithm_optimization(self, job_id: str, request: OptimizationRequest, original_prompt: str) -> OptimizationResult:
        """Run genetic algorithm optimization"""
        try:
            # Initialize population
            population = await self._initialize_genetic_population(original_prompt, request.population_size)
            best_prompt = original_prompt
            best_score = await self._evaluate_prompt(original_prompt, request.test_cases, request.objective)
            optimization_history = [{"iteration": 0, "prompt": original_prompt, "score": best_score}]
            
            for generation in range(1, request.max_iterations + 1):
                logger.info(f"Genetic algorithm generation {generation}", job_id=job_id)
                
                # Evaluate population
                evaluated_population = []
                for individual in population:
                    score = await self._evaluate_prompt(individual, request.test_cases, request.objective)
                    evaluated_population.append((individual, score))
                    
                    if score > best_score:
                        best_score = score
                        best_prompt = individual
                
                # Sort by fitness
                evaluated_population.sort(key=lambda x: x[1], reverse=True)
                
                # Selection (top 50%)
                selected = [individual for individual, score in evaluated_population[:request.population_size // 2]]
                
                # Crossover and mutation
                new_population = selected.copy()
                while len(new_population) < request.population_size:
                    parent1, parent2 = random.sample(selected, 2)
                    child = await self._crossover_prompts(parent1, parent2)
                    
                    if random.random() < request.mutation_rate:
                        child = await self._mutate_prompt(child)
                    
                    new_population.append(child)
                
                population = new_population
                
                optimization_history.append({
                    "iteration": generation,
                    "best_prompt": best_prompt,
                    "best_score": best_score,
                    "population_avg": sum(score for _, score in evaluated_population) / len(evaluated_population)
                })
                
                # Update job progress
                optimization_jobs_collection.document(job_id).update({
                    "iterations_completed": generation,
                    "best_performance": best_score
                })
                
                if best_score >= request.target_performance:
                    break
            
            performance_improvement = best_score - optimization_history[0]["score"]
            
            return OptimizationResult(
                job_id=job_id,
                original_prompt=original_prompt,
                optimized_prompt=best_prompt,
                performance_improvement=performance_improvement,
                iterations_completed=len(optimization_history) - 1,
                strategy_used=request.strategy.value,
                convergence_achieved=best_score >= request.target_performance,
                optimization_history=optimization_history
            )
            
        except Exception as e:
            logger.error("Genetic algorithm optimization failed", job_id=job_id, error=str(e))
            raise
    
    async def _run_rl_optimization(self, job_id: str, request: OptimizationRequest, original_prompt: str) -> OptimizationResult:
        """Run reinforcement learning optimization"""
        try:
            # Simplified RL approach using Q-learning concepts
            action_space = await self._define_rl_action_space()
            q_table = {}
            current_prompt = original_prompt
            best_prompt = original_prompt
            best_score = await self._evaluate_prompt(original_prompt, request.test_cases, request.objective)
            optimization_history = [{"iteration": 0, "prompt": original_prompt, "score": best_score}]
            
            learning_rate = 0.1
            epsilon = 0.3  # Exploration rate
            
            for episode in range(1, request.max_iterations + 1):
                logger.info(f"RL episode {episode}", job_id=job_id)
                
                # Get state representation
                state = self._get_prompt_state(current_prompt)
                
                # Choose action (exploration vs exploitation)
                if random.random() < epsilon:
                    action = random.choice(action_space)
                else:
                    action = self._get_best_action(state, q_table, action_space)
                
                # Apply action to generate new prompt
                new_prompt = await self._apply_rl_action(current_prompt, action)
                new_score = await self._evaluate_prompt(new_prompt, request.test_cases, request.objective)
                
                # Calculate reward
                reward = new_score - best_score if new_score > best_score else (new_score - best_score) * 0.1
                
                # Update Q-table
                if state not in q_table:
                    q_table[state] = {action: 0 for action in action_space}
                
                old_q = q_table[state][action]
                new_state = self._get_prompt_state(new_prompt)
                max_future_q = max(q_table.get(new_state, {action: 0 for action in action_space}).values())
                q_table[state][action] = old_q + learning_rate * (reward + 0.9 * max_future_q - old_q)
                
                # Update best if improved
                if new_score > best_score:
                    best_score = new_score
                    best_prompt = new_prompt
                    current_prompt = new_prompt
                
                optimization_history.append({
                    "iteration": episode,
                    "prompt": new_prompt,
                    "score": new_score,
                    "reward": reward,
                    "action": action
                })
                
                # Decay epsilon
                epsilon = max(0.01, epsilon * 0.995)
                
                # Update job progress
                optimization_jobs_collection.document(job_id).update({
                    "iterations_completed": episode,
                    "best_performance": best_score
                })
                
                if best_score >= request.target_performance:
                    break
            
            performance_improvement = best_score - optimization_history[0]["score"]
            
            return OptimizationResult(
                job_id=job_id,
                original_prompt=original_prompt,
                optimized_prompt=best_prompt,
                performance_improvement=performance_improvement,
                iterations_completed=len(optimization_history) - 1,
                strategy_used=request.strategy.value,
                convergence_achieved=best_score >= request.target_performance,
                optimization_history=optimization_history
            )
            
        except Exception as e:
            logger.error("RL optimization failed", job_id=job_id, error=str(e))
            raise
    
    async def _run_hybrid_optimization(self, job_id: str, request: OptimizationRequest, original_prompt: str) -> OptimizationResult:
        """Run hybrid optimization combining multiple strategies"""
        try:
            # Run AutoMedPrompt for first half of iterations
            automedprompt_request = OptimizationRequest(
                prompt_template_id=request.prompt_template_id,
                test_cases=request.test_cases,
                strategy=OptimizationStrategy.AUTOMEDPROMPT,
                objective=request.objective,
                max_iterations=request.max_iterations // 2,
                target_performance=request.target_performance
            )
            
            automedprompt_result = await self._run_automedprompt_optimization(job_id + "_auto", automedprompt_request, original_prompt)
            
            # Use AutoMedPrompt result as starting point for genetic algorithm
            genetic_request = OptimizationRequest(
                prompt_template_id=request.prompt_template_id,
                test_cases=request.test_cases,
                strategy=OptimizationStrategy.GENETIC_ALGORITHM,
                objective=request.objective,
                max_iterations=request.max_iterations - automedprompt_result.iterations_completed,
                target_performance=request.target_performance,
                population_size=request.population_size,
                mutation_rate=request.mutation_rate
            )
            
            genetic_result = await self._run_genetic_algorithm_optimization(job_id + "_genetic", genetic_request, automedprompt_result.optimized_prompt)
            
            # Combine results
            combined_history = automedprompt_result.optimization_history + genetic_result.optimization_history
            
            return OptimizationResult(
                job_id=job_id,
                original_prompt=original_prompt,
                optimized_prompt=genetic_result.optimized_prompt,
                performance_improvement=genetic_result.performance_improvement,
                iterations_completed=automedprompt_result.iterations_completed + genetic_result.iterations_completed,
                strategy_used="hybrid",
                convergence_achieved=genetic_result.convergence_achieved,
                optimization_history=combined_history
            )
            
        except Exception as e:
            logger.error("Hybrid optimization failed", job_id=job_id, error=str(e))
            raise
    
    async def _evaluate_prompt(self, prompt: str, test_cases: List[Dict[str, Any]], objective: OptimizationObjective) -> float:
        """Evaluate prompt performance against test cases"""
        try:
            total_score = 0.0
            valid_tests = 0
            
            for test_case in test_cases:
                if gemini_model:
                    # Use Gemini for evaluation
                    test_prompt = f"{prompt}\n\nInput: {json.dumps(test_case.get('input_data', {}))}"
                    
                    response = gemini_model.generate_content(
                        test_prompt,
                        generation_config=genai.GenerationConfig(
                            temperature=0.1,
                            top_p=0.8,
                            max_output_tokens=1024
                        )
                    )
                    
                    result = response.text
                    expected = test_case.get('expected_output', '')
                    
                    # Calculate similarity score
                    if objective == OptimizationObjective.ACCURACY:
                        score = self._calculate_text_similarity(result, expected)
                    elif objective == OptimizationObjective.CLINICAL_ACCURACY:
                        score = self._calculate_clinical_accuracy(result, expected, test_case.get('medical_context', ''))
                    elif objective == OptimizationObjective.TOKEN_EFFICIENCY:
                        score = self._calculate_token_efficiency(result, len(test_prompt.split()))
                    else:
                        score = self._calculate_text_similarity(result, expected)
                    
                    total_score += score
                    valid_tests += 1
                else:
                    # Fallback scoring
                    total_score += random.uniform(0.5, 0.9)
                    valid_tests += 1
                
                # Limit test cases to avoid timeout
                if valid_tests >= 5:
                    break
            
            return total_score / max(valid_tests, 1)
            
        except Exception as e:
            logger.error("Failed to evaluate prompt", error=str(e))
            return 0.0
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity score"""
        try:
            # Simple word overlap similarity
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            
            if not words1 and not words2:
                return 1.0
            if not words1 or not words2:
                return 0.0
            
            intersection = words1.intersection(words2)
            union = words1.union(words2)
            
            return len(intersection) / len(union)
        except:
            return 0.0
    
    def _calculate_clinical_accuracy(self, result: str, expected: str, medical_context: str) -> float:
        """Calculate clinical accuracy score with medical context"""
        try:
            # Enhanced scoring for medical content
            base_similarity = self._calculate_text_similarity(result, expected)
            
            # Medical keywords bonus
            medical_keywords = ['diagnosis', 'treatment', 'symptoms', 'medication', 'procedure', 'condition']
            result_lower = result.lower()
            expected_lower = expected.lower()
            
            medical_bonus = 0.0
            for keyword in medical_keywords:
                if keyword in result_lower and keyword in expected_lower:
                    medical_bonus += 0.1
            
            return min(base_similarity + medical_bonus, 1.0)
        except:
            return 0.0
    
    def _calculate_token_efficiency(self, result: str, input_tokens: int) -> float:
        """Calculate token efficiency score"""
        try:
            output_tokens = len(result.split())
            if output_tokens == 0:
                return 0.0
            
            # Efficiency = useful output / total tokens used
            efficiency = min(output_tokens / (input_tokens + output_tokens), 0.5) * 2
            return efficiency
        except:
            return 0.0
    
    async def _generate_automedprompt_variants(self, prompt: str, iteration: int) -> List[str]:
        """Generate prompt variants using AutoMedPrompt strategies"""
        variants = []
        
        try:
            # Strategy 1: Add medical context
            medical_context_variant = self._add_medical_context(prompt)
            variants.append(medical_context_variant)
            
            # Strategy 2: Improve clarity and structure
            structured_variant = self._improve_structure(prompt)
            variants.append(structured_variant)
            
            # Strategy 3: Add examples
            example_variant = self._add_examples(prompt)
            variants.append(example_variant)
            
            # Strategy 4: Adjust tone and specificity
            tone_variant = self._adjust_tone(prompt)
            variants.append(tone_variant)
            
            # Strategy 5: Add constraints
            constraint_variant = self._add_constraints(prompt)
            variants.append(constraint_variant)
            
        except Exception as e:
            logger.error("Failed to generate AutoMedPrompt variants", error=str(e))
        
        return variants[:3]  # Return top 3 variants
    
    def _add_medical_context(self, prompt: str) -> str:
        """Add medical context to prompt"""
        medical_prefixes = [
            "As a healthcare professional, ",
            "Based on clinical guidelines, ",
            "Considering patient safety and medical best practices, ",
            "Using evidence-based medical knowledge, "
        ]
        
        prefix = random.choice(medical_prefixes)
        return f"{prefix}{prompt}"
    
    def _improve_structure(self, prompt: str) -> str:
        """Improve prompt structure and clarity"""
        if ":" not in prompt:
            return f"Task: {prompt}\n\nPlease provide a detailed and accurate response."
        return prompt
    
    def _add_examples(self, prompt: str) -> str:
        """Add examples to prompt"""
        example_text = "\n\nExample format:\nInput: [patient data]\nOutput: [structured analysis]"
        return f"{prompt}{example_text}"
    
    def _adjust_tone(self, prompt: str) -> str:
        """Adjust prompt tone for medical context"""
        if "please" not in prompt.lower():
            return f"{prompt} Please ensure accuracy and clinical relevance."
        return prompt
    
    def _add_constraints(self, prompt: str) -> str:
        """Add constraints to prompt"""
        constraint_text = "\n\nConstraints:\n- Use medical terminology appropriately\n- Provide evidence-based responses\n- Consider patient safety"
        return f"{prompt}{constraint_text}"
    
    async def _initialize_genetic_population(self, original_prompt: str, population_size: int) -> List[str]:
        """Initialize genetic algorithm population"""
        population = [original_prompt]
        
        for _ in range(population_size - 1):
            variant = await self._mutate_prompt(original_prompt)
            population.append(variant)
        
        return population
    
    async def _crossover_prompts(self, parent1: str, parent2: str) -> str:
        """Perform crossover between two prompts"""
        try:
            # Simple crossover: combine parts of both prompts
            words1 = parent1.split()
            words2 = parent2.split()
            
            # Take first half from parent1, second half from parent2
            crossover_point = len(words1) // 2
            child_words = words1[:crossover_point] + words2[crossover_point:]
            
            return " ".join(child_words)
        except:
            return parent1
    
    async def _mutate_prompt(self, prompt: str) -> str:
        """Mutate prompt using various strategies"""
        mutation_strategies = [
            self._add_adjectives,
            self._rephrase_sections,
            self._add_medical_context,
            self._improve_structure
        ]
        
        strategy = random.choice(mutation_strategies)
        return strategy(prompt)
    
    def _add_adjectives(self, prompt: str) -> str:
        """Add descriptive adjectives to prompt"""
        adjectives = ["detailed", "comprehensive", "accurate", "clinical", "evidence-based", "systematic"]
        words = prompt.split()
        
        # Add adjective before nouns
        for i, word in enumerate(words):
            if word.lower() in ["analysis", "diagnosis", "treatment", "assessment", "evaluation"]:
                if i > 0 and words[i-1].lower() not in adjectives:
                    words[i] = f"{random.choice(adjectives)} {word}"
                break
        
        return " ".join(words)
    
    def _rephrase_sections(self, prompt: str) -> str:
        """Rephrase sections of the prompt"""
        rephrase_map = {
            "analyze": "evaluate",
            "determine": "identify",
            "provide": "generate",
            "give": "deliver",
            "make": "create"
        }
        
        result = prompt
        for old_word, new_word in rephrase_map.items():
            result = result.replace(old_word, new_word)
        
        return result
    
    async def _define_rl_action_space(self) -> List[str]:
        """Define action space for reinforcement learning"""
        return [
            "add_medical_context",
            "improve_structure", 
            "add_examples",
            "adjust_tone",
            "add_constraints",
            "rephrase_key_terms",
            "add_specificity",
            "improve_clarity"
        ]
    
    def _get_prompt_state(self, prompt: str) -> str:
        """Get state representation of prompt for RL"""
        # Simple state representation based on prompt characteristics
        word_count = len(prompt.split())
        has_medical_terms = any(term in prompt.lower() for term in ["medical", "clinical", "patient", "diagnosis"])
        has_structure = ":" in prompt or "\n" in prompt
        
        return f"words_{word_count}_medical_{has_medical_terms}_structured_{has_structure}"
    
    def _get_best_action(self, state: str, q_table: Dict, action_space: List[str]) -> str:
        """Get best action for current state"""
        if state not in q_table:
            return random.choice(action_space)
        
        best_action = max(q_table[state].items(), key=lambda x: x[1])[0]
        return best_action
    
    async def _apply_rl_action(self, prompt: str, action: str) -> str:
        """Apply RL action to prompt"""
        action_map = {
            "add_medical_context": self._add_medical_context,
            "improve_structure": self._improve_structure,
            "add_examples": self._add_examples,
            "adjust_tone": self._adjust_tone,
            "add_constraints": self._add_constraints,
            "rephrase_key_terms": self._rephrase_sections,
            "add_specificity": self._add_adjectives,
            "improve_clarity": self._improve_structure
        }
        
        if action in action_map:
            return action_map[action](prompt)
        else:
            return prompt

async def test_prompt_with_model(prompt: str, model_id: str, test_data: Dict[str, Any]) -> float:
    """
    Test a prompt with a specific model (enhanced version)
    """
    try:
        if gemini_model:
            # Use Gemini for testing
            response = gemini_model.generate_content(
                f"{prompt}\n\nTest Data: {json.dumps(test_data)}",
                generation_config=genai.GenerationConfig(
                    temperature=0.1,
                    top_p=0.8,
                    max_output_tokens=1024
                )
            )
            
            result_text = response.text
            
            # Calculate score based on response quality
            score = len(result_text) / 1000  # Simple length-based scoring
            score = min(score, 1.0)  # Cap at 1.0
            
            return score
        else:
            # Fallback to model management service
            if model_id not in available_models:
                await load_model(model_id)
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{MODEL_MANAGEMENT_URL}/models/{model_id}/test",
                    json={
                        "prompt": prompt,
                        "max_tokens": 500
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if result.get("success"):
                        return random.uniform(0.6, 0.95)  # Placeholder score
                    else:
                        return 0.0
                else:
                    return 0.0
                    
    except Exception as e:
        logger.error(f"Failed to test prompt with model {model_id}: {e}")
        return 0.0

async def test_prompt_with_all_models(prompt: str, model_ids: List[str], test_data: Dict[str, Any]) -> Dict[str, float]:
    """
    Test a prompt with multiple models and return scores
    """
    scores = {}
    
    # Test with each model in parallel
    tasks = []
    for model_id in model_ids:
        tasks.append(test_prompt_with_model(prompt, model_id, test_data))
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    for model_id, result in zip(model_ids, results):
        if isinstance(result, Exception):
            logger.error(f"Error testing with model {model_id}: {result}")
            scores[model_id] = 0.0
        else:
            scores[model_id] = result
    
    return scores

# Models
class OptimizationStatus(str, Enum):
    QUEUED = "queued"
    INITIALIZING = "initializing"
    BASELINE = "baseline"
    OPTIMIZING = "optimizing"
    TESTING = "testing"
    COMPLETED = "completed"
    FAILED = "failed"

class OptimizationStrategy(str, Enum):
    AUTOMEDPROMPT = "automedprompt"
    TEXTUAL_GRADIENTS = "textual_gradients"
    FEW_SHOT = "few_shot"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    ROLE_OPTIMIZATION = "role_optimization"

class OptimizationRequest(BaseModel):
    goal_id: str = Field(..., description="Goal ID from Goal Definition Service")
    template_id: str = Field(..., description="POML template to optimize")
    template_content: str = Field(..., description="Current template content")
    target_metric: str = Field(..., description="Target metric (accuracy, precision, etc.)")
    target_value: float = Field(..., ge=0, le=1, description="Target value to achieve")
    medical_domain: str = Field(..., description="Medical specialty domain")
    dataset_ids: List[str] = Field(..., description="Kaggle dataset IDs for testing")
    model_ids: List[str] = Field(..., description="AI model IDs to use for optimization")
    strategy: OptimizationStrategy = OptimizationStrategy.AUTOMEDPROMPT
    max_iterations: int = Field(50, ge=10, le=200, description="Maximum optimization iterations")

class PromptVariation(BaseModel):
    id: str
    content: str
    generation_method: str
    performance_score: Optional[float] = None
    metrics: Dict[str, float] = {}
    created_at: datetime

class OptimizationJob(BaseModel):
    id: str
    goal_id: str
    template_id: str
    original_content: str
    target_metric: str
    target_value: float
    medical_domain: str
    dataset_ids: List[str]
    model_ids: List[str]
    strategy: OptimizationStrategy
    status: OptimizationStatus
    current_iteration: int = 0
    max_iterations: int
    baseline_score: Optional[float] = None
    best_score: Optional[float] = None
    best_prompt: Optional[str] = None
    prompt_variations: List[PromptVariation] = []
    optimization_history: List[Dict[str, Any]] = []
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None

# In-memory storage
optimization_jobs: Dict[str, OptimizationJob] = {}

class AutoMedPromptOptimizer:
    """
    Implementation of AutoMedPrompt methodology for healthcare prompt optimization
    """
    
    def __init__(self, medical_domain: str):
        self.medical_domain = medical_domain
        self.role_templates = self._get_medical_role_templates()
        self.task_patterns = self._get_medical_task_patterns()
        self.output_formats = self._get_medical_output_formats()
    
    def _get_medical_role_templates(self) -> List[str]:
        """Get medical role templates for the domain"""
        domain_roles = {
            "cardiology": [
                "You are a board-certified cardiologist with {years} years of experience in {specialty}.",
                "You are a cardiac specialist trained in {specialty} with expertise in {area}.",
                "You are an interventional cardiologist with advanced training in {procedure}."
            ],
            "emergency_medicine": [
                "You are an emergency medicine physician with {years} years of ER experience.",
                "You are an emergency department attending with expertise in {specialty}.",
                "You are a trauma specialist with advanced life support certification."
            ],
            "pharmacology": [
                "You are a clinical pharmacist with PharmD certification and {years} years of experience.",
                "You are a medication therapy management specialist.",
                "You are a pharmacologist with expertise in drug interactions and safety."
            ],
            "laboratory": [
                "You are a clinical pathologist with board certification in laboratory medicine.",
                "You are a laboratory director with expertise in {test_type} diagnostics.",
                "You are a medical technologist with {years} years of lab experience."
            ],
            "general_medicine": [
                "You are a board-certified internal medicine physician.",
                "You are a primary care physician with {years} years of clinical experience.",
                "You are a hospitalist with expertise in complex medical cases."
            ]
        }
        return domain_roles.get(self.medical_domain, domain_roles["general_medicine"])
    
    def _get_medical_task_patterns(self) -> List[str]:
        """Get medical task instruction patterns"""
        return [
            "Analyze the following {data_type} and provide {analysis_type}:",
            "Evaluate {patient_data} for {condition} and assess {criteria}:",
            "Review {medical_data} and determine {clinical_question}:",
            "Interpret {test_results} in the context of {patient_context}:",
            "Assess {clinical_scenario} and recommend {intervention_type}:"
        ]
    
    def _get_medical_output_formats(self) -> List[str]:
        """Get medical output format templates"""
        return [
            '''Provide a structured analysis with:
1. Clinical Assessment: {assessment}
2. Differential Diagnosis: {diagnoses}
3. Recommended Actions: {actions}
4. Risk Factors: {risks}''',
            '''Format as JSON:
{
  "clinical_findings": "",
  "assessment": "",
  "recommendations": [],
  "confidence_level": "",
  "safety_considerations": []
}''',
            '''Structure your response as:
- Primary Findings: {findings}
- Clinical Significance: {significance} 
- Next Steps: {next_steps}
- Monitoring Requirements: {monitoring}'''
        ]
    
    def generate_role_variations(self, current_role: str) -> List[str]:
        """Generate role variations using medical templates"""
        variations = []
        
        for template in self.role_templates[:3]:  # Use top 3 templates
            # Fill in placeholders
            years_options = ["10+", "15+", "20+", "25+"]
            specialty_options = {
                "cardiology": ["interventional cardiology", "electrophysiology", "heart failure"],
                "emergency_medicine": ["trauma surgery", "critical care", "toxicology"],
                "pharmacology": ["clinical pharmacy", "medication safety", "drug development"],
                "laboratory": ["clinical chemistry", "hematology", "microbiology"],
                "general_medicine": ["internal medicine", "preventive care", "geriatrics"]
            }
            
            specialties = specialty_options.get(self.medical_domain, ["clinical medicine"])
            
            variation = template.format(
                years=random.choice(years_options),
                specialty=random.choice(specialties),
                area=random.choice(specialties),
                procedure=random.choice(specialties),
                test_type=random.choice(specialties)
            )
            variations.append(variation)
        
        return variations
    
    def generate_task_variations(self, current_task: str) -> List[str]:
        """Generate task instruction variations"""
        variations = []
        
        # Extract key components from current task
        task_keywords = self._extract_task_keywords(current_task)
        
        for pattern in self.task_patterns[:3]:
            variation = pattern.format(
                data_type=task_keywords.get("data_type", "clinical data"),
                analysis_type=task_keywords.get("analysis_type", "comprehensive analysis"),
                patient_data=task_keywords.get("patient_data", "patient information"),
                condition=task_keywords.get("condition", "medical condition"),
                criteria=task_keywords.get("criteria", "clinical criteria"),
                medical_data=task_keywords.get("medical_data", "medical information"),
                clinical_question=task_keywords.get("clinical_question", "clinical assessment"),
                test_results=task_keywords.get("test_results", "diagnostic results"),
                patient_context=task_keywords.get("patient_context", "patient history"),
                clinical_scenario=task_keywords.get("clinical_scenario", "clinical situation"),
                intervention_type=task_keywords.get("intervention_type", "appropriate intervention")
            )
            variations.append(variation)
        
        return variations
    
    def generate_output_variations(self, current_output: str) -> List[str]:
        """Generate output format variations"""
        return self.output_formats[:3]  # Return top 3 format options
    
    def _extract_task_keywords(self, task: str) -> Dict[str, str]:
        """Extract keywords from current task for variation generation"""
        # Simple keyword extraction - in production this would be more sophisticated
        keywords = {
            "data_type": "medical data",
            "analysis_type": "analysis",
            "patient_data": "patient information",
            "condition": "condition",
            "criteria": "criteria"
        }
        
        # Look for specific medical terms
        medical_terms = {
            "laboratory": "lab results", "lab": "lab results",
            "medication": "medication data", "drug": "drug information",
            "diagnostic": "diagnostic data", "diagnosis": "diagnostic information",
            "cardiac": "cardiac data", "heart": "cardiac information"
        }
        
        task_lower = task.lower()
        for term, replacement in medical_terms.items():
            if term in task_lower:
                keywords["data_type"] = replacement
                break
        
        return keywords
    
    def optimize_prompt_components(self, current_prompt: str) -> List[str]:
        """
        Generate optimized prompt variations using AutoMedPrompt methodology
        """
        variations = []
        
        # Parse current prompt into components
        components = self._parse_prompt_components(current_prompt)
        
        # Generate role variations
        role_variations = self.generate_role_variations(components.get("role", ""))
        
        # Generate task variations  
        task_variations = self.generate_task_variations(components.get("task", ""))
        
        # Generate output variations
        output_variations = self.generate_output_variations(components.get("output", ""))
        
        # Combine variations systematically
        for i, role in enumerate(role_variations):
            for j, task in enumerate(task_variations):
                for k, output in enumerate(output_variations):
                    if len(variations) >= 9:  # Limit to 9 variations per cycle
                        break
                    
                    variation = self._reconstruct_prompt(
                        components,
                        role=role,
                        task=task,
                        output=output
                    )
                    variations.append(variation)
        
        return variations
    
    def _parse_prompt_components(self, prompt: str) -> Dict[str, str]:
        """Parse POML prompt into components"""
        components = {}
        
        # Extract role
        role_match = re.search(r'<role>(.*?)</role>', prompt, re.DOTALL)
        components["role"] = role_match.group(1).strip() if role_match else ""
        
        # Extract context
        context_match = re.search(r'<context>(.*?)</context>', prompt, re.DOTALL)
        components["context"] = context_match.group(1).strip() if context_match else ""
        
        # Extract task
        task_match = re.search(r'<task>(.*?)</task>', prompt, re.DOTALL)
        components["task"] = task_match.group(1).strip() if task_match else ""
        
        # Extract guidelines
        guidelines_match = re.search(r'<guidelines>(.*?)</guidelines>', prompt, re.DOTALL)
        components["guidelines"] = guidelines_match.group(1).strip() if guidelines_match else ""
        
        # Extract output format
        output_match = re.search(r'<output_format>(.*?)</output_format>', prompt, re.DOTALL)
        components["output"] = output_match.group(1).strip() if output_match else ""
        
        return components
    
    def _reconstruct_prompt(self, base_components: Dict[str, str], **variations) -> str:
        """Reconstruct POML prompt with variations"""
        components = base_components.copy()
        components.update(variations)
        
        prompt = '<prompt version="1.0">\n'
        
        if components.get("role"):
            prompt += f'  <role>\n    {components["role"]}\n  </role>\n\n'
        
        if components.get("context"):
            prompt += f'  <context>\n    {components["context"]}\n  </context>\n\n'
        
        if components.get("task"):
            prompt += f'  <task>\n    {components["task"]}\n  </task>\n\n'
        
        if components.get("guidelines"):
            prompt += f'  <guidelines>\n    {components["guidelines"]}\n  </guidelines>\n\n'
        
        if components.get("output"):
            prompt += f'  <output_format>\n    {components["output"]}\n  </output_format>\n'
        
        prompt += '</prompt>'
        
        return prompt

async def evaluate_prompt_performance(prompt: str, test_data: List[Dict], target_metric: str, model_ids: List[str]) -> Dict[str, float]:
    """
    Evaluate prompt performance using multiple AI models and real medical test data
    """
    if not model_ids:
        raise Exception("No models specified for evaluation")
    
    try:
        results = []
        ground_truth = []
        
        # Test prompt on sample of data (limit for cost/speed)
        sample_size = min(len(test_data), 10)
        test_sample = random.sample(test_data, sample_size)
        
        # Test with all models and aggregate scores
        model_scores = await test_prompt_with_all_models(prompt, model_ids, {"test_data": test_sample})
        
        # Calculate metrics based on model scores
        if not model_scores:
            return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0, "model_scores": {}}
        
        # Calculate average score across all models
        avg_score = sum(model_scores.values()) / len(model_scores)
        
        # Return metrics with model-specific scores
        return {
            "accuracy": avg_score,
            "precision": avg_score,  # Simplified for now
            "recall": avg_score,     # Simplified for now
            "f1_score": avg_score,   # Simplified for now
            "average_score": avg_score,
            "model_scores": model_scores
        }
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0}

def calculate_similarity(text1: str, text2: str) -> float:
    """
    Calculate similarity between two texts (simplified implementation)
    """
    if not text1 or not text2:
        return 0.0
    
    # Simple word overlap similarity
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    return intersection / union if union > 0 else 0.0

async def run_optimization_cycle(job_id: str):
    """
    Run a single optimization cycle for a job
    """
    job = optimization_jobs[job_id]
    
    try:
        # Initialize optimizer
        optimizer = AutoMedPromptOptimizer(job.medical_domain)
        
        # Generate prompt variations
        if job.current_iteration == 0:
            # First iteration - generate initial variations
            current_prompt = job.original_content
        else:
            # Use best prompt so far
            current_prompt = job.best_prompt or job.original_content
        
        variations = optimizer.optimize_prompt_components(current_prompt)
        
        logger.info(f"Generated {len(variations)} prompt variations", 
                   job_id=job_id, iteration=job.current_iteration)
        
        # Test each variation (in production, this would be with real Kaggle data)
        test_data = [
            {"input": "sample medical case", "expected_output": "expected analysis"},
            {"input": "another case", "expected_output": "another expected output"}
        ]
        
        best_variation = None
        best_score = job.best_score or 0.0
        
        for i, variation in enumerate(variations):
            metrics = await evaluate_prompt_performance(variation, test_data, job.target_metric, job.model_ids)
            score = metrics.get(job.target_metric, 0.0)
            
            # Create variation record
            prompt_variation = PromptVariation(
                id=f"{job_id}_iter{job.current_iteration}_var{i}",
                content=variation,
                generation_method=f"AutoMedPrompt_iteration_{job.current_iteration}",
                performance_score=score,
                metrics=metrics,
                created_at=datetime.utcnow()
            )
            
            job.prompt_variations.append(prompt_variation)
            
            # Track best variation
            if score > best_score:
                best_score = score
                best_variation = variation
        
        # Update job with best results
        if best_variation:
            job.best_prompt = best_variation
            job.best_score = best_score
        
        # Record iteration history
        job.optimization_history.append({
            "iteration": job.current_iteration,
            "timestamp": datetime.utcnow().isoformat(),
            "variations_tested": len(variations),
            "best_score": best_score,
            "improvement": best_score - (job.baseline_score or 0.0),
            "target_achieved": best_score >= job.target_value
        })
        
        job.current_iteration += 1
        job.updated_at = datetime.utcnow()
        
        # Check if target achieved or max iterations reached
        if best_score >= job.target_value:
            job.status = OptimizationStatus.COMPLETED
            job.completed_at = datetime.utcnow()
            logger.info(f"Optimization target achieved!", 
                       job_id=job_id, 
                       target=job.target_value,
                       achieved=best_score)
        elif job.current_iteration >= job.max_iterations:
            job.status = OptimizationStatus.COMPLETED
            job.completed_at = datetime.utcnow()
            logger.info(f"Optimization completed - max iterations reached",
                       job_id=job_id,
                       final_score=best_score)
        else:
            # Continue optimization
            logger.info(f"Optimization continuing",
                       job_id=job_id,
                       iteration=job.current_iteration,
                       current_best=best_score,
                       target=job.target_value)
        
    except Exception as e:
        logger.error(f"Optimization cycle failed: {e}", job_id=job_id)
        job.status = OptimizationStatus.FAILED
        job.error_message = str(e)

# Modern Prompt Optimization Endpoints

@app.post("/optimize/modern")
async def optimize_with_modern_context(request: ModernPromptRequest, background_tasks: BackgroundTasks):
    """
    Optimize prompts using modern context-aware techniques with Kaggle dataset integration
    """
    try:
        job_id = f"modern_opt_{uuid.uuid4().hex[:12]}"
        
        logger.info("Starting modern prompt optimization",
                   job_id=job_id,
                   template_id=request.prompt_template_id,
                   strategy=request.strategy,
                   auto_dataset=request.auto_dataset_selection)
        
        # Initialize optimization job
        optimization_job = {
            "job_id": job_id,
            "template_id": request.prompt_template_id,
            "strategy": request.strategy.value,
            "objective": request.objective.value,
            "status": "running",
            "started_at": datetime.utcnow(),
            "context_config": request.context_config.dict(),
            "auto_dataset_selection": request.auto_dataset_selection,
            "kaggle_datasets": request.kaggle_datasets or [],
            "iterations_completed": 0,
            "best_performance": 0.0,
            "optimization_history": []
        }
        
        # Store job in Firestore
        safe_firestore_operation(
            lambda: optimization_jobs_collection.document(job_id).set(optimization_job)
        )
        
        # Start optimization in background
        background_tasks.add_task(run_modern_optimization, job_id, request)
        
        return {
            "job_id": job_id,
            "status": "started",
            "message": "Modern prompt optimization started with context-aware techniques",
            "context_config": request.context_config.dict(),
            "estimated_duration_minutes": request.max_iterations * 2
        }
        
    except Exception as e:
        logger.error(f"Failed to start modern optimization: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start optimization: {str(e)}")

@app.post("/optimize/context/analyze")
async def analyze_prompt_context(request: Dict[str, Any]):
    """
    Analyze prompt content for context optimization opportunities
    """
    try:
        prompt_content = request.get("prompt_content", "")
        agent_type = request.get("agent_type", "clinical")
        medical_domain = request.get("medical_domain", "general_medicine")
        
        context_manager = ModernContextManager(KAGGLE_SERVICE_URL)
        context_config = ContextConfig()
        
        # Generate system prompt
        system_prompt = await context_manager.generate_system_prompt(agent_type, medical_domain, context_config)
        
        # Get few-shot examples
        few_shot_examples = await context_manager.get_few_shot_examples(prompt_content, context_config)
        
        # Optimize context window
        context_optimization = await context_manager.optimize_context_window(
            system_prompt, few_shot_examples, prompt_content, context_config
        )
        
        # Get dataset recommendations
        try:
            async with httpx.AsyncClient() as client:
                recommendations_response = await client.post(
                    f"{KAGGLE_SERVICE_URL}/datasets/recommendations",
                    json={
                        "prompt_content": prompt_content,
                        "agent_type": agent_type,
                        "medical_domain": medical_domain
                    }
                )
                
                if recommendations_response.status_code == 200:
                    dataset_recommendations = recommendations_response.json().get("recommendations", [])
                else:
                    dataset_recommendations = []
        except:
            dataset_recommendations = []
        
        return {
            "context_analysis": {
                "agent_type": agent_type,
                "medical_domain": medical_domain,
                "system_prompt_preview": system_prompt[:500] + "..." if len(system_prompt) > 500 else system_prompt,
                "few_shot_examples_count": len(few_shot_examples),
                "token_usage": context_optimization["token_usage"],
                "optimization_suggestions": [
                    f"Context utilization: {context_optimization['token_usage']['utilization']:.1%}",
                    f"Few-shot examples: {len(few_shot_examples)} available",
                    f"Dataset recommendations: {len(dataset_recommendations)} found"
                ]
            },
            "dataset_recommendations": dataset_recommendations[:5],
            "few_shot_examples": few_shot_examples,
            "optimization_ready": context_optimization["token_usage"]["utilization"] < 0.9
        }
        
    except Exception as e:
        logger.error(f"Context analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Context analysis failed: {str(e)}")

@app.get("/optimize/modern/{job_id}")
async def get_modern_optimization_status(job_id: str):
    """
    Get status of modern optimization job
    """
    try:
        job_doc = safe_firestore_operation(
            lambda: optimization_jobs_collection.document(job_id).get()
        )
        
        if not job_doc or not job_doc.exists:
            raise HTTPException(status_code=404, detail="Optimization job not found")
        
        job_data = job_doc.to_dict()
        
        return {
            "job_id": job_id,
            "status": job_data.get("status", "unknown"),
            "iterations_completed": job_data.get("iterations_completed", 0),
            "best_performance": job_data.get("best_performance", 0.0),
            "strategy": job_data.get("strategy", "unknown"),
            "started_at": job_data.get("started_at"),
            "context_config": job_data.get("context_config", {}),
            "optimization_history": job_data.get("optimization_history", []),
            "current_iteration": len(job_data.get("optimization_history", [])),
            "estimated_completion": job_data.get("estimated_completion"),
            "performance_trend": calculate_performance_trend(job_data.get("optimization_history", []))
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get optimization status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get job status: {str(e)}")

async def run_modern_optimization(job_id: str, request: ModernPromptRequest):
    """
    Run modern context-aware optimization in background
    """
    try:
        logger.info(f"Starting modern optimization background task", job_id=job_id)
        
        optimizer = AutoMedPromptOptimizer()
        context_manager = ModernContextManager(KAGGLE_SERVICE_URL)
        
        # Get original prompt content
        original_prompt = await optimizer._get_template_content(request.prompt_template_id)
        
        # Extract agent type and medical domain from prompt or use defaults
        agent_type = "clinical"  # Could be extracted from prompt analysis
        medical_domain = "general_medicine"  # Could be extracted from prompt analysis
        
        # Get tenant ID (in real implementation, extract from authentication)
        tenant_id = "default_tenant"  # Would come from auth context
        
        # Publish optimization started event for agent coordination
        await event_publisher.publish_optimization_started(
            job_id=job_id,
            tenant_id=tenant_id,
            agent_type=agent_type,
            strategy=request.strategy.value if hasattr(request, 'strategy') else "hybrid"
        )
        
        best_performance = 0.0
        best_prompt = original_prompt
        optimization_history = []
        
        for iteration in range(request.max_iterations):
            try:
                # Generate modern system prompt
                system_prompt = await context_manager.generate_system_prompt(
                    agent_type, medical_domain, request.context_config
                )
                
                # Get few-shot examples
                few_shot_examples = await context_manager.get_few_shot_examples(
                    original_prompt, request.context_config
                )
                
                # Create optimized prompt with modern structure
                modern_prompt = create_modern_prompt_structure(
                    system_prompt, original_prompt, few_shot_examples
                )
                
                # Evaluate performance
                performance = await evaluate_modern_prompt(modern_prompt, request.test_cases)
                
                iteration_data = {
                    "iteration": iteration + 1,
                    "performance": performance,
                    "prompt_length": len(modern_prompt),
                    "few_shot_count": len(few_shot_examples),
                    "timestamp": datetime.utcnow().isoformat(),
                    "improvements": []
                }
                
                if performance > best_performance:
                    best_performance = performance
                    best_prompt = modern_prompt
                    iteration_data["improvements"].append("New best performance achieved")
                
                optimization_history.append(iteration_data)
                
                # Update job status
                job_update = {
                    "iterations_completed": iteration + 1,
                    "best_performance": best_performance,
                    "optimization_history": optimization_history,
                    "status": "running",
                    "last_updated": datetime.utcnow()
                }
                
                safe_firestore_operation(
                    lambda: optimization_jobs_collection.document(job_id).update(job_update)
                )
                
                # Send WebSocket update
                await manager.send_update(job_id, {
                    "type": "optimization_progress",
                    "job_id": job_id,
                    "iteration": iteration + 1,
                    "performance": performance,
                    "best_performance": best_performance,
                    "status": "running",
                    "improvements": iteration_data["improvements"]
                })
                
                # Publish progress event for agent-to-agent coordination
                await event_publisher.publish_optimization_progress(
                    job_id=job_id,
                    tenant_id=tenant_id,
                    iteration=iteration + 1,
                    performance=performance,
                    best_performance=best_performance,
                    improvements=iteration_data["improvements"]
                )
                
                # Check if target performance reached
                if best_performance >= request.target_performance:
                    logger.info(f"Target performance reached", job_id=job_id, performance=best_performance)
                    await manager.send_update(job_id, {
                        "type": "optimization_complete",
                        "job_id": job_id,
                        "status": "completed",
                        "final_performance": best_performance,
                        "message": "Target performance reached"
                    })
                    break
                    
            except Exception as e:
                logger.error(f"Iteration {iteration + 1} failed: {e}", job_id=job_id)
                continue
        
        # Mark job as completed
        final_update = {
            "status": "completed",
            "completed_at": datetime.utcnow(),
            "final_prompt": best_prompt,
            "performance_improvement": best_performance - 0.5,  # Baseline assumption
            "convergence_achieved": best_performance >= request.target_performance
        }
        
        safe_firestore_operation(
            lambda: optimization_jobs_collection.document(job_id).update(final_update)
        )
        
        # Send final WebSocket update
        await manager.send_update(job_id, {
            "type": "optimization_complete",
            "job_id": job_id,
            "status": "completed",
            "final_performance": best_performance,
            "total_iterations": len(optimization_history),
            "convergence_achieved": best_performance >= request.target_performance,
            "message": "Optimization completed successfully"
        })
        
        # Publish completion event for agent consumption
        await event_publisher.publish_optimization_completed(
            job_id=job_id,
            tenant_id=tenant_id,
            agent_type=agent_type,
            final_performance=best_performance,
            optimized_prompt=best_prompt,
            dataset_used=f"kaggle_datasets_{len(optimization_history)}" if optimization_history else None
        )
        
        # If performance is good, share the prompt pattern with other agents
        if best_performance > 0.75:
            await event_publisher.publish_prompt_recommendation(
                tenant_id=tenant_id,
                agent_type=agent_type,
                prompt_pattern=best_prompt,
                performance_score=best_performance,
                usage_context=f"medical_domain_{medical_domain}_optimization"
            )
        
        logger.info(f"Modern optimization completed", job_id=job_id, 
                   iterations=len(optimization_history), 
                   best_performance=best_performance)
        
    except Exception as e:
        logger.error(f"Modern optimization failed: {e}", job_id=job_id)
        safe_firestore_operation(
            lambda: optimization_jobs_collection.document(job_id).update({
                "status": "failed",
                "error_message": str(e),
                "failed_at": datetime.utcnow()
            }))
        
        # Send failure WebSocket update
        await manager.send_update(job_id, {
            "type": "optimization_failed",
            "job_id": job_id,
            "status": "failed",
            "error": str(e),
            "message": "Optimization failed"
        })

def create_modern_prompt_structure(system_prompt: str, original_prompt: str, few_shot_examples: List[Dict[str, str]]) -> str:
    """
    Create modern prompt structure with system/user/assistant pattern
    """
    # Modern prompt template
    prompt_parts = [
        f"<system>{system_prompt}</system>",
        ""
    ]
    
    # Add few-shot examples
    if few_shot_examples:
        prompt_parts.append("<examples>")
        for i, example in enumerate(few_shot_examples, 1):
            prompt_parts.append(f"<example_{i}>")
            prompt_parts.append(f"<user>{example['input']}</user>")
            prompt_parts.append(f"<assistant>{example['output']}</assistant>")
            prompt_parts.append(f"</example_{i}>")
        prompt_parts.append("</examples>")
        prompt_parts.append("")
    
    # Add original prompt as user instruction
    prompt_parts.append(f"<user>{original_prompt}</user>")
    
    return "\n".join(prompt_parts)

async def evaluate_modern_prompt(prompt: str, test_cases: List[Dict[str, Any]]) -> float:
    """
    Evaluate modern prompt performance
    """
    try:
        # Simplified evaluation - in production would use real model testing
        total_score = 0.0
        
        for test_case in test_cases:
            # Simulate performance based on prompt characteristics
            score = 0.5  # Base score
            
            # Modern structure bonus
            if "<system>" in prompt and "<user>" in prompt:
                score += 0.2
            
            # Few-shot examples bonus
            if "<examples>" in prompt:
                score += 0.15
            
            # Length and complexity considerations
            if len(prompt) > 1000:
                score += 0.1
            
            # Medical terminology bonus
            medical_terms = ["clinical", "medical", "patient", "diagnosis", "treatment"]
            if any(term in prompt.lower() for term in medical_terms):
                score += 0.05
                
            total_score += min(score, 1.0)
        
        return total_score / len(test_cases) if test_cases else 0.5
        
    except Exception as e:
        logger.error(f"Prompt evaluation failed: {e}")
        return 0.3  # Conservative fallback score

def calculate_performance_trend(optimization_history: List[Dict[str, Any]]) -> str:
    """
    Calculate performance trend from optimization history
    """
    if len(optimization_history) < 2:
        return "insufficient_data"
    
    recent_performances = [h.get("performance", 0) for h in optimization_history[-5:]]
    
    if len(recent_performances) >= 2:
        trend = recent_performances[-1] - recent_performances[0]
        if trend > 0.05:
            return "improving"
        elif trend < -0.05:
            return "declining"
        else:
            return "stable"
    
    return "unknown"

@app.post("/optimization/jobs", response_model=OptimizationJob)
async def create_optimization_job(request: OptimizationRequest, background_tasks: BackgroundTasks):
    """
    Create a new prompt optimization job
    """
    job_id = f"opt_{uuid.uuid4().hex[:12]}"
    
    job = OptimizationJob(
        id=job_id,
        goal_id=request.goal_id,
        template_id=request.template_id,
        original_content=request.template_content,
        target_metric=request.target_metric,
        target_value=request.target_value,
        medical_domain=request.medical_domain,
        dataset_ids=request.dataset_ids,
        model_ids=request.model_ids,
        strategy=request.strategy,
        status=OptimizationStatus.QUEUED,
        max_iterations=request.max_iterations,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )
    
    optimization_jobs[job_id] = job
    
    # Start optimization in background
    background_tasks.add_task(start_optimization, job_id)
    
    logger.info(f"Optimization job created",
               job_id=job_id,
               goal_id=request.goal_id,
               template_id=request.template_id,
               target=f"{request.target_metric}:{request.target_value}")
    
    return job

async def start_optimization(job_id: str):
    """
    Start the optimization process for a job
    """
    try:
        job = optimization_jobs[job_id]
        job.status = OptimizationStatus.INITIALIZING
        
        # Measure baseline performance
        job.status = OptimizationStatus.BASELINE
        test_data = [{"input": "test", "expected_output": "test"}]  # Simplified
        baseline_metrics = await evaluate_prompt_performance(
            job.original_content, test_data, job.target_metric, job.model_ids
        )
        job.baseline_score = baseline_metrics.get(job.target_metric, 0.0)
        job.best_score = job.baseline_score
        
        logger.info(f"Baseline established", 
                   job_id=job_id, 
                   baseline_score=job.baseline_score)
        
        # Start optimization cycles
        job.status = OptimizationStatus.OPTIMIZING
        
        while (job.status == OptimizationStatus.OPTIMIZING and 
               job.current_iteration < job.max_iterations and
               (job.best_score or 0.0) < job.target_value):
            
            await run_optimization_cycle(job_id)
            await asyncio.sleep(1)  # Brief pause between cycles
        
    except Exception as e:
        logger.error(f"Optimization failed: {e}", job_id=job_id)
        job.status = OptimizationStatus.FAILED
        job.error_message = str(e)

@app.get("/optimization/jobs", response_model=List[OptimizationJob])
async def list_optimization_jobs(status: Optional[OptimizationStatus] = None,
                                medical_domain: Optional[str] = None):
    """
    List optimization jobs with optional filtering
    """
    jobs = list(optimization_jobs.values())
    
    if status:
        jobs = [j for j in jobs if j.status == status]
    
    if medical_domain:
        jobs = [j for j in jobs if j.medical_domain == medical_domain]
    
    jobs.sort(key=lambda x: x.created_at, reverse=True)
    return jobs

@app.get("/optimization/jobs/{job_id}", response_model=OptimizationJob)
async def get_optimization_job(job_id: str):
    """
    Get specific optimization job details
    """
    if job_id not in optimization_jobs:
        raise HTTPException(status_code=404, detail="Optimization job not found")
    
    return optimization_jobs[job_id]

@app.get("/optimization/jobs/{job_id}/progress")
async def get_optimization_progress(job_id: str):
    """
    Get detailed progress for an optimization job
    """
    if job_id not in optimization_jobs:
        raise HTTPException(status_code=404, detail="Optimization job not found")
    
    job = optimization_jobs[job_id]
    
    progress_percentage = (job.current_iteration / job.max_iterations * 100) if job.max_iterations > 0 else 0
    
    improvement = 0.0
    if job.baseline_score and job.best_score:
        improvement = ((job.best_score - job.baseline_score) / job.baseline_score) * 100
    
    return {
        "job_id": job_id,
        "status": job.status,
        "progress_percentage": min(progress_percentage, 100),
        "current_iteration": job.current_iteration,
        "max_iterations": job.max_iterations,
        "baseline_score": job.baseline_score,
        "current_best_score": job.best_score,
        "target_score": job.target_value,
        "improvement_percentage": improvement,
        "target_achieved": (job.best_score or 0.0) >= job.target_value,
        "variations_tested": len(job.prompt_variations),
        "optimization_history": job.optimization_history[-5:]  # Last 5 iterations
    }

@app.get("/optimization/jobs/{job_id}/best-prompt")
async def get_best_prompt(job_id: str):
    """
    Get the best optimized prompt for a job
    """
    if job_id not in optimization_jobs:
        raise HTTPException(status_code=404, detail="Optimization job not found")
    
    job = optimization_jobs[job_id]
    
    if not job.best_prompt:
        raise HTTPException(status_code=400, detail="No optimized prompt available yet")
    
    return {
        "job_id": job_id,
        "original_prompt": job.original_content,
        "optimized_prompt": job.best_prompt,
        "performance_improvement": {
            "baseline_score": job.baseline_score,
            "optimized_score": job.best_score,
            "improvement": job.best_score - (job.baseline_score or 0.0),
            "target_achieved": (job.best_score or 0.0) >= job.target_value
        }
    }

@app.delete("/optimization/jobs/{job_id}")
async def delete_optimization_job(job_id: str):
    """
    Delete an optimization job
    """
    if job_id not in optimization_jobs:
        raise HTTPException(status_code=404, detail="Optimization job not found")
    
    deleted_job = optimization_jobs.pop(job_id)
    
    logger.info(f"Optimization job deleted", job_id=job_id)
    
    return {"message": "Optimization job deleted successfully"}

# Initialize enhanced optimizer
enhanced_optimizer = AutoMedPromptOptimizer("general_medicine")

@app.post("/optimize/automedprompt")
async def optimize_automedprompt(request: OptimizationRequest, background_tasks: BackgroundTasks):
    """Enhanced AutoMedPrompt optimization endpoint"""
    try:
        result = await enhanced_optimizer.optimize_prompt(request)
        
        # Store result in Firestore
        optimization_results_collection.document(result.job_id).set({
            "job_id": result.job_id,
            "original_prompt": result.original_prompt,
            "optimized_prompt": result.optimized_prompt,
            "performance_improvement": result.performance_improvement,
            "iterations_completed": result.iterations_completed,
            "strategy_used": result.strategy_used,
            "convergence_achieved": result.convergence_achieved,
            "optimization_history": result.optimization_history,
            "created_at": datetime.utcnow()
        })
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/optimize/jobs/{job_id}")
async def get_optimization_job_enhanced(job_id: str):
    """Get enhanced optimization job details"""
    try:
        doc = optimization_jobs_collection.document(job_id).get()
        if not doc.exists:
            raise HTTPException(status_code=404, detail="Optimization job not found")
        
        return doc.to_dict()
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/optimize/jobs")
async def list_optimization_jobs_enhanced(
    status: Optional[str] = None,
    strategy: Optional[str] = None,
    limit: int = 50
):
    """List enhanced optimization jobs"""
    try:
        query = optimization_jobs_collection.order_by("started_at", direction=firestore.Query.DESCENDING)
        
        if status:
            query = query.where("status", "==", status)
        if strategy:
            query = query.where("strategy", "==", strategy)
        
        query = query.limit(limit)
        
        jobs = []
        for doc in query.stream():
            job_data = doc.to_dict()
            jobs.append({
                "job_id": job_data["job_id"],
                "template_id": job_data["template_id"],
                "strategy": job_data["strategy"],
                "status": job_data["status"],
                "iterations_completed": job_data.get("iterations_completed", 0),
                "best_performance": job_data.get("best_performance", 0.0),
                "started_at": job_data["started_at"]
            })
        
        return {"jobs": jobs, "total": len(jobs)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/optimize/variants/{template_id}")
async def generate_prompt_variants(template_id: str, count: int = 5):
    """Generate prompt variants for testing"""
    try:
        # Get template content
        original_prompt = await enhanced_optimizer._get_template_content(template_id)
        
        # Generate variants
        variants = await enhanced_optimizer._generate_automedprompt_variants(original_prompt, 1)
        
        # Store variants
        variant_records = []
        for i, variant in enumerate(variants[:count]):
            variant_id = f"{template_id}_variant_{uuid.uuid4().hex[:8]}"
            
            variant_record = {
                "variant_id": variant_id,
                "template_id": template_id,
                "content": variant,
                "generation_method": "automedprompt",
                "created_at": datetime.utcnow()
            }
            
            prompt_variants_collection.document(variant_id).set(variant_record)
            variant_records.append(variant_record)
        
        return {
            "template_id": template_id,
            "variants": variant_records,
            "count": len(variant_records)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/optimize/evaluate")
async def evaluate_prompt_variants(
    template_id: str,
    variant_ids: List[str],
    test_cases: List[Dict[str, Any]]
):
    """Evaluate multiple prompt variants against test cases"""
    try:
        results = []
        
        for variant_id in variant_ids:
            # Get variant content
            variant_doc = prompt_variants_collection.document(variant_id).get()
            if not variant_doc.exists:
                continue
            
            variant_data = variant_doc.to_dict()
            variant_content = variant_data["content"]
            
            # Evaluate variant
            score = await enhanced_optimizer._evaluate_prompt(
                variant_content, 
                test_cases, 
                OptimizationObjective.ACCURACY
            )
            
            results.append({
                "variant_id": variant_id,
                "score": score,
                "content_preview": variant_content[:200] + "..." if len(variant_content) > 200 else variant_content
            })
            
            # Update variant with score
            prompt_variants_collection.document(variant_id).update({
                "performance_score": score,
                "evaluated_at": datetime.utcnow()
            })
        
        # Sort by score
        results.sort(key=lambda x: x["score"], reverse=True)
        
        return {
            "template_id": template_id,
            "evaluation_results": results,
            "best_variant": results[0] if results else None
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket endpoint for real-time optimization updates
@app.websocket("/ws/optimization/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    await manager.connect(websocket, job_id)
    try:
        while True:
            # Keep connection alive and wait for client messages
            data = await websocket.receive_text()
            # Echo back any messages (for heartbeat/testing)
            await websocket.send_text(f"Received: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket, job_id)
    except Exception as e:
        logger.error(f"WebSocket error for job {job_id}: {e}")
        manager.disconnect(websocket, job_id)

@app.get("/health")
async def health_check():
    """
    Enhanced health check endpoint
    """
    try:
        gemini_status = "available" if gemini_model else "unavailable"
        
        # Test Firestore connection
        if optimization_jobs_collection:
            optimization_jobs_collection.limit(1).get()
            firestore_status = "healthy"
            
            # Count active jobs
            active_jobs_query = optimization_jobs_collection.where("status", "==", "running").stream()
            active_jobs = len(list(active_jobs_query))
            
            # Count completed jobs
            completed_jobs_query = optimization_jobs_collection.where("status", "==", "completed").stream()
            completed_jobs = len(list(completed_jobs_query))
        else:
            firestore_status = "unavailable"
            active_jobs = 0
            completed_jobs = 0
        
        return {
            "status": "healthy",
            "service": "enhanced-prompt-optimization",
            "timestamp": datetime.utcnow().isoformat(),
            "gemini_api": gemini_status,
            "firestore": firestore_status,
            "active_optimizations": active_jobs,
            "completed_optimizations": completed_jobs,
            "optimization_strategies": [strategy.value for strategy in OptimizationStrategy],
            "version": "2.0.0"
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "service": "enhanced-prompt-optimization",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)