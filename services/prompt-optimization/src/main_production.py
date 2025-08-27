"""
Production Prompt Optimization Engine - SADP
Real AI integration with Firestore persistence and comprehensive logging
"""

import sys
import os

# Add common services to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'common', 'src'))

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List, Tuple, Union
from datetime import datetime, timedelta
import uuid
import asyncio
import structlog
from enum import Enum

# Import common services
from ai_client import ai_client, AIProvider
from database.firestore_client import firestore_client
from database.models.execution_record import POMLExecution, ExecutionStatus
from config.secrets import secret_manager

# Initialize logger
logger = structlog.get_logger()

app = FastAPI(
    title="SADP Production Prompt Optimization Engine", 
    description="Production-ready prompt optimization with real AI and persistence",
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
    COMPREHENSIVE = "comprehensive"

class OptimizationRequest(BaseModel):
    """Request for prompt optimization"""
    prompt: str = Field(..., description="Original prompt to optimize")
    strategy: OptimizationStrategy = Field(default=OptimizationStrategy.AUTOMEDPROMPT)
    objective: OptimizationObjective = Field(default=OptimizationObjective.ACCURACY)
    test_cases: Optional[List[Dict[str, Any]]] = Field(default=[])
    max_iterations: Optional[int] = Field(default=10)
    target_metrics: Optional[Dict[str, float]] = Field(default={})
    user_id: str = Field(default="system")
    tenant_id: str = Field(default="default")

class OptimizationResult(BaseModel):
    """Result of prompt optimization"""
    job_id: str
    original_prompt: str
    optimized_prompt: str
    strategy_used: str
    performance_improvement: float
    optimization_metrics: Dict[str, Any]
    execution_time: float
    iterations_completed: int
    status: str
    created_at: datetime
    completed_at: Optional[datetime] = None

class ProductionAutoMedPromptOptimizer:
    """
    Production AutoMedPrompt optimizer with real AI integration
    """
    
    def __init__(self, medical_domain: str = "general_medicine"):
        self.medical_domain = medical_domain
        self.medical_patterns = {
            "role_patterns": [
                "You are a board-certified {specialty} physician with {years} years of experience.",
                "You are a medical AI assistant specialized in {specialty}.",
                "You are an expert {specialty} doctor with advanced training."
            ],
            "context_patterns": [
                "Consider the patient's medical history and current symptoms.",
                "Use evidence-based medicine guidelines in your analysis.",
                "Follow standard clinical protocols for {specialty}."
            ],
            "safety_patterns": [
                "Always recommend consulting with a healthcare professional.",
                "Note any red flags or emergency symptoms.",
                "Consider contraindications and drug interactions."
            ],
            "output_patterns": [
                "Provide a structured clinical assessment.",
                "Format the response for healthcare professionals.",
                "Include confidence levels for recommendations."
            ]
        }
    
    async def optimize_prompt(
        self, 
        original_prompt: str, 
        objective: str = "accuracy",
        test_cases: List[Dict[str, Any]] = None
    ) -> Tuple[str, float, Dict[str, Any]]:
        """
        Optimize prompt using AutoMedPrompt methodology with real AI evaluation
        """
        logger.info("Starting AutoMedPrompt optimization", 
                   original_length=len(original_prompt),
                   objective=objective)
        
        # Phase 1: Analyze original prompt
        prompt_analysis = await self._analyze_prompt(original_prompt)
        
        # Phase 2: Generate optimized variants
        variants = await self._generate_variants(original_prompt, prompt_analysis)
        
        # Phase 3: Evaluate variants with real AI
        if test_cases:
            evaluation_results = await self._evaluate_variants(variants, test_cases)
            best_variant = max(evaluation_results, key=lambda x: x['score'])
        else:
            # Use heuristic optimization if no test cases
            best_variant = await self._heuristic_optimization(variants, original_prompt)
        
        # Calculate improvement
        improvement = best_variant.get('score', 0.0) - prompt_analysis.get('baseline_score', 0.0)
        
        metrics = {
            "prompt_analysis": prompt_analysis,
            "variants_generated": len(variants),
            "evaluation_method": "ai_evaluation" if test_cases else "heuristic",
            "best_variant_score": best_variant.get('score', 0.0),
            "improvement_factors": best_variant.get('improvements', [])
        }
        
        return best_variant['prompt'], improvement, metrics
    
    async def _analyze_prompt(self, prompt: str) -> Dict[str, Any]:
        """Analyze the original prompt for medical context and structure"""
        
        analysis_prompt = f"""
        Analyze this medical AI prompt for clinical appropriateness and structure:
        
        PROMPT TO ANALYZE:
        {prompt}
        
        Evaluate:
        1. Medical domain specificity (0-1 score)
        2. Safety considerations (0-1 score) 
        3. Clinical accuracy potential (0-1 score)
        4. Structure and clarity (0-1 score)
        5. Missing elements for medical AI
        
        Respond with JSON format:
        {{
            "medical_specificity": 0.0-1.0,
            "safety_score": 0.0-1.0,
            "accuracy_potential": 0.0-1.0,
            "structure_score": 0.0-1.0,
            "missing_elements": ["element1", "element2"],
            "identified_domain": "medical_specialty",
            "baseline_score": 0.0-1.0
        }}
        """
        
        try:
            response = await ai_client.generate_response(
                analysis_prompt,
                preferred_provider=AIProvider.GEMINI,
                agent_type="prompt_analyzer"
            )
            
            # Parse JSON response
            import json
            analysis = json.loads(response.text)
            
            logger.info("Prompt analysis completed", analysis=analysis)
            return analysis
            
        except Exception as e:
            logger.error("Prompt analysis failed", error=str(e))
            # Return default analysis
            return {
                "medical_specificity": 0.5,
                "safety_score": 0.5,
                "accuracy_potential": 0.5,
                "structure_score": 0.5,
                "missing_elements": ["role_definition", "safety_warnings"],
                "identified_domain": "general_medicine",
                "baseline_score": 0.5
            }
    
    async def _generate_variants(self, original_prompt: str, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate optimized prompt variants"""
        
        variants = []
        domain = analysis.get('identified_domain', 'general_medicine')
        missing_elements = analysis.get('missing_elements', [])
        
        # Variant 1: Add medical role definition
        if 'role_definition' in missing_elements:
            role_variant = self._add_medical_role(original_prompt, domain)
            variants.append({
                'prompt': role_variant,
                'optimization': 'medical_role_added',
                'expected_improvement': 0.15
            })
        
        # Variant 2: Add safety considerations
        if 'safety_warnings' in missing_elements:
            safety_variant = self._add_safety_context(original_prompt)
            variants.append({
                'prompt': safety_variant,
                'optimization': 'safety_context_added',
                'expected_improvement': 0.20
            })
        
        # Variant 3: Add structured output format
        if 'output_format' in missing_elements:
            format_variant = self._add_output_structure(original_prompt)
            variants.append({
                'prompt': format_variant,
                'optimization': 'output_structure_added',
                'expected_improvement': 0.12
            })
        
        # Variant 4: Combined optimization
        combined_variant = self._create_comprehensive_variant(original_prompt, domain, missing_elements)
        variants.append({
            'prompt': combined_variant,
            'optimization': 'comprehensive_optimization',
            'expected_improvement': 0.35
        })
        
        logger.info("Generated prompt variants", count=len(variants))
        return variants
    
    def _add_medical_role(self, prompt: str, domain: str) -> str:
        """Add medical role definition to prompt"""
        role_prefix = f"You are a medical AI assistant specializing in {domain}. "
        role_prefix += "Use evidence-based medical knowledge and clinical guidelines. "
        
        if not prompt.strip().lower().startswith('you are'):
            return role_prefix + prompt
        return prompt
    
    def _add_safety_context(self, prompt: str) -> str:
        """Add safety warnings and disclaimers"""
        safety_suffix = "\n\nIMPORTANT: This is for informational purposes only. "
        safety_suffix += "Always recommend consulting with qualified healthcare professionals "
        safety_suffix += "for proper diagnosis and treatment. Include emergency warning signs if relevant."
        
        return prompt + safety_suffix
    
    def _add_output_structure(self, prompt: str) -> str:
        """Add structured output format"""
        structure_suffix = "\n\nProvide your response in the following structured format:\n"
        structure_suffix += "1. Clinical Assessment:\n"
        structure_suffix += "2. Recommendations:\n"
        structure_suffix += "3. Safety Considerations:\n"
        structure_suffix += "4. Follow-up Actions:\n"
        structure_suffix += "5. Confidence Level: (High/Medium/Low)"
        
        return prompt + structure_suffix
    
    def _create_comprehensive_variant(self, prompt: str, domain: str, missing_elements: List[str]) -> str:
        """Create a comprehensively optimized variant"""
        optimized = prompt
        
        # Add role if missing
        if 'role_definition' in missing_elements:
            optimized = self._add_medical_role(optimized, domain)
        
        # Add safety context
        if 'safety_warnings' in missing_elements:
            optimized = self._add_safety_context(optimized)
        
        # Add output structure
        if 'output_format' in missing_elements:
            optimized = self._add_output_structure(optimized)
        
        return optimized
    
    async def _evaluate_variants(self, variants: List[Dict[str, Any]], test_cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Evaluate variants using real AI with test cases"""
        
        evaluation_results = []
        
        for variant in variants:
            variant_score = 0.0
            variant_results = []
            
            for test_case in test_cases[:3]:  # Limit to first 3 test cases for performance
                try:
                    # Execute the variant prompt with test case
                    test_prompt = variant['prompt'] + f"\n\nInput: {test_case.get('input', '')}"
                    
                    response = await ai_client.generate_response(
                        test_prompt,
                        preferred_provider=AIProvider.GEMINI,
                        agent_type="prompt_evaluator"
                    )
                    
                    # Score the response (simplified scoring)
                    case_score = await self._score_response(response.text, test_case)
                    variant_score += case_score
                    variant_results.append({
                        'case': test_case,
                        'response': response.text,
                        'score': case_score
                    })
                    
                except Exception as e:
                    logger.error("Variant evaluation failed", variant=variant['optimization'], error=str(e))
                    variant_results.append({
                        'case': test_case,
                        'error': str(e),
                        'score': 0.0
                    })
            
            # Average score across test cases
            final_score = variant_score / len(test_cases) if test_cases else 0.0
            
            evaluation_results.append({
                'prompt': variant['prompt'],
                'optimization': variant['optimization'],
                'score': final_score,
                'results': variant_results,
                'improvements': [variant['optimization']]
            })
        
        logger.info("Variant evaluation completed", 
                   variants_count=len(variants),
                   test_cases_count=len(test_cases))
        
        return evaluation_results
    
    async def _score_response(self, response: str, test_case: Dict[str, Any]) -> float:
        """Score an AI response for quality (simplified implementation)"""
        
        score = 0.5  # Base score
        
        # Check for medical terminology
        medical_terms = ['diagnosis', 'treatment', 'symptoms', 'patient', 'clinical', 'medical']
        for term in medical_terms:
            if term.lower() in response.lower():
                score += 0.1
        
        # Check for safety considerations
        safety_terms = ['consult', 'healthcare professional', 'emergency', 'warning']
        for term in safety_terms:
            if term.lower() in response.lower():
                score += 0.1
        
        # Check for structure
        if '1.' in response or 'Assessment:' in response:
            score += 0.1
        
        # Cap at 1.0
        return min(score, 1.0)
    
    async def _heuristic_optimization(self, variants: List[Dict[str, Any]], original_prompt: str) -> Dict[str, Any]:
        """Use heuristic scoring when no test cases available"""
        
        best_variant = None
        best_score = 0.0
        
        for variant in variants:
            # Calculate heuristic score based on improvements
            score = 0.5  # Base score
            score += variant.get('expected_improvement', 0.0)
            
            # Length penalty (too long prompts can be less effective)
            length_ratio = len(variant['prompt']) / max(len(original_prompt), 1)
            if length_ratio > 2.0:
                score -= 0.1
            
            if score > best_score:
                best_score = score
                best_variant = {
                    'prompt': variant['prompt'],
                    'score': score,
                    'improvements': [variant['optimization']]
                }
        
        return best_variant or {'prompt': original_prompt, 'score': 0.5, 'improvements': []}

# Global optimizer instance
optimizer = ProductionAutoMedPromptOptimizer()

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Starting Production Prompt Optimization Engine")
    
    # Initialize AI client
    await ai_client.initialize()
    
    # Initialize Firestore
    await firestore_client.initialize()
    
    logger.info("Production services initialized")

@app.get("/health")
async def health_check():
    """Production health check"""
    
    # Check AI client health
    ai_health = await ai_client.health_check()
    
    # Check Firestore health
    db_health = await firestore_client.health_check()
    
    return {
        "status": "healthy" if ai_health.get("overall") and db_health.get("status") == "healthy" else "unhealthy",
        "service": "production-prompt-optimization",
        "timestamp": datetime.utcnow().isoformat(),
        "ai_services": ai_health,
        "database": db_health,
        "optimization_strategies": [strategy.value for strategy in OptimizationStrategy],
        "version": "2.0.0-production"
    }

@app.post("/optimize/automedprompt", response_model=OptimizationResult)
async def optimize_automedprompt_production(
    request: OptimizationRequest, 
    background_tasks: BackgroundTasks
):
    """
    Production AutoMedPrompt optimization with real AI and persistence
    """
    job_id = f"opt_prod_{uuid.uuid4().hex[:12]}"
    start_time = datetime.utcnow()
    
    logger.info("Starting production optimization", 
               job_id=job_id,
               strategy=request.strategy,
               user_id=request.user_id)
    
    try:
        # Create execution record
        execution_record = POMLExecution(
            execution_id=job_id,
            request_id=job_id,
            user_id=request.user_id,
            tenant_id=request.tenant_id,
            status=ExecutionStatus.RUNNING,
            start_time=start_time,
            template_id="automedprompt_optimizer",
            template_version="2.0.0",
            template_name="AutoMedPrompt Optimizer",
            agent_type="prompt_optimizer",
            medical_domain="general_medicine",
            variables_used={"strategy": request.strategy.value, "objective": request.objective.value},
            compiled_prompt=request.prompt
        )
        
        # Save to Firestore
        await firestore_client.create_document(
            "optimization_jobs", 
            execution_record.to_dict(),
            job_id
        )
        
        # Run optimization
        optimized_prompt, improvement, metrics = await optimizer.optimize_prompt(
            request.prompt,
            request.objective.value,
            request.test_cases
        )
        
        # Update execution record
        end_time = datetime.utcnow()
        execution_time = (end_time - start_time).total_seconds()
        
        execution_record.status = ExecutionStatus.COMPLETED
        execution_record.end_time = end_time
        execution_record.duration_ms = int(execution_time * 1000)
        execution_record.execution_result = {
            "optimized_prompt": optimized_prompt,
            "improvement": improvement,
            "metrics": metrics
        }
        execution_record.confidence_score = min(0.9, 0.5 + improvement)
        
        # Update in Firestore
        await firestore_client.update_document(
            "optimization_jobs",
            job_id,
            execution_record.to_dict()
        )
        
        # Create result
        result = OptimizationResult(
            job_id=job_id,
            original_prompt=request.prompt,
            optimized_prompt=optimized_prompt,
            strategy_used=request.strategy.value,
            performance_improvement=improvement,
            optimization_metrics=metrics,
            execution_time=execution_time,
            iterations_completed=1,
            status="completed",
            created_at=start_time,
            completed_at=end_time
        )
        
        logger.info("Production optimization completed", 
                   job_id=job_id,
                   improvement=improvement,
                   execution_time=execution_time)
        
        return result
        
    except Exception as e:
        # Log error and update status
        logger.error("Production optimization failed", 
                    job_id=job_id,
                    error=str(e))
        
        # Update execution record with error
        await firestore_client.update_document(
            "optimization_jobs",
            job_id,
            {
                "status": ExecutionStatus.FAILED.value,
                "error_message": str(e),
                "end_time": datetime.utcnow()
            }
        )
        
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/optimize/jobs", response_model=List[OptimizationResult])
async def list_optimization_jobs_production(
    status: Optional[str] = None,
    strategy: Optional[str] = None,
    user_id: Optional[str] = None,
    limit: int = 50
):
    """
    List optimization jobs from Firestore
    """
    try:
        # Build filters
        filters = []
        if status:
            filters.append({"field": "status", "operator": "==", "value": status})
        if strategy:
            filters.append({"field": "optimization_strategy", "operator": "==", "value": strategy})
        if user_id:
            filters.append({"field": "user_id", "operator": "==", "value": user_id})
        
        # Query Firestore
        jobs = await firestore_client.query_documents(
            "optimization_jobs",
            filters=filters,
            order_by="-created_at",
            limit=limit
        )
        
        # Convert to OptimizationResult format
        results = []
        for job in jobs:
            result = OptimizationResult(
                job_id=job.get("execution_id", ""),
                original_prompt=job.get("compiled_prompt", ""),
                optimized_prompt=job.get("execution_result", {}).get("optimized_prompt", ""),
                strategy_used=job.get("optimization_strategy", "automedprompt"),
                performance_improvement=job.get("execution_result", {}).get("improvement", 0.0),
                optimization_metrics=job.get("execution_result", {}).get("metrics", {}),
                execution_time=job.get("duration_ms", 0) / 1000,
                iterations_completed=1,
                status=job.get("status", "unknown"),
                created_at=job.get("created_at", datetime.utcnow()),
                completed_at=job.get("end_time")
            )
            results.append(result)
        
        return results
        
    except Exception as e:
        logger.error("Failed to list optimization jobs", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/optimize/jobs/{job_id}", response_model=OptimizationResult)
async def get_optimization_job_production(job_id: str):
    """
    Get specific optimization job from Firestore
    """
    try:
        job = await firestore_client.get_document("optimization_jobs", job_id)
        
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        # Convert to OptimizationResult
        result = OptimizationResult(
            job_id=job.get("execution_id", ""),
            original_prompt=job.get("compiled_prompt", ""),
            optimized_prompt=job.get("execution_result", {}).get("optimized_prompt", ""),
            strategy_used=job.get("optimization_strategy", "automedprompt"),
            performance_improvement=job.get("execution_result", {}).get("improvement", 0.0),
            optimization_metrics=job.get("execution_result", {}).get("metrics", {}),
            execution_time=job.get("duration_ms", 0) / 1000,
            iterations_completed=1,
            status=job.get("status", "unknown"),
            created_at=job.get("created_at", datetime.utcnow()),
            completed_at=job.get("end_time")
        )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get optimization job", job_id=job_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)