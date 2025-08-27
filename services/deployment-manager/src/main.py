"""
SADP Deployment Manager - Progressive Rollout & A/B Testing
Advanced deployment strategies for POML templates with automated rollback
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timedelta
import os
import json
import asyncio
import uuid
import structlog
import httpx
import random
import math
from enum import Enum
from google.cloud import firestore, storage, pubsub_v1

# Initialize logger
logger = structlog.get_logger()

app = FastAPI(
    title="SADP Deployment Manager",
    description="Progressive rollout and A/B testing for POML templates",
    version="1.0.0"
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
POML_ORCHESTRATOR_URL = os.environ.get("POML_ORCHESTRATOR_URL", "https://sadp-poml-orchestrator-xonau6hybq-uc.a.run.app")
LEARNING_PIPELINE_URL = os.environ.get("LEARNING_PIPELINE_URL", "https://sadp-learning-pipeline-xonau6hybq-uc.a.run.app")

# Initialize clients
firestore_client = firestore.Client(project=PROJECT_ID)
storage_client = storage.Client(project=PROJECT_ID)
publisher = pubsub_v1.PublisherClient()

# Collections
deployments_collection = firestore_client.collection('template_deployments')
experiments_collection = firestore_client.collection('ab_experiments')
traffic_collection = firestore_client.collection('traffic_routing')
rollback_collection = firestore_client.collection('rollback_events')

# Enums
class DeploymentStrategy(str, Enum):
    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    ROLLING = "rolling"
    A_B_TEST = "ab_test"
    SHADOW = "shadow"

class DeploymentStatus(str, Enum):
    PENDING = "pending"
    DEPLOYING = "deploying"
    DEPLOYED = "deployed"
    ROLLING_BACK = "rolling_back"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"
    PAUSED = "paused"

class RollbackTrigger(str, Enum):
    PERFORMANCE_DEGRADATION = "performance_degradation"
    ERROR_RATE_SPIKE = "error_rate_spike"
    MANUAL = "manual"
    TIMEOUT = "timeout"
    A_B_TEST_FAILURE = "ab_test_failure"

# Models
class DeploymentRequest(BaseModel):
    template_id: str = Field(..., description="Template to deploy")
    strategy: DeploymentStrategy = Field(default=DeploymentStrategy.CANARY)
    target_environments: List[str] = Field(default=["production"], description="Target environments")
    initial_traffic_percentage: float = Field(default=5.0, ge=0, le=100)
    rollout_schedule: Optional[List[Dict[str, Any]]] = Field(None, description="Automatic rollout schedule")
    rollback_config: Dict[str, Any] = Field(default={}, description="Rollback configuration")
    health_checks: List[Dict[str, Any]] = Field(default=[], description="Health check configurations")
    
class ABTestRequest(BaseModel):
    experiment_name: str = Field(..., description="A/B test experiment name")
    template_a_id: str = Field(..., description="Control template ID")
    template_b_id: str = Field(..., description="Variant template ID")
    traffic_split: float = Field(default=0.5, ge=0, le=1, description="Traffic split for variant B")
    success_metrics: List[str] = Field(..., description="Metrics to track for success")
    minimum_sample_size: int = Field(default=1000, description="Minimum sample size")
    max_duration_hours: int = Field(default=168, description="Maximum test duration (hours)")
    significance_threshold: float = Field(default=0.05, description="Statistical significance threshold")
    
class RollbackConfig(BaseModel):
    enabled: bool = Field(default=True)
    error_rate_threshold: float = Field(default=0.05, description="Error rate threshold for auto-rollback")
    latency_threshold_ms: int = Field(default=5000, description="Latency threshold for auto-rollback")
    success_rate_threshold: float = Field(default=0.95, description="Success rate threshold")
    monitoring_window_minutes: int = Field(default=10, description="Monitoring window for metrics")

class HealthCheckConfig(BaseModel):
    endpoint: str = Field(..., description="Health check endpoint")
    interval_seconds: int = Field(default=30, description="Check interval")
    timeout_seconds: int = Field(default=10, description="Request timeout")
    failure_threshold: int = Field(default=3, description="Consecutive failures before marking unhealthy")
    success_threshold: int = Field(default=2, description="Consecutive successes before marking healthy")

class DeploymentManager:
    """Manages progressive deployments and A/B testing"""
    
    def __init__(self):
        self.http_client = httpx.AsyncClient()
        self.active_deployments = {}
        self.active_experiments = {}
        
    async def start_deployment(self, request: DeploymentRequest, deployment_id: str) -> Dict[str, Any]:
        """Start a new deployment with progressive rollout"""
        try:
            logger.info("Starting deployment", 
                       deployment_id=deployment_id, 
                       template_id=request.template_id,
                       strategy=request.strategy)
            
            # Create deployment record
            deployment_record = {
                "deployment_id": deployment_id,
                "template_id": request.template_id,
                "strategy": request.strategy.value,
                "target_environments": request.target_environments,
                "current_traffic_percentage": 0.0,
                "target_traffic_percentage": request.initial_traffic_percentage,
                "status": DeploymentStatus.PENDING.value,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
                "rollback_config": request.rollback_config,
                "health_checks": request.health_checks,
                "rollout_history": []
            }
            
            deployments_collection.document(deployment_id).set(deployment_record)
            
            # Execute deployment strategy
            if request.strategy == DeploymentStrategy.BLUE_GREEN:
                await self._execute_blue_green_deployment(deployment_id, request)
            elif request.strategy == DeploymentStrategy.CANARY:
                await self._execute_canary_deployment(deployment_id, request)
            elif request.strategy == DeploymentStrategy.ROLLING:
                await self._execute_rolling_deployment(deployment_id, request)
            elif request.strategy == DeploymentStrategy.A_B_TEST:
                await self._execute_ab_test_deployment(deployment_id, request)
            elif request.strategy == DeploymentStrategy.SHADOW:
                await self._execute_shadow_deployment(deployment_id, request)
            
            return {"deployment_id": deployment_id, "status": "started"}
            
        except Exception as e:
            logger.error("Failed to start deployment", deployment_id=deployment_id, error=str(e))
            await self._mark_deployment_failed(deployment_id, str(e))
            raise HTTPException(status_code=500, detail=f"Deployment failed: {str(e)}")
    
    async def _execute_canary_deployment(self, deployment_id: str, request: DeploymentRequest):
        """Execute canary deployment strategy"""
        try:
            # Phase 1: Deploy to canary environment with initial traffic
            await self._update_deployment_status(deployment_id, DeploymentStatus.DEPLOYING)
            
            # Route initial traffic to new template
            await self._route_traffic(deployment_id, request.template_id, request.initial_traffic_percentage)
            
            # Start health monitoring
            await self._start_health_monitoring(deployment_id)
            
            # If rollout schedule is provided, execute it
            if request.rollout_schedule:
                await self._execute_rollout_schedule(deployment_id, request.rollout_schedule)
            
            await self._update_deployment_status(deployment_id, DeploymentStatus.DEPLOYED)
            
        except Exception as e:
            logger.error("Canary deployment failed", deployment_id=deployment_id, error=str(e))
            await self._trigger_rollback(deployment_id, RollbackTrigger.MANUAL, str(e))
    
    async def _execute_blue_green_deployment(self, deployment_id: str, request: DeploymentRequest):
        """Execute blue-green deployment strategy"""
        try:
            await self._update_deployment_status(deployment_id, DeploymentStatus.DEPLOYING)
            
            # Deploy to green environment (0% traffic initially)
            await self._route_traffic(deployment_id, request.template_id, 0.0)
            
            # Validate green environment
            health_check_passed = await self._validate_environment(deployment_id, request.template_id)
            
            if health_check_passed:
                # Switch traffic to green (100%)
                await self._route_traffic(deployment_id, request.template_id, 100.0)
                await self._update_deployment_status(deployment_id, DeploymentStatus.DEPLOYED)
            else:
                raise Exception("Green environment health checks failed")
                
        except Exception as e:
            logger.error("Blue-green deployment failed", deployment_id=deployment_id, error=str(e))
            await self._trigger_rollback(deployment_id, RollbackTrigger.MANUAL, str(e))
    
    async def _execute_rolling_deployment(self, deployment_id: str, request: DeploymentRequest):
        """Execute rolling deployment strategy"""
        try:
            await self._update_deployment_status(deployment_id, DeploymentStatus.DEPLOYING)
            
            # Rolling deployment in 25% increments
            rollout_steps = [25.0, 50.0, 75.0, 100.0]
            
            for step_percentage in rollout_steps:
                await self._route_traffic(deployment_id, request.template_id, step_percentage)
                
                # Wait and monitor each step
                await asyncio.sleep(300)  # 5 minutes between steps
                
                # Check health metrics
                health_ok = await self._check_deployment_health(deployment_id)
                if not health_ok:
                    raise Exception(f"Health check failed at {step_percentage}% traffic")
            
            await self._update_deployment_status(deployment_id, DeploymentStatus.DEPLOYED)
            
        except Exception as e:
            logger.error("Rolling deployment failed", deployment_id=deployment_id, error=str(e))
            await self._trigger_rollback(deployment_id, RollbackTrigger.PERFORMANCE_DEGRADATION, str(e))
    
    async def _execute_ab_test_deployment(self, deployment_id: str, request: DeploymentRequest):
        """Execute A/B test deployment strategy"""
        try:
            await self._update_deployment_status(deployment_id, DeploymentStatus.DEPLOYING)
            
            # For A/B test, we need both templates
            # This would be part of an AB test request, but for now we'll use initial traffic percentage
            await self._route_traffic(deployment_id, request.template_id, request.initial_traffic_percentage)
            
            # Start A/B test monitoring
            await self._start_ab_test_monitoring(deployment_id)
            
            await self._update_deployment_status(deployment_id, DeploymentStatus.DEPLOYED)
            
        except Exception as e:
            logger.error("A/B test deployment failed", deployment_id=deployment_id, error=str(e))
            await self._trigger_rollback(deployment_id, RollbackTrigger.A_B_TEST_FAILURE, str(e))
    
    async def _execute_shadow_deployment(self, deployment_id: str, request: DeploymentRequest):
        """Execute shadow deployment strategy"""
        try:
            await self._update_deployment_status(deployment_id, DeploymentStatus.DEPLOYING)
            
            # Shadow deployment: route duplicate traffic to new template (0% user-facing)
            await self._setup_shadow_traffic(deployment_id, request.template_id)
            
            # Monitor shadow performance
            await self._monitor_shadow_deployment(deployment_id)
            
            await self._update_deployment_status(deployment_id, DeploymentStatus.DEPLOYED)
            
        except Exception as e:
            logger.error("Shadow deployment failed", deployment_id=deployment_id, error=str(e))
            await self._trigger_rollback(deployment_id, RollbackTrigger.MANUAL, str(e))
    
    async def _route_traffic(self, deployment_id: str, template_id: str, percentage: float):
        """Route traffic to template"""
        try:
            traffic_record = {
                "deployment_id": deployment_id,
                "template_id": template_id,
                "traffic_percentage": percentage,
                "updated_at": datetime.utcnow(),
                "active": True
            }
            
            traffic_collection.document(f"{deployment_id}_{template_id}").set(traffic_record)
            
            # Update deployment record
            deployments_collection.document(deployment_id).update({
                "current_traffic_percentage": percentage,
                "updated_at": datetime.utcnow()
            })
            
            logger.info("Traffic routed", 
                       deployment_id=deployment_id, 
                       template_id=template_id, 
                       percentage=percentage)
            
        except Exception as e:
            logger.error("Failed to route traffic", deployment_id=deployment_id, error=str(e))
            raise
    
    async def _start_health_monitoring(self, deployment_id: str):
        """Start health monitoring for deployment"""
        try:
            # Get deployment config
            doc = deployments_collection.document(deployment_id).get()
            if not doc.exists:
                return
            
            deployment_data = doc.to_dict()
            health_checks = deployment_data.get("health_checks", [])
            rollback_config = deployment_data.get("rollback_config", {})
            
            # Start background monitoring task
            asyncio.create_task(self._monitor_deployment_health(deployment_id, health_checks, rollback_config))
            
        except Exception as e:
            logger.error("Failed to start health monitoring", deployment_id=deployment_id, error=str(e))
    
    async def _monitor_deployment_health(self, deployment_id: str, health_checks: List[Dict], rollback_config: Dict):
        """Monitor deployment health and trigger rollback if needed"""
        try:
            monitoring_window = rollback_config.get("monitoring_window_minutes", 10)
            check_interval = 60  # Check every minute
            
            for _ in range(monitoring_window):
                await asyncio.sleep(check_interval)
                
                # Check if deployment is still active
                doc = deployments_collection.document(deployment_id).get()
                if not doc.exists or doc.to_dict().get("status") != DeploymentStatus.DEPLOYED.value:
                    break
                
                # Perform health checks
                health_ok = await self._check_deployment_health(deployment_id)
                
                if not health_ok:
                    await self._trigger_rollback(
                        deployment_id, 
                        RollbackTrigger.PERFORMANCE_DEGRADATION,
                        "Health check failed during monitoring"
                    )
                    break
                    
        except Exception as e:
            logger.error("Health monitoring failed", deployment_id=deployment_id, error=str(e))
    
    async def _check_deployment_health(self, deployment_id: str) -> bool:
        """Check deployment health metrics"""
        try:
            # Simulate health check by calling template performance metrics
            doc = deployments_collection.document(deployment_id).get()
            if not doc.exists:
                return False
            
            deployment_data = doc.to_dict()
            template_id = deployment_data["template_id"]
            
            # Call POML orchestrator for performance metrics
            try:
                response = await self.http_client.get(
                    f"{POML_ORCHESTRATOR_URL}/templates/{template_id}/performance",
                    timeout=10.0
                )
                
                if response.status_code == 200:
                    metrics = response.json()
                    performance_metrics = metrics.get("performance_metrics", {})
                    
                    # Check key health indicators
                    success_rate = performance_metrics.get("success_rate", 0.0)
                    avg_latency = performance_metrics.get("average_latency_ms", 0.0)
                    
                    # Apply thresholds from rollback config
                    rollback_config = deployment_data.get("rollback_config", {})
                    success_threshold = rollback_config.get("success_rate_threshold", 0.95)
                    latency_threshold = rollback_config.get("latency_threshold_ms", 5000)
                    
                    if success_rate < success_threshold:
                        logger.warning("Success rate below threshold", 
                                     deployment_id=deployment_id,
                                     current=success_rate,
                                     threshold=success_threshold)
                        return False
                    
                    if avg_latency > latency_threshold:
                        logger.warning("Latency above threshold",
                                     deployment_id=deployment_id,
                                     current=avg_latency,
                                     threshold=latency_threshold)
                        return False
                    
                    return True
                else:
                    return False
                    
            except httpx.TimeoutException:
                logger.warning("Health check timeout", deployment_id=deployment_id)
                return False
            except Exception as e:
                logger.error("Health check error", deployment_id=deployment_id, error=str(e))
                return False
                
        except Exception as e:
            logger.error("Failed to check deployment health", deployment_id=deployment_id, error=str(e))
            return False
    
    async def _validate_environment(self, deployment_id: str, template_id: str) -> bool:
        """Validate deployment environment"""
        try:
            # Test template execution
            response = await self.http_client.post(
                f"{POML_ORCHESTRATOR_URL}/templates/{template_id}/execute",
                json={
                    "variables": {"test": "validation"},
                    "context": {},
                    "execution_mode": "validation",
                    "timeout_seconds": 30
                },
                timeout=35.0
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("status") == "completed"
            else:
                return False
                
        except Exception as e:
            logger.error("Environment validation failed", deployment_id=deployment_id, error=str(e))
            return False
    
    async def _trigger_rollback(self, deployment_id: str, trigger: RollbackTrigger, reason: str):
        """Trigger deployment rollback"""
        try:
            logger.warning("Triggering rollback", 
                          deployment_id=deployment_id, 
                          trigger=trigger.value,
                          reason=reason)
            
            # Update deployment status
            await self._update_deployment_status(deployment_id, DeploymentStatus.ROLLING_BACK)
            
            # Create rollback event
            rollback_event = {
                "deployment_id": deployment_id,
                "trigger": trigger.value,
                "reason": reason,
                "triggered_at": datetime.utcnow(),
                "status": "in_progress"
            }
            
            rollback_id = f"rollback_{uuid.uuid4().hex[:12]}"
            rollback_collection.document(rollback_id).set(rollback_event)
            
            # Execute rollback
            await self._execute_rollback(deployment_id, rollback_id)
            
        except Exception as e:
            logger.error("Failed to trigger rollback", deployment_id=deployment_id, error=str(e))
    
    async def _execute_rollback(self, deployment_id: str, rollback_id: str):
        """Execute rollback to previous version"""
        try:
            # Get previous deployment state
            doc = deployments_collection.document(deployment_id).get()
            if not doc.exists:
                raise Exception("Deployment not found")
            
            deployment_data = doc.to_dict()
            
            # Rollback traffic to 0%
            await self._route_traffic(deployment_id, deployment_data["template_id"], 0.0)
            
            # Update status
            await self._update_deployment_status(deployment_id, DeploymentStatus.ROLLED_BACK)
            
            # Update rollback event
            rollback_collection.document(rollback_id).update({
                "status": "completed",
                "completed_at": datetime.utcnow()
            })
            
            logger.info("Rollback completed", deployment_id=deployment_id, rollback_id=rollback_id)
            
        except Exception as e:
            logger.error("Rollback execution failed", deployment_id=deployment_id, error=str(e))
            
            # Update rollback event with failure
            rollback_collection.document(rollback_id).update({
                "status": "failed",
                "error": str(e),
                "completed_at": datetime.utcnow()
            })
    
    async def _execute_rollout_schedule(self, deployment_id: str, schedule: List[Dict[str, Any]]):
        """Execute automatic rollout schedule"""
        try:
            for step in schedule:
                # Wait for specified time
                delay_minutes = step.get("delay_minutes", 30)
                await asyncio.sleep(delay_minutes * 60)
                
                # Check if deployment is still active
                doc = deployments_collection.document(deployment_id).get()
                if not doc.exists or doc.to_dict().get("status") != DeploymentStatus.DEPLOYED.value:
                    break
                
                # Increase traffic
                target_percentage = step.get("traffic_percentage", 100.0)
                template_id = doc.to_dict()["template_id"]
                
                await self._route_traffic(deployment_id, template_id, target_percentage)
                
                # Health check after each step
                await asyncio.sleep(300)  # Wait 5 minutes
                health_ok = await self._check_deployment_health(deployment_id)
                
                if not health_ok:
                    await self._trigger_rollback(
                        deployment_id,
                        RollbackTrigger.PERFORMANCE_DEGRADATION,
                        f"Health check failed at {target_percentage}% during scheduled rollout"
                    )
                    break
                    
        except Exception as e:
            logger.error("Rollout schedule execution failed", deployment_id=deployment_id, error=str(e))
    
    async def _setup_shadow_traffic(self, deployment_id: str, template_id: str):
        """Setup shadow traffic for deployment"""
        # Shadow deployment implementation would duplicate traffic
        # For now, we'll just mark it as shadow
        await self._route_traffic(deployment_id, template_id, 0.0)
        
        # Mark as shadow deployment
        deployments_collection.document(deployment_id).update({
            "shadow_mode": True,
            "updated_at": datetime.utcnow()
        })
    
    async def _monitor_shadow_deployment(self, deployment_id: str):
        """Monitor shadow deployment"""
        # Monitor shadow deployment performance vs production
        # For now, just log that monitoring started
        logger.info("Shadow deployment monitoring started", deployment_id=deployment_id)
    
    async def _start_ab_test_monitoring(self, deployment_id: str):
        """Start A/B test monitoring"""
        logger.info("A/B test monitoring started", deployment_id=deployment_id)
    
    async def _update_deployment_status(self, deployment_id: str, status: DeploymentStatus):
        """Update deployment status"""
        deployments_collection.document(deployment_id).update({
            "status": status.value,
            "updated_at": datetime.utcnow()
        })
    
    async def _mark_deployment_failed(self, deployment_id: str, error: str):
        """Mark deployment as failed"""
        deployments_collection.document(deployment_id).update({
            "status": DeploymentStatus.FAILED.value,
            "error": error,
            "updated_at": datetime.utcnow()
        })

# Initialize deployment manager
deployment_manager = DeploymentManager()

@app.post("/deployments")
async def create_deployment(request: DeploymentRequest, background_tasks: BackgroundTasks):
    """Create a new deployment"""
    try:
        deployment_id = f"deploy_{uuid.uuid4().hex[:12]}"
        
        # Start deployment in background
        background_tasks.add_task(deployment_manager.start_deployment, request, deployment_id)
        
        return {
            "deployment_id": deployment_id,
            "status": "queued",
            "message": f"Deployment started for template {request.template_id}"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/deployments/{deployment_id}")
async def get_deployment(deployment_id: str):
    """Get deployment details"""
    try:
        doc = deployments_collection.document(deployment_id).get()
        
        if not doc.exists:
            raise HTTPException(status_code=404, detail="Deployment not found")
        
        return doc.to_dict()
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/deployments")
async def list_deployments(
    status: Optional[str] = None,
    template_id: Optional[str] = None,
    limit: int = 50
):
    """List deployments with optional filters"""
    try:
        query = deployments_collection.order_by("created_at", direction=firestore.Query.DESCENDING)
        
        if status:
            query = query.where("status", "==", status)
        if template_id:
            query = query.where("template_id", "==", template_id)
        
        query = query.limit(limit)
        
        deployments = []
        for doc in query.stream():
            deployment_data = doc.to_dict()
            deployments.append({
                "deployment_id": deployment_data["deployment_id"],
                "template_id": deployment_data["template_id"],
                "strategy": deployment_data["strategy"],
                "status": deployment_data["status"],
                "current_traffic_percentage": deployment_data.get("current_traffic_percentage", 0),
                "created_at": deployment_data["created_at"],
                "updated_at": deployment_data["updated_at"]
            })
        
        return {"deployments": deployments, "total": len(deployments)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/deployments/{deployment_id}/rollback")
async def rollback_deployment(deployment_id: str, reason: str = "Manual rollback"):
    """Manually trigger deployment rollback"""
    try:
        await deployment_manager._trigger_rollback(
            deployment_id, 
            RollbackTrigger.MANUAL, 
            reason
        )
        
        return {"message": "Rollback triggered", "deployment_id": deployment_id}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/deployments/{deployment_id}/traffic")
async def update_traffic(deployment_id: str, percentage: float):
    """Manually update traffic percentage"""
    try:
        if not 0 <= percentage <= 100:
            raise HTTPException(status_code=400, detail="Percentage must be between 0 and 100")
        
        doc = deployments_collection.document(deployment_id).get()
        if not doc.exists:
            raise HTTPException(status_code=404, detail="Deployment not found")
        
        deployment_data = doc.to_dict()
        template_id = deployment_data["template_id"]
        
        await deployment_manager._route_traffic(deployment_id, template_id, percentage)
        
        return {"message": "Traffic updated", "percentage": percentage}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ab-tests")
async def create_ab_test(request: ABTestRequest, background_tasks: BackgroundTasks):
    """Create a new A/B test"""
    try:
        experiment_id = f"exp_{uuid.uuid4().hex[:12]}"
        
        experiment_record = {
            "experiment_id": experiment_id,
            "experiment_name": request.experiment_name,
            "template_a_id": request.template_a_id,
            "template_b_id": request.template_b_id,
            "traffic_split": request.traffic_split,
            "success_metrics": request.success_metrics,
            "minimum_sample_size": request.minimum_sample_size,
            "max_duration_hours": request.max_duration_hours,
            "significance_threshold": request.significance_threshold,
            "status": "running",
            "created_at": datetime.utcnow(),
            "results": {
                "template_a": {"samples": 0, "conversions": 0, "metrics": {}},
                "template_b": {"samples": 0, "conversions": 0, "metrics": {}}
            }
        }
        
        experiments_collection.document(experiment_id).set(experiment_record)
        
        return {
            "experiment_id": experiment_id,
            "status": "created",
            "message": f"A/B test '{request.experiment_name}' created"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ab-tests/{experiment_id}")
async def get_ab_test(experiment_id: str):
    """Get A/B test details and results"""
    try:
        doc = experiments_collection.document(experiment_id).get()
        
        if not doc.exists:
            raise HTTPException(status_code=404, detail="A/B test not found")
        
        experiment_data = doc.to_dict()
        
        # Calculate statistical significance if enough samples
        results = experiment_data.get("results", {})
        template_a_results = results.get("template_a", {})
        template_b_results = results.get("template_b", {})
        
        a_samples = template_a_results.get("samples", 0)
        b_samples = template_b_results.get("samples", 0)
        
        statistical_analysis = {}
        if a_samples >= 100 and b_samples >= 100:
            # Simple statistical analysis
            a_rate = template_a_results.get("conversions", 0) / a_samples
            b_rate = template_b_results.get("conversions", 0) / b_samples
            
            statistical_analysis = {
                "template_a_rate": a_rate,
                "template_b_rate": b_rate,
                "lift": ((b_rate - a_rate) / a_rate * 100) if a_rate > 0 else 0,
                "winner": "template_b" if b_rate > a_rate else "template_a",
                "confidence": "high" if abs(b_rate - a_rate) > 0.1 else "low"
            }
        
        return {
            **experiment_data,
            "statistical_analysis": statistical_analysis
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test Firestore connection
        deployments_collection.limit(1).get()
        
        # Count active deployments
        active_deployments = len(list(
            deployments_collection.where("status", "==", "deployed").stream()
        ))
        
        # Count active experiments
        active_experiments = len(list(
            experiments_collection.where("status", "==", "running").stream()
        ))
        
        return {
            "status": "healthy",
            "service": "deployment-manager",
            "timestamp": datetime.utcnow().isoformat(),
            "active_deployments": active_deployments,
            "active_experiments": active_experiments,
            "deployment_strategies": [strategy.value for strategy in DeploymentStrategy],
            "version": "1.0.0"
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "service": "deployment-manager",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)