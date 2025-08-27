"""
SADP Evaluation Service
Comprehensive agent performance evaluation and metrics collection
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer
from pydantic import BaseModel, ValidationError
import uvicorn

from evaluator import EvaluationEngine
from metrics import MetricsCollector, PerformanceAnalyzer
from validators import ResultValidator
from models import (
    EvaluationRequest, EvaluationResult, EvaluationConfig,
    AgentPerformanceMetrics, BenchmarkResult
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
    logger.info("Starting SADP Evaluation Service", version="1.0.0")
    
    # Initialize services
    app.state.evaluation_engine = EvaluationEngine(settings)
    app.state.metrics_collector = MetricsCollector(settings)
    app.state.performance_analyzer = PerformanceAnalyzer(settings)
    app.state.result_validator = ResultValidator(settings)
    app.state.pubsub_client = PubSubClient(settings)
    
    # Setup telemetry
    setup_telemetry(settings)
    
    # Start background tasks
    evaluation_task = asyncio.create_task(
        app.state.evaluation_engine.start_continuous_evaluation()
    )
    metrics_task = asyncio.create_task(
        app.state.metrics_collector.start_collection()
    )
    
    logger.info("Evaluation service started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down evaluation service")
    evaluation_task.cancel()
    metrics_task.cancel()
    
    try:
        await evaluation_task
        await metrics_task
    except asyncio.CancelledError:
        pass
    
    await app.state.pubsub_client.close()
    logger.info("Evaluation service shutdown complete")

# Create FastAPI application
app = FastAPI(
    title="SADP Evaluation Service",
    description="Agent performance evaluation and metrics collection service",
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

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint for load balancer"""
    return {
        "status": "healthy",
        "service": "evaluation",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/ready")
async def readiness_check():
    """Readiness check for Kubernetes"""
    try:
        # Check database connection
        await app.state.metrics_collector.health_check()
        return {"status": "ready"}
    except Exception as e:
        logger.error("Readiness check failed", error=str(e))
        raise HTTPException(status_code=503, detail="Service not ready")

# Evaluation endpoints
@app.post("/evaluate", response_model=EvaluationResult)
async def evaluate_agent(
    request: EvaluationRequest,
    background_tasks: BackgroundTasks,
    current_user: TokenData = Depends(require_permission("agent:evaluate"))
):
    """Evaluate agent performance on given test cases"""
    try:
        logger.info(
            "Starting agent evaluation",
            agent_id=request.agent_id,
            user_id=current_user.user_id,
            organization_id=current_user.organization_id
        )
        
        # Validate request
        if not request.test_cases:
            raise HTTPException(
                status_code=400,
                detail="At least one test case is required"
            )
        
        # Perform evaluation
        result = await app.state.evaluation_engine.evaluate_agent(
            agent_id=request.agent_id,
            test_cases=request.test_cases,
            config=request.config,
            user_id=current_user.user_id,
            organization_id=current_user.organization_id
        )
        
        # Schedule background analysis
        background_tasks.add_task(
            app.state.performance_analyzer.analyze_evaluation_result,
            result
        )
        
        logger.info(
            "Agent evaluation completed",
            evaluation_id=result.evaluation_id,
            accuracy=result.overall_accuracy,
            duration_ms=result.evaluation_duration_ms
        )
        
        return result
        
    except ValidationError as e:
        logger.error("Validation error in evaluation request", error=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Failed to evaluate agent", error=str(e))
        raise HTTPException(status_code=500, detail="Evaluation failed")

@app.post("/evaluate/batch", response_model=List[EvaluationResult])
async def evaluate_agents_batch(
    requests: List[EvaluationRequest],
    background_tasks: BackgroundTasks,
    current_user: TokenData = Depends(require_permission("agent:evaluate"))
):
    """Batch evaluate multiple agents"""
    try:
        logger.info(
            "Starting batch evaluation",
            count=len(requests),
            user_id=current_user.user_id
        )
        
        if len(requests) > 10:
            raise HTTPException(
                status_code=400,
                detail="Maximum 10 evaluations per batch"
            )
        
        results = []
        for request in requests:
            result = await app.state.evaluation_engine.evaluate_agent(
                agent_id=request.agent_id,
                test_cases=request.test_cases,
                config=request.config,
                user_id=current_user.user_id,
                organization_id=current_user.organization_id
            )
            results.append(result)
            
            # Schedule background analysis
            background_tasks.add_task(
                app.state.performance_analyzer.analyze_evaluation_result,
                result
            )
        
        logger.info("Batch evaluation completed", count=len(results))
        return results
        
    except Exception as e:
        logger.error("Failed to evaluate agents batch", error=str(e))
        raise HTTPException(status_code=500, detail="Batch evaluation failed")

@app.get("/evaluations/{evaluation_id}", response_model=EvaluationResult)
async def get_evaluation_result(
    evaluation_id: str,
    current_user: TokenData = Depends(require_permission("agent:read"))
):
    """Get evaluation result by ID"""
    try:
        result = await app.state.evaluation_engine.get_evaluation_result(
            evaluation_id, current_user.organization_id
        )
        
        if not result:
            raise HTTPException(status_code=404, detail="Evaluation not found")
        
        return result
        
    except Exception as e:
        logger.error("Failed to get evaluation result", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve evaluation")

@app.get("/agents/{agent_id}/metrics", response_model=AgentPerformanceMetrics)
async def get_agent_metrics(
    agent_id: str,
    days: int = 30,
    current_user: TokenData = Depends(require_permission("agent:read"))
):
    """Get performance metrics for an agent"""
    try:
        since = datetime.utcnow() - timedelta(days=days)
        
        metrics = await app.state.metrics_collector.get_agent_metrics(
            agent_id=agent_id,
            organization_id=current_user.organization_id,
            since=since
        )
        
        if not metrics:
            raise HTTPException(status_code=404, detail="Agent metrics not found")
        
        return metrics
        
    except Exception as e:
        logger.error("Failed to get agent metrics", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve metrics")

@app.post("/benchmark", response_model=BenchmarkResult)
async def run_benchmark(
    agent_ids: List[str],
    benchmark_config: Dict[str, Any],
    background_tasks: BackgroundTasks,
    current_user: TokenData = Depends(require_permission("agent:evaluate"))
):
    """Run performance benchmark across multiple agents"""
    try:
        logger.info(
            "Starting benchmark",
            agent_count=len(agent_ids),
            user_id=current_user.user_id
        )
        
        if len(agent_ids) > 5:
            raise HTTPException(
                status_code=400,
                detail="Maximum 5 agents per benchmark"
            )
        
        benchmark_result = await app.state.evaluation_engine.run_benchmark(
            agent_ids=agent_ids,
            config=benchmark_config,
            organization_id=current_user.organization_id
        )
        
        # Schedule detailed analysis
        background_tasks.add_task(
            app.state.performance_analyzer.analyze_benchmark_result,
            benchmark_result
        )
        
        logger.info(
            "Benchmark completed",
            benchmark_id=benchmark_result.benchmark_id,
            winner=benchmark_result.best_performing_agent
        )
        
        return benchmark_result
        
    except Exception as e:
        logger.error("Failed to run benchmark", error=str(e))
        raise HTTPException(status_code=500, detail="Benchmark failed")

@app.get("/analytics/performance-trends")
async def get_performance_trends(
    agent_id: Optional[str] = None,
    days: int = 30,
    current_user: TokenData = Depends(require_permission("agent:read"))
):
    """Get performance trends analysis"""
    try:
        since = datetime.utcnow() - timedelta(days=days)
        
        trends = await app.state.performance_analyzer.get_performance_trends(
            agent_id=agent_id,
            organization_id=current_user.organization_id,
            since=since
        )
        
        return trends
        
    except Exception as e:
        logger.error("Failed to get performance trends", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve trends")

@app.get("/analytics/accuracy-report")
async def get_accuracy_report(
    days: int = 30,
    current_user: TokenData = Depends(require_permission("agent:read"))
):
    """Get accuracy analysis report"""
    try:
        since = datetime.utcnow() - timedelta(days=days)
        
        report = await app.state.performance_analyzer.generate_accuracy_report(
            organization_id=current_user.organization_id,
            since=since
        )
        
        return report
        
    except Exception as e:
        logger.error("Failed to generate accuracy report", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to generate report")

@app.post("/validate-results")
async def validate_evaluation_results(
    evaluation_id: str,
    current_user: TokenData = Depends(require_permission("agent:evaluate"))
):
    """Validate evaluation results for accuracy"""
    try:
        validation_result = await app.state.result_validator.validate_evaluation(
            evaluation_id=evaluation_id,
            organization_id=current_user.organization_id
        )
        
        return validation_result
        
    except Exception as e:
        logger.error("Failed to validate results", error=str(e))
        raise HTTPException(status_code=500, detail="Validation failed")

# WebSocket endpoint for real-time evaluation updates
@app.websocket("/ws/evaluations")
async def evaluation_websocket(websocket):
    """WebSocket endpoint for real-time evaluation updates"""
    await websocket.accept()
    
    try:
        # Subscribe to evaluation events
        async for message in app.state.pubsub_client.subscribe("evaluation-events"):
            await websocket.send_json(message)
            
    except Exception as e:
        logger.error("WebSocket error", error=str(e))
    finally:
        await websocket.close()

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.PORT,
        log_config=None,  # Use structlog instead
        access_log=False
    )