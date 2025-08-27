"""
SADP Monitoring Service
Real-time performance monitoring, alerting, and analytics
"""

import asyncio
import json
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer
import uvicorn

from telemetry import TelemetryCollector, MetricsAggregator
from alerting import AlertManager
from dashboards import DashboardGenerator
from analytics import PerformanceAnalyzer
from models import (
    MetricData, AlertRule, Alert, Dashboard, 
    PerformanceReport, SystemHealth
)
from config import Settings
from auth import get_current_user, require_permission, TokenData
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
    logger.info("Starting SADP Monitoring Service", version="1.0.0")
    
    # Initialize services
    app.state.telemetry_collector = TelemetryCollector(settings)
    app.state.metrics_aggregator = MetricsAggregator(settings)
    app.state.alert_manager = AlertManager(settings)
    app.state.dashboard_generator = DashboardGenerator(settings)
    app.state.performance_analyzer = PerformanceAnalyzer(settings)
    app.state.pubsub_client = PubSubClient(settings)
    
    # Start background tasks
    telemetry_task = asyncio.create_task(
        app.state.telemetry_collector.start_collection()
    )
    alert_task = asyncio.create_task(
        app.state.alert_manager.start_monitoring()
    )
    
    logger.info("Monitoring service started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down monitoring service")
    telemetry_task.cancel()
    alert_task.cancel()
    
    try:
        await telemetry_task
        await alert_task
    except asyncio.CancelledError:
        pass
    
    await app.state.pubsub_client.close()
    logger.info("Monitoring service shutdown complete")

# Create FastAPI application
app = FastAPI(
    title="SADP Monitoring Service",
    description="Real-time monitoring, alerting, and analytics service",
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
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "monitoring",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/system-health", response_model=SystemHealth)
async def get_system_health(
    current_user: TokenData = Depends(require_permission("org:read"))
):
    """Get overall system health status"""
    try:
        health = await app.state.telemetry_collector.get_system_health(
            current_user.organization_id
        )
        return health
    except Exception as e:
        logger.error("Failed to get system health", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve system health")

# Metrics endpoints
@app.get("/metrics")
async def get_metrics(
    metric_names: Optional[List[str]] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    current_user: TokenData = Depends(require_permission("agent:read"))
):
    """Get metrics data"""
    try:
        if not start_time:
            start_time = datetime.utcnow() - timedelta(hours=1)
        if not end_time:
            end_time = datetime.utcnow()
        
        metrics = await app.state.metrics_aggregator.get_metrics(
            organization_id=current_user.organization_id,
            metric_names=metric_names,
            start_time=start_time,
            end_time=end_time
        )
        
        return metrics
    except Exception as e:
        logger.error("Failed to get metrics", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve metrics")

@app.post("/metrics")
async def submit_metrics(
    metrics: List[MetricData],
    current_user: TokenData = Depends(require_permission("agent:write"))
):
    """Submit custom metrics"""
    try:
        for metric in metrics:
            metric.organization_id = current_user.organization_id
        
        await app.state.telemetry_collector.store_metrics(metrics)
        
        logger.info(
            "Custom metrics submitted",
            count=len(metrics),
            user_id=current_user.user_id
        )
        
        return {"message": "Metrics submitted successfully"}
    except Exception as e:
        logger.error("Failed to submit metrics", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to submit metrics")

# Alert management
@app.get("/alerts", response_model=List[Alert])
async def list_alerts(
    status: Optional[str] = None,
    severity: Optional[str] = None,
    current_user: TokenData = Depends(require_permission("org:read"))
):
    """List alerts"""
    try:
        alerts = await app.state.alert_manager.list_alerts(
            organization_id=current_user.organization_id,
            status=status,
            severity=severity
        )
        return alerts
    except Exception as e:
        logger.error("Failed to list alerts", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve alerts")

@app.post("/alert-rules", response_model=AlertRule)
async def create_alert_rule(
    rule: AlertRule,
    current_user: TokenData = Depends(require_permission("org:write"))
):
    """Create alert rule"""
    try:
        rule.organization_id = current_user.organization_id
        rule.created_by = current_user.user_id
        
        created_rule = await app.state.alert_manager.create_rule(rule)
        
        logger.info(
            "Alert rule created",
            rule_id=created_rule.rule_id,
            user_id=current_user.user_id
        )
        
        return created_rule
    except Exception as e:
        logger.error("Failed to create alert rule", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to create alert rule")

@app.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(
    alert_id: str,
    current_user: TokenData = Depends(require_permission("org:write"))
):
    """Acknowledge alert"""
    try:
        success = await app.state.alert_manager.acknowledge_alert(
            alert_id=alert_id,
            organization_id=current_user.organization_id,
            acknowledged_by=current_user.user_id
        )
        
        if not success:
            raise HTTPException(status_code=404, detail="Alert not found")
        
        logger.info(
            "Alert acknowledged",
            alert_id=alert_id,
            user_id=current_user.user_id
        )
        
        return {"message": "Alert acknowledged"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to acknowledge alert", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to acknowledge alert")

# Dashboard management
@app.get("/dashboards", response_model=List[Dashboard])
async def list_dashboards(
    current_user: TokenData = Depends(require_permission("org:read"))
):
    """List available dashboards"""
    try:
        dashboards = await app.state.dashboard_generator.list_dashboards(
            current_user.organization_id
        )
        return dashboards
    except Exception as e:
        logger.error("Failed to list dashboards", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve dashboards")

@app.get("/dashboards/{dashboard_id}", response_model=Dashboard)
async def get_dashboard(
    dashboard_id: str,
    current_user: TokenData = Depends(require_permission("org:read"))
):
    """Get specific dashboard"""
    try:
        dashboard = await app.state.dashboard_generator.get_dashboard(
            dashboard_id, current_user.organization_id
        )
        
        if not dashboard:
            raise HTTPException(status_code=404, detail="Dashboard not found")
        
        return dashboard
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get dashboard", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve dashboard")

@app.post("/dashboards", response_model=Dashboard)
async def create_dashboard(
    dashboard: Dashboard,
    current_user: TokenData = Depends(require_permission("org:write"))
):
    """Create custom dashboard"""
    try:
        dashboard.organization_id = current_user.organization_id
        dashboard.created_by = current_user.user_id
        
        created_dashboard = await app.state.dashboard_generator.create_dashboard(dashboard)
        
        logger.info(
            "Dashboard created",
            dashboard_id=created_dashboard.dashboard_id,
            user_id=current_user.user_id
        )
        
        return created_dashboard
    except Exception as e:
        logger.error("Failed to create dashboard", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to create dashboard")

# Analytics endpoints
@app.get("/analytics/performance-report", response_model=PerformanceReport)
async def get_performance_report(
    days: int = 7,
    current_user: TokenData = Depends(require_permission("org:read"))
):
    """Get performance analysis report"""
    try:
        report = await app.state.performance_analyzer.generate_performance_report(
            organization_id=current_user.organization_id,
            days=days
        )
        return report
    except Exception as e:
        logger.error("Failed to generate performance report", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to generate report")

@app.get("/analytics/cost-analysis")
async def get_cost_analysis(
    days: int = 30,
    current_user: TokenData = Depends(require_permission("org:read"))
):
    """Get cost analysis"""
    try:
        analysis = await app.state.performance_analyzer.analyze_costs(
            organization_id=current_user.organization_id,
            days=days
        )
        return analysis
    except Exception as e:
        logger.error("Failed to analyze costs", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to analyze costs")

@app.get("/analytics/usage-trends")
async def get_usage_trends(
    metric: str = "requests",
    days: int = 30,
    current_user: TokenData = Depends(require_permission("org:read"))
):
    """Get usage trends analysis"""
    try:
        trends = await app.state.performance_analyzer.analyze_usage_trends(
            organization_id=current_user.organization_id,
            metric=metric,
            days=days
        )
        return trends
    except Exception as e:
        logger.error("Failed to analyze usage trends", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to analyze trends")

# Real-time endpoints
@app.websocket("/ws/metrics")
async def metrics_websocket(websocket: WebSocket):
    """WebSocket for real-time metrics"""
    await websocket.accept()
    
    try:
        # Subscribe to real-time metrics
        async for metric_data in app.state.telemetry_collector.stream_metrics():
            await websocket.send_json(metric_data)
    except Exception as e:
        logger.error("WebSocket error", error=str(e))
    finally:
        await websocket.close()

@app.websocket("/ws/alerts")
async def alerts_websocket(websocket: WebSocket):
    """WebSocket for real-time alerts"""
    await websocket.accept()
    
    try:
        # Subscribe to real-time alerts
        async for alert in app.state.alert_manager.stream_alerts():
            await websocket.send_json(alert.dict())
    except Exception as e:
        logger.error("WebSocket error", error=str(e))
    finally:
        await websocket.close()

# Log aggregation
@app.get("/logs")
async def get_logs(
    service: Optional[str] = None,
    level: str = "INFO",
    limit: int = 100,
    current_user: TokenData = Depends(require_permission("org:read"))
):
    """Get application logs"""
    try:
        logs = await app.state.telemetry_collector.get_logs(
            organization_id=current_user.organization_id,
            service=service,
            level=level,
            limit=limit
        )
        return logs
    except Exception as e:
        logger.error("Failed to get logs", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve logs")

# Traces
@app.get("/traces/{trace_id}")
async def get_trace(
    trace_id: str,
    current_user: TokenData = Depends(require_permission("org:read"))
):
    """Get distributed trace"""
    try:
        trace = await app.state.telemetry_collector.get_trace(
            trace_id, current_user.organization_id
        )
        
        if not trace:
            raise HTTPException(status_code=404, detail="Trace not found")
        
        return trace
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get trace", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve trace")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.PORT,
        log_config=None,  # Use structlog instead
        access_log=False
    )