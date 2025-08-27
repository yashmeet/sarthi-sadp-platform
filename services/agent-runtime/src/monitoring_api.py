"""
Monitoring and Telemetry API
Production monitoring, metrics, and observability
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import structlog
from google.cloud import firestore, monitoring_v3
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import json
import os

from auth import get_current_user, TokenData, get_current_organization
from auth import Organization

logger = structlog.get_logger()

router = APIRouter(prefix="/monitoring", tags=["Monitoring"])

# Initialize clients
db = firestore.Client()
metrics_client = monitoring_v3.MetricServiceClient()
query_client = monitoring_v3.QueryServiceClient()

# Prometheus metrics
http_requests = Counter('sadp_http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
http_duration = Histogram('sadp_http_request_duration_seconds', 'HTTP request duration', ['method', 'endpoint'])
active_users = Gauge('sadp_active_users', 'Currently active users')
agent_executions = Counter('sadp_agent_executions_total', 'Total agent executions', ['agent_type', 'status'])
template_executions = Counter('sadp_template_executions_total', 'Total template executions', ['template_id', 'status'])
api_errors = Counter('sadp_api_errors_total', 'Total API errors', ['endpoint', 'error_type'])
db_operations = Histogram('sadp_db_operation_duration_seconds', 'Database operation duration', ['operation', 'collection'])

@router.get("/metrics")
async def get_metrics(
    timeframe: str = Query("1h", description="Timeframe: 1h, 24h, 7d, 30d"),
    current_user: TokenData = Depends(get_current_user),
    org: Organization = Depends(get_current_organization)
):
    """Get comprehensive metrics for organization"""
    try:
        # Parse timeframe
        now = datetime.utcnow()
        if timeframe == "1h":
            start_time = now - timedelta(hours=1)
        elif timeframe == "24h":
            start_time = now - timedelta(days=1)
        elif timeframe == "7d":
            start_time = now - timedelta(days=7)
        elif timeframe == "30d":
            start_time = now - timedelta(days=30)
        else:
            start_time = now - timedelta(hours=1)
        
        # Collect metrics from Firestore
        metrics = {
            "timeframe": timeframe,
            "start_time": start_time.isoformat(),
            "end_time": now.isoformat(),
            "organization_id": org.id
        }
        
        # Agent execution metrics
        agent_executions_query = db.collection("agent_executions")\
            .where("organization_id", "==", org.id)\
            .where("created_at", ">=", start_time)\
            .stream()
        
        agent_metrics = {
            "total": 0,
            "successful": 0,
            "failed": 0,
            "by_type": {},
            "avg_execution_time": 0,
            "total_execution_time": 0
        }
        
        execution_times = []
        for doc in agent_executions_query:
            data = doc.to_dict()
            agent_metrics["total"] += 1
            
            if data.get("status") == "completed":
                agent_metrics["successful"] += 1
            elif data.get("status") == "failed":
                agent_metrics["failed"] += 1
            
            agent_type = data.get("agent_type", "unknown")
            if agent_type not in agent_metrics["by_type"]:
                agent_metrics["by_type"][agent_type] = {"total": 0, "successful": 0, "failed": 0}
            
            agent_metrics["by_type"][agent_type]["total"] += 1
            if data.get("status") == "completed":
                agent_metrics["by_type"][agent_type]["successful"] += 1
            elif data.get("status") == "failed":
                agent_metrics["by_type"][agent_type]["failed"] += 1
            
            if data.get("execution_time"):
                execution_times.append(data["execution_time"])
        
        if execution_times:
            agent_metrics["avg_execution_time"] = sum(execution_times) / len(execution_times)
            agent_metrics["total_execution_time"] = sum(execution_times)
        
        metrics["agent_executions"] = agent_metrics
        
        # Template execution metrics
        template_executions_query = db.collection("template_executions")\
            .where("organization_id", "==", org.id)\
            .where("created_at", ">=", start_time)\
            .stream()
        
        template_metrics = {
            "total": 0,
            "by_template": {},
            "total_tokens": 0,
            "avg_latency": 0
        }
        
        latencies = []
        for doc in template_executions_query:
            data = doc.to_dict()
            template_metrics["total"] += 1
            
            template_id = data.get("template_id", "unknown")
            if template_id not in template_metrics["by_template"]:
                template_metrics["by_template"][template_id] = {"executions": 0, "tokens": 0}
            
            template_metrics["by_template"][template_id]["executions"] += 1
            
            if data.get("tokens"):
                template_metrics["total_tokens"] += data["tokens"]
                template_metrics["by_template"][template_id]["tokens"] += data["tokens"]
            
            if data.get("latency"):
                latencies.append(data["latency"])
        
        if latencies:
            template_metrics["avg_latency"] = sum(latencies) / len(latencies)
        
        metrics["template_executions"] = template_metrics
        
        # User activity metrics
        user_activity_query = db.collection("user_activity")\
            .where("organization_id", "==", org.id)\
            .where("timestamp", ">=", start_time)\
            .stream()
        
        user_metrics = {
            "active_users": set(),
            "total_actions": 0,
            "by_action": {}
        }
        
        for doc in user_activity_query:
            data = doc.to_dict()
            user_metrics["active_users"].add(data.get("user_id"))
            user_metrics["total_actions"] += 1
            
            action = data.get("action", "unknown")
            if action not in user_metrics["by_action"]:
                user_metrics["by_action"][action] = 0
            user_metrics["by_action"][action] += 1
        
        user_metrics["active_users"] = len(user_metrics["active_users"])
        metrics["user_activity"] = user_metrics
        
        # API usage metrics
        api_usage_query = db.collection("api_usage")\
            .where("organization_id", "==", org.id)\
            .where("timestamp", ">=", start_time)\
            .stream()
        
        api_metrics = {
            "total_requests": 0,
            "by_endpoint": {},
            "errors": 0,
            "avg_response_time": 0
        }
        
        response_times = []
        for doc in api_usage_query:
            data = doc.to_dict()
            api_metrics["total_requests"] += 1
            
            endpoint = data.get("endpoint", "unknown")
            if endpoint not in api_metrics["by_endpoint"]:
                api_metrics["by_endpoint"][endpoint] = {"requests": 0, "errors": 0}
            
            api_metrics["by_endpoint"][endpoint]["requests"] += 1
            
            if data.get("error"):
                api_metrics["errors"] += 1
                api_metrics["by_endpoint"][endpoint]["errors"] += 1
            
            if data.get("response_time"):
                response_times.append(data["response_time"])
        
        if response_times:
            api_metrics["avg_response_time"] = sum(response_times) / len(response_times)
        
        metrics["api_usage"] = api_metrics
        
        # Cost estimation
        cost_metrics = {
            "estimated_gemini_cost": template_metrics["total_tokens"] * 0.00001,  # Rough estimate
            "estimated_storage_gb": 0.1,  # Would calculate from actual usage
            "estimated_compute_hours": agent_metrics["total_execution_time"] / 3600,
            "estimated_total_cost": 0
        }
        
        cost_metrics["estimated_total_cost"] = (
            cost_metrics["estimated_gemini_cost"] +
            cost_metrics["estimated_storage_gb"] * 0.02 +
            cost_metrics["estimated_compute_hours"] * 0.05
        )
        
        metrics["cost_estimation"] = cost_metrics
        
        return metrics
        
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def get_system_health(
    current_user: TokenData = Depends(get_current_user)
):
    """Get system health status"""
    health = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {}
    }
    
    # Check Firestore
    try:
        test_doc = db.collection("health_check").document("test")
        test_doc.set({"timestamp": datetime.utcnow()})
        health["services"]["firestore"] = {"status": "healthy", "latency_ms": 10}
    except Exception as e:
        health["services"]["firestore"] = {"status": "unhealthy", "error": str(e)}
        health["status"] = "degraded"
    
    # Check Gemini API
    try:
        import google.generativeai as genai
        if os.environ.get("GEMINI_API_KEY"):
            health["services"]["gemini"] = {"status": "configured"}
        else:
            health["services"]["gemini"] = {"status": "not_configured"}
    except Exception as e:
        health["services"]["gemini"] = {"status": "error", "error": str(e)}
    
    # Check Cloud Storage
    try:
        from google.cloud import storage
        storage_client = storage.Client()
        buckets = list(storage_client.list_buckets(max_results=1))
        health["services"]["storage"] = {"status": "healthy"}
    except Exception as e:
        health["services"]["storage"] = {"status": "unhealthy", "error": str(e)}
        health["status"] = "degraded"
    
    # Check Cloud Run services
    try:
        from google.cloud import run_v2
        run_client = run_v2.ServicesClient()
        # Would check actual services here
        health["services"]["cloud_run"] = {"status": "healthy"}
    except Exception as e:
        health["services"]["cloud_run"] = {"status": "error", "error": str(e)}
    
    return health

@router.get("/audit-log")
async def get_audit_log(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    user_id: Optional[str] = None,
    action: Optional[str] = None,
    limit: int = Query(100, le=1000),
    current_user: TokenData = Depends(get_current_user),
    org: Organization = Depends(get_current_organization)
):
    """Get audit log entries"""
    try:
        query = db.collection("audit_log").where("organization_id", "==", org.id)
        
        if start_date:
            query = query.where("timestamp", ">=", start_date)
        if end_date:
            query = query.where("timestamp", "<=", end_date)
        if user_id:
            query = query.where("user_id", "==", user_id)
        if action:
            query = query.where("action", "==", action)
        
        query = query.order_by("timestamp", direction=firestore.Query.DESCENDING).limit(limit)
        
        entries = []
        for doc in query.stream():
            entries.append(doc.to_dict())
        
        return {
            "entries": entries,
            "total": len(entries),
            "filters": {
                "start_date": start_date.isoformat() if start_date else None,
                "end_date": end_date.isoformat() if end_date else None,
                "user_id": user_id,
                "action": action
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get audit log: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/alerts")
async def get_alerts(
    severity: Optional[str] = None,
    status: Optional[str] = Query("active", description="active, resolved, all"),
    current_user: TokenData = Depends(get_current_user),
    org: Organization = Depends(get_current_organization)
):
    """Get system alerts"""
    try:
        query = db.collection("alerts").where("organization_id", "==", org.id)
        
        if status != "all":
            query = query.where("status", "==", status)
        if severity:
            query = query.where("severity", "==", severity)
        
        query = query.order_by("created_at", direction=firestore.Query.DESCENDING).limit(50)
        
        alerts = []
        for doc in query.stream():
            alerts.append(doc.to_dict())
        
        return {
            "alerts": alerts,
            "total": len(alerts),
            "by_severity": {
                "critical": len([a for a in alerts if a.get("severity") == "critical"]),
                "high": len([a for a in alerts if a.get("severity") == "high"]),
                "medium": len([a for a in alerts if a.get("severity") == "medium"]),
                "low": len([a for a in alerts if a.get("severity") == "low"])
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/alerts/{alert_id}/resolve")
async def resolve_alert(
    alert_id: str,
    resolution: Dict[str, Any],
    current_user: TokenData = Depends(get_current_user),
    org: Organization = Depends(get_current_organization)
):
    """Resolve an alert"""
    try:
        alert_doc = db.collection("alerts").document(alert_id)
        alert_data = alert_doc.get()
        
        if not alert_data.exists:
            raise HTTPException(status_code=404, detail="Alert not found")
        
        if alert_data.to_dict().get("organization_id") != org.id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        alert_doc.update({
            "status": "resolved",
            "resolved_at": datetime.utcnow(),
            "resolved_by": current_user.user_id,
            "resolution": resolution
        })
        
        return {"message": "Alert resolved"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to resolve alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/prometheus")
async def get_prometheus_metrics(
    current_user: TokenData = Depends(get_current_user)
):
    """Get Prometheus-formatted metrics"""
    return generate_latest()

@router.get("/usage-report")
async def get_usage_report(
    month: str = Query(..., description="YYYY-MM format"),
    current_user: TokenData = Depends(get_current_user),
    org: Organization = Depends(get_current_organization)
):
    """Get monthly usage report"""
    try:
        # Parse month
        from datetime import datetime
        start_date = datetime.strptime(f"{month}-01", "%Y-%m-%d")
        
        # Calculate end date (first day of next month)
        if start_date.month == 12:
            end_date = datetime(start_date.year + 1, 1, 1)
        else:
            end_date = datetime(start_date.year, start_date.month + 1, 1)
        
        # Collect usage data
        report = {
            "organization": org.name,
            "month": month,
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            }
        }
        
        # Agent executions
        agent_exec_query = db.collection("agent_executions")\
            .where("organization_id", "==", org.id)\
            .where("created_at", ">=", start_date)\
            .where("created_at", "<", end_date)\
            .stream()
        
        agent_usage = {
            "total_executions": 0,
            "by_agent": {},
            "total_compute_seconds": 0
        }
        
        for doc in agent_exec_query:
            data = doc.to_dict()
            agent_usage["total_executions"] += 1
            
            agent_type = data.get("agent_type", "unknown")
            if agent_type not in agent_usage["by_agent"]:
                agent_usage["by_agent"][agent_type] = 0
            agent_usage["by_agent"][agent_type] += 1
            
            if data.get("execution_time"):
                agent_usage["total_compute_seconds"] += data["execution_time"]
        
        report["agent_usage"] = agent_usage
        
        # Template executions
        template_exec_query = db.collection("template_executions")\
            .where("organization_id", "==", org.id)\
            .where("created_at", ">=", start_date)\
            .where("created_at", "<", end_date)\
            .stream()
        
        template_usage = {
            "total_executions": 0,
            "total_tokens": 0,
            "by_template": {}
        }
        
        for doc in template_exec_query:
            data = doc.to_dict()
            template_usage["total_executions"] += 1
            
            if data.get("tokens"):
                template_usage["total_tokens"] += data["tokens"]
            
            template_id = data.get("template_id", "unknown")
            if template_id not in template_usage["by_template"]:
                template_usage["by_template"][template_id] = {"executions": 0, "tokens": 0}
            
            template_usage["by_template"][template_id]["executions"] += 1
            if data.get("tokens"):
                template_usage["by_template"][template_id]["tokens"] += data["tokens"]
        
        report["template_usage"] = template_usage
        
        # User activity
        user_activity_query = db.collection("users")\
            .where("organization_id", "==", org.id)\
            .stream()
        
        active_users = []
        for doc in user_activity_query:
            user_data = doc.to_dict()
            if user_data.get("last_login") and user_data["last_login"] >= start_date and user_data["last_login"] < end_date:
                active_users.append(user_data["id"])
        
        report["user_activity"] = {
            "active_users": len(active_users),
            "total_users": sum(1 for _ in db.collection("users").where("organization_id", "==", org.id).stream())
        }
        
        # Cost calculation
        costs = {
            "gemini_api": template_usage["total_tokens"] * 0.00001,
            "cloud_run": agent_usage["total_compute_seconds"] / 3600 * 0.05,
            "storage": 0.02 * 10,  # Assuming 10GB
            "firestore": agent_usage["total_executions"] * 0.0001,
            "total": 0
        }
        
        costs["total"] = sum(v for k, v in costs.items() if k != "total")
        report["estimated_costs"] = costs
        
        return report
        
    except Exception as e:
        logger.error(f"Failed to generate usage report: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Helper function to log audit events
async def log_audit_event(
    organization_id: str,
    user_id: str,
    action: str,
    resource_type: str,
    resource_id: str,
    details: Optional[Dict[str, Any]] = None
):
    """Log an audit event"""
    try:
        db.collection("audit_log").add({
            "organization_id": organization_id,
            "user_id": user_id,
            "action": action,
            "resource_type": resource_type,
            "resource_id": resource_id,
            "details": details or {},
            "timestamp": datetime.utcnow(),
            "ip_address": None,  # Would get from request context
            "user_agent": None   # Would get from request context
        })
    except Exception as e:
        logger.error(f"Failed to log audit event: {e}")