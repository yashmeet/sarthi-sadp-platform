"""
SADP Monitoring Service - Simple Working Version
"""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List, Optional

app = FastAPI(
    title="SADP Monitoring Service",
    description="System monitoring and performance metrics",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MetricData(BaseModel):
    metric_name: str
    value: float
    timestamp: str
    labels: Dict[str, str] = {}

class Alert(BaseModel):
    alert_id: str
    severity: str
    message: str
    status: str
    triggered_at: str

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "monitoring"}

@app.get("/")
async def root():
    return {"message": "SADP Monitoring Service is running"}

@app.get("/metrics")
async def get_metrics(
    metric_names: Optional[List[str]] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None
):
    from datetime import datetime
    import random
    
    # Mock metrics data
    metrics = []
    metric_list = metric_names or ["cpu_usage", "memory_usage", "request_count", "response_time"]
    
    for metric_name in metric_list:
        value = random.uniform(0.1, 0.9) if "usage" in metric_name else random.uniform(50, 200)
        metrics.append({
            "metric_name": metric_name,
            "value": value,
            "timestamp": datetime.utcnow().isoformat(),
            "labels": {"service": "sadp", "environment": "production"}
        })
    
    return {"metrics": metrics}

@app.get("/alerts")
async def list_alerts(
    status: Optional[str] = None,
    severity: Optional[str] = None
):
    from datetime import datetime
    
    # Mock alerts
    alerts = [
        {
            "alert_id": "alert_001",
            "severity": "medium",
            "message": "High memory usage detected",
            "status": "active",
            "triggered_at": datetime.utcnow().isoformat()
        },
        {
            "alert_id": "alert_002", 
            "severity": "low",
            "message": "Service response time increased",
            "status": "acknowledged",
            "triggered_at": datetime.utcnow().isoformat()
        }
    ]
    
    if status:
        alerts = [a for a in alerts if a["status"] == status]
    if severity:
        alerts = [a for a in alerts if a["severity"] == severity]
    
    return alerts

@app.get("/system/status")
async def get_system_status():
    return {
        "status": "operational",
        "services": {
            "agent_runtime": "healthy",
            "evaluation": "healthy", 
            "development": "healthy",
            "monitoring": "healthy"
        },
        "uptime": "99.9%",
        "last_check": "2025-01-22T00:00:00Z"
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)