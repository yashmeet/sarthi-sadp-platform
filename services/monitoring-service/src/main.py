"""
Production Monitoring Service for SADP
Real-time system monitoring, metrics, and alerting
"""

import os
import sys
import logging
import asyncio
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import httpx
import time

from monitoring import MonitoringSystem, monitoring
from circuit_breaker import circuit_manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="SADP Monitoring Service",
    description="Real-time monitoring, metrics, and alerting for SADP",
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

# Service endpoints to monitor
MONITORED_SERVICES = {
    "auth": "https://sadp-auth-service-prod-xonau6hybq-uc.a.run.app",
    "phi_protection": "https://sadp-phi-protection-prod-xonau6hybq-uc.a.run.app",
    "audit": "https://sadp-audit-service-prod-xonau6hybq-uc.a.run.app",
    "prompt_optimization": "https://sadp-prompt-optimization-355881591332.us-central1.run.app",
    "unified_dashboard": "https://sadp-unified-dashboard-355881591332.us-central1.run.app"
}

class MetricQuery(BaseModel):
    metric_name: str
    duration_minutes: int = 60
    labels: Optional[Dict[str, str]] = None

class AlertRequest(BaseModel):
    severity: str
    message: str
    component: Optional[str] = None

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "monitoring-service",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "monitored_services": len(MONITORED_SERVICES),
        "uptime_seconds": time.time() - monitoring.metrics_collector.start_time
    }

@app.get("/metrics/dashboard")
async def get_dashboard():
    """Get comprehensive dashboard data"""
    return monitoring.get_dashboard_data()

@app.get("/metrics/system")
async def get_system_metrics():
    """Get current system metrics"""
    return monitoring.metrics_collector.get_system_metrics()

@app.get("/metrics/health")
async def get_health_summary():
    """Get service health summary"""
    return monitoring.metrics_collector.get_health_summary()

@app.get("/metrics/compliance")
async def get_compliance_metrics():
    """Get compliance monitoring metrics"""
    return monitoring.compliance_monitor.get_compliance_summary()

@app.post("/metrics/query")
async def query_metric(query: MetricQuery):
    """Query specific metric data"""
    return monitoring.metrics_collector.get_metric_summary(
        query.metric_name, 
        query.duration_minutes
    )

@app.get("/circuit-breakers")
async def get_circuit_breaker_status():
    """Get circuit breaker status for all services"""
    return circuit_manager.get_all_stats()

@app.post("/alerts")
async def create_alert(alert: AlertRequest):
    """Create a new alert"""
    monitoring.create_alert(
        severity=alert.severity,
        message=alert.message,
        component=alert.component
    )
    return {"status": "alert_created", "timestamp": datetime.utcnow().isoformat()}

@app.get("/alerts")
async def get_alerts():
    """Get recent alerts"""
    return {
        "alerts": monitoring.alerts[-50:],  # Last 50 alerts
        "total_alerts": len(monitoring.alerts),
        "unresolved_alerts": len([a for a in monitoring.alerts if not a.get("resolved", False)])
    }

async def check_service_health(service_name: str, base_url: str):
    """Check health of a single service"""
    start_time = time.time()
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{base_url}/health")
            duration_ms = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                monitoring.metrics_collector.update_service_health(
                    service_name, "healthy", duration_ms
                )
                monitoring.performance_tracker.track_api_call(
                    "/health", "GET", response.status_code, duration_ms
                )
            else:
                monitoring.metrics_collector.update_service_health(
                    service_name, "unhealthy", duration_ms, 
                    f"HTTP {response.status_code}"
                )
                monitoring.create_alert(
                    "warning", 
                    f"Service {service_name} returned HTTP {response.status_code}",
                    service_name
                )
    
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        monitoring.metrics_collector.update_service_health(
            service_name, "unhealthy", duration_ms, str(e)
        )
        monitoring.create_alert(
            "error",
            f"Service {service_name} health check failed: {str(e)}",
            service_name
        )

async def health_check_loop():
    """Background task to continuously check service health"""
    while True:
        try:
            logger.info("Running health checks for all services")
            
            # Check all services in parallel
            tasks = [
                check_service_health(name, url) 
                for name, url in MONITORED_SERVICES.items()
            ]
            await asyncio.gather(*tasks, return_exceptions=True)
            
            # Record monitoring metrics
            monitoring.metrics_collector.record_metric("health_checks_completed", 1)
            
            # Sleep for 30 seconds before next check
            await asyncio.sleep(30)
            
        except Exception as e:
            logger.error(f"Error in health check loop: {e}")
            monitoring.create_alert("error", f"Health check loop error: {str(e)}", "monitoring")
            await asyncio.sleep(60)  # Wait longer on error

@app.on_event("startup")
async def startup_event():
    """Start background monitoring tasks"""
    logger.info("Starting monitoring service background tasks")
    
    # Start health check loop
    asyncio.create_task(health_check_loop())
    
    # Record startup
    monitoring.create_alert("info", "Monitoring service started", "monitoring")

@app.get("/dashboard", response_class=HTMLResponse)
async def monitoring_dashboard():
    """Simple HTML dashboard for monitoring"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>SADP Monitoring Dashboard</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .container { max-width: 1200px; margin: 0 auto; }
            .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
            .card { border: 1px solid #ddd; border-radius: 8px; padding: 15px; background: #f9f9f9; }
            .status-healthy { color: green; font-weight: bold; }
            .status-unhealthy { color: red; font-weight: bold; }
            .status-degraded { color: orange; font-weight: bold; }
            .metric { margin: 10px 0; }
            .metric-value { font-size: 1.2em; font-weight: bold; color: #333; }
            .alert-error { background: #ffebee; border-left: 4px solid #f44336; }
            .alert-warning { background: #fff3e0; border-left: 4px solid #ff9800; }
            .alert-info { background: #e3f2fd; border-left: 4px solid #2196f3; }
            .refresh-btn { padding: 10px 20px; background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer; }
            .refresh-btn:hover { background: #0056b3; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>SADP Production Monitoring Dashboard</h1>
            <button class="refresh-btn" onclick="location.reload()">Refresh Dashboard</button>
            
            <div id="dashboard-content">
                <p>Loading dashboard data...</p>
            </div>
        </div>

        <script>
            async function loadDashboard() {
                try {
                    const response = await fetch('/metrics/dashboard');
                    const data = await response.json();
                    
                    const content = document.getElementById('dashboard-content');
                    content.innerHTML = `
                        <div class="grid">
                            <div class="card">
                                <h3>System Health</h3>
                                <div class="metric">
                                    <div>Services: ${data.health_summary.healthy_services}/${data.health_summary.total_services} healthy</div>
                                    <div class="metric-value">${data.health_summary.health_percentage.toFixed(1)}%</div>
                                </div>
                            </div>
                            
                            <div class="card">
                                <h3>System Resources</h3>
                                <div class="metric">CPU: <span class="metric-value">${data.system_metrics.cpu_percent.toFixed(1)}%</span></div>
                                <div class="metric">Memory: <span class="metric-value">${data.system_metrics.memory_percent.toFixed(1)}%</span></div>
                                <div class="metric">Disk: <span class="metric-value">${data.system_metrics.disk_percent.toFixed(1)}%</span></div>
                            </div>
                            
                            <div class="card">
                                <h3>Compliance</h3>
                                <div class="metric">PHI Access Events: <span class="metric-value">${data.compliance_summary.phi_access_events}</span></div>
                                <div class="metric">Compliance Rate: <span class="metric-value">${data.compliance_summary.compliance_rate.toFixed(1)}%</span></div>
                            </div>
                            
                            <div class="card">
                                <h3>Service Status</h3>
                                ${Object.entries(data.health_summary.services).map(([name, service]) => `
                                    <div class="metric">
                                        ${name}: <span class="status-${service.status}">${service.status}</span>
                                        ${service.response_time_ms ? `(${service.response_time_ms.toFixed(0)}ms)` : ''}
                                    </div>
                                `).join('')}
                            </div>
                            
                            <div class="card">
                                <h3>Recent Alerts</h3>
                                ${data.alerts.slice(-5).map(alert => `
                                    <div class="metric alert-${alert.severity}">
                                        <strong>${alert.severity.toUpperCase()}</strong> ${alert.component || 'System'}: ${alert.message}
                                        <br><small>${new Date(alert.timestamp).toLocaleString()}</small>
                                    </div>
                                `).join('')}
                            </div>
                            
                            <div class="card">
                                <h3>Uptime</h3>
                                <div class="metric">
                                    <div class="metric-value">${(data.uptime_seconds / 3600).toFixed(1)} hours</div>
                                </div>
                            </div>
                        </div>
                        
                        <p><small>Last updated: ${new Date(data.timestamp).toLocaleString()}</small></p>
                    `;
                } catch (error) {
                    document.getElementById('dashboard-content').innerHTML = `
                        <div class="card alert-error">
                            <h3>Error Loading Dashboard</h3>
                            <p>${error.message}</p>
                        </div>
                    `;
                }
            }
            
            // Load dashboard on page load
            loadDashboard();
            
            // Auto-refresh every 30 seconds
            setInterval(loadDashboard, 30000);
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "SADP Monitoring Service",
        "version": "1.0.0",
        "status": "running",
        "monitored_services": list(MONITORED_SERVICES.keys()),
        "endpoints": [
            "/health",
            "/dashboard",
            "/metrics/dashboard",
            "/metrics/system",
            "/metrics/health",
            "/metrics/compliance",
            "/circuit-breakers",
            "/alerts"
        ],
        "features": [
            "real_time_monitoring",
            "service_health_checks",
            "system_metrics",
            "compliance_tracking",
            "circuit_breaker_monitoring",
            "alerting"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)