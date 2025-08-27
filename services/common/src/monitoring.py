"""
Comprehensive Monitoring System for SADP
Tracks metrics, health, performance, and compliance
"""

import time
import psutil
import asyncio
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json

logger = logging.getLogger(__name__)

@dataclass
class MetricValue:
    """Individual metric value with timestamp"""
    value: float
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)

@dataclass
class ServiceHealth:
    """Health status for a service"""
    service_name: str
    status: str  # healthy, unhealthy, degraded
    last_check: datetime
    response_time_ms: Optional[float] = None
    error_message: Optional[str] = None
    uptime_seconds: Optional[float] = None

class MetricsCollector:
    """
    Collects and stores system metrics
    """
    
    def __init__(self, retention_hours: int = 24):
        self.retention_hours = retention_hours
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))  # Max 10k points per metric
        self.service_health: Dict[str, ServiceHealth] = {}
        self.start_time = time.time()
        
    def record_metric(self, name: str, value: float, labels: Dict[str, str] = None):
        """Record a metric value"""
        metric = MetricValue(
            value=value,
            timestamp=datetime.utcnow(),
            labels=labels or {}
        )
        self.metrics[name].append(metric)
        self._cleanup_old_metrics()
    
    def increment_counter(self, name: str, labels: Dict[str, str] = None):
        """Increment a counter metric"""
        self.record_metric(name, 1, labels)
    
    def record_duration(self, name: str, duration_ms: float, labels: Dict[str, str] = None):
        """Record a duration metric in milliseconds"""
        self.record_metric(f"{name}_duration_ms", duration_ms, labels)
    
    def update_service_health(self, service_name: str, status: str, 
                            response_time_ms: float = None, error_message: str = None):
        """Update service health status"""
        self.service_health[service_name] = ServiceHealth(
            service_name=service_name,
            status=status,
            last_check=datetime.utcnow(),
            response_time_ms=response_time_ms,
            error_message=error_message,
            uptime_seconds=time.time() - self.start_time
        )
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_used_mb": memory.used / 1024 / 1024,
                "memory_available_mb": memory.available / 1024 / 1024,
                "disk_percent": disk.percent,
                "disk_used_gb": disk.used / 1024 / 1024 / 1024,
                "disk_free_gb": disk.free / 1024 / 1024 / 1024,
                "uptime_seconds": time.time() - self.start_time
            }
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            return {"error": str(e)}
    
    def get_metric_summary(self, metric_name: str, duration_minutes: int = 60) -> Dict[str, Any]:
        """Get summary statistics for a metric over a time period"""
        if metric_name not in self.metrics:
            return {"error": f"Metric {metric_name} not found"}
        
        cutoff_time = datetime.utcnow() - timedelta(minutes=duration_minutes)
        recent_values = [
            m.value for m in self.metrics[metric_name] 
            if m.timestamp >= cutoff_time
        ]
        
        if not recent_values:
            return {"count": 0, "period_minutes": duration_minutes}
        
        return {
            "count": len(recent_values),
            "min": min(recent_values),
            "max": max(recent_values),
            "avg": sum(recent_values) / len(recent_values),
            "sum": sum(recent_values),
            "period_minutes": duration_minutes,
            "latest": recent_values[-1] if recent_values else None
        }
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get overall health summary"""
        total_services = len(self.service_health)
        healthy_services = len([s for s in self.service_health.values() if s.status == "healthy"])
        
        return {
            "total_services": total_services,
            "healthy_services": healthy_services,
            "unhealthy_services": total_services - healthy_services,
            "health_percentage": (healthy_services / total_services * 100) if total_services > 0 else 0,
            "services": {name: {
                "status": health.status,
                "last_check": health.last_check.isoformat(),
                "response_time_ms": health.response_time_ms,
                "error_message": health.error_message
            } for name, health in self.service_health.items()}
        }
    
    def _cleanup_old_metrics(self):
        """Remove metrics older than retention period"""
        cutoff_time = datetime.utcnow() - timedelta(hours=self.retention_hours)
        
        for metric_name in self.metrics:
            # Remove old metrics from the beginning of the deque
            while (self.metrics[metric_name] and 
                   self.metrics[metric_name][0].timestamp < cutoff_time):
                self.metrics[metric_name].popleft()

class PerformanceTracker:
    """
    Tracks performance metrics for API calls and operations
    """
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self.active_operations: Dict[str, float] = {}  # operation_id -> start_time
    
    def start_operation(self, operation_name: str, operation_id: str = None) -> str:
        """Start tracking an operation"""
        if operation_id is None:
            operation_id = f"{operation_name}_{int(time.time() * 1000)}"
        
        self.active_operations[operation_id] = time.time()
        self.metrics.increment_counter("operations_started", {"operation": operation_name})
        return operation_id
    
    def end_operation(self, operation_id: str, operation_name: str, 
                     success: bool = True, error_message: str = None):
        """End tracking an operation"""
        if operation_id not in self.active_operations:
            logger.warning(f"Operation {operation_id} not found in active operations")
            return
        
        start_time = self.active_operations.pop(operation_id)
        duration_ms = (time.time() - start_time) * 1000
        
        # Record metrics
        self.metrics.record_duration(operation_name, duration_ms)
        self.metrics.increment_counter(
            "operations_completed", 
            {"operation": operation_name, "status": "success" if success else "failure"}
        )
        
        if not success:
            self.metrics.increment_counter(
                "operations_failed", 
                {"operation": operation_name, "error": error_message or "unknown"}
            )
    
    def track_api_call(self, endpoint: str, method: str, status_code: int, 
                      duration_ms: float, user_id: str = None):
        """Track API call metrics"""
        labels = {
            "endpoint": endpoint,
            "method": method,
            "status_code": str(status_code),
            "status_class": f"{status_code // 100}xx"
        }
        
        if user_id:
            labels["user_id"] = user_id
        
        self.metrics.record_duration("api_request", duration_ms, labels)
        self.metrics.increment_counter("api_requests", labels)
        
        if status_code >= 400:
            self.metrics.increment_counter("api_errors", labels)

class ComplianceMonitor:
    """
    Monitors compliance-related metrics and events
    """
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self.phi_access_log: List[Dict[str, Any]] = []
    
    def record_phi_access(self, user_id: str, operation: str, phi_detected: bool, 
                         sanitized: bool = False):
        """Record PHI access for compliance tracking"""
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "operation": operation,
            "phi_detected": phi_detected,
            "sanitized": sanitized
        }
        
        self.phi_access_log.append(event)
        
        # Keep only last 1000 PHI access events
        if len(self.phi_access_log) > 1000:
            self.phi_access_log = self.phi_access_log[-1000:]
        
        # Record metrics
        self.metrics.increment_counter("phi_access", {
            "operation": operation,
            "phi_detected": str(phi_detected),
            "sanitized": str(sanitized)
        })
    
    def get_compliance_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get compliance summary for the specified time period"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        cutoff_str = cutoff_time.isoformat()
        
        recent_phi_access = [
            event for event in self.phi_access_log 
            if event["timestamp"] >= cutoff_str
        ]
        
        phi_detected_count = len([e for e in recent_phi_access if e["phi_detected"]])
        sanitized_count = len([e for e in recent_phi_access if e["sanitized"]])
        
        return {
            "period_hours": hours,
            "phi_access_events": len(recent_phi_access),
            "phi_detected_events": phi_detected_count,
            "phi_sanitized_events": sanitized_count,
            "compliance_rate": (sanitized_count / phi_detected_count * 100) if phi_detected_count > 0 else 100,
            "unique_users": len(set(e["user_id"] for e in recent_phi_access)),
            "operations": list(set(e["operation"] for e in recent_phi_access))
        }

class MonitoringSystem:
    """
    Central monitoring system that coordinates all monitoring components
    """
    
    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.performance_tracker = PerformanceTracker(self.metrics_collector)
        self.compliance_monitor = ComplianceMonitor(self.metrics_collector)
        self.alerts: List[Dict[str, Any]] = []
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data"""
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "system_metrics": self.metrics_collector.get_system_metrics(),
            "health_summary": self.metrics_collector.get_health_summary(),
            "compliance_summary": self.compliance_monitor.get_compliance_summary(),
            "performance_summary": {
                "api_requests_1h": self.metrics_collector.get_metric_summary("api_requests", 60),
                "api_errors_1h": self.metrics_collector.get_metric_summary("api_errors", 60),
                "avg_response_time_1h": self.metrics_collector.get_metric_summary("api_request_duration_ms", 60)
            },
            "alerts": self.alerts[-10:],  # Last 10 alerts
            "uptime_seconds": time.time() - self.metrics_collector.start_time
        }
    
    def create_alert(self, severity: str, message: str, component: str = None):
        """Create a monitoring alert"""
        alert = {
            "id": f"alert_{int(time.time() * 1000)}",
            "timestamp": datetime.utcnow().isoformat(),
            "severity": severity,  # info, warning, error, critical
            "message": message,
            "component": component,
            "resolved": False
        }
        
        self.alerts.append(alert)
        
        # Keep only last 100 alerts
        if len(self.alerts) > 100:
            self.alerts = self.alerts[-100:]
        
        logger.log(
            logging.CRITICAL if severity == "critical" else
            logging.ERROR if severity == "error" else
            logging.WARNING if severity == "warning" else
            logging.INFO,
            f"ALERT [{severity.upper()}] {component}: {message}"
        )

# Global monitoring system
monitoring = MonitoringSystem()