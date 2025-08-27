"""
Telemetry and Distributed Tracing
OpenTelemetry integration for production observability
"""

from opentelemetry import trace, metrics
from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.instrumentation.grpc import GrpcInstrumentorClient
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.semconv.resource import ResourceAttributes
import structlog
import os

logger = structlog.get_logger()

def setup_telemetry(app):
    """Setup OpenTelemetry for the application"""
    
    # Configure resource
    resource = Resource.create({
        ResourceAttributes.SERVICE_NAME: "sadp-production",
        ResourceAttributes.SERVICE_VERSION: "1.0.0",
        ResourceAttributes.DEPLOYMENT_ENVIRONMENT: os.environ.get("ENVIRONMENT", "production"),
        ResourceAttributes.CLOUD_PROVIDER: "gcp",
        ResourceAttributes.CLOUD_PLATFORM: "gcp_cloud_run",
        ResourceAttributes.CLOUD_REGION: os.environ.get("REGION", "us-central1")
    })
    
    # Setup tracing
    tracer_provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(tracer_provider)
    
    # Add Cloud Trace exporter
    if os.environ.get("ENABLE_CLOUD_TRACE", "true").lower() == "true":
        cloud_trace_exporter = CloudTraceSpanExporter(
            project_id=os.environ.get("GCP_PROJECT_ID", "sarthi-patient-experience-hub")
        )
        tracer_provider.add_span_processor(
            BatchSpanProcessor(cloud_trace_exporter)
        )
    
    # Setup metrics (simplified without CloudMonitoringMetricsExporter)
    # Metrics will be exposed via Prometheus endpoint instead
    
    # Instrument libraries
    FastAPIInstrumentor.instrument_app(app)
    RequestsInstrumentor().instrument()
    GrpcInstrumentorClient().instrument()
    
    # Instrument SQLAlchemy if used
    try:
        from sqlalchemy import create_engine
        SQLAlchemyInstrumentor().instrument()
    except ImportError:
        pass
    
    logger.info("Telemetry setup complete")
    
    return tracer_provider

def get_tracer(name: str = "sadp"):
    """Get a tracer instance"""
    return trace.get_tracer(name)

def get_meter(name: str = "sadp"):
    """Get a meter instance"""
    return metrics.get_meter(name)

# Custom span decorators
def trace_method(name: str = None):
    """Decorator to trace a method"""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            tracer = get_tracer()
            span_name = name or f"{func.__module__}.{func.__name__}"
            
            with tracer.start_as_current_span(span_name) as span:
                try:
                    # Add attributes
                    span.set_attribute("function.module", func.__module__)
                    span.set_attribute("function.name", func.__name__)
                    
                    # Execute function
                    result = await func(*args, **kwargs)
                    
                    # Mark success
                    span.set_attribute("function.success", True)
                    
                    return result
                    
                except Exception as e:
                    # Record exception
                    span.record_exception(e)
                    span.set_attribute("function.success", False)
                    span.set_attribute("error.type", type(e).__name__)
                    span.set_attribute("error.message", str(e))
                    raise
        
        def sync_wrapper(*args, **kwargs):
            tracer = get_tracer()
            span_name = name or f"{func.__module__}.{func.__name__}"
            
            with tracer.start_as_current_span(span_name) as span:
                try:
                    # Add attributes
                    span.set_attribute("function.module", func.__module__)
                    span.set_attribute("function.name", func.__name__)
                    
                    # Execute function
                    result = func(*args, **kwargs)
                    
                    # Mark success
                    span.set_attribute("function.success", True)
                    
                    return result
                    
                except Exception as e:
                    # Record exception
                    span.record_exception(e)
                    span.set_attribute("function.success", False)
                    span.set_attribute("error.type", type(e).__name__)
                    span.set_attribute("error.message", str(e))
                    raise
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

# Metric helpers
class MetricsCollector:
    """Helper class for collecting custom metrics"""
    
    def __init__(self):
        self.meter = get_meter()
        
        # Create metrics
        self.request_counter = self.meter.create_counter(
            name="sadp.requests",
            description="Number of requests",
            unit="1"
        )
        
        self.request_duration = self.meter.create_histogram(
            name="sadp.request.duration",
            description="Request duration",
            unit="ms"
        )
        
        self.agent_execution_counter = self.meter.create_counter(
            name="sadp.agent.executions",
            description="Number of agent executions",
            unit="1"
        )
        
        self.template_execution_counter = self.meter.create_counter(
            name="sadp.template.executions",
            description="Number of template executions",
            unit="1"
        )
        
        self.active_users_gauge = self.meter.create_up_down_counter(
            name="sadp.users.active",
            description="Number of active users",
            unit="1"
        )
        
        self.error_counter = self.meter.create_counter(
            name="sadp.errors",
            description="Number of errors",
            unit="1"
        )
    
    def record_request(self, method: str, endpoint: str, status: int, duration_ms: float):
        """Record an HTTP request"""
        attributes = {
            "http.method": method,
            "http.target": endpoint,
            "http.status_code": status
        }
        
        self.request_counter.add(1, attributes)
        self.request_duration.record(duration_ms, attributes)
    
    def record_agent_execution(self, agent_type: str, success: bool, duration_ms: float):
        """Record an agent execution"""
        attributes = {
            "agent.type": agent_type,
            "execution.success": success
        }
        
        self.agent_execution_counter.add(1, attributes)
    
    def record_template_execution(self, template_id: str, success: bool, tokens: int):
        """Record a template execution"""
        attributes = {
            "template.id": template_id,
            "execution.success": success,
            "tokens.used": tokens
        }
        
        self.template_execution_counter.add(1, attributes)
    
    def update_active_users(self, delta: int):
        """Update active users count"""
        self.active_users_gauge.add(delta)
    
    def record_error(self, error_type: str, endpoint: str):
        """Record an error"""
        attributes = {
            "error.type": error_type,
            "endpoint": endpoint
        }
        
        self.error_counter.add(1, attributes)

# Global metrics collector instance
metrics_collector = MetricsCollector()