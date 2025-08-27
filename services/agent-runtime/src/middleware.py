from fastapi import FastAPI, Request
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from starlette.responses import Response
import time
import structlog

logger = structlog.get_logger()

# Prometheus metrics
request_count = Counter(
    'agent_runtime_requests_total',
    'Total number of requests',
    ['method', 'endpoint', 'status']
)

request_duration = Histogram(
    'agent_runtime_request_duration_seconds',
    'Request duration in seconds',
    ['method', 'endpoint']
)

active_requests = Gauge(
    'agent_runtime_active_requests',
    'Number of active requests'
)

agent_executions = Counter(
    'agent_executions_total',
    'Total number of agent executions',
    ['agent_type', 'status']
)

def setup_monitoring(app: FastAPI):
    """Setup monitoring middleware"""
    
    @app.middleware("http")
    async def monitoring_middleware(request: Request, call_next):
        """Track request metrics"""
        start_time = time.time()
        active_requests.inc()
        
        try:
            # Process request
            response = await call_next(request)
            
            # Record metrics
            duration = time.time() - start_time
            request_count.labels(
                method=request.method,
                endpoint=request.url.path,
                status=response.status_code
            ).inc()
            
            request_duration.labels(
                method=request.method,
                endpoint=request.url.path
            ).observe(duration)
            
            # Add custom headers
            response.headers["X-Request-Duration"] = str(duration)
            
            return response
            
        except Exception as e:
            # Record error metrics
            duration = time.time() - start_time
            request_count.labels(
                method=request.method,
                endpoint=request.url.path,
                status=500
            ).inc()
            
            request_duration.labels(
                method=request.method,
                endpoint=request.url.path
            ).observe(duration)
            
            logger.error(f"Request failed",
                        method=request.method,
                        path=request.url.path,
                        error=str(e))
            raise
            
        finally:
            active_requests.dec()
    
    @app.get("/prometheus")
    async def prometheus_metrics():
        """Expose Prometheus metrics"""
        return Response(
            content=generate_latest(),
            media_type="text/plain"
        )
    
    @app.middleware("http")
    async def logging_middleware(request: Request, call_next):
        """Log all requests"""
        logger.info(f"Request received",
                   method=request.method,
                   path=request.url.path,
                   client=request.client.host if request.client else None)
        
        response = await call_next(request)
        
        logger.info(f"Request completed",
                   method=request.method,
                   path=request.url.path,
                   status=response.status_code)
        
        return response
    
    @app.middleware("http")
    async def error_handling_middleware(request: Request, call_next):
        """Handle errors gracefully"""
        try:
            return await call_next(request)
        except Exception as e:
            logger.error(f"Unhandled error",
                        method=request.method,
                        path=request.url.path,
                        error=str(e),
                        exc_info=True)
            
            return Response(
                content=json.dumps({
                    "error": "Internal server error",
                    "message": str(e) if app.debug else "An error occurred"
                }),
                status_code=500,
                media_type="application/json"
            )

def track_agent_execution(agent_type: str, status: str):
    """Track agent execution metrics"""
    agent_executions.labels(
        agent_type=agent_type,
        status=status
    ).inc()