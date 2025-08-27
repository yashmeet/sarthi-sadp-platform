"""
SADP Production API
Complete production-ready implementation with authentication, real agent execution, and monitoring
"""

from fastapi import FastAPI, Depends, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import os
import structlog
from datetime import datetime
from typing import Dict, Any, List
import asyncio
from prometheus_client import make_asgi_app, Counter, Histogram, Gauge
import time

# Import routers
from auth_api import router as auth_router
from poml_api_production import router as poml_router
from agent_api_production import router as agent_router
from monitoring_api import router as monitoring_router

# Import middleware and dependencies
from auth import get_current_user, TokenData, check_rate_limit
from telemetry import setup_telemetry

# Setup logging
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
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Metrics
request_count = Counter('sadp_requests_total', 'Total requests', ['method', 'endpoint', 'status'])
request_duration = Histogram('sadp_request_duration_seconds', 'Request duration', ['method', 'endpoint'])
active_connections = Gauge('sadp_websocket_connections', 'Active WebSocket connections')
agent_executions = Counter('sadp_agent_executions_total', 'Total agent executions', ['agent_type', 'status'])
template_executions = Counter('sadp_template_executions_total', 'Total template executions', ['status'])

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, organization_id: str):
        await websocket.accept()
        if organization_id not in self.active_connections:
            self.active_connections[organization_id] = []
        self.active_connections[organization_id].append(websocket)
        active_connections.inc()

    def disconnect(self, websocket: WebSocket, organization_id: str):
        if organization_id in self.active_connections:
            self.active_connections[organization_id].remove(websocket)
            if not self.active_connections[organization_id]:
                del self.active_connections[organization_id]
        active_connections.dec()

    async def broadcast_to_organization(self, message: dict, organization_id: str):
        if organization_id in self.active_connections:
            for connection in self.active_connections[organization_id]:
                try:
                    await connection.send_json(message)
                except:
                    pass

manager = ConnectionManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting SADP Production API")
    
    # Initialize services
    from agent_executor_production import AgentExecutor
    from poml_executor_production import POMLExecutor
    
    app.state.agent_executor = AgentExecutor()
    app.state.poml_executor = POMLExecutor()
    
    await app.state.agent_executor.initialize()
    await app.state.poml_executor.initialize()
    
    # Setup telemetry
    setup_telemetry(app)
    
    logger.info("SADP Production API started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down SADP Production API")
    await app.state.agent_executor.shutdown()
    await app.state.poml_executor.shutdown()

# Create FastAPI app
app = FastAPI(
    title="SADP Production API",
    description="Sarthi AI Agent Development Platform - Production API",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Security middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=os.environ.get("ALLOWED_HOSTS", "*").split(",")
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request tracking middleware
@app.middleware("http")
async def track_requests(request, call_next):
    start_time = time.time()
    
    # Add request ID
    request_id = request.headers.get("X-Request-ID", os.urandom(16).hex())
    
    # Log request
    logger.info(
        "Request started",
        request_id=request_id,
        method=request.method,
        path=request.url.path,
        client=request.client.host if request.client else "unknown"
    )
    
    try:
        response = await call_next(request)
        
        # Track metrics
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
        
        # Log response
        logger.info(
            "Request completed",
            request_id=request_id,
            status=response.status_code,
            duration=duration
        )
        
        # Add response headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = str(duration)
        
        return response
        
    except Exception as e:
        duration = time.time() - start_time
        
        # Log error
        logger.error(
            "Request failed",
            request_id=request_id,
            error=str(e),
            duration=duration
        )
        
        # Track error metrics
        request_count.labels(
            method=request.method,
            endpoint=request.url.path,
            status=500
        ).inc()
        
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "request_id": request_id
            }
        )

# Include routers
app.include_router(auth_router)
app.include_router(poml_router, dependencies=[Depends(get_current_user)])
app.include_router(agent_router, dependencies=[Depends(get_current_user)])
app.include_router(monitoring_router, dependencies=[Depends(get_current_user)])

# Mount metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "service": "SADP Production API",
        "version": "1.0.0",
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "environment": os.environ.get("ENVIRONMENT", "production")
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "checks": {}
    }
    
    # Check database
    try:
        from google.cloud import firestore
        db = firestore.Client()
        test_doc = db.collection("health_check").document("test")
        test_doc.set({"timestamp": datetime.utcnow()})
        health_status["checks"]["database"] = "healthy"
    except Exception as e:
        health_status["checks"]["database"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"
    
    # Check Gemini API
    try:
        import google.generativeai as genai
        if os.environ.get("GEMINI_API_KEY"):
            health_status["checks"]["gemini_api"] = "configured"
        else:
            health_status["checks"]["gemini_api"] = "not configured"
    except Exception as e:
        health_status["checks"]["gemini_api"] = f"error: {str(e)}"
    
    # Check storage
    try:
        from google.cloud import storage
        storage_client = storage.Client()
        buckets = list(storage_client.list_buckets(max_results=1))
        health_status["checks"]["storage"] = "healthy"
    except Exception as e:
        health_status["checks"]["storage"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"
    
    return health_status

@app.websocket("/ws/{organization_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    organization_id: str,
    token: str
):
    """WebSocket endpoint for real-time updates"""
    try:
        # Verify token
        from jose import jwt, JWTError
        from auth import SECRET_KEY, ALGORITHM
        
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            if payload.get("organization_id") != organization_id:
                await websocket.close(code=1008, reason="Unauthorized")
                return
        except JWTError:
            await websocket.close(code=1008, reason="Invalid token")
            return
        
        # Connect
        await manager.connect(websocket, organization_id)
        
        try:
            while True:
                # Keep connection alive
                data = await websocket.receive_json()
                
                # Handle different message types
                if data.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
                elif data.get("type") == "subscribe":
                    # Handle subscription to specific events
                    pass
                    
        except WebSocketDisconnect:
            manager.disconnect(websocket, organization_id)
            
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close(code=1011, reason="Internal error")

@app.post("/events/publish")
async def publish_event(
    event: Dict[str, Any],
    current_user: TokenData = Depends(get_current_user)
):
    """Publish event to organization's WebSocket connections"""
    await manager.broadcast_to_organization(
        {
            "type": "event",
            "data": event,
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": current_user.user_id
        },
        current_user.organization_id
    )
    
    return {"status": "published"}

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.environ.get("PORT", 8080))
    workers = int(os.environ.get("WORKERS", 4))
    
    uvicorn.run(
        "main_production:app",
        host="0.0.0.0",
        port=port,
        workers=workers,
        log_config={
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                },
                "json": {
                    "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
                    "format": "%(asctime)s %(name)s %(levelname)s %(message)s"
                }
            },
            "handlers": {
                "default": {
                    "formatter": "json",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stdout"
                }
            },
            "root": {
                "level": "INFO",
                "handlers": ["default"]
            }
        }
    )