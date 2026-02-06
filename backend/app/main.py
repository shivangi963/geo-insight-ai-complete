from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from fastapi.staticfiles import StaticFiles
import logging
import os
import asyncio
from datetime import datetime
from pydantic import BaseModel

# ==================== CONFIGURATION ====================

from .security_config import CORSSettings
from .middleware import (
    RequestValidationMiddleware,
    RequestLoggingMiddleware,
    SecurityHeadersMiddleware,
    RateLimitHeaderMiddleware
)

class Settings(BaseModel):
    """Application settings"""
    app_name: str = "GeoInsight AI"
    app_version: str = "4.3.0"  # Updated version
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000
    environment: str = os.getenv("ENVIRONMENT", "development")
    
    # Database
    mongo_url: str = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
    mongo_db: str = os.getenv("DATABASE_NAME", "geoinsight_ai")

settings = Settings()

# ==================== LOGGING ====================
logging.basicConfig(
    level=logging.INFO if not settings.debug else logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# ==================== FEATURE DETECTION ====================

# Celery
CELERY_AVAILABLE = False
try:
    from celery.result import AsyncResult
    from celery_config import celery_app
    CELERY_AVAILABLE = True
    logger.info("✅ Celery available")
except ImportError:
    logger.info("⚠️ Celery not available - using sync mode")

# Vector DB
VECTOR_DB_AVAILABLE = False
try:
    from .supabase_client import vector_db
    if vector_db and getattr(vector_db, 'enabled', False):
        VECTOR_DB_AVAILABLE = True
        logger.info("✅ Vector database available")
    else:
        logger.info("⚠️ Vector database not enabled")
except ImportError:
    logger.info("⚠️ Vector database not available")

# AI Agent
AI_AGENT_AVAILABLE = False
try:
    from .agents.local_expert import agent
    AI_AGENT_AVAILABLE = True
    logger.info("✅ AI Agent available")
except ImportError:
    logger.info("⚠️ AI Agent not available")

# Workflow
WORKFLOW_ENABLED = False
try:
    from .workflow_endpoints import router as workflow_router
    WORKFLOW_ENABLED = True
    logger.info("✅ Workflow endpoints available")
except ImportError:
    logger.info("⚠️ Workflow endpoints not available")

# Rate Limiting
RATE_LIMITING_AVAILABLE = False
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    
    limiter = Limiter(key_func=get_remote_address)
    RATE_LIMITING_AVAILABLE = True
    logger.info("✅ Rate limiting available")
except ImportError:
    logger.info("⚠️ Rate limiting not available")
    class DummyLimiter:
        def limit(self, *args, **kwargs):
            def decorator(func):
                return func
            return decorator
    limiter = DummyLimiter()

# ==================== DATABASE ====================

from .database import Database

async def periodic_cleanup():
    """Periodic cleanup task"""
    while True:
        await asyncio.sleep(3600)  # Every hour
        logger.info("Running periodic cleanup...")
        # Add cleanup tasks here

# ==================== LIFESPAN ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"📊 Features: Celery={CELERY_AVAILABLE}, VectorDB={VECTOR_DB_AVAILABLE}, AI={AI_AGENT_AVAILABLE}")
    
    # Connect to database
    max_retries = 3
    for attempt in range(max_retries):
        try:
            await Database.connect()
            logger.info("✅ Database connected")
            break
        except Exception as e:
            logger.error(f"Database connection attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                logger.critical("Failed to connect to database")
                raise
            await asyncio.sleep(2 ** attempt)
    
    # Initialize app state
    app.state.startup_time = datetime.now()
    app.state.total_requests = 0
    
    # Start cleanup task
    cleanup_task = asyncio.create_task(periodic_cleanup())
    
    yield
    
    # Cleanup
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass
    
    logger.info(f"Shutting down {settings.app_name}")
    try:
        await Database.close()
        logger.info("✅ Database closed")
    except Exception as e:
        logger.error(f"Error closing database: {e}")

# ==================== APPLICATION ====================

app = FastAPI(
    title=settings.app_name,
    description="Advanced Real Estate Intelligence & Geospatial Analysis Platform",
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# ==================== MIDDLEWARE ====================

# CORS
cors_config = CORSSettings.get_cors_config(environment=settings.environment)
app.add_middleware(CORSMiddleware, **cors_config)

# Security & Logging
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(RequestValidationMiddleware)
app.add_middleware(RateLimitHeaderMiddleware)

# Compression
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Rate limiter state
if RATE_LIMITING_AVAILABLE:
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

logger.info(f"✅ Middleware initialized for environment: {settings.environment}")

# ==================== ROUTERS ====================

# Import all routers
from .routers import (
    properties,
    neighborhood,
    ai_agent,
    image_analysis,
    vector_search,
    tasks,
    debug_stats,
    green_space
)

# Include routers
app.include_router(properties.router)
app.include_router(neighborhood.router)
app.include_router(ai_agent.router)
app.include_router(image_analysis.router)
app.include_router(vector_search.router)
app.include_router(tasks.router)
app.include_router(debug_stats.router)
app.include_router(green_space.router)

# Include workflow router if available
if WORKFLOW_ENABLED:
    app.include_router(workflow_router, prefix="/api/workflow", tags=["workflow"])

logger.info(" All routers registered")

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
MAPS_DIR = os.path.join(os.path.dirname(BACKEND_DIR), "maps")
os.makedirs(MAPS_DIR, exist_ok=True)

app.mount("/static/maps", StaticFiles(directory=MAPS_DIR), name="maps")
logger.info(f"Static maps mounted: {MAPS_DIR}")

# ==================== ROOT ENDPOINTS ====================

@app.get("/")
async def root():
    """Root endpoint - API information"""
    startup_time = app.state.startup_time
    uptime = str(datetime.now() - startup_time)
    
    return {
        "application": settings.app_name,
        "version": settings.app_version,
        "status": "operational",
        "uptime": uptime,
        "docs": "/docs",
        "health": "/health",
        "features": {
            "celery": CELERY_AVAILABLE,
            "vector_db": VECTOR_DB_AVAILABLE,
            "ai_agent": AI_AGENT_AVAILABLE,
            "rate_limiting": RATE_LIMITING_AVAILABLE,
            "workflow": WORKFLOW_ENABLED
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        db_connected = await Database.is_connected()
        
        return {
            "status": "healthy" if db_connected else "degraded",
            "timestamp": datetime.now().isoformat(),
            "version": settings.app_version,
            "database": "connected" if db_connected else "disconnected",
            "features": {
                "celery": CELERY_AVAILABLE,
                "vector_db": VECTOR_DB_AVAILABLE,
                "workflow": WORKFLOW_ENABLED,
                "geospatial": True,
                "ai_agent": AI_AGENT_AVAILABLE,
                "rate_limiting": RATE_LIMITING_AVAILABLE
            }
        }
    except Exception as e:
        return {
            "status": "degraded",
            "timestamp": datetime.now().isoformat(),
            "version": settings.app_version,
            "database": "unknown",
            "features": {},
            "error": str(e)
        }
    
# ==================== ERROR HANDLERS ====================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """HTTP exception handler"""
    logger.warning(f"HTTP {exc.status_code}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat(),
            "path": request.url.path
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "details": str(exc) if settings.debug else "Contact administrator",
            "timestamp": datetime.now().isoformat(),
            "request_id": getattr(request.state, "request_id", "unknown")
        }
    )

# ==================== STARTUP ====================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="debug" if settings.debug else "info",
        access_log=True,
        timeout_keep_alive=30
    )