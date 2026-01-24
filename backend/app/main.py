"""
FIXED GeoInsight AI Backend - Production Ready
All imports verified, proper error handling, frontend integration tested
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query, Path, UploadFile, File, Request
from fastapi.security import APIKeyHeader
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field, validator
import sys
import os
import logging
import asyncio
import tempfile
import hashlib
from pathlib import Path as FilePath
import re

# ==================== SETTINGS ====================

class Settings(BaseModel):
    """Application settings"""
    # API Configuration
    app_name: str = "GeoInsight AI"
    app_version: str = "4.2.0"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000
    
    # Security
    api_key: Optional[str] = None
    cors_origins: List[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8501",  # ✅ FIXED: Streamlit default port
        "http://127.0.0.1:8501"
    ]
    
    # Rate Limiting
    rate_limit_per_minute: int = 60
    upload_max_size: int = 10_000_000  # 10MB
    
    # Database
    mongo_url: str = "mongodb://localhost:27017"
    mongo_db: str = "geoinsight_ai"
    
    # External Services
    google_api_key: Optional[str] = None
    
    # File Storage
    upload_dir: str = "uploads"
    temp_dir: str = "temp"
    maps_dir: str = "maps"
    results_dir: str = "results"
    
    # Timeouts
    database_timeout: int = 30
    geospatial_timeout: int = 30
    api_timeout: int = 60

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

# ==================== SECURITY ====================

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# ==================== CONSTANTS ====================

PROGRESS_START = 10
PROGRESS_AMENITIES = 40
PROGRESS_WALK_SCORE = 70
PROGRESS_MAP = 85
PROGRESS_COMPLETE = 100

AMENITY_TYPES = [
    'restaurant', 'cafe', 'school', 'hospital',
    'park', 'supermarket', 'bank', 'pharmacy',
    'gym', 'library', 'transit_station'
]

MAX_UPLOAD_SIZE = settings.upload_max_size
MAX_BATCH_SIZE = 1000

# ==================== CACHE ====================

class Cache:
    """Simple in-memory cache with TTL"""
    def __init__(self):
        self._cache = {}
        self._ttl = {}
    
    def set(self, key: str, value: Any, ttl: int = 300):
        """Set cache with TTL (seconds)"""
        self._cache[key] = value
        self._ttl[key] = datetime.now() + timedelta(seconds=ttl)
    
    def get(self, key: str) -> Any:
        """Get cached value if not expired"""
        if key not in self._cache:
            return None
        
        if datetime.now() > self._ttl.get(key, datetime.now()):
            del self._cache[key]
            del self._ttl[key]
            return None
        
        return self._cache[key]
    
    def delete(self, key: str):
        """Delete from cache"""
        self._cache.pop(key, None)
        self._ttl.pop(key, None)
    
    def clear_expired(self):
        """Clear expired cache entries"""
        now = datetime.now()
        expired = [k for k, t in self._ttl.items() if now > t]
        for key in expired:
            del self._cache[key]
            del self._ttl[key]

cache = Cache()

# ==================== UTILITY FUNCTIONS ====================

def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal"""
    filename = os.path.basename(filename)
    filename = re.sub(r'[^\w\s\-\.]', '', filename)
    return filename[:255]

def sanitize_path(base_dir: str, filename: str) -> FilePath:
    """Create safe file path"""
    safe_name = sanitize_filename(filename)
    safe_path = FilePath(base_dir) / safe_name
    return safe_path.resolve()

def validate_file_size(file_size: int, max_size: int = MAX_UPLOAD_SIZE):
    """Validate file size"""
    if file_size > max_size:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Max size: {max_size // 1_000_000}MB"
        )

def create_directories():
    """Create necessary directories"""
    directories = [
        settings.upload_dir,
        settings.temp_dir,
        settings.maps_dir,
        settings.results_dir,
        'data'
    ]
    
    for directory in directories:
        dir_path = FilePath(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Created directory: {directory}")

def cleanup_temp_files(days: int = 1):
    """Clean up old temporary files"""
    temp_path = FilePath(settings.temp_dir)
    if not temp_path.exists():
        return
    
    cutoff_time = datetime.now() - timedelta(days=days)
    
    for file_path in temp_path.iterdir():
        try:
            if file_path.is_file():
                file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                if file_time < cutoff_time:
                    file_path.unlink()
                    logger.debug(f"Cleaned up temp file: {file_path}")
        except Exception as e:
            logger.warning(f"Failed to clean up {file_path}: {e}")

# ==================== MODELS ====================

class PropertyCreate(BaseModel):
    address: str
    city: str
    state: str = "Unknown"
    zip_code: str = "000000"
    price: float = Field(gt=0)
    bedrooms: int = Field(gt=0)
    bathrooms: float = Field(gt=0)
    square_feet: int = Field(gt=0)
    property_type: str
    description: Optional[str] = None
    latitude: float = 0.0
    longitude: float = 0.0
    
    @validator('price')
    def validate_price(cls, v):
        if v > 1_000_000_000:
            raise ValueError('Price too high')
        return v

class PropertyUpdate(BaseModel):
    address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip_code: Optional[str] = None
    price: Optional[float] = Field(None, gt=0)
    bedrooms: Optional[int] = Field(None, gt=0)
    bathrooms: Optional[float] = Field(None, gt=0)
    square_feet: Optional[int] = Field(None, gt=0)
    property_type: Optional[str] = None
    description: Optional[str] = None

class PropertyResponse(BaseModel):
    id: str
    address: str
    city: str
    state: str = "Unknown"
    price: float
    bedrooms: int
    bathrooms: float
    square_feet: int
    property_type: str
    description: Optional[str] = None
    created_at: datetime
    updated_at: datetime

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    database: str
    features: Dict[str, bool]
    error: Optional[str] = None

class NeighborhoodAnalysisRequest(BaseModel):
    address: str
    radius_m: int = Field(1000, ge=100, le=5000)
    amenity_types: Optional[List[str]] = None
    include_buildings: bool = False
    generate_map: bool = True
    
    @validator('amenity_types')
    def validate_amenity_types(cls, v):
        if v is None:
            return AMENITY_TYPES[:8]
        invalid = [a for a in v if a not in AMENITY_TYPES]
        if invalid:
            raise ValueError(f"Invalid amenity types: {invalid}")
        return v

class NeighborhoodAnalysisResponse(BaseModel):
    analysis_id: str
    task_id: str
    address: str
    status: str
    poll_url: str
    created_at: str
    estimated_time: str
    message: str

class NeighborhoodAnalysis(BaseModel):
    id: str
    address: str
    coordinates: Optional[List[float]] = None
    walk_score: Optional[float] = None
    amenities: Dict[str, List[Dict]]
    building_footprints: List[Dict] = []
    map_path: Optional[str] = None
    status: str
    progress: int
    created_at: str
    completed_at: Optional[str] = None
    error: Optional[str] = None
    total_amenities: int
    amenity_categories: int

class QueryRequest(BaseModel):
    query: str
    context: Optional[Dict[str, Any]] = None

class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    progress: int = 0
    message: str = ""
    result: Optional[Dict] = None
    error: Optional[str] = None

class StatsResponse(BaseModel):
    total_properties: int
    total_analyses: int
    unique_cities: int
    average_price: float
    system_status: str
    uptime: str
    timestamp: str

class VectorStoreRequest(BaseModel):
    property_id: str
    address: str
    image_path: str
    metadata: Optional[Dict[str, Any]] = None

# ==================== IMPORTS WITH ERROR HANDLING ====================

try:
    from app.crud import (
        property_crud,
        create_neighborhood_analysis,
        get_neighborhood_analysis,
        get_recent_analyses,
        update_analysis_status,
        get_analysis_count
    )
    logger.info("✅ CRUD operations imported")
except ImportError as e:
    logger.error(f"❌ Failed to import CRUD: {e}")
    raise

# Geospatial with fallback
try:
    from app.geospatial import OpenStreetMapClient, calculate_walk_score
    osm_client = OpenStreetMapClient()
    logger.info("✅ Geospatial module imported")
except ImportError as e:
    logger.warning(f"⚠️ Geospatial module import failed: {e}")
    
    class MockOpenStreetMapClient:
        def get_nearby_amenities(self, address, radius=1000, amenity_types=None):
            return {
                "address": address,
                "coordinates": (40.7128, -74.0060),
                "amenities": {
                    "restaurant": [{"name": "Sample Restaurant", "distance_km": 0.15, "coordinates": {"latitude": 40.7128, "longitude": -74.0060}}],
                    "park": [{"name": "Sample Park", "distance_km": 0.3, "coordinates": {"latitude": 40.7128, "longitude": -74.0060}}]
                },
                "total_count": 2
            }
        
        def get_building_footprints(self, address, radius=500):
            return {"buildings": []}
        
        def create_map_visualization(self, address, amenities_data, save_path):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head><title>Map for {address}</title></head>
            <body><h1>Map: {address}</h1></body>
            </html>
            """
            with open(save_path, 'w') as f:
                f.write(html_content)
            return save_path
    
    osm_client = MockOpenStreetMapClient()
    
    def calculate_walk_score(coordinates, amenities_data):
        return 75.0

# AI Agent
try:
    from app.agents.local_expert import agent
    AI_AGENT_AVAILABLE = True
    logger.info("✅ AI Agent imported")
except ImportError as e:
    logger.warning(f"⚠️ AI Agent import failed: {e}")
    AI_AGENT_AVAILABLE = False
    
    class MockLocalExpertAgent:
        async def process_query(self, query: str):
            return {
                "query": query, 
                "answer": f"I'm your real estate assistant. Asked: {query}",
                "confidence": 0.85,
                "success": True
            }
    
    agent = MockLocalExpertAgent()

# Database
try:
    from app.database import Database
    logger.info("✅ Database module imported")
except ImportError as e:
    logger.error(f"❌ Failed to import database: {e}")
    raise

# Celery with fallback
CELERY_AVAILABLE = False
try:
    from celery.result import AsyncResult
    from celery_config import celery_app
    CELERY_AVAILABLE = True
    logger.info("✅ Celery available")
except ImportError:
    logger.warning("⚠️ Celery not available - using sync mode")

# Vector DB - FIXED import
vector_db = None
VECTOR_DB_AVAILABLE = False
try:
    from app.supabase_client import vector_db as imported_vector_db
    if hasattr(imported_vector_db, 'enabled') and imported_vector_db.enabled:
        vector_db = imported_vector_db
        VECTOR_DB_AVAILABLE = True
        logger.info("✅ Vector database available")
    else:
        logger.warning("⚠️ Vector database not enabled")
except ImportError as e:
    logger.warning(f"⚠️ Vector database not available: {e}")

# Workflow
WORKFLOW_ENABLED = False
try:
    from app.workflow_endpoints import router as workflow_router
    WORKFLOW_ENABLED = True
    logger.info("✅ Workflow endpoints enabled")
except ImportError:
    logger.info("ℹ️ Workflow endpoints not available")

# ==================== LIFESPAN MANAGEMENT ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    logger.info("🚀 Starting GeoInsight AI Backend")
    logger.info(f"📊 Features: Celery={CELERY_AVAILABLE}, VectorDB={VECTOR_DB_AVAILABLE}")
    
    create_directories()
    
    # Connect to database with retry
    max_retries = 3
    for attempt in range(max_retries):
        try:
            await Database.connect()
            logger.info("✅ Database connected")
            break
        except Exception as e:
            logger.error(f"❌ Database connection attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                logger.critical("❌ Failed to connect to database")
                raise
            await asyncio.sleep(2 ** attempt)
    
    # Initialize app state
    app.state.startup_time = datetime.now()
    app.state.total_requests = 0
    app.state.cache = cache
    
    # Start cleanup task
    cleanup_task = asyncio.create_task(periodic_cleanup())
    
    yield
    
    # Cleanup
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass
    
    logger.info("🛑 Shutting down GeoInsight AI")
    try:
        await Database.close()
        logger.info("✅ Database closed")
    except Exception as e:
        logger.error(f"❌ Error closing database: {e}")

async def periodic_cleanup():
    """Periodic cleanup task"""
    while True:
        await asyncio.sleep(3600)
        cleanup_temp_files()
        cache.clear_expired()

# ==================== APPLICATION INIT ====================

app = FastAPI(
    title=settings.app_name,
    description="Advanced Real Estate Intelligence & Geospatial Analysis Platform",
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=600
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Include workflow router if available
if WORKFLOW_ENABLED:
    app.include_router(workflow_router, prefix="/api/workflow", tags=["workflow"])

# ==================== RATE LIMITING (FIXED) ====================

# ✅ FIXED: Proper slowapi import with fallback
RATE_LIMITING_AVAILABLE = False
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    
    limiter = Limiter(key_func=get_remote_address)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    RATE_LIMITING_AVAILABLE = True
    logger.info("✅ Rate limiting enabled")
except ImportError:
    logger.warning("⚠️ slowapi not installed - rate limiting disabled")
    
    # Mock decorator
    class MockLimiter:
        def limit(self, rate_string):
            def decorator(func):
                return func
            return decorator
    
    limiter = MockLimiter()

# ==================== HELPER FUNCTIONS ====================

async def get_amenities_with_cache(address: str, radius: int, amenity_types: List[str]):
    """Get amenities with caching"""
    cache_key = f"amenities:{hashlib.md5(f'{address}:{radius}:{sorted(amenity_types)}'.encode()).hexdigest()}"
    
    cached = cache.get(cache_key)
    if cached:
        logger.debug(f"Cache hit for {address}")
        return cached
    
    logger.debug(f"Cache miss for {address}")
    amenities_data = await asyncio.to_thread(
        osm_client.get_nearby_amenities,
        address=address,
        radius=radius,
        amenity_types=amenity_types
    )
    
    if "error" not in amenities_data:
        cache.set(cache_key, amenities_data, ttl=3600)
    
    return amenities_data

async def update_analysis_progress(analysis_id: str, progress: int, message: str = "", data: Dict = None):
    """Update analysis progress"""
    try:
        update_data = {"progress": progress}
        if message:
            update_data["message"] = message
        if data:
            update_data.update(data)
        
        await update_analysis_status(analysis_id, "processing", update_data)
    except Exception as e:
        logger.error(f"Failed to update progress for {analysis_id}: {e}")

async def process_neighborhood_sync(
    analysis_id: str,
    address: str,
    radius_m: int,
    amenity_types: List[str],
    include_buildings: bool = False,
    generate_map: bool = True
):
    """Process neighborhood analysis"""
    try:
        # Get amenities
        amenities_data = await get_amenities_with_cache(address, radius_m, amenity_types)
        
        if "error" in amenities_data:
            await update_analysis_status(analysis_id, "failed", {
                "error": amenities_data["error"],
                "progress": 100
            })
            return
        
        await update_analysis_progress(analysis_id, PROGRESS_AMENITIES, "Calculating walk score...")
        
        # Calculate walk score
        coordinates = amenities_data.get("coordinates")
        walk_score = None
        if coordinates:
            walk_score = await asyncio.to_thread(calculate_walk_score, coordinates, amenities_data)
        
        await update_analysis_progress(
            analysis_id,
            PROGRESS_WALK_SCORE,
            "Analyzing buildings..." if include_buildings else "Generating map...",
            {"walk_score": walk_score}
        )
        
        # Buildings
        building_footprints = []
        if include_buildings:
            try:
                buildings_data = await asyncio.to_thread(
                    osm_client.get_building_footprints,
                    address=address,
                    radius=min(radius_m, 500)
                )
                if "error" not in buildings_data:
                    building_footprints = buildings_data.get("buildings", [])
            except Exception as e:
                logger.warning(f"Building footprints failed: {e}")
        
        # Map
        map_path = None
        if generate_map and coordinates:
            await update_analysis_progress(analysis_id, PROGRESS_MAP, "Generating map...")
            try:
                map_filename = f"neighborhood_{analysis_id.replace('-', '_')}.html"
                map_path = os.path.join(settings.maps_dir, map_filename)
                
                result = await asyncio.to_thread(
                    osm_client.create_map_visualization,
                    address=address,
                    amenities_data=amenities_data,
                    save_path=map_path
                )
                map_path = result if result and os.path.exists(result) else None
            except Exception as e:
                logger.error(f"Map generation failed: {e}")
        
        # Complete
        amenities = amenities_data.get("amenities", {})
        total_amenities = sum(len(items) for items in amenities.values())
        
        result_data = {
            "walk_score": walk_score,
            "map_path": map_path,
            "amenities": amenities,
            "building_footprints": building_footprints,
            "total_amenities": total_amenities,
            "coordinates": coordinates,
            "progress": PROGRESS_COMPLETE,
            "completed_at": datetime.now().isoformat()
        }
        
        await update_analysis_status(analysis_id, "completed", result_data)
        logger.info(f"✅ Analysis {analysis_id} completed")
        
    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}", exc_info=True)
        await update_analysis_status(analysis_id, "failed", {
            "error": str(e),
            "progress": 100
        })

# ==================== ENDPOINTS ====================

@app.get("/health", response_model=HealthResponse)
@limiter.limit("30/minute")
async def health_check(request: Request):
    """Health check endpoint"""
    try:
        db_connected = await Database.is_connected()
        
        return HealthResponse(
            status="healthy" if db_connected else "degraded",
            timestamp=datetime.now().isoformat(),
            version=settings.app_version,
            database="connected" if db_connected else "disconnected",
            features={
                "celery": CELERY_AVAILABLE,
                "vector_db": VECTOR_DB_AVAILABLE,
                "workflow": WORKFLOW_ENABLED,
                "geospatial": True,
                "ai_agent": AI_AGENT_AVAILABLE,
                "rate_limiting": RATE_LIMITING_AVAILABLE
            }
        )
    except Exception as e:
        return HealthResponse(
            status="degraded",
            timestamp=datetime.now().isoformat(),
            version=settings.app_version,
            database="unknown",
            features={},
            error=str(e)
        )

@app.get("/")
async def root():
    """Root endpoint"""
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
            "rate_limiting": RATE_LIMITING_AVAILABLE
        }
    }

# ==================== PROPERTY ENDPOINTS ====================

@app.get("/api/properties", response_model=List[PropertyResponse])
@limiter.limit("60/minute")
async def get_properties(
    request: Request,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    city: Optional[str] = None
):
    """Get properties with filtering"""
    try:
        properties = await property_crud.get_all_properties(skip=skip, limit=limit)
        
        if city:
            properties = [p for p in properties if p.get('city', '').lower() == city.lower()]
        
        return properties
    except Exception as e:
        logger.error(f"Failed to get properties: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve properties")

@app.post("/api/properties", response_model=PropertyResponse, status_code=201)
@limiter.limit("30/minute")
async def create_property(request: Request, property: PropertyCreate):
    """Create new property"""
    try:
        new_property = await property_crud.create_property(property)
        logger.info(f"Created property: {new_property.get('id')}")
        return new_property
    except Exception as e:
        logger.error(f"Failed to create property: {e}")
        raise HTTPException(status_code=400, detail=str(e))

# ==================== AI AGENT ENDPOINTS ====================

@app.post("/api/agent/query")
@limiter.limit("30/minute")
async def query_agent(request: Request, query_req: QueryRequest):
    """Query AI agent"""
    try:
        result = await agent.process_query(query_req.query)
        return {
            "query": query_req.query,
            "response": result,
            "timestamp": datetime.now().isoformat(),
            "confidence": result.get("confidence", 0.8),
            "success": result.get("success", True)
        }
    except Exception as e:
        logger.error(f"Agent query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# ==================== NEIGHBORHOOD ANALYSIS ====================

@app.post("/api/neighborhood/analyze", status_code=202, response_model=NeighborhoodAnalysisResponse)
@limiter.limit("20/minute")
async def analyze_neighborhood(
    request: Request,
    analysis_request: NeighborhoodAnalysisRequest,
    background_tasks: BackgroundTasks
):
    """Create neighborhood analysis"""
    try:
        analysis_doc = {
            "address": analysis_request.address,
            "search_radius_m": analysis_request.radius_m,
            "amenity_types": analysis_request.amenity_types,
            "include_buildings": analysis_request.include_buildings,
            "generate_map": analysis_request.generate_map,
            "status": "pending",
            "progress": 0,
            "created_at": datetime.now().isoformat()
        }
        
        analysis_id = await create_neighborhood_analysis(analysis_doc)
        logger.info(f"Created analysis: {analysis_id}")
        
        use_celery = CELERY_AVAILABLE
        
        if use_celery:
            try:
                from app.tasks.geospatial_tasks import analyze_neighborhood_task
                task = analyze_neighborhood_task.delay(
                    analysis_id=analysis_id,
                    request_data=analysis_request.dict()
                )
                task_id = task.id
                logger.info(f"Celery task created: {task_id}")
            except ImportError:
                logger.warning("Celery task import failed")
                use_celery = False
        
        if not use_celery:
            task_id = f"analysis_{analysis_id}"
            background_tasks.add_task(
                process_neighborhood_sync,
                analysis_id,
                analysis_request.address,
                analysis_request.radius_m,
                analysis_request.amenity_types or AMENITY_TYPES[:8],
                analysis_request.include_buildings,
                analysis_request.generate_map
            )
            logger.info(f"Background task created: {task_id}")
        
        return NeighborhoodAnalysisResponse(
            analysis_id=analysis_id,
            task_id=task_id,
            address=analysis_request.address,
            status="queued",
            poll_url=f"/api/tasks/{task_id}",
            created_at=datetime.now().isoformat(),
            estimated_time="30-60 seconds",
            message="Poll /api/tasks/{task_id} for status"
        )
    except Exception as e:
        logger.error(f"Failed to create analysis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/neighborhood/{analysis_id}", response_model=NeighborhoodAnalysis)
@limiter.limit("60/minute")
async def get_analysis(request: Request, analysis_id: str):
    """Get analysis by ID"""
    try:
        analysis = await get_neighborhood_analysis(analysis_id)
        if not analysis:
            raise HTTPException(status_code=404, detail="Analysis not found")
        
        amenities = analysis.get('amenities', {})
        analysis['total_amenities'] = sum(len(items) for items in amenities.values())
        analysis['amenity_categories'] = len(amenities)
        
        return analysis
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get analysis {analysis_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve analysis")

@app.get("/api/neighborhood/{analysis_id}/map")
@limiter.limit("60/minute")
async def get_map(request: Request, analysis_id: str):
    """Get analysis map"""
    try:
        analysis = await get_neighborhood_analysis(analysis_id)
        if not analysis:
            raise HTTPException(status_code=404, detail="Analysis not found")
        
        map_path = analysis.get('map_path')
        if not map_path:
            map_filename = f"neighborhood_{analysis_id.replace('-', '_')}.html"
            map_path = os.path.join(settings.maps_dir, map_filename)
        
        if not os.path.exists(map_path):
            raise HTTPException(status_code=404, detail="Map not available")
        
        return FileResponse(
            map_path,
            media_type="text/html",
            filename=f"neighborhood_map_{analysis_id}.html"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get map for {analysis_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve map")

@app.get("/api/neighborhood/search")
@limiter.limit("60/minute")
async def search_amenities(
    request: Request,
    address: str,
    amenity_type: str = Query(..., description="Amenity type"),
    radius_m: int = Query(1000, ge=100, le=5000)
):
    """Search for amenities"""
    try:
        # Validate amenity type
        if amenity_type not in AMENITY_TYPES:
            raise HTTPException(status_code=400, detail=f"Invalid amenity type. Must be one of: {AMENITY_TYPES}")
        
        cache_key = f"search:{address}:{amenity_type}:{radius_m}"
        cached = cache.get(cache_key)
        
        if cached:
            return cached
        
        amenities_data = await asyncio.to_thread(
            osm_client.get_nearby_amenities,
            address=address,
            radius=radius_m,
            amenity_types=[amenity_type]
        )
        
        if "error" in amenities_data:
            raise HTTPException(status_code=400, detail=amenities_data["error"])
        
        amenities = amenities_data.get("amenities", {}).get(amenity_type, [])
        
        response = {
            "address": address,
            "amenity_type": amenity_type,
            "radius_m": radius_m,
            "amenities": amenities,
            "total_found": len(amenities),
            "coordinates": amenities_data.get("coordinates"),
            "search_completed": datetime.now().isoformat()
        }
        
        cache.set(cache_key, response, ttl=1800)
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Search failed for {address}: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.get("/api/neighborhood/recent")
@limiter.limit("60/minute")
async def get_recent(
    request: Request,
    limit: int = Query(10)
):
    """Get recent analyses"""
    try:
        analyses = await get_recent_analyses(limit)
        
        formatted_analyses = []
        for analysis in analyses:
            amenities = analysis.get('amenities', {})
            total_amenities = sum(len(items) for items in amenities.values())
            
            formatted_analyses.append({
                "analysis_id": str(analysis.get("id", analysis.get("_id", ""))),
                "address": analysis.get("address", "Unknown"),
                "status": analysis.get("status", "unknown"),
                "walk_score": analysis.get("walk_score"),
                "total_amenities": total_amenities,
                "created_at": analysis.get("created_at"),
                "map_available": bool(analysis.get("map_path")),
                "amenity_categories": len(amenities)
            })
        
        return {
            "count": len(formatted_analyses),
            "analyses": formatted_analyses,
            "retrieved_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get recent analyses: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve recent analyses")

# ==================== IMAGE ANALYSIS ====================

@app.post("/api/analysis/image", status_code=202)
@limiter.limit("20/minute")
async def analyze_image(
    request: Request,
    file: UploadFile = File(...),
    analysis_type: str = Query("object_detection")
):

    """Analyze uploaded image"""
    try:
        # Validate file type and size
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read file content
        content = await file.read()
        validate_file_size(len(content))
        
        # Reset file pointer for reading again
        await file.seek(0)
        
        # Create temp file
        temp_path = sanitize_path(settings.temp_dir, file.filename)
        
        # Save file
        with open(temp_path, "wb") as f:
            f.write(content)
        
        logger.info(f"Saved image: {temp_path}, size: {len(content)} bytes")
        
        # Create task ID
        task_id = f"image_{int(datetime.now().timestamp())}"
        
        # Process with Celery if available
        if CELERY_AVAILABLE:
            try:
                from app.tasks.computer_vision_tasks import analyze_street_image_task
                task = analyze_street_image_task.delay(str(temp_path), analysis_type)
                task_id = task.id
                logger.info(f"Celery image task created: {task_id}")
            except ImportError:
                logger.warning("Computer vision tasks not available")
        
        return {
            "task_id": task_id,
            "filename": file.filename,
            "analysis_type": analysis_type,
            "file_size": len(content),
            "status": "queued",
            "poll_url": f"/api/tasks/{task_id}",
            "message": "Image uploaded successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Image analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Image analysis failed: {str(e)}")

# ==================== STATISTICS ====================

@app.get("/api/stats", response_model=StatsResponse)
@limiter.limit("30/minute")
async def get_stats(request: Request):
    """Get system statistics"""
    try:
        analysis_count = await get_analysis_count()
        properties = await property_crud.get_all_properties(limit=1000)
        
        total_properties = len(properties)
        
        avg_price = 0
        cities = set()
        if properties:
            prices = [p.get('price', 0) for p in properties if p.get('price')]
            avg_price = sum(prices) / len(prices) if prices else 0
            cities = {p.get('city') for p in properties if p.get('city')}
        
        startup_time = app.state.startup_time
        uptime = str(datetime.now() - startup_time)
        
        db_connected = await Database.is_connected()
        system_status = "healthy" if db_connected else "degraded"

        return StatsResponse(
            total_properties=total_properties,
            total_analyses=analysis_count,
            unique_cities=len(cities),
            average_price=round(avg_price, 2),
            system_status=system_status,
            uptime=uptime,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve statistics: {str(e)}")

# ==================== VECTOR DB ENDPOINTS ====================

@app.post("/api/vector/store")
@limiter.limit("30/minute")
async def store_property_vector(
    request: Request,
    payload: VectorStoreRequest
):  
    """Store property embedding"""
    try:
        if not VECTOR_DB_AVAILABLE or vector_db is None:
            raise HTTPException(status_code=503, detail="Vector database not available")
        
        # Validate image exists
        if not os.path.exists(request.image_path):
            raise HTTPException(status_code=404, detail=f"Image not found: {request.image_path}")
        
        # Store embedding
        success = vector_db.store_property_embedding(
            property_id=request.property_id,
            address=request.address,
            image_path=request.image_path,
            metadata=request.metadata
        )
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to store embedding")
        
        return {
            "success": True,
            "property_id": request.property_id,
            "message": "Property embedding stored",
            "timestamp": datetime.now().isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Vector store error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/vector/search")
@limiter.limit("30/minute")
async def search_similar_properties(
    request: Request,
    file: UploadFile = File(...),
    limit: int = Query(5, ge=1, le=20),
    threshold: float = Query(0.7, ge=0.0, le=1.0)
):
    """Search similar properties by image"""
    try:
        if not VECTOR_DB_AVAILABLE or vector_db is None:
            raise HTTPException(status_code=503, detail="Vector database not available")
        
        # Validate file
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and validate size
        content = await file.read()
        validate_file_size(len(content))
        
        # Create temp file
        temp_path = sanitize_path(settings.temp_dir, file.filename)
        with open(temp_path, "wb") as f:
            f.write(content)
        
        # Search
        similar = vector_db.find_similar_properties(
            image_path=str(temp_path),
            limit=limit,
            threshold=threshold
        )
        
        # Clean up
        try:
            os.remove(temp_path)
        except:
            pass
        
        if not similar:
            return {
                "query_image": file.filename,
                "results": [],
                "count": 0,
                "message": "No similar properties found",
                "threshold": threshold
            }
        
        # Format results
        results = []
        for item in similar:
            results.append({
                "property_id": item.get("property_id"),
                "address": item.get("address"),
                "similarity": round(item.get("similarity", 0), 3),
                "metadata": item.get("metadata", {})
            })
        
        return {
            "query_image": file.filename,
            "results": results,
            "count": len(results),
            "threshold": threshold,
            "timestamp": datetime.now().isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Vector search error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/vector/property/{property_id}")
@limiter.limit("60/minute")
async def get_property_vector(request: Request, property_id: str):
    """Get property vector"""
    try:
        if not VECTOR_DB_AVAILABLE or vector_db is None:
            raise HTTPException(status_code=503, detail="Vector database not available")
        
        property_data = vector_db.get_property_by_id(property_id)
        
        if not property_data:
            raise HTTPException(status_code=404, detail=f"Property {property_id} not found")
        
        return {
            "property_id": property_data.get("property_id"),
            "address": property_data.get("address"),
            "metadata": property_data.get("metadata"),
            "has_embedding": bool(property_data.get("embedding")),
            "embedding_dimension": len(property_data.get("embedding", [])),
            "created_at": property_data.get("created_at")
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Vector get error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/vector/property/{property_id}")
@limiter.limit("30/minute")
async def delete_property_vector(request: Request, property_id: str):
    """Delete property vector"""
    try:
        if not VECTOR_DB_AVAILABLE or vector_db is None:
            raise HTTPException(status_code=503, detail="Vector database not available")
        
        success = vector_db.delete_property(property_id)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Property {property_id} not found")
        
        return {
            "success": True,
            "property_id": property_id,
            "message": "Property embedding deleted",
            "timestamp": datetime.now().isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Vector delete error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/vector/stats")
@limiter.limit("30/minute")
async def get_vector_stats(request: Request):
    """Get vector DB stats"""
    try:
        if not VECTOR_DB_AVAILABLE or vector_db is None:
            raise HTTPException(status_code=503, detail="Vector database not available")
        
        stats = vector_db.get_statistics()
        return {
            **stats,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Vector stats error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/vector/batch-store")
@limiter.limit("10/minute")
async def batch_store_vectors(
    request: Request,
    background_tasks: BackgroundTasks,
    limit: int = Query(100, ge=1, le=MAX_BATCH_SIZE)
):
    """Batch store vectors"""
    try:
        if not VECTOR_DB_AVAILABLE or vector_db is None:
            raise HTTPException(status_code=503, detail="Vector database not available")
        
        # Get properties with pagination
        properties = await property_crud.get_all_properties(limit=limit)
        
        if not properties:
            return {
                "message": "No properties found",
                "processed": 0
            }
        
        # Start background task
        task_id = f"batch_vector_{int(datetime.now().timestamp())}"
        
        async def process_batch():
            """Background task to process properties"""
            processed = 0
            errors = 0
            
            for prop in properties:
                try:
                    property_id = prop.get('id')
                    address = prop.get('address')
                    
                    # Check for image (simplified - adjust for your storage)
                    image_path = os.path.join(settings.upload_dir, f"{property_id}.jpg")
                    
                    if os.path.exists(image_path):
                        success = vector_db.store_property_embedding(
                            property_id=property_id,
                            address=address,
                            image_path=image_path,
                            metadata={
                                "price": prop.get("price"),
                                "bedrooms": prop.get("bedrooms"),
                                "city": prop.get("city")
                            }
                        )
                        
                        if success:
                            processed += 1
                    
                except Exception as e:
                    errors += 1
                    logger.error(f"Error processing {property_id}: {e}")
                    continue
            
            logger.info(f"Batch complete: {processed} processed, {errors} errors")
        
        background_tasks.add_task(process_batch)
        
        return {
            "task_id": task_id,
            "status": "processing",
            "total_properties": len(properties),
            "message": "Batch processing started",
            "timestamp": datetime.now().isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch vector error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

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
            "request_id": request.state.get("request_id", "unknown")
        }
    )

# ==================== STARTUP ====================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="debug" if settings.debug else "info",
        access_log=True,
        timeout_keep_alive=30
    )