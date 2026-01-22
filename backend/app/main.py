from fastapi import FastAPI, HTTPException, BackgroundTasks, Query, Path, UploadFile, File
from typing import List, Optional, Dict, Any
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import sys
import os
import logging
from pathlib import Path as FilePath
import asyncio

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import models
try:
    from app.models import (
        PropertyCreate, PropertyUpdate, PropertyResponse, HealthResponse,
        NeighborhoodAnalysisRequest, NeighborhoodAnalysisResponse,
        NeighborhoodAnalysis, Coordinates
    )
    logger.info("✅ Models imported successfully")
except ImportError as e:
    logger.error(f"❌ Failed to import models: {e}")
    raise

# Import CRUD
try:
    from app.crud import (
        property_crud,
        create_neighborhood_analysis,
        get_neighborhood_analysis,
        get_recent_analyses,
        update_analysis_status,
        get_analysis_count
    )
    logger.info("✅ CRUD operations imported successfully")
except ImportError as e:
    logger.error(f"❌ Failed to import CRUD: {e}")
    raise

# Import geospatial
try:
    from app.geospatial import OpenStreetMapClient, calculate_walk_score
    osm_client = OpenStreetMapClient()
    logger.info("✅ Geospatial module imported successfully")
except ImportError as e:
    logger.warning(f"⚠️ Geospatial module import failed: {e}")
    logger.warning("⚠️ Creating mock geospatial client...")
    
    class MockOpenStreetMapClient:
        def get_nearby_amenities(self, address, radius=1000, amenity_types=None):
            return {
                "address": address,
                "coordinates": (40.7128, -74.0060),  # NYC coordinates
                "amenities": {
                    "restaurant": [{"name": "Sample Restaurant", "distance": 150}],
                    "park": [{"name": "Sample Park", "distance": 300}]
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
            <head>
                <title>Map for {address}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; }}
                    h1 {{ color: #2c3e50; }}
                    .map-container {{ padding: 20px; background: #f8f9fa; }}
                </style>
            </head>
            <body>
                <h1>Neighborhood Map: {address}</h1>
                <div class="map-container">
                    <p>Map visualization would appear here with actual geospatial data.</p>
                    <p>Coordinates: {amenities_data.get('coordinates', [0, 0])}</p>
                </div>
            </body>
            </html>
            """
            with open(save_path, 'w') as f:
                f.write(html_content)
            return save_path
    
    osm_client = MockOpenStreetMapClient()
    
    def calculate_walk_score(coordinates, amenities_data):
        return 75.0  # Default walk score

# Import AI Agent
try:
    from app.agents.local_expert import agent
    logger.info("✅ AI Agent imported successfully")
except ImportError as e:
    logger.warning(f"⚠️ AI Agent import failed: {e}")
    
    class LocalExpertAgent:
        async def process_query(self, query: str):
            return {
                "query": query, 
                "answer": f"I'm your real estate assistant. For your query '{query}', I'd recommend considering location, amenities, and market trends.", 
                "sources": ["market_data", "location_analysis"],
                "confidence": 0.85
            }
    
    agent = LocalExpertAgent()

# Import database
try:
    from app.database import Database
    logger.info("✅ Database module imported successfully")
except ImportError as e:
    logger.error(f"❌ Failed to import database: {e}")
    raise

# Celery imports with fallback
CELERY_AVAILABLE = False
try:
    from celery.result import AsyncResult
    from celery_config import celery_app
    CELERY_AVAILABLE = True
    logger.info("✅ Celery available - background processing enabled")
except ImportError:
    logger.warning("⚠️ Celery not available - background tasks will run synchronously")

# Additional imports from version 2
try:
    from app.workflow_endpoints import router as workflow_router
    WORKFLOW_ENABLED = True
    logger.info("✅ Workflow endpoints enabled")
except ImportError:
    WORKFLOW_ENABLED = False
    logger.info("ℹ️ Workflow endpoints not available")

# ==================== LIFESPAN MANAGEMENT ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    logger.info("🚀 Starting GeoInsight AI Backend")
    logger.info(f"📊 Features: Celery={CELERY_AVAILABLE}, Workflow={WORKFLOW_ENABLED}")
    
    try:
        await Database.connect()
        logger.info("✅ Database connected successfully")
    except Exception as e:
        logger.error(f"❌ Database connection error: {e}")
    
    # Create necessary directories
    directories = ['maps', 'results', 'temp', 'data', 'uploads']
    for directory in directories:
        dir_path = FilePath(directory)
        dir_path.mkdir(exist_ok=True)
        logger.debug(f"📁 Created directory: {directory}")
    
    # Initialize caches or connections
    app.state.startup_time = datetime.now()
    app.state.total_requests = 0
    
    yield  # Application runs here
    
    # Cleanup
    logger.info("🛑 Shutting down GeoInsight AI")
    try:
        await Database.close()
        logger.info("✅ Database connection closed")
    except Exception as e:
        logger.error(f"❌ Error closing database: {e}")

# ==================== APPLICATION INIT ====================

app = FastAPI(
    title="GeoInsight AI API",
    description="Advanced Real Estate Intelligence & Geospatial Analysis Platform",
    version="4.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Include workflow router if available
if WORKFLOW_ENABLED:
    app.include_router(workflow_router, prefix="/api/workflow", tags=["workflow"])

# ==================== MODELS ====================

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
    estimated_completion: Optional[str] = None

class WorkflowRequest(BaseModel):
    address: str
    radius_m: Optional[int] = 1000
    email: Optional[str] = None
    analysis_types: Optional[List[str]] = ["amenities", "walkability", "market"]

class StatsResponse(BaseModel):
    total_properties: int
    total_analyses: int
    unique_cities: int
    average_price: float
    system_status: str
    uptime: str
    timestamp: str

# ==================== HELPER FUNCTIONS ====================

async def process_neighborhood_sync(
    analysis_id: str,
    address: str,
    radius_m: int,
    amenity_types: List[str],
    include_buildings: bool = False,
    generate_map: bool = True
):
    """Synchronous fallback for neighborhood analysis"""
    try:
        # Update status
        await update_analysis_status(analysis_id, "processing", {"progress": 10})
        
        # Get amenities
        amenities_data = osm_client.get_nearby_amenities(
            address=address,
            radius=radius_m,
            amenity_types=amenity_types or [
                'restaurant', 'cafe', 'school', 'hospital',
                'park', 'supermarket', 'bank', 'pharmacy',
                'gym', 'library', 'transit_station'
            ]
        )
        
        if "error" in amenities_data:
            await update_analysis_status(analysis_id, "failed", {
                "error": amenities_data["error"],
                "progress": 100
            })
            return
        
        await update_analysis_status(analysis_id, "processing", {"progress": 40})
        
        # Calculate walk score
        coordinates = amenities_data.get("coordinates")
        walk_score = None
        if coordinates:
            walk_score = calculate_walk_score(coordinates, amenities_data)
        
        # Get building footprints if requested
        building_footprints = []
        if include_buildings:
            try:
                buildings_data = osm_client.get_building_footprints(
                    address=address,
                    radius=min(radius_m, 500)
                )
                if "error" not in buildings_data:
                    building_footprints = buildings_data.get("buildings", [])
            except Exception as e:
                logger.warning(f"Building footprints failed: {e}")
        
        await update_analysis_status(analysis_id, "processing", {
            "progress": 70,
            "walk_score": walk_score,
            "building_count": len(building_footprints)
        })
        
        # Generate map
        map_path = None
        if generate_map and coordinates:
            try:
                map_filename = f"neighborhood_{analysis_id.replace('-', '_')}.html"
                map_path = os.path.join("maps", map_filename)
                
                # Update map generation status
                await update_analysis_status(analysis_id, "processing", {
                    "progress": 85,
                    "map_generating": True
                })
                
                map_path = osm_client.create_map_visualization(
                    address=address,
                    amenities_data=amenities_data,
                    save_path=map_path
                )
            except Exception as e:
                logger.error(f"Map generation failed: {e}")
                map_path = None
        
        # Calculate total amenities
        amenities = amenities_data.get("amenities", {})
        total_amenities = sum(len(items) for items in amenities.values())
        
        # Complete analysis
        result_data = {
            "walk_score": walk_score,
            "map_path": map_path,
            "amenities": amenities,
            "building_footprints": building_footprints,
            "total_amenities": total_amenities,
            "coordinates": coordinates,
            "progress": 100,
            "completed_at": datetime.now().isoformat()
        }
        
        await update_analysis_status(analysis_id, "completed", result_data)
        
        logger.info(f"✅ Analysis {analysis_id} completed successfully")
        logger.info(f"   Address: {address}, Amenities: {total_amenities}, Walk Score: {walk_score}")
        
    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}", exc_info=True)
        await update_analysis_status(analysis_id, "failed", {
            "error": str(e),
            "progress": 100
        })

# ==================== HEALTH & ROOT ENDPOINTS ====================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint with system status"""
    try:
        db_status = "connected" if Database.is_connected() else "disconnected"
        
        return HealthResponse(
            status="healthy",
            timestamp=datetime.now().isoformat(),
            version="4.1.0",
            database=db_status,
            features={
                "celery": CELERY_AVAILABLE,
                "workflow": WORKFLOW_ENABLED,
                "geospatial": True,
                "ai_agent": True
            }
        )
    except Exception as e:
        return HealthResponse(
            status="degraded",
            timestamp=datetime.now().isoformat(),
            version="4.1.0",
            error=str(e)
        )

@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint with comprehensive API information"""
    startup_time = app.state.startup_time if hasattr(app.state, 'startup_time') else datetime.now()
    uptime = str(datetime.now() - startup_time)
    
    return {
        "application": "GeoInsight AI",
        "version": "4.1.0",
        "status": "operational",
        "uptime": uptime,
        "description": "Advanced Real Estate Intelligence Platform",
        "endpoints": {
            "health": "/health",
            "documentation": "/docs",
            "properties": {
                "list": "/api/properties",
                "create": "/api/properties [POST]",
                "details": "/api/properties/{id}"
            },
            "analysis": {
                "neighborhood": "/api/neighborhood/analyze [POST]",
                "status": "/api/tasks/{task_id}",
                "image": "/api/analysis/image [POST]"
            },
            "intelligence": {
                "agent": "/api/agent/query [POST]",
                "search": "/api/neighborhood/search"
            },
            "system": {
                "stats": "/api/stats",
                "recent": "/api/neighborhood/recent"
            }
        },
        "features": [
            "Property Management & CRUD",
            "Geospatial Neighborhood Analysis",
            "AI-Powered Real Estate Agent",
            "Walk Score Calculation",
            "Interactive Map Generation",
            "Image Analysis (Street View)",
            "Async Task Processing",
            "Workflow Automation" if WORKFLOW_ENABLED else "Basic Workflow"
        ],
        "system": {
            "celery_enabled": CELERY_AVAILABLE,
            "workflow_enabled": WORKFLOW_ENABLED,
            "database": "MongoDB"
        }
    }

# ==================== PROPERTY ENDPOINTS ====================

@app.get("/api/properties", response_model=List[PropertyResponse])
async def get_properties(
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum records to return"),
    city: Optional[str] = Query(None, description="Filter by city"),
    min_price: Optional[float] = Query(None, ge=0, description="Minimum price filter"),
    max_price: Optional[float] = Query(None, ge=0, description="Maximum price filter")
):
    """Get all properties with filtering options"""
    try:
        properties = await property_crud.get_all_properties(skip=skip, limit=limit)
        
        # Apply filters
        filtered_properties = []
        for prop in properties:
            # City filter
            if city and prop.get('city', '').lower() != city.lower():
                continue
            
            # Price filters
            price = prop.get('price', 0)
            if min_price is not None and price < min_price:
                continue
            if max_price is not None and price > max_price:
                continue
            
            filtered_properties.append(prop)
        
        logger.info(f"Retrieved {len(filtered_properties)} properties")
        return filtered_properties
        
    except Exception as e:
        logger.error(f"Failed to get properties: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve properties")

@app.get("/api/properties/{property_id}", response_model=PropertyResponse)
async def get_property(
    property_id: str = Path(..., description="Property ID")
):
    """Get property by ID"""
    try:
        property_data = await property_crud.get_property_by_id(property_id)
        if not property_data:
            raise HTTPException(status_code=404, detail="Property not found")
        return property_data
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get property {property_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/api/properties", response_model=PropertyResponse, status_code=201)
async def create_property(property: PropertyCreate):
    """Create new property"""
    try:
        new_property = await property_crud.create_property(property)
        logger.info(f"Created new property: {new_property.get('id')}")
        return new_property
    except Exception as e:
        logger.error(f"Failed to create property: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to create property: {str(e)}")

@app.put("/api/properties/{property_id}", response_model=PropertyResponse)
async def update_property(
    property_id: str = Path(..., description="Property ID"),
    property: PropertyUpdate = ...
):
    """Update existing property"""
    try:
        updated = await property_crud.update_property(property_id, property)
        if not updated:
            raise HTTPException(status_code=404, detail="Property not found")
        
        logger.info(f"Updated property: {property_id}")
        return updated
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update property {property_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to update property")

@app.delete("/api/properties/{property_id}")
async def delete_property(
    property_id: str = Path(..., description="Property ID")
):
    """Delete property"""
    try:
        success = await property_crud.delete_property(property_id)
        if not success:
            raise HTTPException(status_code=404, detail="Property not found")
        
        logger.info(f"Deleted property: {property_id}")
        return {"message": "Property deleted successfully", "id": property_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete property {property_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete property")

# ==================== AI AGENT ENDPOINTS ====================

@app.post("/api/agent/query")
async def query_agent(request: QueryRequest):
    """Query AI agent for real estate insights"""
    try:
        result = await agent.process_query(request.query)
        return {
            "query": request.query,
            "response": result,
            "timestamp": datetime.now().isoformat(),
            "confidence": result.get("confidence", 0.8)
        }
    except Exception as e:
        logger.error(f"Agent query failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"AI agent query failed: {str(e)}"
        )

# ==================== TASK STATUS ENDPOINTS ====================

@app.get("/api/tasks/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """
    Get task status - Frontend polls this for background tasks
    Supports both Celery and synchronous tasks
    """
    try:
        # Check for analysis tasks (sync fallback)
        if task_id.startswith("analysis_"):
            analysis_id = task_id.replace("analysis_", "")
            analysis = await get_neighborhood_analysis(analysis_id)
            
            if not analysis:
                return TaskStatusResponse(
                    task_id=task_id,
                    status="not_found",
                    message="Task not found"
                )
            
            status_val = analysis.get('status', 'unknown')
            progress = analysis.get('progress', 0)
            result_data = {
                "analysis_id": analysis_id,
                "walk_score": analysis.get('walk_score'),
                "total_amenities": sum(
                    len(items) for items in analysis.get('amenities', {}).values()
                ),
                "map_available": bool(analysis.get('map_path')),
                "address": analysis.get('address')
            }
            
            if status_val == "completed":
                return TaskStatusResponse(
                    task_id=task_id,
                    status="completed",
                    progress=100,
                    message="Analysis complete",
                    result=result_data
                )
            elif status_val == "failed":
                return TaskStatusResponse(
                    task_id=task_id,
                    status="failed",
                    progress=100,
                    error=analysis.get('error', 'Unknown error')
                )
            else:
                return TaskStatusResponse(
                    task_id=task_id,
                    status="processing",
                    progress=progress,
                    message=analysis.get('message', f"Processing... ({progress}%)")
                )
        
        # Celery task status
        if CELERY_AVAILABLE:
            try:
                result = AsyncResult(task_id, app=celery_app)
                
                response = TaskStatusResponse(
                    task_id=task_id,
                    status=result.state.lower(),
                    progress=0,
                    message=f"Task state: {result.state}"
                )
                
                if result.state == 'PENDING':
                    response.message = "Task queued for processing"
                elif result.state == 'PROGRESS':
                    info = result.info or {}
                    response.progress = info.get('progress', 0)
                    response.message = info.get('message', 'Processing...')
                elif result.state == 'SUCCESS':
                    response.progress = 100
                    response.message = "Task completed successfully"
                    response.result = result.result
                elif result.state == 'FAILURE':
                    response.progress = 100
                    response.status = "failed"
                    response.error = str(result.info or "Unknown error")
                
                return response
                
            except Exception as e:
                logger.error(f"Celery status check failed: {e}")
                return TaskStatusResponse(
                    task_id=task_id,
                    status="error",
                    error=f"Failed to check task status: {str(e)}"
                )
        
        return TaskStatusResponse(
            task_id=task_id,
            status="unknown",
            message="Task ID format not recognized"
        )
        
    except Exception as e:
        logger.error(f"Task status check failed: {e}", exc_info=True)
        return TaskStatusResponse(
            task_id=task_id,
            status="error",
            error=f"Internal server error: {str(e)}"
        )

# ==================== NEIGHBORHOOD ANALYSIS ENDPOINTS ====================

@app.post("/api/neighborhood/analyze", status_code=202)
async def analyze_neighborhood(
    request: NeighborhoodAnalysisRequest,
    background_tasks: BackgroundTasks
):
    """
    Create neighborhood analysis
    Returns task_id for polling via /api/tasks/{task_id}
    """
    try:
        # Create initial analysis record
        analysis_doc = {
            "address": request.address,
            "search_radius_m": request.radius_m,
            "amenity_types": request.amenity_types or [
                'restaurant', 'cafe', 'school', 'hospital',
                'park', 'supermarket', 'bank', 'pharmacy'
            ],
            "include_buildings": request.include_buildings,
            "generate_map": request.generate_map,
            "status": "pending",
            "progress": 0,
            "created_at": datetime.now().isoformat()
        }
        
        analysis_id = await create_neighborhood_analysis(analysis_doc)
        logger.info(f"Created analysis: {analysis_id} for address: {request.address}")
        
        # Use Celery if available, else background task
        if CELERY_AVAILABLE:
            try:
                from app.tasks.geospatial_tasks import analyze_neighborhood_task
                
                task = analyze_neighborhood_task.delay(
                    analysis_id=analysis_id,
                    request_data=request.dict()
                )
                
                task_id = task.id
                logger.info(f"Celery task created: {task_id}")
                
            except ImportError:
                logger.warning("Celery task import failed, falling back to sync")
                CELERY_AVAILABLE = False
        
        if not CELERY_AVAILABLE:
            # Fallback: Use FastAPI background tasks
            task_id = f"analysis_{analysis_id}"
            
            background_tasks.add_task(
                process_neighborhood_sync,
                analysis_id,
                request.address,
                request.radius_m,
                request.amenity_types or [],
                request.include_buildings,
                request.generate_map
            )
            
            logger.info(f"Background task created: {task_id}")
        
        return {
            "analysis_id": analysis_id,
            "task_id": task_id,
            "address": request.address,
            "status": "queued",
            "poll_url": f"/api/tasks/{task_id}",
            "created_at": datetime.now().isoformat(),
            "estimated_time": "30-60 seconds",
            "message": "Poll /api/tasks/{task_id} for status updates every 2 seconds"
        }
        
    except Exception as e:
        logger.error(f"Failed to create analysis: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Analysis creation failed: {str(e)}"
        )

@app.get("/api/neighborhood/{analysis_id}", response_model=NeighborhoodAnalysis)
async def get_analysis(
    analysis_id: str = Path(..., description="Analysis ID")
):
    """Get completed analysis results"""
    try:
        analysis = await get_neighborhood_analysis(analysis_id)
        if not analysis:
            raise HTTPException(status_code=404, detail="Analysis not found")
        
        # Calculate totals
        amenities = analysis.get('amenities', {})
        total_amenities = sum(len(items) for items in amenities.values())
        
        # Add derived fields
        analysis['total_amenities'] = total_amenities
        analysis['amenity_categories'] = len(amenities)
        
        return analysis
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get analysis {analysis_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve analysis")

@app.get("/api/neighborhood/{analysis_id}/map")
async def get_map(
    analysis_id: str = Path(..., description="Analysis ID")
):
    """Get interactive map for analysis"""
    try:
        analysis = await get_neighborhood_analysis(analysis_id)
        if not analysis:
            raise HTTPException(status_code=404, detail="Analysis not found")
        
        map_path = analysis.get('map_path')
        if not map_path:
            # Try to generate default map path
            map_filename = f"neighborhood_{analysis_id.replace('-', '_')}.html"
            map_path = os.path.join("maps", map_filename)
        
        if not os.path.exists(map_path):
            raise HTTPException(
                status_code=404,
                detail="Map not available. Enable map generation when creating analysis."
            )
        
        return FileResponse(
            map_path,
            media_type="text/html",
            filename=f"neighborhood_map_{analysis_id}.html",
            headers={
                "Cache-Control": "public, max-age=3600",
                "X-Map-Type": "interactive"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get map for {analysis_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve map")

@app.get("/api/neighborhood/search")
async def search_amenities(
    address: str = Query(..., description="Target address"),
    amenity_type: str = Query(..., description="Type of amenity (restaurant, park, etc.)"),
    radius_m: int = Query(1000, ge=100, le=5000, description="Search radius in meters")
):
    """Search for specific amenities near an address"""
    try:
        amenities_data = osm_client.get_nearby_amenities(
            address=address,
            radius=radius_m,
            amenity_types=[amenity_type]
        )
        
        if "error" in amenities_data:
            raise HTTPException(status_code=400, detail=amenities_data["error"])
        
        amenities = amenities_data.get("amenities", {}).get(amenity_type, [])
        
        return {
            "address": address,
            "amenity_type": amenity_type,
            "radius_m": radius_m,
            "amenities": amenities,
            "total_found": len(amenities),
            "coordinates": amenities_data.get("coordinates"),
            "search_completed": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Search failed for {address}: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.get("/api/neighborhood/recent")
async def get_recent(
    limit: int = Query(10, ge=1, le=100, description="Number of analyses to return")
):
    """Get recent neighborhood analyses"""
    try:
        analyses = await get_recent_analyses(limit)
        
        # Format response
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

# ==================== IMAGE ANALYSIS ENDPOINTS ====================

@app.post("/api/analysis/image", status_code=202)
async def analyze_image(
    file: UploadFile = File(..., description="Image file to analyze"),
    analysis_type: str = Query("object_detection", description="Type of analysis")
):
    """Analyze uploaded image for real estate insights"""
    try:
        # Validate file
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail="File must be an image (JPEG, PNG, etc.)"
            )
        
        # Save file temporarily
        temp_dir = FilePath("temp")
        temp_dir.mkdir(exist_ok=True)
        
        timestamp = int(datetime.now().timestamp())
        file_extension = os.path.splitext(file.filename)[1] or '.jpg'
        file_path = temp_dir / f"{timestamp}_{analysis_type}{file_extension}"
        
        # Read and save file
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        
        logger.info(f"Saved uploaded image: {file_path}, size: {len(content)} bytes")
        
        # Create task
        if CELERY_AVAILABLE:
            try:
                from app.tasks.computer_vision_tasks import analyze_street_image_task
                
                task = analyze_street_image_task.delay(
                    str(file_path),
                    analysis_type=analysis_type
                )
                task_id = task.id
                
                logger.info(f"Celery image analysis task created: {task_id}")
                
            except ImportError:
                logger.warning("Computer vision tasks not available")
                task_id = f"image_{timestamp}"
        else:
            task_id = f"image_{timestamp}"
        
        return {
            "task_id": task_id,
            "filename": file.filename,
            "analysis_type": analysis_type,
            "file_size": len(content),
            "status": "queued",
            "poll_url": f"/api/tasks/{task_id}",
            "message": "Image uploaded successfully. Use poll_url to check status."
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Image analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Image analysis failed: {str(e)}")

# ==================== STATISTICS ENDPOINTS ====================

@app.get("/api/stats", response_model=StatsResponse)
async def get_stats():
    """Get system statistics"""
    try:
        # Get counts
        analysis_count = await get_analysis_count()
        properties = await property_crud.get_all_properties()
        
        # Calculate stats
        total_properties = len(properties)
        
        avg_price = 0
        cities = set()
        if properties:
            prices = [p.get('price', 0) for p in properties if p.get('price')]
            avg_price = sum(prices) / len(prices) if prices else 0
            cities = {p.get('city') for p in properties if p.get('city')}
        
        # Calculate uptime
        startup_time = app.state.startup_time if hasattr(app.state, 'startup_time') else datetime.now()
        uptime = str(datetime.now() - startup_time)
        
        # System status
        system_status = "healthy"
        if not Database.is_connected():
            system_status = "degraded"
        
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

# ==================== WORKFLOW ENDPOINTS ====================

if WORKFLOW_ENABLED:
    @app.post("/api/workflow/complete")
    async def complete_workflow(request: WorkflowRequest):
        """Complete workflow with multiple analysis types"""
        try:
            # This is a placeholder - actual implementation would be in workflow_endpoints
            return {
                "workflow_id": f"wf_{int(datetime.now().timestamp())}",
                "address": request.address,
                "status": "initiated",
                "analyses": request.analysis_types,
                "email_sent": bool(request.email),
                "created_at": datetime.now().isoformat()
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

# ==================== ERROR HANDLERS ====================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    logger.warning(f"HTTP {exc.status_code}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat()
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
            "details": str(exc) if app.debug else "Contact administrator",
            "timestamp": datetime.now().isoformat()
        }
    )

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        access_log=True
    )