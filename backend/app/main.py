from fastapi import FastAPI, HTTPException, BackgroundTasks, Query, Path, status
from typing import List, Optional, Dict, Any
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from fastapi.responses import FileResponse
from pydantic import BaseModel
import sys
import os
import asyncio
from celery import current_app as celery_app
from celery.result import AsyncResult
from app.tasks.computer_vision_tasks import analyze_street_image_task, calculate_green_space_task
from app.tasks.geospatial_tasks import analyze_neighborhood_task
from app.workflow_endpoints import router as workflow_router


current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

print(f" Starting GeoInsight AI API - Phase 4")
print(f" Current directory: {os.getcwd()}")


try:
    from .models import (
        PropertyCreate, PropertyUpdate, PropertyResponse, HealthResponse,
        NeighborhoodAnalysisRequest, NeighborhoodAnalysisResponse,
        NeighborhoodAnalysis, Coordinates, Amenity, BuildingFootprint
    )
    print("Models imported successfully")
except ImportError as e:
    print(f" Error importing models: {e}")
    raise

try:
    from .crud import (
        property_crud,
        create_neighborhood_analysis,
        get_neighborhood_analysis,
        get_recent_analyses,
        update_analysis_status,
        delete_neighborhood_analysis,
        get_analysis_count
    )
    print(" CRUD operations imported successfully")
except ImportError as e:
    print(f"Error importing CRUD: {e}")
    raise

try:
    from .geospatial import OpenStreetMapClient, calculate_walk_score
    print("Geospatial module imported successfully")
    osm_client = OpenStreetMapClient()
except ImportError as e:
    print(f"Geospatial module import failed: {e}")
    print("Creating mock geospatial client...")
    
    class MockOpenStreetMapClient:
        def get_nearby_amenities(self, address, radius=1000, amenity_types=None):
            return {
                "address": address,
                "coordinates": (12.9716, 77.5946),
                "amenities": {}
            }
        
        def get_building_footprints(self, address, radius=500):
            return {"buildings": []}
        
        def create_map_visualization(self, address, amenities_data, save_path):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w') as f:
                f.write(f"<html><body><h1>Map for {address}</h1></body></html>")
            return save_path
    
    osm_client = MockOpenStreetMapClient()
    
    def calculate_walk_score(coordinates, amenities_data):
        return 75.0

try:
    from .agents.local_expert import agent
    print(" AI Agent imported successfully")
except ImportError as e:
    print(f" AI Agent import failed: {e}")
    
    class LocalExpertAgent:
        def process_query(self, query: str):
            return {"query": query, "answer": f"Demo: {query}", "success": True}
    
    agent = LocalExpertAgent()

try:
    from .database import Database
    print(" Database module imported successfully")
except ImportError as e:
    print(f"Error importing database: {e}")
    raise

class QueryRequest(BaseModel):
    query: str

class TaskStatusResponse(BaseModel):

    task_id: str
    status: str
    result: Optional[Dict] = None
    error: Optional[str] = None

class WorkflowRequest(BaseModel):
    address: str
    radius_m: Optional[int] = 1000
    email: Optional[str] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(" Starting GeoInsight AI API (Phase 4)...")
    
    try:
        await Database.connect()
        print("Database connected successfully")
    except Exception as e:
        print(f"Database connection note: {e}")
    
    os.makedirs("maps", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("temp", exist_ok=True)
    print(" Created necessary directories")
    
    yield
    
    print(" Shutting down GeoInsight AI API...")
    
    try:
        await Database.close()
        print(" Database connection closed")
    except Exception as e:
        print(f" Database close note: {e}")



app = FastAPI(
    title="GeoInsight AI API",
    description="Phase 4: Geospatial Analysis & Async Processing",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(workflow_router)


async def process_map_generation(
    analysis_id: str,
    address: str,
    amenities_data: Dict
):

    try:
        map_filename = f"neighborhood_{analysis_id.replace('-', '_')}.html"
        map_path = os.path.join("maps", map_filename)
        
        map_path = osm_client.create_map_visualization(
            address=address,
            amenities_data=amenities_data,
            save_path=map_path
        )
        
        if map_path:
            await update_analysis_status(
                analysis_id,
                "completed",
                {"map_path": map_path}
            )
            print(f" Map generated for analysis {analysis_id}")
        else:
            await update_analysis_status(analysis_id, "completed")
            print(f"  Map generation skipped")
            
    except Exception as e:
        print(f" Error in map generation: {e}")
        await update_analysis_status(analysis_id, "failed", {"error": str(e)})


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="ok",
        timestamp=datetime.now().isoformat(),
        version="2.0.0"
    )

@app.get("/")
async def root():

    return {
        "message": "Welcome to GeoInsight AI API - Phase 4",
        "phase": 4,
        "status": "running",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "properties": "/api/properties",
            "neighborhood": "/api/neighborhood/analyze",
            "agent": "/api/agent/query"
        }
    }

@app.get("/api/properties", response_model=List[PropertyResponse])
async def get_properties(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000)
):
    properties = property_crud.get_all_properties(skip=skip, limit=limit)
    return properties

@app.get("/api/properties/{property_id}", response_model=PropertyResponse)
async def get_property(property_id: str = Path(...)):
  
    property = property_crud.get_property_by_id(property_id)
    if not property:
        raise HTTPException(status_code=404, detail="Property not found")
    return property

@app.post("/api/properties", response_model=PropertyResponse, status_code=201)
async def create_property(property: PropertyCreate):
   
    try:
        new_property = property_crud.create_property(property)
        return new_property
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.put("/api/properties/{property_id}", response_model=PropertyResponse)
async def update_property(
    property_id: str = Path(...),
    property: PropertyUpdate = ...
):
    
    updated_property = property_crud.update_property(property_id, property)
    if not updated_property:
        raise HTTPException(status_code=404, detail="Property not found")
    return updated_property

@app.delete("/api/properties/{property_id}")
async def delete_property(property_id: str = Path(...)):
    success = property_crud.delete_property(property_id)
    if not success:
        raise HTTPException(status_code=404, detail="Property not found")
    return {"message": "Property deleted successfully"}

class QueryRequest(BaseModel):
    query: str

@app.post("/api/agent/query")
async def query_agent(request: QueryRequest):

    try:
        result = agent.process_query(request.query)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/neighborhood/analyze", response_model=NeighborhoodAnalysisResponse, status_code=202)
async def analyze_neighborhood(
    request: NeighborhoodAnalysisRequest,
    background_tasks: BackgroundTasks
):
 
    if osm_client is None:
        raise HTTPException(status_code=503, detail="Geospatial services unavailable")
    
    try:

        amenities_data = osm_client.get_nearby_amenities(
            address=request.address,
            radius=request.radius_m,
            amenity_types=request.amenity_types
        )
        
        if "error" in amenities_data:
            raise HTTPException(status_code=400, detail=amenities_data["error"])
        
        coordinates = amenities_data.get("coordinates")
        walk_score = None
        if coordinates:
            walk_score = calculate_walk_score(coordinates, amenities_data)
        
        building_footprints = []
        if request.include_buildings:
            buildings_data = osm_client.get_building_footprints(
                address=request.address,
                radius=min(request.radius_m, 500)
            )
            if "error" not in buildings_data:
                building_footprints = buildings_data.get("buildings", [])
        
        analysis_doc = {
            "address": request.address,
            "coordinates": {
                "latitude": coordinates[0] if coordinates else 0,
                "longitude": coordinates[1] if coordinates else 0
            },
            "search_radius_m": request.radius_m,
            "amenities": amenities_data.get("amenities", {}),
            "building_footprints": building_footprints,
            "walk_score": walk_score,
            "status": "processing"
        }
        
        analysis_id = await create_neighborhood_analysis(analysis_doc)
        
        map_url = None
        if request.generate_map:
            background_tasks.add_task(
                process_map_generation,
                analysis_id,
                request.address,
                amenities_data
            )
            map_url = f"/api/neighborhood/map/{analysis_id}"
        else:
            await update_analysis_status(analysis_id, "completed")

        total_amenities = sum(
            len(items) for items in amenities_data.get("amenities", {}).values()
        )
        
        return NeighborhoodAnalysisResponse(
            analysis_id=analysis_id,
            address=request.address,
            status="processing" if request.generate_map else "completed",
            walk_score=walk_score,
            total_amenities=total_amenities,
            map_url=map_url,
            created_at=datetime.now()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.get(
    "/api/neighborhood/recent",
    response_model=List[NeighborhoodAnalysisResponse],
    summary="Get recent analyses",
    description="Get list of recent neighborhood analyses"
)
async def get_recent_neighborhood_analyses(
    limit: int = Query(10, ge=1, le=50, description="Number of analyses to return")
):

    analyses = await get_recent_analyses(limit)
    
    response_list = []
    for analysis in analyses:
        total_amenities = sum(
            len(items) for items in analysis.get("amenities", {}).values()
        )
        
        response_list.append(NeighborhoodAnalysisResponse(
            analysis_id=str(analysis["_id"]),
            address=analysis.get("address", "Unknown"),
            status=analysis.get("status", "unknown"),
            walk_score=analysis.get("walk_score"),
            total_amenities=total_amenities,
            map_url=f"/api/neighborhood/map/{analysis['_id']}" if analysis.get("map_path") else None,
            created_at=analysis.get("created_at")
        ))
    
    return response_list


@app.get(
    "/api/neighborhood/search",
    summary="Search for specific amenities",
    description="Search for specific types of amenities near an address"
)
async def search_amenities(
    address: str = Query(..., description="Target address"),
    amenity_type: str = Query(..., description="Type of amenity (restaurant, park, etc.)"),
    radius_m: int = Query(1000, ge=100, le=5000, description="Search radius in meters")
):

    try:
        amenities_data = osm_client.get_nearby_amenities(
            address=address,
            radius=radius_m,
            amenity_types=[amenity_type]
        )
        
        if "error" in amenities_data:
            raise HTTPException(status_code=400, detail=amenities_data["error"])
        
        return {
            "address": address,
            "amenity_type": amenity_type,
            "radius_m": radius_m,
            "amenities": amenities_data.get("amenities", {}).get(amenity_type, []),
            "total_found": len(amenities_data.get("amenities", {}).get(amenity_type, []))
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
    
@app.get("/api/neighborhood/{analysis_id}", response_model=NeighborhoodAnalysis)
async def get_analysis_results(analysis_id: str = Path(...)):
    analysis = await get_neighborhood_analysis(analysis_id)
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return analysis

@app.get("/api/neighborhood/map/{analysis_id}")
async def get_analysis_map(analysis_id: str = Path(...)):

    analysis = await get_neighborhood_analysis(analysis_id)
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    map_path = analysis.get("map_path")
    if not map_path or not os.path.exists(map_path):
        map_filename = f"neighborhood_{analysis_id.replace('-', '_')}.html"
        map_path = os.path.join("maps", map_filename)
        
        if not os.path.exists(map_path):
            raise HTTPException(status_code=404, detail="Map not available")
    
    return FileResponse(
        map_path,
        media_type="text/html",
        filename=f"neighborhood_map_{analysis_id}.html"
    )


@app.get("/api/stats")
async def get_statistics():
    try:
        analysis_count = await get_analysis_count()
        
        return {
            "analyses": analysis_count,
            "properties": len(property_crud.get_all_properties()),
            "phase": 4,
            "status": "running"
        }
    except Exception as e:
        return {"error": str(e), "phase": 4}

@app.post("/api/tasks/analyze-street")
async def create_street_analysis_task(image_url: str):

    try:
   
        import requests
        response = requests.get(image_url)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to download image")
      
        import uuid
        filename = f"temp_{uuid.uuid4().hex}.jpg"
        with open(filename, "wb") as f:
            f.write(response.content)

        task = analyze_street_image_task.delay(filename)
        
        return {
            "task_id": task.id,
            "status": "PENDING",
            "message": "Street analysis task created"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/tasks/{task_id}")
async def get_task_status(task_id: str):

    task_result = AsyncResult(task_id, app=celery_app)
    
    response = {
        "task_id": task_id,
        "status": task_result.status,
        "result": None
    }
    
    if task_result.ready():
        response["result"] = task_result.result
        if task_result.failed():
            response["error"] = str(task_result.result)
    
    return response

@app.post("/api/tasks/neighborhood/async")
async def create_async_neighborhood_analysis(request: NeighborhoodAnalysisRequest):
   
    try:
        analysis_doc = {
            "address": request.address,
            "search_radius_m": request.radius_m,
            "status": "queued",
            "created_at": datetime.now()
        }
        
        analysis_id = await create_neighborhood_analysis(analysis_doc)
        

        task = analyze_neighborhood_task.delay(
            analysis_id,
            request.dict()
        )
        
        return {
            "task_id": task.id,
            "analysis_id": analysis_id,
            "status": "queued",
            "message": "Neighborhood analysis task created"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    


if __name__ == "__main__":
    import uvicorn
    
    print("=" * 50)
    print("GeoInsight AI API - Phase 4")
    print("=" * 50)
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )