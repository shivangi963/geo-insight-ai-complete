"""
Neighborhood Analysis Router
Extracted from main.py for better organization
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from typing import List, Optional
import logging
from datetime import datetime
import asyncio
import os

from ..models import NeighborhoodAnalysisRequest, NeighborhoodAnalysisResponse, NeighborhoodAnalysis
from ..crud import (
    create_neighborhood_analysis,
    get_neighborhood_analysis,
    get_recent_analyses,
    update_analysis_status
)
from ..geospatial import OpenStreetMapClient, calculate_walk_score

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/neighborhood", tags=["neighborhood"])

# Initialize OSM client
osm_client = OpenStreetMapClient()

# Constants
PROGRESS_START = 10
PROGRESS_AMENITIES = 40
PROGRESS_WALK_SCORE = 70
PROGRESS_MAP = 85
PROGRESS_COMPLETE = 100
AMENITY_TYPES = ['restaurant', 'cafe', 'school', 'hospital', 'park', 'supermarket']


async def update_analysis_progress(analysis_id: str, progress: int, message: str = "", data: dict = None):
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
        amenities_data = await asyncio.to_thread(
            osm_client.get_nearby_amenities,
            address=address,
            radius=radius_m,
            amenity_types=amenity_types
        )
        
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
            "Generating map..." if generate_map else "Finalizing...",
            {"walk_score": walk_score}
        )
        
        # Map generation
        map_path = None
        if generate_map and coordinates:
            await update_analysis_progress(analysis_id, PROGRESS_MAP, "Generating map...")
            try:
                map_filename = f"neighborhood_{analysis_id.replace('-', '_')}.html"
                map_path = os.path.join("maps", map_filename)
                
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
            "total_amenities": total_amenities,
            "coordinates": coordinates,
            "progress": PROGRESS_COMPLETE,
            "completed_at": datetime.now().isoformat()
        }
        
        await update_analysis_status(analysis_id, "completed", result_data)
        logger.info(f"Analysis {analysis_id} completed")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        await update_analysis_status(analysis_id, "failed", {
            "error": str(e),
            "progress": 100
        })


@router.post("/analyze", status_code=202, response_model=NeighborhoodAnalysisResponse)
async def analyze_neighborhood(
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
        
        # Check for Celery
        CELERY_AVAILABLE = False
        try:
            from celery.result import AsyncResult
            from celery_config import celery_app
            CELERY_AVAILABLE = True
        except ImportError:
            pass
        
        use_celery = CELERY_AVAILABLE
        
        if use_celery:
            try:
                from ..tasks.geospatial_tasks import analyze_neighborhood_task
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
            logger.info(f"Background task scheduled: {task_id}")
        
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


@router.get("/{analysis_id}", response_model=NeighborhoodAnalysis)
async def get_analysis(analysis_id: str):
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


@router.get("/recent")
async def get_recent(limit: int = Query(10)):
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