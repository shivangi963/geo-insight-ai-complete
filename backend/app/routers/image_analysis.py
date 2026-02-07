"""
Image Analysis Router - Fixed Imports
Handles both green space analysis and street scene detection
"""
from fastapi import APIRouter, HTTPException, UploadFile, File, Query, BackgroundTasks
from typing import Dict, Any, Optional
import logging
from datetime import datetime
import tempfile
import os

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/analysis", tags=["image-analysis"])


# ==================== GREEN SPACE ANALYSIS ====================

@router.post("/green-space", status_code=202)
async def analyze_green_space(
    address: str = Query(..., description="Address to analyze"),
    radius_m: int = Query(500, ge=100, le=4000, description="Search radius in meters"),
    background_tasks: BackgroundTasks = None
):
    try:
        from ..crud import create_satellite_analysis
        
        logger.info(f"🌳 Green space analysis request: {address}, radius={radius_m}m")
        
        # Create analysis document
        analysis_doc = {
            "address": address,
            "search_radius_m": radius_m,
            "calculate_green_space": True,
            "map_source": "OpenStreetMap",
            "status": "pending",
            "progress": 0,
            "created_at": datetime.now().isoformat()
        }
        
        analysis_id = await create_satellite_analysis(analysis_doc)
        logger.info(f"✅ Created green space analysis: {analysis_id}")
        
        # Check for Celery
        CELERY_AVAILABLE = False
        try:
            from celery.result import AsyncResult
            from celery_config import celery_app
            CELERY_AVAILABLE = True
        except ImportError:
            pass
        
        use_celery = CELERY_AVAILABLE
        task_id = None
        
        if use_celery:
            try:
                from ..tasks.satellite_tasks import analyze_satellite_task
                task = analyze_satellite_task.delay(
                    analysis_id=analysis_id,
                    request_data={
                        "address": address,
                        "radius_m": radius_m,
                        "calculate_green_space": True
                    }
                )
                task_id = task.id
                logger.info(f"✅ Celery task created: {task_id}")
            except ImportError as e:
                logger.warning(f"Celery task import failed: {e}, using background task")
                use_celery = False
        
        if not use_celery:
            
            task_id = f"green_space_{analysis_id}"
            background_tasks.add_task(
                process_green_space_analysis,
                analysis_id,
                address,
                radius_m
            )
            logger.info(f"✅ Background task scheduled: {task_id}")
        
        return {
            "analysis_id": analysis_id,
            "task_id": task_id,
            "address": address,
            "status": "queued",
            "message": "Green space analysis started",
            "created_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start green space analysis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


async def process_green_space_analysis(
    analysis_id: str,
    address: str,
    radius_m: int
):
    """
    Background task to process green space analysis
    """
    try:
        from ..crud import update_analysis_status
        from ..geospatial import LocationGeocoder
        from ..tasks.computer_vision_tasks import analyze_osm_green_spaces
        import asyncio
        
        # Update status
        await update_analysis_status(analysis_id, "processing", {
            "progress": 10,
            "message": "Geocoding address..."
        })
        
        # Geocode address
        geocoder = LocationGeocoder()
        coordinates = await asyncio.to_thread(
            geocoder.address_to_coordinates, 
            address
        )
        
        if not coordinates:
            await update_analysis_status(analysis_id, "failed", {
                "error": "Could not geocode address",
                "progress": 100
            })
            return
        
        lat, lon = coordinates
        
        await update_analysis_status(analysis_id, "processing", {
            "progress": 30,
            "message": "Fetching OpenStreetMap tiles...",
            "coordinates": {"latitude": lat, "longitude": lon}
        })
        
        # Import OSM functions
        import sys
        import os
        
        # Add backend directory to path if needed
        backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if backend_dir not in sys.path:
            sys.path.insert(0, backend_dir)
        
        # Import from osm_green_space router
        from app.routers.green_space import get_osm_map_area
        
        # Fetch OSM map
        map_image = await asyncio.to_thread(
            get_osm_map_area,
            lat, lon, radius_m
        )
        
        if not map_image:
            await update_analysis_status(analysis_id, "failed", {
                "error": "Failed to fetch OpenStreetMap tiles",
                "progress": 100
            })
            return
        
        # Save temporary image
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
            map_image.save(temp_file, format='PNG')
            temp_path = temp_file.name
        
        try:
            await update_analysis_status(analysis_id, "processing", {
                "progress": 60,
                "message": "Analyzing green spaces..."
            })
            
            # Analyze green spaces
            result = await asyncio.to_thread(
                analyze_osm_green_spaces,
                temp_path,
                analysis_id
            )
            
            if result:
                green_space_pct = result.get('green_space_percentage', 0)
                green_pixels = result.get('green_pixels', 0)
                total_pixels = result.get('total_pixels', 0)
                visualization_path = result.get('visualization_path')
                breakdown = result.get('breakdown', {})
                
                # Complete analysis
                result_data = {
                    "address": address,
                    "coordinates": {"latitude": lat, "longitude": lon},
                    "search_radius_m": radius_m,
                    "green_space_percentage": green_space_pct,
                    "green_pixels": green_pixels,
                    "total_pixels": total_pixels,
                    "visualization_path": visualization_path,
                    "breakdown": breakdown,
                    "map_source": "OpenStreetMap",
                    "status": "completed",
                    "progress": 100,
                    "completed_at": datetime.now().isoformat()
                }
                
                await update_analysis_status(analysis_id, "completed", result_data)
                
                logger.info(f"✅ Green space analysis {analysis_id} completed")
                logger.info(f"   Green Space: {green_space_pct:.1f}%")
            else:
                await update_analysis_status(analysis_id, "failed", {
                    "error": "Green space calculation failed",
                    "progress": 100
                })
        
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_path)
            except:
                pass
    
    except Exception as e:
        logger.error(f"Green space analysis failed: {e}", exc_info=True)
        await update_analysis_status(analysis_id, "failed", {
            "error": str(e),
            "progress": 100
        })


@router.get("/green-space/{analysis_id}")
async def get_green_space_analysis(analysis_id: str):
    """
    Get green space analysis results by ID
    
    Returns:
        {
            "analysis_id": str,
            "status": str,
            "address": str,
            "green_space_percentage": float,
            "coordinates": dict,
            "visualization_path": str,
            "breakdown": dict
        }
    """
    try:
        from ..crud import get_satellite_analysis
        
        analysis = await get_satellite_analysis(analysis_id)
        
        if not analysis:
            raise HTTPException(status_code=404, detail="Analysis not found")
        
        return analysis
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get analysis {analysis_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve analysis")


@router.get("/green-space/recent")
async def get_recent_green_space_analyses(limit: int = Query(10, ge=1, le=50)):
    """Get recent green space analyses"""
    try:
        from ..crud import get_recent_satellite_analyses
        
        analyses = await get_recent_satellite_analyses(limit)
        
        return {"analyses": analyses}
        
    except Exception as e:
        logger.error(f"Failed to get recent analyses: {e}")
        return {"analyses": []}


# ==================== STREET SCENE DETECTION ====================

@router.post("/street-scene", status_code=200)
async def analyze_street_scene(
    file: UploadFile = File(...),
):
    """
    Detect objects in street scene images using YOLO
    
    Upload a street scene photo to detect:
    - Vehicles (cars, trucks, buses)
    - Pedestrians
    - Other urban objects
    
    Returns:
        Detection results with object counts and bounding boxes
    """
    try:
        logger.info(f"🚗 Street scene analysis request")
        
        # Validate file
        if not file or not hasattr(file, 'content_type') or file.content_type is None:
            raise HTTPException(
                status_code=400,
                detail="Invalid file upload"
            )
        
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type: {file.content_type}. Please upload an image."
            )
        
        # Read file content
        image_data = await file.read()
        
        # Validate size
        max_size = 10 * 1024 * 1024  # 10MB
        if len(image_data) > max_size:
            raise HTTPException(
                status_code=400,
                detail="Image file too large. Maximum size is 10MB."
            )
        
        logger.info(f"📊 Image size: {len(image_data)} bytes")
        
        # Convert to numpy array
        import numpy as np
        from PIL import Image
        import io
        
        image = Image.open(io.BytesIO(image_data))
        image_np = np.array(image)
        
        logger.info(f"🖼️ Image dimensions: {image_np.shape}")
        
        # Perform detection
        result = await perform_street_detection(image_np, file.filename)
        
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Street scene analysis failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )


async def perform_street_detection(image_np, filename: str) -> Dict:
    """
    Perform YOLO object detection on street scene
    """
    try:
        from ultralytics import YOLO
        import asyncio
        
        def detect():
            # Load YOLO model
            model = YOLO('yolov8n.pt')
            
            # Run detection
            results = model(image_np)
            
            # Extract detections
            detections = []
            class_counts = {}
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    detection = {
                        "class": result.names[int(box.cls[0])],
                        "confidence": float(box.conf[0]),
                        "bbox": box.xyxy[0].tolist()
                    }
                    detections.append(detection)
                    
                    # Count classes
                    class_name = detection["class"]
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            return detections, class_counts
        
        # Run detection in thread
        detections, class_counts = await asyncio.to_thread(detect)
        
        logger.info(f"✅ Detected {len(detections)} objects")
        
        return {
            "status": "success",
            "analysis_type": "street_scene",
            "detections": detections,
            "class_counts": class_counts,
            "total_objects": len(detections),
            "image_size": {
                "width": image_np.shape[1],
                "height": image_np.shape[0]
            },
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Street detection failed: {e}")
        raise

