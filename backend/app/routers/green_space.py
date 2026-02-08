"""
REPLACE your backend/app/routers/green_space.py with this
Router now imports from geospatial.py instead of defining functions
"""
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
import io
import os
import logging

from app.geospatial import get_osm_map_area, download_osm_tile

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/osm", tags=["osm-green-space"])


@router.get("/tile/{zoom}/{tile_x}/{tile_y}")
async def get_tile(zoom: int, tile_x: int, tile_y: int):
    """
    Get single OSM tile
    
    Args:
        zoom: Zoom level (1-19)
        tile_x: X tile coordinate
        tile_y: Y tile coordinate
    
    Returns:
        StreamingResponse: PNG image
    """
    try:
        tile_img = download_osm_tile(tile_x, tile_y, zoom)
        
        if not tile_img:
            raise HTTPException(status_code=404, detail="Tile not found")
        
        # Convert to bytes
        img_byte_arr = io.BytesIO()
        tile_img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        return StreamingResponse(img_byte_arr, media_type="image/png")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching tile: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/map")
async def get_map_image(
    latitude: float = Query(..., ge=-90, le=90, description="Latitude"),
    longitude: float = Query(..., ge=-180, le=180, description="Longitude"),
    radius_m: int = Query(500, ge=100, le=5000, description="Radius in meters")
):
    """
    Get map image for coordinates
    
    Args:
        latitude: Center latitude
        longitude: Center longitude
        radius_m: Search radius in meters
    
    Returns:
        StreamingResponse: PNG image
    """
    try:
        # Get map from geospatial utility
        map_path = get_osm_map_area(latitude, longitude, radius_m)
        
        if not map_path:
            raise HTTPException(status_code=500, detail="Failed to generate map")
        
        # Read from temp file
        with open(map_path, 'rb') as f:
            img_data = f.read()
        
        # Clean up temp file
        try:
            os.unlink(map_path)
        except:
            pass
        
        return StreamingResponse(io.BytesIO(img_data), media_type="image/png")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating map: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/info")
async def get_osm_info():
    """
    Get information about OSM tile service
    
    Returns:
        dict: Service information
    """
    return {
        "service": "OpenStreetMap Tile Service",
        "tile_size": 256,
        "max_zoom": 19,
        "attribution": "© OpenStreetMap contributors",
        "license": "ODbL",
        "usage_policy": "https://operations.osmfoundation.org/policies/tiles/",
        "note": "Please respect OSM tile usage policy - avoid bulk downloads"
    }