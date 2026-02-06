"""
OpenStreetMap Green Space Router
Provides endpoints and utilities for fetching OSM map tiles
"""
from fastapi import APIRouter, HTTPException, Query
from typing import Optional, Tuple
import requests
from PIL import Image
import io
import math
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/osm", tags=["osm-green-space"])


def get_osm_map_area(
    latitude: float,
    longitude: float,
    radius_meters: int = 500,
    zoom: int = 17
) -> Optional[Image.Image]:
    """
    Fetch OpenStreetMap tiles for a given area and return as PIL Image
    
    Args:
        latitude: Center latitude
        longitude: Center longitude
        radius_meters: Radius around center point in meters
        zoom: OSM zoom level (higher = more detail, max 19)
    
    Returns:
        PIL Image of the map area, or None if failed
    """
    try:
        # Calculate tile coordinates for center point
        center_tile_x, center_tile_y = lat_lon_to_tile(latitude, longitude, zoom)
        
        # Calculate how many tiles we need based on radius
        # At zoom 17, each tile is ~150m wide at equator
        # So for 500m radius, we need ~7 tiles (3.33 tiles each direction)
        meters_per_tile = 156543.03 * math.cos(math.radians(latitude)) / (2 ** zoom)
        tiles_needed = math.ceil(radius_meters / meters_per_tile)
        
        # Ensure we get at least a 3x3 grid
        tiles_needed = max(tiles_needed, 1)
        
        # Calculate tile range
        min_tile_x = center_tile_x - tiles_needed
        max_tile_x = center_tile_x + tiles_needed
        min_tile_y = center_tile_y - tiles_needed
        max_tile_y = center_tile_y + tiles_needed
        
        logger.info(f"Fetching OSM tiles: center=({center_tile_x}, {center_tile_y}), "
                   f"range: x={min_tile_x}-{max_tile_x}, y={min_tile_y}-{max_tile_y}, zoom={zoom}")
        
        # Download tiles
        tiles = {}
        for tile_x in range(min_tile_x, max_tile_x + 1):
            for tile_y in range(min_tile_y, max_tile_y + 1):
                tile_img = download_osm_tile(tile_x, tile_y, zoom)
                if tile_img:
                    tiles[(tile_x, tile_y)] = tile_img
        
        if not tiles:
            logger.error("Failed to download any OSM tiles")
            return None
        
        # Stitch tiles together
        stitched_image = stitch_tiles(tiles, min_tile_x, min_tile_y, max_tile_x, max_tile_y)
        
        logger.info(f"Created stitched map image: {stitched_image.size}")
        return stitched_image
        
    except Exception as e:
        logger.error(f"Error fetching OSM map area: {e}", exc_info=True)
        return None


def lat_lon_to_tile(latitude: float, longitude: float, zoom: int) -> Tuple[int, int]:
    """
    Convert latitude/longitude to OSM tile coordinates
    
    Args:
        latitude: Latitude in degrees
        longitude: Longitude in degrees
        zoom: Zoom level
    
    Returns:
        Tuple of (tile_x, tile_y)
    """
    lat_rad = math.radians(latitude)
    n = 2.0 ** zoom
    
    tile_x = int((longitude + 180.0) / 360.0 * n)
    tile_y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    
    return tile_x, tile_y


def download_osm_tile(tile_x: int, tile_y: int, zoom: int, 
                      tile_server: str = "https://tile.openstreetmap.org") -> Optional[Image.Image]:
    """
    Download a single OSM tile
    
    Args:
        tile_x: Tile X coordinate
        tile_y: Tile Y coordinate
        zoom: Zoom level
        tile_server: OSM tile server URL
    
    Returns:
        PIL Image of the tile, or None if failed
    """
    try:
        # OSM tile URL format: https://tile.openstreetmap.org/{z}/{x}/{y}.png
        url = f"{tile_server}/{zoom}/{tile_x}/{tile_y}.png"
        
        # Add user agent (required by OSM)
        headers = {
            'User-Agent': 'GeoInsightAI/1.0 (Educational Project)'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            tile_image = Image.open(io.BytesIO(response.content))
            return tile_image
        else:
            logger.warning(f"Failed to download tile ({tile_x}, {tile_y}, {zoom}): HTTP {response.status_code}")
            return None
            
    except Exception as e:
        logger.warning(f"Error downloading tile ({tile_x}, {tile_y}, {zoom}): {e}")
        return None


def stitch_tiles(tiles: dict, min_x: int, min_y: int, max_x: int, max_y: int) -> Image.Image:
    """
    Stitch downloaded tiles into a single image
    
    Args:
        tiles: Dictionary mapping (x, y) -> PIL Image
        min_x: Minimum tile X
        min_y: Minimum tile Y
        max_x: Maximum tile X
        max_y: Maximum tile Y
    
    Returns:
        Stitched PIL Image
    """
    # OSM tiles are 256x256 pixels
    tile_size = 256
    
    # Calculate output image size
    width = (max_x - min_x + 1) * tile_size
    height = (max_y - min_y + 1) * tile_size
    
    # Create blank canvas
    stitched = Image.new('RGB', (width, height), color=(240, 240, 240))
    
    # Paste each tile
    for (tile_x, tile_y), tile_img in tiles.items():
        # Calculate position in stitched image
        x_offset = (tile_x - min_x) * tile_size
        y_offset = (tile_y - min_y) * tile_size
        
        stitched.paste(tile_img, (x_offset, y_offset))
    
    return stitched


# ==================== API ENDPOINTS ====================

@router.get("/tile/{zoom}/{tile_x}/{tile_y}")
async def get_tile(
    zoom: int,
    tile_x: int,
    tile_y: int
):
    """
    Proxy endpoint to fetch a single OSM tile
    
    Useful for debugging or direct tile access
    """
    try:
        tile_img = download_osm_tile(tile_x, tile_y, zoom)
        
        if not tile_img:
            raise HTTPException(status_code=404, detail="Tile not found")
        
        # Convert to bytes
        img_byte_arr = io.BytesIO()
        tile_img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        from fastapi.responses import StreamingResponse
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
    radius_m: int = Query(500, ge=100, le=5000, description="Radius in meters"),
    zoom: int = Query(17, ge=1, le=19, description="Zoom level")
):
    """
    Fetch a map image for a given location
    
    Returns a stitched OSM map as PNG
    """
    try:
        map_image = get_osm_map_area(latitude, longitude, radius_m, zoom)
        
        if not map_image:
            raise HTTPException(status_code=500, detail="Failed to generate map")
        
        # Convert to bytes
        img_byte_arr = io.BytesIO()
        map_image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        from fastapi.responses import StreamingResponse
        return StreamingResponse(img_byte_arr, media_type="image/png")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating map: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/info")
async def get_osm_info():
    """
    Get information about OSM tile service
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