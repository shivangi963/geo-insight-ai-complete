from fastapi import APIRouter, HTTPException, Query
from typing import Optional, Tuple
import requests
from PIL import Image
import io
import math
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/osm", tags=["osm-green-space"])


def lat_lon_to_tile(latitude: float, longitude: float, zoom: int) -> Tuple[int, int]:
    
    lat_rad = math.radians(latitude)
    n = 2.0 ** zoom
    
    tile_x = int((longitude + 180.0) / 360.0 * n)
    tile_y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    
    return tile_x, tile_y


def download_osm_tile(tile_x: int, tile_y: int, zoom: int) -> Optional[Image.Image]:

    try:
        url = f"https://tile.openstreetmap.org/{zoom}/{tile_x}/{tile_y}.png"
        headers = {'User-Agent': 'GeoInsightAI/1.0 (Educational Project)'}
        
        # CRITICAL: 5 second timeout per tile
        response = requests.get(url, headers=headers, timeout=5)
        
        if response.status_code == 200:
            tile_image = Image.open(io.BytesIO(response.content))
            return tile_image
        else:
            logger.warning(f"OSM tile ({tile_x},{tile_y}) returned {response.status_code}")
            return None
    except requests.exceptions.Timeout:
        logger.warning(f"OSM tile ({tile_x},{tile_y}) timeout after 5s")
        return None
    except Exception as e:
        logger.warning(f"OSM tile ({tile_x},{tile_y}) error: {e}")
        return None

def stitch_tiles(tiles: dict, min_x: int, min_y: int, max_x: int, max_y: int) -> Image.Image:
    
    tile_size = 256
    
    width = (max_x - min_x + 1) * tile_size
    height = (max_y - min_y + 1) * tile_size
    
    stitched = Image.new('RGB', (width, height), color=(240, 240, 240))
    
    for (tile_x, tile_y), tile_img in tiles.items():
        x_offset = (tile_x - min_x) * tile_size
        y_offset = (tile_y - min_y) * tile_size
        
        stitched.paste(tile_img, (x_offset, y_offset))
    
    return stitched



@router.get("/tile/{zoom}/{tile_x}/{tile_y}")
async def get_tile(
    zoom: int,
    tile_x: int,
    tile_y: int
):
    
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
    
    try:
        map_image = get_osm_map_area(latitude, longitude, radius_m)
        
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