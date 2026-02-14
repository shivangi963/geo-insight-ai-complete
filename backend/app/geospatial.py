import osmnx as ox
import networkx as nx
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from typing import Dict, List, Optional, Tuple
import folium
from shapely.geometry import Point, Polygon
import json
from datetime import datetime
import time
from functools import wraps
import math
import requests
from PIL import Image
import io
import tempfile
import os

# ✅ FIXED: Configure OSM with timeouts
ox.settings.log_console = True
ox.settings.use_cache = True
ox.settings.cache_folder = "./.osmnx_cache"
ox.settings.timeout = 30 
KNOWN_COORDINATES = {
        'indiranagar, bengaluru, karnataka, india': (12.9716, 77.6412),
        'indiranagar, bangalore, karnataka, india': (12.9716, 77.6412),
        'koramangala, bengaluru, karnataka, india': (12.9352, 77.6245),
        'koramangala, bangalore, karnataka, india': (12.9352, 77.6245),
        'whitefield, bengaluru, karnataka, india': (12.9698, 77.7499),
        'hsr layout, bengaluru, karnataka, india': (12.9116, 77.6389),
        'jayanagar, bengaluru, karnataka, india': (12.9308, 77.5838),
        'jp nagar, bengaluru, karnataka, india': (12.9063, 77.5857),
        'electronic city, bengaluru, karnataka, india': (12.8399, 77.6770),
        'marathahalli, bengaluru, karnataka, india': (12.9591, 77.6974),
        'btm layout, bengaluru, karnataka, india': (12.9166, 77.6101),
        'malleshwaram, bengaluru, karnataka, india': (13.0035, 77.5710),
    }


def timeout_decorator(timeout_seconds: int = 30):
    """
    Decorator to add timeout to functions
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError(f"Function {func.__name__} timed out after {timeout_seconds} seconds")
            
            # Set timeout (Unix-like systems only)
            try:
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(timeout_seconds)
                result = func(*args, **kwargs)
                signal.alarm(0)  # Cancel alarm
                return result
            except AttributeError:
                # Windows doesn't support SIGALRM, just run normally
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


class LocationGeocoder:
    
    
    def __init__(self, user_agent: str = "geo_insight_ai"):
        self.geolocator = Nominatim(
            user_agent=user_agent,
            timeout=10  # ✅ Add timeout
        )
        self.last_request_time = 0
        self.min_request_interval = 1.0  # ✅ Rate limit: 1 request per second
    
    def _rate_limit(self):
        """Enforce rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        
        self.last_request_time = time.time()
    
    def address_to_coordinates(self, address: str) -> Optional[Tuple[float, float]]:
        if not address or not isinstance(address, str):
            print(f" Invalid address provided: {address!r}")
            return None
        address_lower = address.lower().strip()
        if address_lower in KNOWN_COORDINATES:
            print(f"✅ Using cached coordinates for: {address}")
            return KNOWN_COORDINATES[address_lower] 
        try:
            self._rate_limit() 
            
            location = self.geolocator.geocode(address)
            if location:
                return (location.latitude, location.longitude)
            return None
    
        except Exception as e:
            print(f"❌ Geocoding error for '{address}': {e}")
            return None
        
    
    def coordinates_to_address(self, lat: float, lon: float) -> Optional[str]:
        """
        Convert coordinates to address with rate limiting
        """
        try:
            self._rate_limit()  # ✅ Enforce rate limit
            
            location = self.geolocator.reverse(f"{lat}, {lon}")
            return location.address if location else None
        
        except Exception as e:
            print(f"❌ Reverse geocoding error: {e}")
            return None
        
_geocoder_instance = None

def get_geocoder() -> LocationGeocoder:
    global _geocoder_instance
    if _geocoder_instance is None:
        _geocoder_instance = LocationGeocoder()
    return _geocoder_instance


class OpenStreetMapClient:
    def __init__(self):
        ox.settings.log_console = True
        ox.settings.use_cache = True
        ox.settings.cache_folder = "./.osmnx_cache"
        ox.settings.timeout = 30  # ✅ 30 second timeout
        ox.settings.max_query_area_size = 50000000  # ✅ Limit query size
    
    def get_nearby_amenities(
    self,
    address: str,
    radius: float = 1000,
    amenity_types: Optional[List[str]] = None,
    max_results_per_type: int = None
    ) -> Dict:
    
        if amenity_types is None:
            amenity_types = [
                'restaurant', 'cafe', 'school', 'hospital',
                'park', 'supermarket', 'bank', 'pharmacy'
            ]
        
        if max_results_per_type is None:
            if radius >= 5000:
                max_results_per_type = 50  
            elif radius >= 2000:
                max_results_per_type = 30  
            else:
                max_results_per_type = 20  

        max_amenity_types = 6
        if radius >= 5000:
            max_amenity_types = 3  
        elif radius >= 2000:
            max_amenity_types = 4  
        
        if len(amenity_types) > max_amenity_types:
            print(f"⏱️ Limiting {len(amenity_types)} amenity types to {max_amenity_types} for timeout prevention (radius: {radius}m)")
            amenity_types = amenity_types[:max_amenity_types]
        
        try:
            geocoder = get_geocoder()
            coordinates = geocoder.address_to_coordinates(address)
            
            if not coordinates:
                return {
                    "error": "Could not geocode address",
                    "address": address
                }
            
            lat, lon = coordinates
            
            lat_rad = math.radians(lat)
            delta_lat = radius / 111320
            delta_lon = radius / (111320 * math.cos(lat_rad))
            north = lat + delta_lat
            south = lat - delta_lat
            east  = lon + delta_lon
            west  = lon - delta_lon
                        
            amenities_data = {}
            errors = []
            timeout_count = 0
            
            for amenity in amenity_types:
                try:
                    print(f"Fetching {amenity}...")
                    
      
                    if radius >= 5000:
                        query_timeout = 15  
                    elif radius >= 2000:
                        query_timeout = 12
                    else:
                        query_timeout = 10 
                    
                    import signal
                    
                    def timeout_handler(signum, frame):
                        raise TimeoutError(f"Query timeout for {amenity} after {query_timeout}s")
                    
                    try:
                
                        signal.signal(signal.SIGALRM, timeout_handler)
                        signal.alarm(query_timeout)
                        
                        amenities = ox.features.features_from_bbox(
                            north, south, east, west,
                            tags={'amenity': amenity}
                        )
                        
                        signal.alarm(0)  
                        
                    except AttributeError:
                
                        amenities = ox.features.features_from_bbox(
                            north, south, east, west,
                            tags={'amenity': amenity}
                        )
                    
                    if not amenities.empty:
                        amenities_list = []
                        
                        for idx, row in amenities.iterrows():
                            try:
                                centroid = row.geometry.centroid
                                amenity_lat = centroid.y
                                amenity_lon = centroid.x
                            except Exception:
                                continue
                            
                          
                            distance = geodesic(
                                coordinates,
                                (amenity_lat, amenity_lon)
                            ).km
                            
                            amenity_info = {
                                'name': row.get('name', f'Unknown {amenity}'),
                                'type': amenity,
                                'coordinates': {
                                    'latitude': float(amenity_lat),
                                    'longitude': float(amenity_lon)
                                },
                                'distance_km': round(distance, 2)
                            }
                            amenities_list.append(amenity_info)
                     
                        amenities_list.sort(key=lambda x: x['distance_km'])
                        amenities_data[amenity] = amenities_list[:max_results_per_type]
                    else:
                        amenities_data[amenity] = []
                
                except TimeoutError as te:
                    timeout_count += 1
                    error_msg = f"Timeout fetching {amenity} (took >10s)"
                    print(f"⏱️ {error_msg}")
                    errors.append(error_msg)
                    amenities_data[amenity] = []
                    
                    if timeout_count >= 3:
                        print(f"⚠️ Too many timeouts ({timeout_count}), stopping early")
                        break
                
                except Exception as e:
                    error_msg = f"Error fetching {amenity}: {str(e)}"
                    print(f"{error_msg}")
                    errors.append(error_msg)
                    amenities_data[amenity] = []
            
            return {
                "address": address,
                "coordinates": coordinates,
                "search_radius_m": radius,
                "amenities": amenities_data,
                "errors": errors if errors else None,
                "timestamp": datetime.now().isoformat(),
                "timeout_count": timeout_count
            }
        
        except Exception as e:
            return {
                "error": f"Failed to get amenities: {str(e)}",
                "address": address
            }
    
    def get_building_footprints(
        self,
        address: str,
        radius: float = 500
    ) -> Dict:
        
        try:
            # Geocode
            geocoder = get_geocoder()
            coordinates = geocoder.address_to_coordinates(address)
            
            if not coordinates:
                return {"error": "Could not geocode address"}
            
            lat, lon = coordinates
            
            try:
                buildings = ox.features.features_from_point(
                    (lat, lon),
                    dist=radius,
                    tags={'building': True}
                )
            except TimeoutError:
                return {
                    "error": "Timeout fetching buildings",
                    "address": address,
                    "coordinates": coordinates
                }
            
            building_data = []
            
            if not buildings.empty:
                for idx, row in buildings.iterrows():
                    try:
                        centroid = row.geometry.centroid
                        area = row.geometry.area if hasattr(row.geometry, 'area') else None
                        
                        building_info = {
                            'building_id': str(idx),
                            'building_type': row.get('building', 'unknown'),
                            'geometry_type': row.geometry.geom_type,
                            'area_sq_m': round(area, 2) if area else None,
                            'centroid': {
                                'latitude': float(centroid.y),
                                'longitude': float(centroid.x)
                            }
                        }
                        building_data.append(building_info)
                    except Exception as e:
                        continue
            
            return {
                "address": address,
                "coordinates": coordinates,
                "total_buildings": len(building_data),
                "buildings": building_data
            }
        
        except Exception as e:
            return {
                "error": f"Failed to get buildings: {str(e)}",
                "address": address
            }
    
    def create_map_visualization(
        self,
        address: str,
        amenities_data: Dict,
        save_path: str = "map.html"
    ) -> Optional[str]:
    
        try:
            coordinates = amenities_data.get("coordinates")
            if not coordinates:
                print(" No coordinates in amenities_data")
                return None
            
            lat, lon = coordinates
 
            if not os.path.isabs(save_path):
         
                backend_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                save_path = os.path.join(backend_root, save_path)
            
          
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            print(f" Creating map for: {address}")
            print(f" Coordinates: ({lat:.4f}, {lon:.4f})")
            print(f" Saving to: {save_path}")
            
            m = folium.Map(
                location=[lat, lon],
                zoom_start=15,
                tiles='OpenStreetMap',
                control_scale=True,
                prefer_canvas=True 
            )
            
            folium.Marker(
                [lat, lon],
                popup=folium.Popup(f"<b>📍 Target Location</b><br>{address}", max_width=300),
                tooltip="Target Location",
                icon=folium.Icon(color="red", icon="star", prefix='fa')
            ).add_to(m)
            
     
            colors = {
                'restaurant': 'blue',
                'cafe': 'green',
                'school': 'orange',
                'hospital': 'red',
                'park': 'darkgreen',
                'supermarket': 'purple',
                'bank': 'darkblue',
                'pharmacy': 'pink',
                'gym': 'cadetblue',
                'library': 'lightblue',
                'transit_station': 'gray'
            }
            
            amenities = amenities_data.get("amenities", {})
            marker_count = 0
            
            for amenity_type, items in amenities.items():
                color = colors.get(amenity_type, 'gray')
                
                for item in items:
                    try:
                        item_coords = item.get('coordinates', {})
                        item_lat = item_coords.get('latitude')
                        item_lon = item_coords.get('longitude')
                        
                        if item_lat and item_lon:
                            popup_html = f"""
                            <div style="font-family: Arial; min-width: 150px;">
                                <h4 style="margin: 0 0 5px 0;">{item.get('name', 'Unknown')}</h4>
                                <p style="margin: 0;"><b>Type:</b> {amenity_type.title()}</p>
                                <p style="margin: 0;"><b>Distance:</b> {item.get('distance_km', 0):.2f} km</p>
                            </div>
                            """
                            
                            folium.Marker(
                                [item_lat, item_lon],
                                popup=folium.Popup(popup_html, max_width=300),
                                tooltip=item.get('name', 'Unknown'),
                                icon=folium.Icon(color=color, icon="info-sign")
                            ).add_to(m)
                            marker_count += 1
                    except Exception as e:
                        print(f" Skipping marker: {e}")
                        continue
            
            print(f" Added {marker_count} amenity markers")
            
            # Add search radius circle
            search_radius = amenities_data.get("search_radius_m", 1000)
            folium.Circle(
                radius=search_radius,
                location=[lat, lon],
                color='crimson',
                fill=True,
                fill_color='crimson',
                fill_opacity=0.1,
                weight=2,
                popup=f"Search Radius: {search_radius}m",
                tooltip=f"{search_radius}m radius"
            ).add_to(m)
            
            folium.LayerControl().add_to(m)
            

            try:
                m.save(save_path)
                print(f" Map saved successfully")
            except Exception as save_error:
                print(f" Error saving map: {save_error}")
                return None

            if os.path.exists(save_path):
                file_size = os.path.getsize(save_path)
                if file_size < 1000:  # File too small, likely error
                    print(f" Map file seems too small: {file_size} bytes")
                    return None
                print(f" Map verified: {save_path} ({file_size:,} bytes)")
                return save_path
            else:
                print(f" Map file not found after save: {save_path}")
                return None
        
        except Exception as e:
            print(f" Error creating map: {e}")
            import traceback
            traceback.print_exc()
            return None


def calculate_walk_score(coordinates: Tuple[float, float], amenities_data: Dict) -> float:

    try:
        amenities = amenities_data.get("amenities", {})
        
        if not amenities:
            return 0.0
        
        score = 0
        max_points = 0
        
        # Weights for different amenity types
        weights = {
            'restaurant': 10,
            'cafe': 8,
            'supermarket': 15,
            'pharmacy': 12,
            'school': 8,
            'hospital': 5,
            'park': 10,
            'bank': 5,
            'gym': 7,
            'library': 6,
            'transit_station': 12
        }
        
        for amenity_type, items in amenities.items():
            weight = weights.get(amenity_type, 5)
            max_points += weight * 5  # Max 5 items per type
            
            for item in items[:5]:
                distance_km = item.get('distance_km', 10)
                
                # Handle NaN or invalid distances
                if not isinstance(distance_km, (int, float)) or math.isnan(distance_km):
                    distance_km = 10  # Default to 10km if invalid
                
                distance_m = distance_km * 1000
                
                # Distance-based scoring
                if distance_m <= 500:  # Within 500m
                    score += weight
                elif distance_m <= 1000:  # 500m-1km
                    score += weight * 0.7
                elif distance_m <= 2000:  # 1-2km
                    score += weight * 0.3
                # Beyond 2km: no points
        
        if max_points > 0:
            normalized_score = (score / max_points) * 100
            final_score = round(min(normalized_score, 100), 1)
            # Ensure it's not NaN
            if math.isnan(final_score):
                return 0.0
            return final_score
        
        return 0.0
    
    except Exception as e:
        print(f"❌ Error calculating walk score: {e}")
        return 0.0



def lat_lon_to_tile(latitude: float, longitude: float, zoom: int) -> Tuple[int, int]:
    """
    Convert lat/lon to OSM tile coordinates
    
    Args:
        latitude: Latitude in degrees
        longitude: Longitude in degrees
        zoom: Zoom level (1-19)
    
    Returns:
        Tuple[int, int]: (tile_x, tile_y)
    """
    lat_rad = math.radians(latitude)
    n = 2.0 ** zoom
    
    tile_x = int((longitude + 180.0) / 360.0 * n)
    tile_y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    
    return tile_x, tile_y


def download_osm_tile(tile_x: int, tile_y: int, zoom: int, timeout: int = 5) -> Optional[Image.Image]:
    """
    Download single OSM tile with timeout
    
    Args:
        tile_x: X tile coordinate
        tile_y: Y tile coordinate
        zoom: Zoom level
        timeout: Request timeout in seconds
    
    Returns:
        Optional[Image.Image]: PIL Image or None if failed
    """
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        url = f"https://tile.openstreetmap.org/{zoom}/{tile_x}/{tile_y}.png"
        headers = {'User-Agent': 'GeoInsightAI/1.0 (Educational Project)'}
        
        response = requests.get(url, headers=headers, timeout=timeout)
        
        if response.status_code == 200:
            tile_image = Image.open(io.BytesIO(response.content))
            return tile_image
        else:
            logger.warning(f"OSM tile ({tile_x},{tile_y}) returned {response.status_code}")
            return None
            
    except requests.exceptions.Timeout:
        logger.warning(f"OSM tile ({tile_x},{tile_y}) timeout after {timeout}s")
        return None
    except Exception as e:
        logger.warning(f"OSM tile ({tile_x},{tile_y}) error: {e}")
        return None


def get_osm_map_area(latitude: float, longitude: float, radius_meters: int = 500) -> Optional[str]:
    """
    Fetch OSM tiles and save to temp file
    
    Args:
        latitude: Center latitude
        longitude: Center longitude
        radius_meters: Search radius in meters
    
    Returns:
        Optional[str]: Path to temporary PNG file, or None if failed
    """
    import time
    import logging
    logger = logging.getLogger(__name__)
    
    logger.info(f"Fetching OSM map for ({latitude:.4f}, {longitude:.4f}), radius={radius_meters}m")

    # Determine zoom level based on radius
    if radius_meters <= 500:
        zoom = 17
    elif radius_meters <= 1000:
        zoom = 16
    else:
        zoom = 15
    
    logger.info(f"Using zoom level {zoom}")
    
    # Get center tile coordinates
    center_x, center_y = lat_lon_to_tile(latitude, longitude, zoom)
    logger.info(f"Center tile: ({center_x}, {center_y})")
    
    # For small radius, download single tile
    if radius_meters <= 600:
        logger.info("Small radius - downloading 1 tile only")
        
        start_time = time.time()
        tile = download_osm_tile(center_x, center_y, zoom)
        elapsed = time.time() - start_time
        
        if not tile:
            logger.error(f"Failed to download center tile after {elapsed:.1f}s")
            return None
        
        logger.info(f"✅ Downloaded 1 tile in {elapsed:.1f}s")
        
        # Save to temp file and return PATH
        try:
            fd, temp_path = tempfile.mkstemp(suffix='.png', prefix='osm_map_')
            os.close(fd)  # Close file descriptor immediately
            
            tile.save(temp_path, format='PNG')
            logger.info(f"✅ Saved map to {temp_path}")
            
            return temp_path
        except Exception as e:
            logger.error(f"Failed to save tile: {e}")
            return None

    # For larger areas, download 2x2 grid (4 tiles max)
    else:
        logger.info("Large radius - downloading 2x2 grid (4 tiles)")
        
        tiles = {}
        start_time = time.time()
        
        for dx in [0, 1]:
            for dy in [0, 1]:
                tx = center_x + dx
                ty = center_y + dy
                
                tile = download_osm_tile(tx, ty, zoom)
                if tile:
                    tiles[(tx, ty)] = tile
        
        elapsed = time.time() - start_time
        
        if not tiles:
            logger.error(f"Failed to download any tiles after {elapsed:.1f}s")
            return None
        
        logger.info(f"✅ Downloaded {len(tiles)}/4 tiles in {elapsed:.1f}s")
        
        # If only got 1 tile, use it
        if len(tiles) == 1:
            logger.warning("Only got 1 tile, using it")
            tile = list(tiles.values())[0]
            
            try:
                fd, temp_path = tempfile.mkstemp(suffix='.png', prefix='osm_map_')
                os.close(fd)
                
                tile.save(temp_path, format='PNG')
                logger.info(f"✅ Saved map to {temp_path}")
                
                return temp_path
            except Exception as e:
                logger.error(f"Failed to save tile: {e}")
                return None

        # Stitch tiles together
        logger.info("Stitching tiles...")
        
        tile_size = 256
        stitched = Image.new('RGB', (tile_size * 2, tile_size * 2), color=(240, 240, 240))
        
        for (tx, ty), tile in tiles.items():
            x_offset = (tx - center_x) * tile_size
            y_offset = (ty - center_y) * tile_size
            stitched.paste(tile, (x_offset, y_offset))
        
        # Save stitched image to temp file
        try:
            fd, temp_path = tempfile.mkstemp(suffix='.png', prefix='osm_map_')
            os.close(fd)
            
            stitched.save(temp_path, format='PNG')
            logger.info(f"✅ Saved stitched map to {temp_path}")
            
            return temp_path
        except Exception as e:
            logger.error(f"Failed to save stitched image: {e}")
            return None