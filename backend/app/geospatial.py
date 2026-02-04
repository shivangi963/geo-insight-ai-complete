"""
FIXED: Geospatial module with proper timeouts and error handling
"""
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

# ✅ FIXED: Configure OSM with timeouts
ox.settings.log_console = True
ox.settings.use_cache = True
ox.settings.cache_folder = "./.osmnx_cache"
ox.settings.timeout = 30  # ✅ Add 30 second timeout


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
    """
    FIXED: Geocoder with proper rate limiting and error handling
    """
    
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
        """
        Convert address to coordinates with rate limiting
        """
        try:
            self._rate_limit()  # ✅ Enforce rate limit
            
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


class OpenStreetMapClient:
    """
    FIXED: OSM client with timeouts and better error handling
    """
    
    def __init__(self):
        # Configure OSMnx
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
        
        # Scale max results based on radius
        # Larger radius = more amenities expected
        if max_results_per_type is None:
            if radius >= 5000:
                max_results_per_type = 50  # Large radius: up to 50 per type
            elif radius >= 2000:
                max_results_per_type = 30  # Medium radius: up to 30 per type
            else:
                max_results_per_type = 20  # Small radius: up to 20 per type
        
        # Limit amenity types based on radius to prevent timeouts
        # Larger radius = fewer queries needed
        max_amenity_types = 6
        if radius >= 5000:
            max_amenity_types = 3  # Large radius: only 3 amenity types for speed
        elif radius >= 2000:
            max_amenity_types = 4  # Medium radius: 4 types
        
        if len(amenity_types) > max_amenity_types:
            print(f"⏱️ Limiting {len(amenity_types)} amenity types to {max_amenity_types} for timeout prevention (radius: {radius}m)")
            amenity_types = amenity_types[:max_amenity_types]
        
        try:
            # Geocode with timeout
            geocoder = LocationGeocoder()
            coordinates = geocoder.address_to_coordinates(address)
            
            if not coordinates:
                return {
                    "error": "Could not geocode address",
                    "address": address
                }
            
            lat, lon = coordinates
            
            # Calculate bounding box
            north = lat + (radius / 111320)
            south = lat - (radius / 111320)
            east = lon + (radius / (111320 * abs(lat) if lat != 0 else 111320))
            west = lon - (radius / (111320 * abs(lat) if lat != 0 else 111320))
            
            amenities_data = {}
            errors = []
            timeout_count = 0
            
            # Fetch amenities with per-type timeout
            for amenity in amenity_types:
                try:
                    print(f"Fetching {amenity}...")
                    
                    # Adaptive timeout based on radius
                    if radius >= 5000:
                        query_timeout = 15  # Longer timeout for large radius
                    elif radius >= 2000:
                        query_timeout = 12
                    else:
                        query_timeout = 10  # Default timeout
                    
                    # Use signal for timeout (Unix only)
                    import signal
                    
                    def timeout_handler(signum, frame):
                        raise TimeoutError(f"Query timeout for {amenity} after {query_timeout}s")
                    
                    try:
                        # Set adaptive timeout per amenity type
                        signal.signal(signal.SIGALRM, timeout_handler)
                        signal.alarm(query_timeout)
                        
                        amenities = ox.features.features_from_bbox(
                            north, south, east, west,
                            tags={'amenity': amenity}
                        )
                        
                        signal.alarm(0)  # Cancel alarm
                        
                    except AttributeError:
                        # Windows doesn't support SIGALRM
                        # Just run without timeout
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
                            
                            # Calculate distance
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
                        
                        # Sort and limit
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
                    
                    # If too many timeouts, stop
                    if timeout_count >= 3:
                        print(f"⚠️ Too many timeouts ({timeout_count}), stopping early")
                        break
                
                except Exception as e:
                    error_msg = f"Error fetching {amenity}: {str(e)}"
                    print(f"❌ {error_msg}")
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
        """
        FIXED: Get building footprints with timeout
        """
        try:
            # Geocode
            geocoder = LocationGeocoder()
            coordinates = geocoder.address_to_coordinates(address)
            
            if not coordinates:
                return {"error": "Could not geocode address"}
            
            lat, lon = coordinates
            
            # ✅ Fetch buildings with timeout
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
                        # Skip problematic buildings
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
        """
        FIXED: Create map visualization with better error handling
        """
        try:
            coordinates = amenities_data.get("coordinates")
            if not coordinates:
                print("❌ No coordinates in amenities_data")
                return None
            
            lat, lon = coordinates
            
            # Create map
            m = folium.Map(
                location=[lat, lon],
                zoom_start=15,
                tiles='OpenStreetMap'
            )
            
            # Add target marker
            folium.Marker(
                [lat, lon],
                popup=f"<b>Target:</b> {address}",
                icon=folium.Icon(color="red", icon="info-sign")
            ).add_to(m)
            
            # Amenity colors
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
                'transit_station': 'lightgray'
            }
            
            # Add amenity markers
            amenities = amenities_data.get("amenities", {})
            marker_count = 0
            
            for amenity_type, items in amenities.items():
                color = colors.get(amenity_type, 'gray')
                
                # Show all items from the search, no arbitrary limit
                for item in items:
                    try:
                        item_coords = item.get('coordinates', {})
                        item_lat = item_coords.get('latitude')
                        item_lon = item_coords.get('longitude')
                        
                        if item_lat and item_lon:
                            folium.Marker(
                                [item_lat, item_lon],
                                popup=f"""
                                <b>{item.get('name', 'Unknown')}</b><br>
                                Type: {item.get('type', 'N/A')}<br>
                                Distance: {item.get('distance_km', 0):.2f} km
                                """,
                                icon=folium.Icon(color=color, icon="info-sign")
                            ).add_to(m)
                            marker_count += 1
                    except Exception as e:
                        # Skip problematic markers
                        continue
            
            print(f"✅ Added {marker_count} amenity markers to map")
            
            # Add search radius circle
            search_radius = amenities_data.get("search_radius_m", 1000)
            folium.Circle(
                radius=search_radius,
                location=[lat, lon],
                color='crimson',
                fill=True,
                fill_color='crimson',
                fill_opacity=0.1,
                popup=f"Search Radius: {search_radius}m"
            ).add_to(m)
            
            # Save map
            import os
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            m.save(save_path)
            
            # Verify file was created
            if os.path.exists(save_path):
                file_size = os.path.getsize(save_path)
                print(f"✅ Map saved to: {save_path} ({file_size} bytes)")
                return save_path
            else:
                print(f"❌ Map file was not created at: {save_path}")
                return None
        
        except Exception as e:
            print(f"❌ Error creating map: {e}")
            return None


def calculate_walk_score(coordinates: Tuple[float, float], amenities_data: Dict) -> float:
    """
    FIXED: Calculate walk score with better error handling
    """
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