import osmnx as ox
import networkx as nx
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from typing import Dict, List, Optional, Tuple
import folium
from shapely.geometry import Point, Polygon
import json
from datetime import datetime

class LocationGeocoder:
    
    def __init__(self, user_agent: str = "geo_insight_ai"):
    
        self.geolocator = Nominatim(user_agent=user_agent)
        
    def address_to_coordinates(self, address: str) -> Optional[Tuple[float, float]]:
        
        try:
            location = self.geolocator.geocode(address)
            if location:
                return (location.latitude, location.longitude)
            return None
        except Exception as e:
            print(f"Geocoding error: {e}")
            return None
    
    def coordinates_to_address(self, lat: float, lon: float) -> Optional[str]:
    
        try:
            location = self.geolocator.reverse(f"{lat}, {lon}")
            return location.address if location else None
        except Exception as e:
            print(f"Reverse geocoding error: {e}")
            return None


class OpenStreetMapClient:
    
    def __init__(self):
        ox.settings.log_console = True
        ox.settings.use_cache = True
        ox.settings.cache_folder = "./.osmnx_cache"
        
    def get_nearby_amenities(
        self,
        address: str,
        radius: float = 1000,
        amenity_types: Optional[List[str]] = None
    ) -> Dict:
    
        if amenity_types is None:
            amenity_types = [
                'restaurant', 'cafe', 'school', 'hospital',
                'park', 'supermarket', 'bank', 'pharmacy'
            ]
        
        try:
            
            geocoder = LocationGeocoder()
            coordinates = geocoder.address_to_coordinates(address)
            
            if not coordinates:
                return {"error": "Could not geocode address"}
            
            lat, lon = coordinates
            
           
            north = lat + (radius / 111320)  
            south = lat - (radius / 111320)
            east = lon + (radius / (111320 * abs(lat)))  
            west = lon - (radius / (111320 * abs(lat)))
            
            amenities_data = {}
            
            for amenity in amenity_types:
                try:
                  
                    amenities = ox.features.features_from_bbox(
                        north, south, east, west,
                        tags={'amenity': amenity}
                    )
                    
                
                    if not amenities.empty:
                        amenities_list = []
                        for idx, row in amenities.iterrows():
                            amenity_info = {
                                'name': row.get('name', f'Unknown {amenity}'),
                                'type': amenity,
                                'coordinates': {
                                'latitude': row.geometry.centroid.y,
                                'longitude': row.geometry.centroid.x
                            },
                                'distance_km': round(geodesic(
                                    coordinates, 
                                    (row.geometry.centroid.y, row.geometry.centroid.x)
                                ).km, 2)
                            }
                            amenities_list.append(amenity_info)
                        
                    
                        amenities_list.sort(key=lambda x: x['distance_km'])
                        amenities_data[amenity] = amenities_list[:10]  
                        
                except Exception as e:
                    print(f"Error fetching {amenity}: {e}")
                    amenities_data[amenity] = []
            
            return {
                "address": address,
                "coordinates": coordinates,
                "search_radius_m": radius,
                "amenities": amenities_data,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": f"Failed to get amenities: {str(e)}"}
    
    def get_building_footprints(
        self,
        address: str,
        radius: float = 500
    ) -> Dict:
      
        try:
           
            geocoder = LocationGeocoder()
            coordinates = geocoder.address_to_coordinates(address)
            
            if not coordinates:
                return {"error": "Could not geocode address"}
            
            lat, lon = coordinates
            
       
            buildings = ox.features.features_from_point(
                (lat, lon),
                dist=radius,
                tags={'building': True}
            )
            
            building_data = []
            
            if not buildings.empty:
                for idx, row in buildings.iterrows():
                    building_info = {
                        'building_id': str(idx),
                        'building_type': row.get('building', 'unknown'),
                        'geometry_type': row.geometry.geom_type,
                        'area_sq_m': round(row.geometry.area, 2) if hasattr(row.geometry, 'area') else None,
                        'centroid': {
                            'latitude': row.geometry.centroid.y,  
                            'longitude': row.geometry.centroid.x
                        }
                    }
                    building_data.append(building_info)
            
            return {
                "address": address,
                "coordinates": coordinates,
                "total_buildings": len(building_data),
                "buildings": building_data
            }
            
        except Exception as e:
            return {"error": f"Failed to get buildings: {str(e)}"}
    
    def create_map_visualization(
        self,
        address: str,
        amenities_data: Dict,
        save_path: str = "map.html"
    ) -> str:
     
        try:
            coordinates = amenities_data.get("coordinates")
            if not coordinates:
                return None
            
            lat, lon = coordinates
            
          
            m = folium.Map(location=[lat, lon], zoom_start=15)
            
       
            folium.Marker(
                [lat, lon],
                popup=f"<b>Target:</b> {address}",
                icon=folium.Icon(color="red", icon="info-sign")
            ).add_to(m)
            
          
            amenities = amenities_data.get("amenities", {})
            
            
            colors = {
                'restaurant': 'blue',
                'cafe': 'green',
                'school': 'orange',
                'hospital': 'red',
                'park': 'darkgreen',
                'supermarket': 'purple',
                'bank': 'darkblue',
                'pharmacy': 'pink'
            }
            
            for amenity_type, items in amenities.items():
                color = colors.get(amenity_type, 'gray')
                for item in items[:5]: 
                    item_lat = item['coordinates']['latitude']
                    item_lon = item['coordinates']['longitude']
                    
                    folium.Marker(
                        [item_lat, item_lon],  
                        popup=f"<b>{item['name']}</b><br>Type: {item['type']}<br>Distance: {item['distance_km']}km",
                        icon=folium.Icon(color=color, icon="info-sign")
                    ).add_to(m)
           
            folium.Circle(
                radius=amenities_data.get("search_radius_m", 1000),
                location=[lat, lon],
                color='crimson',
                fill=True,
                fill_color='crimson',
                fill_opacity=0.1,
                popup=f"Search Radius: {amenities_data.get('search_radius_m', 1000)}m"
            ).add_to(m)
                      
            m.save(save_path)
            return save_path
            
        except Exception as e:
            print(f"Error creating map: {e}")
            return None


def calculate_walk_score(coordinates: Tuple[float, float], amenities_data: Dict) -> float:

    try:
        amenities = amenities_data.get("amenities", {})
        
        score = 0
        max_points = 0
        
        weights = {
            'restaurant': 10,
            'cafe': 8,
            'supermarket': 15,
            'pharmacy': 12,
            'school': 8,
            'hospital': 5,
            'park': 10,
            'bank': 5
        }
        
        for amenity_type, items in amenities.items():
            weight = weights.get(amenity_type, 5)
            max_points += weight * 5  
            for item in items[:5]:  
                distance = item.get('distance_km', 10) * 1000  
                
                if distance <= 500:  
                    score += weight
                elif distance <= 1000:  
                    score += weight * 0.7
                elif distance <= 2000: 
                    score += weight * 0.3

        if max_points > 0:
            normalized_score = (score / max_points) * 100
            return round(min(normalized_score, 100), 1)
        
        return 0.0
        
    except Exception as e:
        print(f"Error calculating walk score: {e}")
        return 0.0