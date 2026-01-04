
from celery import shared_task
from app.geospatial import OpenStreetMapClient, calculate_walk_score
from app.crud import update_analysis_status
from typing import Dict
from datetime import datetime

@shared_task(bind=True, name="analyze_neighborhood")
def analyze_neighborhood_task(self, analysis_id: str, request_data: Dict) -> Dict:
   
    try:
        self.update_state(state='PROGRESS', meta={'status': 'Geocoding address...'})
        
        
        osm_client = OpenStreetMapClient()
      
        address = request_data.get('address')
        radius_m = request_data.get('radius_m', 1000)
        amenity_types = request_data.get('amenity_types')
        include_buildings = request_data.get('include_buildings', True)
        
       
        update_analysis_status(analysis_id, 'processing')
        
        self.update_state(state='PROGRESS', meta={'status': 'Fetching amenities...'})
        amenities_data = osm_client.get_nearby_amenities(
            address=address,
            radius=radius_m,
            amenity_types=amenity_types
        )
        
        if "error" in amenities_data:
            raise Exception(amenities_data["error"])
    
        self.update_state(state='PROGRESS', meta={'status': 'Calculating walkability...'})
        coordinates = amenities_data.get("coordinates")
        walk_score = None
        if coordinates:
            walk_score = calculate_walk_score(coordinates, amenities_data)
  
        building_footprints = []
        if include_buildings:
            self.update_state(state='PROGRESS', meta={'status': 'Analyzing buildings...'})
            buildings_data = osm_client.get_building_footprints(
                address=address,
                radius=min(radius_m, 500)
            )
            if "error" not in buildings_data:
                building_footprints = buildings_data.get("buildings", [])
        
        
        self.update_state(state='PROGRESS', meta={'status': 'Generating map...'})
        map_path = None
        if coordinates:
            map_path = osm_client.create_map_visualization(
                address=address,
                amenities_data=amenities_data,
                save_path=f"maps/neighborhood_{analysis_id}.html"
            )
        
   
        total_amenities = sum(
            len(items) for items in amenities_data.get("amenities", {}).values()
        )
        
        results = {
            'analysis_id': analysis_id,
            'status': 'completed',
            'address': address,
            'walk_score': walk_score,
            'total_amenities': total_amenities,
            'building_count': len(building_footprints),
            'map_path': map_path,
            'coordinates': coordinates,
            'timestamp': datetime.now().isoformat()
        }
        
        update_data = {
            'status': 'completed',
            'walk_score': walk_score,
            'map_path': map_path,
            'completed_at': datetime.now()
        }
        update_analysis_status(analysis_id, 'completed', update_data)
        
        return results
        
    except Exception as e:
        update_analysis_status(analysis_id, 'failed', {'error': str(e)})
        
        return {
            'analysis_id': analysis_id,
            'status': 'failed',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }