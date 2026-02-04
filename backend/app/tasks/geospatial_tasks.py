"""
FIXED Geospatial Tasks - Proper sync database handling
"""
import os
from celery import shared_task
from app.geospatial import OpenStreetMapClient, calculate_walk_score
from app.database import get_sync_database
from typing import Dict
from datetime import datetime
from bson import ObjectId
import traceback

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MAPS_DIR = os.path.join(PROJECT_ROOT, "maps")

def update_analysis_status_sync(db, analysis_id: str, status: str, updates: Dict = None):
    """Synchronous version for Celery tasks"""
    try:
        update_data = {
            "status": status,
            "updated_at": datetime.now()
        }
        
        if updates:
            update_data.update(updates)
        
        # Try ObjectId first
        try:
            obj_id = ObjectId(analysis_id)
            result = db.neighborhood_analyses.update_one(
                {"_id": obj_id},
                {"$set": update_data}
            )
        except:
            result = db.neighborhood_analyses.update_one(
                {"_id": analysis_id},
                {"$set": update_data}
            )
        
        if result.modified_count > 0:
            print(f"✅ Updated analysis {analysis_id} to status: {status}")
        else:
            print(f"⚠️ No documents updated for analysis ID: {analysis_id}")
            
    except Exception as e:
        print(f"❌ Error updating analysis status: {e}")

@shared_task(bind=True, name="analyze_neighborhood")
def analyze_neighborhood_task(self, analysis_id: str, request_data: Dict) -> Dict:
    """
    FIXED: Analyze neighborhood using synchronous database
    """
    db = None
    
    try:
        # Get sync database connection
        db = get_sync_database()
        
        # Update initial status
        self.update_state(state='PROGRESS', meta={
            'status': 'Geocoding address...',
            'progress': 5
        })
        update_analysis_status_sync(db, analysis_id, 'processing', {'progress': 5})
        
        # Initialize OSM client
        osm_client = OpenStreetMapClient()
        
        # Extract request data
        address = request_data.get('address')
        radius_m = request_data.get('radius_m', 1000)
        amenity_types = request_data.get('amenity_types')
        include_buildings = request_data.get('include_buildings', True)
        generate_map = request_data.get('generate_map', True)
        
        # Fetch amenities
        self.update_state(state='PROGRESS', meta={
            'status': 'Fetching amenities from OpenStreetMap...',
            'progress': 20
        })
        update_analysis_status_sync(db, analysis_id, 'processing', {'progress': 20})
        
        amenities_data = osm_client.get_nearby_amenities(
            address=address,
            radius=radius_m,
            amenity_types=amenity_types
        )
        
        if "error" in amenities_data:
            raise Exception(amenities_data["error"])
        
        # Calculate walkability
        self.update_state(state='PROGRESS', meta={
            'status': 'Calculating walk score...',
            'progress': 50
        })
        update_analysis_status_sync(db, analysis_id, 'processing', {'progress': 50})
        
        coordinates = amenities_data.get("coordinates")
        walk_score = None
        if coordinates:
            walk_score = calculate_walk_score(coordinates, amenities_data)
        
        # Get building footprints if requested
        building_footprints = []
        if include_buildings:
            self.update_state(state='PROGRESS', meta={
                'status': 'Analyzing building footprints...',
                'progress': 65
            })
            update_analysis_status_sync(db, analysis_id, 'processing', {'progress': 65})
            
            try:
                buildings_data = osm_client.get_building_footprints(
                    address=address,
                    radius=min(radius_m, 500)
                )
                if "error" not in buildings_data:
                    building_footprints = buildings_data.get("buildings", [])
            except Exception as e:
                print(f"⚠️ Building footprints failed: {e}")
        
        # Generate map
        map_path = None
        if generate_map and coordinates:
            self.update_state(state='PROGRESS', meta={
                'status': 'Generating interactive map...',
                'progress': 80
            })
            update_analysis_status_sync(db, analysis_id, 'processing', {'progress': 80})
            
            try:
                import os
                os.makedirs(MAPS_DIR, exist_ok=True)
                map_filename = f"neighborhood_{analysis_id.replace('-', '_')}.html"
                map_path = os.path.join(MAPS_DIR, map_filename) 
                
                map_path = osm_client.create_map_visualization(
                    address=address,
                    amenities_data=amenities_data,
                    save_path=map_path
                )
            except Exception as e:
                print(f"⚠️ Map generation failed: {e}")
                map_path = None
        
        # Calculate totals
        amenities = amenities_data.get("amenities", {})
        total_amenities = sum(len(items) for items in amenities.values())
        
        # Prepare results
        results = {
            'analysis_id': analysis_id,
            'status': 'completed',
            'address': address,
            'walk_score': walk_score,
            'total_amenities': total_amenities,
            'building_count': len(building_footprints),
            'map_path': map_path,
            'coordinates': coordinates,
            'amenities': amenities,
            'timestamp': datetime.now().isoformat()
        }
        
        # Update database with final results
        update_data = {
            'status': 'completed',
            'walk_score': walk_score,
            'map_path': map_path,
            'amenities': amenities,
            'building_footprints': building_footprints,
            'total_amenities': total_amenities,
            'coordinates': coordinates,
            'completed_at': datetime.now(),
            'progress': 100
        }
        update_analysis_status_sync(db, analysis_id, 'completed', update_data)
        
        print(f"✅ Analysis {analysis_id} completed successfully")
        print(f"   Address: {address}")
        print(f"   Amenities: {total_amenities}")
        print(f"   Walk Score: {walk_score}")
        
        return results
        
    except Exception as e:
        error_msg = str(e)
        error_trace = traceback.format_exc()
        
        print(f"❌ Analysis failed: {error_msg}")
        print(f"Traceback:\n{error_trace}")
        
        if db:
            update_analysis_status_sync(db, analysis_id, 'failed', {
                'error': error_msg,
                'progress': 100
            })
        
        return {
            'analysis_id': analysis_id,
            'status': 'failed',
            'error': error_msg,
            'timestamp': datetime.now().isoformat()
        }