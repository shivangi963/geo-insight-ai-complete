"""
Satellite/Green Space Analysis Celery Tasks
Handles background processing of OSM green space analysis
"""
from celery import shared_task
from datetime import datetime
from bson import ObjectId
import traceback
import os
import tempfile

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@shared_task(bind=True, name="analyze_satellite")
def analyze_satellite_task(self, analysis_id: str, request_data: dict) -> dict:
    """
    Celery task for green space analysis
    
    Args:
        analysis_id: MongoDB document ID
        request_data: Analysis parameters (address, radius_m, etc.)
    
    Returns:
        Analysis results dictionary
    """
    try:
        from app.database import get_sync_database
        from app.geospatial import LocationGeocoder
        from app.tasks.computer_vision_tasks import analyze_osm_green_spaces
        
        # Get synchronous database connection
        db = get_sync_database()
        
        # Update status
        self.update_state(state='PROGRESS', meta={
            'status': 'Geocoding address...',
            'progress': 10
        })
        
        update_analysis_status_sync(db, analysis_id, 'processing', {'progress': 10})
        
        # Extract parameters
        address = request_data.get('address')
        radius_m = request_data.get('radius_m', 500)
        
        # Geocode address
        geocoder = LocationGeocoder()
        coordinates = geocoder.address_to_coordinates(address)
        
        if not coordinates:
            raise Exception("Could not geocode address")
        
        lat, lon = coordinates
        
        # Update status
        self.update_state(state='PROGRESS', meta={
            'status': 'Fetching OpenStreetMap tiles...',
            'progress': 30
        })
        
        update_analysis_status_sync(db, analysis_id, 'processing', {
            'progress': 30,
            'coordinates': {'latitude': lat, 'longitude': lon}
        })
        
        # Fetch OSM map
        from app.routers.osm_green_space import get_osm_map_area
        map_image = get_osm_map_area(lat, lon, radius_m)
        
        if not map_image:
            raise Exception("Failed to fetch OpenStreetMap tiles")
        
        # Save temporary image
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
            map_image.save(temp_file, format='PNG')
            temp_path = temp_file.name
        
        try:
            # Update status
            self.update_state(state='PROGRESS', meta={
                'status': 'Analyzing green spaces...',
                'progress': 60
            })
            
            update_analysis_status_sync(db, analysis_id, 'processing', {'progress': 60})
            
            # Analyze green spaces
            result = analyze_osm_green_spaces(temp_path, analysis_id)
            
            if result:
                green_space_pct = result.get('green_space_percentage', 0)
                green_pixels = result.get('green_pixels', 0)
                total_pixels = result.get('total_pixels', 0)
                visualization_path = result.get('visualization_path')
                breakdown = result.get('breakdown', {})
                
                # Complete analysis
                result_data = {
                    'address': address,
                    'coordinates': {'latitude': lat, 'longitude': lon},
                    'search_radius_m': radius_m,
                    'green_space_percentage': green_space_pct,
                    'green_pixels': green_pixels,
                    'total_pixels': total_pixels,
                    'visualization_path': visualization_path,
                    'breakdown': breakdown,
                    'map_source': 'OpenStreetMap',
                    'status': 'completed',
                    'progress': 100,
                    'completed_at': datetime.now().isoformat()
                }
                
                update_analysis_status_sync(db, analysis_id, 'completed', result_data)
                
                print(f"✅ Satellite analysis {analysis_id} completed")
                print(f"   Green Space: {green_space_pct:.1f}%")
                
                return result_data
            else:
                raise Exception("Green space calculation failed")
        
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_path)
            except:
                pass
    
    except Exception as e:
        error_msg = str(e)
        error_trace = traceback.format_exc()
        
        print(f"❌ Satellite analysis failed: {error_msg}")
        print(f"Traceback:\n{error_trace}")
        
        try:
            from app.database import get_sync_database
            db = get_sync_database()
            update_analysis_status_sync(db, analysis_id, 'failed', {
                'error': error_msg,
                'progress': 100
            })
        except:
            pass
        
        return {
            'analysis_id': analysis_id,
            'status': 'failed',
            'error': error_msg,
            'timestamp': datetime.now().isoformat()
        }


def update_analysis_status_sync(db, analysis_id: str, status: str, updates: dict = None):
    """
    Synchronous version of update_analysis_status for Celery tasks
    
    Args:
        db: Synchronous MongoDB database connection
        analysis_id: Analysis document ID
        status: New status
        updates: Additional fields to update
    """
    try:
        update_data = {
            'status': status,
            'updated_at': datetime.now()
        }
        
        if updates:
            update_data.update(updates)
        
        # Try ObjectId first
        try:
            obj_id = ObjectId(analysis_id)
            result = db.satellite_analyses.update_one(
                {"_id": obj_id},
                {"$set": update_data}
            )
        except:
            result = db.satellite_analyses.update_one(
                {"_id": analysis_id},
                {"$set": update_data}
            )
        
        if result.modified_count > 0:
            print(f"✅ Updated analysis {analysis_id} to status: {status}")
        else:
            print(f"⚠️ No documents updated for analysis ID: {analysis_id}")
    
    except Exception as e:
        print(f"❌ Error updating analysis status: {e}")