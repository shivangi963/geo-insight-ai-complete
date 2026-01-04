
import os
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from bson import ObjectId
from datetime import datetime
from app.database import get_database, get_sync_database

class PropertyCRUD:
    
    def __init__(self):
        print(" Initializing PropertyCRUD with mock data")
        self.properties = self._load_sample_data()
        self.next_id = len(self.properties) + 1
    
    def get_all_properties(self, skip: int = 0, limit: int = 100) -> List[Dict[str, Any]]:
     
        print(f"Getting properties (skip={skip}, limit={limit})")
        properties = self.properties[skip:skip + limit]
        print(f" Returning {len(properties)} properties")
        return properties
    
    def get_property_by_id(self, property_id: str) -> Optional[Dict[str, Any]]:
     
        print(f"Searching for property ID: {property_id}")
        for prop in self.properties:
            if prop["id"] == property_id:
                print(f"Found property: {prop['address']}")
                return prop
        print(f"Property not found: {property_id}")
        return None
    
    def create_property(self, property_data: BaseModel) -> Dict[str, Any]:
   
        print(f"Creating new property: {property_data.address}")
        
        property_dict = property_data.dict()
        property_dict["id"] = str(self.next_id)
        property_dict["created_at"] = "2024-01-20T10:00:00"
        property_dict["updated_at"] = "2024-01-20T10:00:00"
        
        self.properties.append(property_dict)
        self.next_id += 1
        
        print(f"Property created with ID: {property_dict['id']}")
        return property_dict
    
    def update_property(self, property_id: str, property_data: BaseModel) -> Optional[Dict[str, Any]]:
        
        print(f"Updating property ID: {property_id}")
        
        for i, prop in enumerate(self.properties):
            if prop["id"] == property_id:
                update_data = property_data.dict(exclude_unset=True)
                self.properties[i].update(update_data)
                self.properties[i]["updated_at"] = "2024-01-20T11:00:00"
                
                print(f"Property updated: {self.properties[i]['address']}")
                return self.properties[i]
        
        print(f"Property not found for update: {property_id}")
        return None
    
    def delete_property(self, property_id: str) -> bool:
     
        print(f"Deleting property ID: {property_id}")
        
        for i, prop in enumerate(self.properties):
            if prop["id"] == property_id:
                del self.properties[i]
                print(f" Property deleted: {property_id}")
                return True
        
        print(f"Property not found for deletion: {property_id}")
        return False


NEIGHBORHOOD_ANALYSIS_COLLECTION = "neighborhood_analyses"

async def create_neighborhood_analysis(analysis_data: Dict[str, Any]) -> str:

    try:
        db = await get_database()
        
     
        analysis_data["created_at"] = datetime.now()
        analysis_data["updated_at"] = datetime.now()
        analysis_data["status"] = analysis_data.get("status", "processing")
        
        
        result = await db[NEIGHBORHOOD_ANALYSIS_COLLECTION].insert_one(analysis_data)
        
        print(f"Created neighborhood analysis with ID: {result.inserted_id}")
        return str(result.inserted_id)
        
    except Exception as e:
        print(f" Error creating neighborhood analysis: {e}")
      
        import uuid
        mock_id = str(uuid.uuid4())
        print(f" Using mock ID: {mock_id}")
        return mock_id

async def get_neighborhood_analysis(analysis_id: str) -> Optional[Dict]:
    
    try:
        db = await get_database()
    
        try:
            obj_id = ObjectId(analysis_id)
        except:
           
            obj_id = analysis_id
            analysis = await db[NEIGHBORHOOD_ANALYSIS_COLLECTION].find_one(
                {"_id": analysis_id}
            )
        else:
            
            analysis = await db[NEIGHBORHOOD_ANALYSIS_COLLECTION].find_one(
                {"_id": obj_id}
            )
        
        if analysis:
           
            analysis["id"] = str(analysis["_id"])
            return analysis
        return None
        
    except Exception as e:
        print(f"Error getting neighborhood analysis: {e}")
        return None

async def get_recent_analyses(limit: int = 10) -> List[Dict]:
    
    try:
        db = await get_database()
        
        analyses = []
        cursor = db[NEIGHBORHOOD_ANALYSIS_COLLECTION].find().sort(
            "created_at", -1
        ).limit(limit)
        
        async for analysis in cursor:
            analysis["id"] = str(analysis["_id"])
            analyses.append(analysis)
        
        print(f"Retrieved {len(analyses)} recent analyses")
        return analyses
        
    except Exception as e:
        print(f"Error getting recent analyses: {e}")
        return []

async def update_analysis_status(analysis_id: str, status: str, updates: Optional[Dict] = None):
  
    try:
        db = await get_database()
        
        update_data = {
            "status": status,
            "updated_at": datetime.now()
        }
        
        if updates:
            update_data.update(updates)
        
       
        try:
            obj_id = ObjectId(analysis_id)
            result = await db[NEIGHBORHOOD_ANALYSIS_COLLECTION].update_one(
                {"_id": obj_id},
                {"$set": update_data}
            )
        except:
          
            result = await db[NEIGHBORHOOD_ANALYSIS_COLLECTION].update_one(
                {"_id": analysis_id},
                {"$set": update_data}
            )
        
        if result.modified_count > 0:
            print(f"Updated analysis {analysis_id} to status: {status}")
        else:
            print(f" No documents updated for analysis ID: {analysis_id}")
            
    except Exception as e:
        print(f" Error updating analysis status: {e}")

async def delete_neighborhood_analysis(analysis_id: str) -> bool:
    
    try:
        db = await get_database()
        
       
        try:
            obj_id = ObjectId(analysis_id)
            result = await db[NEIGHBORHOOD_ANALYSIS_COLLECTION].delete_one(
                {"_id": obj_id}
            )
        except:
          
            result = await db[NEIGHBORHOOD_ANALYSIS_COLLECTION].delete_one(
                {"_id": analysis_id}
            )
        
        if result.deleted_count > 0:
            print(f"Deleted analysis: {analysis_id}")
            return True
        else:
            print(f" No analysis found to delete: {analysis_id}")
            return False
            
    except Exception as e:
        print(f"Error deleting neighborhood analysis: {e}")
        return False


async def get_analysis_count() -> int:
    
    try:
        db = await get_database()
        count = await db[NEIGHBORHOOD_ANALYSIS_COLLECTION].count_documents({})
        return count
    except Exception as e:
        print(f"Error getting analysis count: {e}")
        return 0

async def initialize_collections():
    
    try:
        db = await get_database()
        
        await db[NEIGHBORHOOD_ANALYSIS_COLLECTION].create_index("created_at")
        await db[NEIGHBORHOOD_ANALYSIS_COLLECTION].create_index("status")
        await db[NEIGHBORHOOD_ANALYSIS_COLLECTION].create_index("address")
        
        print(" Database collections initialized with indexes")
        
    except Exception as e:
        print(f"Error initializing collections: {e}")

property_crud = PropertyCRUD()