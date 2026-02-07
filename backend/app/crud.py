from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from bson import ObjectId
from datetime import datetime, timedelta
import logging

from .database import get_database

logger = logging.getLogger(__name__)

def document_to_dict(doc: Dict) -> Dict:
    """Convert MongoDB document to dict with string ID"""
    if doc and "_id" in doc:
        doc["id"] = str(doc["_id"])
        del doc["_id"]
    return doc

class PropertyCRUD:
    """Property CRUD operations"""
    
    def __init__(self):
        self.collection_name = "properties"
    
    async def get_all_properties(self, skip: int = 0, limit: int = 100) -> List[Dict[str, Any]]:
        """Get all properties with pagination"""
        try:
            db = await get_database()
            logger.debug(f"Fetching properties: skip={skip}, limit={limit}")
            
            count = await db[self.collection_name].count_documents({})
            logger.debug(f"Total properties in DB: {count}")
            
            if count == 0:
                logger.warning("No properties found in database")
                return []
            
            cursor = db[self.collection_name].find().skip(skip).limit(limit)
            properties = []
            
            async for doc in cursor:
                properties.append(document_to_dict(doc))
            
            logger.info(f"Retrieved {len(properties)} properties")
            return properties
            
        except Exception as e:
            logger.error(f"Error getting properties: {e}", exc_info=True)
            return []
    
    async def get_property_by_id(self, property_id: str) -> Optional[Dict[str, Any]]:
        """Get property by ID"""
        try:
            db = await get_database()
            
            try:
                obj_id = ObjectId(property_id)
                doc = await db[self.collection_name].find_one({"_id": obj_id})
            except:
                doc = await db[self.collection_name].find_one({"id": property_id})
            
            return document_to_dict(doc) if doc else None
            
        except Exception as e:
            logger.error(f"Error getting property {property_id}: {e}")
            return None
    
    async def create_property(self, property_data: BaseModel) -> Dict[str, Any]:
        """Create new property"""
        try:
            db = await get_database()
            
            property_dict = property_data.dict()
            property_dict["created_at"] = datetime.now()
            property_dict["updated_at"] = datetime.now()
            
            result = await db[self.collection_name].insert_one(property_dict)
            created_doc = await db[self.collection_name].find_one({"_id": result.inserted_id})
            
            logger.info(f"Created property: {result.inserted_id}")
            return document_to_dict(created_doc)
            
        except Exception as e:
            logger.error(f"Error creating property: {e}")
            raise
    
    async def update_property(self, property_id: str, property_data: BaseModel) -> Optional[Dict[str, Any]]:
        """Update property"""
        try:
            db = await get_database()
            
            update_data = property_data.dict(exclude_unset=True)
            update_data["updated_at"] = datetime.now()
            
            try:
                obj_id = ObjectId(property_id)
                result = await db[self.collection_name].find_one_and_update(
                    {"_id": obj_id},
                    {"$set": update_data},
                    return_document=True
                )
            except:
                result = await db[self.collection_name].find_one_and_update(
                    {"id": property_id},
                    {"$set": update_data},
                    return_document=True
                )
            
            if result:
                logger.info(f"Updated property: {property_id}")
                return document_to_dict(result)
            
            return None
            
        except Exception as e:
            logger.error(f"Error updating property: {e}")
            return None
    
    async def delete_property(self, property_id: str) -> bool:
        """Delete property"""
        try:
            db = await get_database()
            
            try:
                obj_id = ObjectId(property_id)
                result = await db[self.collection_name].delete_one({"_id": obj_id})
            except:
                result = await db[self.collection_name].delete_one({"id": property_id})
            
            if result.deleted_count > 0:
                logger.info(f"Deleted property: {property_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error deleting property: {e}")
            return False

property_crud = PropertyCRUD()


# ==================== NEIGHBORHOOD ANALYSIS CRUD ====================

NEIGHBORHOOD_ANALYSIS_COLLECTION = "neighborhood_analyses"

async def create_neighborhood_analysis(analysis_data: Dict[str, Any]) -> str:
    """Create neighborhood analysis document"""
    try:
        db = await get_database()
        
        # Add timestamps
        analysis_data["created_at"] = datetime.now()
        analysis_data["updated_at"] = datetime.now()
        analysis_data["status"] = analysis_data.get("status", "processing")
        
        # Insert document
        result = await db[NEIGHBORHOOD_ANALYSIS_COLLECTION].insert_one(analysis_data)
        
        print(f"✅ Created neighborhood analysis with ID: {result.inserted_id}")
        return str(result.inserted_id)
        
    except Exception as e:
        print(f"❌ Error creating neighborhood analysis: {e}")
        raise

async def get_neighborhood_analysis(analysis_id: str) -> Optional[Dict]:
    """Get neighborhood analysis by ID"""
    try:
        db = await get_database()
        
        # Try ObjectId first
        try:
            obj_id = ObjectId(analysis_id)
            analysis = await db[NEIGHBORHOOD_ANALYSIS_COLLECTION].find_one({"_id": obj_id})
        except:
            # Fallback to string ID
            analysis = await db[NEIGHBORHOOD_ANALYSIS_COLLECTION].find_one({"_id": analysis_id})
        
        if analysis:
            return document_to_dict(analysis)
        return None
        
    except Exception as e:
        print(f"❌ Error getting neighborhood analysis: {e}")
        return None

async def get_recent_analyses(limit: int = 10) -> List[Dict]:
    """Get recent neighborhood analyses"""
    try:
        db = await get_database()
        
        cursor = db[NEIGHBORHOOD_ANALYSIS_COLLECTION].find().sort(
            "created_at", -1
        ).limit(limit)
        
        analyses = []
        async for doc in cursor:
            analyses.append(document_to_dict(doc))
        
        print(f"✅ Retrieved {len(analyses)} recent analyses")
        return analyses
        
    except Exception as e:
        print(f"❌ Error getting recent analyses: {e}")
        return []

async def update_analysis_status(analysis_id: str, status: str, updates: Optional[Dict] = None):
    """Update analysis status"""
    try:
        db = await get_database()
        
        update_data = {
            "status": status,
            "updated_at": datetime.now()
        }
        
        if updates:
            update_data.update(updates)
        
        # Try ObjectId first
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
            print(f"✅ Updated analysis {analysis_id} to status: {status}")
        else:
            print(f"⚠️ No documents updated for analysis ID: {analysis_id}")
            
    except Exception as e:
        print(f"❌ Error updating analysis status: {e}")

async def delete_neighborhood_analysis(analysis_id: str) -> bool:
    """Delete neighborhood analysis"""
    try:
        db = await get_database()
        
        # Try ObjectId first
        try:
            obj_id = ObjectId(analysis_id)
            result = await db[NEIGHBORHOOD_ANALYSIS_COLLECTION].delete_one({"_id": obj_id})
        except:
            result = await db[NEIGHBORHOOD_ANALYSIS_COLLECTION].delete_one({"_id": analysis_id})
        
        if result.deleted_count > 0:
            print(f"✅ Deleted analysis: {analysis_id}")
            return True
        
        print(f"⚠️ No analysis found to delete: {analysis_id}")
        return False
            
    except Exception as e:
        print(f"❌ Error deleting neighborhood analysis: {e}")
        return False

async def get_analysis_count() -> int:
    """Get total count of analyses"""
    try:
        db = await get_database()
        count = await db[NEIGHBORHOOD_ANALYSIS_COLLECTION].count_documents({})
        return count
    except Exception as e:
        print(f"❌ Error getting analysis count: {e}")
        return 0

async def cleanup_stuck_analyses(max_age_minutes: int = 30):
    """
    Clean up analyses stuck in 'processing' state
    """
    try:
        db = await get_database()
        
        cutoff_time = datetime.now() - timedelta(minutes=max_age_minutes)
        
        result = await db[NEIGHBORHOOD_ANALYSIS_COLLECTION].update_many(
            {
                "status": "processing",
                "created_at": {"$lt": cutoff_time}
            },
            {
                "$set": {
                    "status": "failed",
                    "error": f"Analysis timed out after {max_age_minutes} minutes",
                    "progress": 100,
                    "updated_at": datetime.now()
                }
            }
        )
        
        if result.modified_count > 0:
            print(f"✅ Cleaned up {result.modified_count} stuck analyses")
        
        return result.modified_count
    except Exception as e:
        print(f"❌ Error cleaning up stuck analyses: {e}")
        return 0

SATELLITE_ANALYSIS_COLLECTION = "satellite_analyses"

async def create_satellite_analysis(analysis_data: Dict) -> str:
    """Create new satellite analysis document"""
    db = await get_database()
    analysis_data["created_at"] = datetime.now()
    analysis_data["updated_at"] = datetime.now()
    result = await db[SATELLITE_ANALYSIS_COLLECTION].insert_one(analysis_data)
    return str(result.inserted_id)

async def get_satellite_analysis(analysis_id: str) -> Optional[Dict]:
    """Get satellite analysis by ID"""
    db = await get_database()
    try:
        obj_id = ObjectId(analysis_id)
        analysis = await db[SATELLITE_ANALYSIS_COLLECTION].find_one({"_id": obj_id})
    except:
        analysis = await db[SATELLITE_ANALYSIS_COLLECTION].find_one({"_id": analysis_id})
    
    if analysis and "_id" in analysis:
        analysis["id"] = str(analysis["_id"])
        del analysis["_id"]
    return analysis

async def update_analysis_status(analysis_id: str, status: str, update_data: Dict) -> bool:
    """Update analysis status and data"""
    db = await get_database()
    try:
        obj_id = ObjectId(analysis_id)
        filter_query = {"_id": obj_id}
    except:
        filter_query = {"_id": analysis_id}
    
    update_data["status"] = status
    update_data["updated_at"] = datetime.now()
    
    result = await db[SATELLITE_ANALYSIS_COLLECTION].update_one(
        filter_query,
        {"$set": update_data}
    )
    return result.modified_count > 0

async def get_recent_satellite_analyses(limit: int = 10) -> list:
    """Get recent analyses"""
    db = await get_database()
    cursor = db[SATELLITE_ANALYSIS_COLLECTION].find().sort("created_at", -1).limit(limit)
    analyses = []
    async for doc in cursor:
        if "_id" in doc:
            doc["id"] = str(doc["_id"])
            del doc["_id"]
        analyses.append(doc)
    return analyses