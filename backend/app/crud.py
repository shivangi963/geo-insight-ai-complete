from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from bson import ObjectId
from datetime import datetime
from app.database import get_database

# Helper function to convert MongoDB document to dict
def document_to_dict(doc: Dict) -> Dict:
    """Convert MongoDB document to dictionary with string ID"""
    if doc and "_id" in doc:
        doc["id"] = str(doc["_id"])
        del doc["_id"]
    return doc

# ==================== PROPERTY CRUD ====================

class PropertyCRUD:
    """Property CRUD operations using MongoDB"""
    
    def __init__(self):
        self.collection_name = "properties"
    
    async def get_all_properties(self, skip: int = 0, limit: int = 100) -> List[Dict[str, Any]]:
        """Get all properties with pagination"""
        try:
            db = await get_database()
            cursor = db[self.collection_name].find().skip(skip).limit(limit)
            
            properties = []
            async for doc in cursor:
                properties.append(document_to_dict(doc))
            
            print(f"✅ Retrieved {len(properties)} properties from MongoDB")
            return properties
            
        except Exception as e:
            print(f"❌ Error getting properties: {e}")
            return []
    
    async def get_property_by_id(self, property_id: str) -> Optional[Dict[str, Any]]:
        """Get property by ID"""
        try:
            db = await get_database()
            
            # Try to convert to ObjectId
            try:
                obj_id = ObjectId(property_id)
                doc = await db[self.collection_name].find_one({"_id": obj_id})
            except:
                # If not ObjectId, search by string id
                doc = await db[self.collection_name].find_one({"id": property_id})
            
            if doc:
                return document_to_dict(doc)
            return None
            
        except Exception as e:
            print(f"❌ Error getting property: {e}")
            return None
    
    async def create_property(self, property_data: BaseModel) -> Dict[str, Any]:
        """Create new property"""
        try:
            db = await get_database()
            
            # Convert Pydantic model to dict
            property_dict = property_data.dict()
            property_dict["created_at"] = datetime.now()
            property_dict["updated_at"] = datetime.now()
            
            # Insert into MongoDB
            result = await db[self.collection_name].insert_one(property_dict)
            
            # Fetch the created document
            created_doc = await db[self.collection_name].find_one({"_id": result.inserted_id})
            
            print(f"✅ Created property with ID: {result.inserted_id}")
            return document_to_dict(created_doc)
            
        except Exception as e:
            print(f"❌ Error creating property: {e}")
            raise
    
    async def update_property(self, property_id: str, property_data: BaseModel) -> Optional[Dict[str, Any]]:
        """Update property"""
        try:
            db = await get_database()
            
            # Prepare update data
            update_data = property_data.dict(exclude_unset=True)
            update_data["updated_at"] = datetime.now()
            
            # Try ObjectId first
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
                print(f"✅ Updated property: {property_id}")
                return document_to_dict(result)
            
            return None
            
        except Exception as e:
            print(f"❌ Error updating property: {e}")
            return None
    
    async def delete_property(self, property_id: str) -> bool:
        """Delete property"""
        try:
            db = await get_database()
            
            # Try ObjectId first
            try:
                obj_id = ObjectId(property_id)
                result = await db[self.collection_name].delete_one({"_id": obj_id})
            except:
                result = await db[self.collection_name].delete_one({"id": property_id})
            
            if result.deleted_count > 0:
                print(f"✅ Deleted property: {property_id}")
                return True
            
            return False
            
        except Exception as e:
            print(f"❌ Error deleting property: {e}")
            return False

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

# Create instance
property_crud = PropertyCRUD()