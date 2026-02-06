"""
Property Management Router
Extracted from main.py for better organization
"""
from fastapi import APIRouter, HTTPException, Query, Request, Depends
from typing import List, Optional
import logging
from ..crud import property_crud
from ..models import PropertyCreate, PropertyUpdate, PropertyResponse
from ..database import Database

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/properties", tags=["properties"])


@router.get("/raw")
async def get_properties_raw():
    """Get properties RAW without model validation - DEBUG ONLY"""
    try:
        logger.info("🔍 Raw properties endpoint called")
        
        from backend.app.database import get_database
        db = await get_database()
        
        logger.info(f"Connected to database: {db.name}")
        
        count = await db["properties"].count_documents({})
        logger.info(f"Property count in DB: {count}")
        
        if count == 0:
            logger.warning("⚠️ No properties found in database")
            return []
        
        cursor = db["properties"].find().limit(100)
        properties = []
        async for doc in cursor:
            if "_id" in doc:
                doc["id"] = str(doc["_id"])
                del doc["_id"]
            properties.append(doc)
        
        logger.info(f"Returned {len(properties)} properties")
        return properties
    except Exception as e:
        logger.error(f"Failed to get properties: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("", response_model=List[PropertyResponse])
async def get_properties(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    city: Optional[str] = None
):
    """Get properties with filtering"""
    try:
        logger.info(f"/api/properties called - skip:{skip}, limit:{limit}, city:{city}")
        
        properties = await property_crud.get_all_properties(skip=skip, limit=limit)
        logger.info(f"   CRUD returned {len(properties)} properties")
        
        # Filter by city if specified
        if city:
            properties = [p for p in properties if p.get('city', '').lower() == city.lower()]
            logger.info(f"   After city filter: {len(properties)} properties")
        
        # Validate each property
        valid_props = []
        for p in properties:
            try:
                validated = PropertyResponse.model_validate(p)
                valid_props.append(validated)
            except Exception as ve:
                logger.warning(f"Property validation failed (id={p.get('id')}): {ve}")

        logger.info(f"✅ Validation: {len(valid_props)}/{len(properties)} passed")
        
        return valid_props
        
    except Exception as e:
        logger.error(f"Failed to get properties: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve properties")


@router.post("", response_model=PropertyResponse, status_code=201)
async def create_property(property: PropertyCreate):
    """Create new property"""
    try:
        new_property = await property_crud.create_property(property)
        logger.info(f"Created property: {new_property.get('id')}")
        return new_property
    except Exception as e:
        logger.error(f"Failed to create property: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{property_id}", response_model=PropertyResponse)
async def get_property(property_id: str):
    """Get single property by ID"""
    try:
        property = await property_crud.get_property_by_id(property_id)
        if not property:
            raise HTTPException(status_code=404, detail="Property not found")
        return property
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get property {property_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{property_id}", response_model=PropertyResponse)
async def update_property(property_id: str, property_update: PropertyUpdate):
    """Update property"""
    try:
        updated = await property_crud.update_property(property_id, property_update)
        if not updated:
            raise HTTPException(status_code=404, detail="Property not found")
        return updated
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update property {property_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{property_id}")
async def delete_property(property_id: str):
    """Delete property"""
    try:
        deleted = await property_crud.delete_property(property_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Property not found")
        return {"message": "Property deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete property {property_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))