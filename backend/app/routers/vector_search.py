"""
Vector Search Router
Extracted from main.py - Handles vector similarity search
"""
from fastapi import APIRouter, HTTPException, UploadFile, File, Query, BackgroundTasks
from typing import Dict, Any, Optional
import logging
from datetime import datetime
import tempfile
import os
import asyncio

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/vector", tags=["vector-search"])

# Import vector DB
vector_db = None
VECTOR_DB_AVAILABLE = False

try:
    from ..supabase_client import vector_db as imported_vector_db
    
    if imported_vector_db and getattr(imported_vector_db, 'enabled', False):
        vector_db = imported_vector_db
        VECTOR_DB_AVAILABLE = True
        logger.info("✅ Vector database available and enabled")
    else:
        logger.warning("⚠️ Vector database not enabled")
except ImportError as e:
    logger.warning(f"Vector database import failed: {e}")


@router.post("/search")
async def search_similar_properties(
    file: UploadFile = File(...),
    limit: int = Query(3, ge=1, le=20),
    threshold: float = Query(0.7, ge=0.0, le=1.0)
):
    """
    Search for visually similar properties using image embeddings
    
    Args:
        file: Query image
        limit: Maximum number of results
        threshold: Similarity threshold (0-1)
    
    Returns:
        List of similar properties with similarity scores
    """
    try:
        logger.info(f"🔍 Vector search request: limit={limit}, threshold={threshold}")
        
        # Check if vector DB is available
        if not VECTOR_DB_AVAILABLE or vector_db is None:
            raise HTTPException(
                status_code=503,
                detail="Vector database not available. Check SUPABASE configuration."
            )
        
        # Validate file exists
        if file is None:
            logger.error("No file uploaded for vector search")
            raise HTTPException(
                status_code=400,
                detail="No image file provided. Please upload an image to search."
            )
        
        # Validate content_type
        if not hasattr(file, 'content_type') or file.content_type is None:
            logger.error(f"File upload missing content_type. File: {file}")
            raise HTTPException(
                status_code=400,
                detail="Invalid file upload. Please ensure you're uploading a valid image file."
            )
        
        # Check if it's an image
        if not file.content_type.startswith('image/'):
            logger.warning(f"Invalid file type for vector search: {file.content_type}")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type: {file.content_type}. Please upload an image."
            )
        
        logger.info(f"✅ Valid query image: {file.filename}, type: {file.content_type}")
        
        # Read image data
        image_data = await file.read()
        
        # Validate size
        max_size = 10 * 1024 * 1024  # 10MB
        if len(image_data) > max_size:
            raise HTTPException(
                status_code=400,
                detail="Image file too large. Maximum size is 10MB."
            )
        
        # Save image temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            temp_file.write(image_data)
            temp_path = temp_file.name
        
        try:
            # Use vector_db to find similar properties
            results = await asyncio.to_thread(
                vector_db.find_similar_properties,
                image_path=temp_path,
                limit=limit,
                threshold=threshold
            )
            
            logger.info(f"✅ Found {len(results)} similar properties")
            
            return {
                "status": "success",
                "query_image": file.filename,
                "results": results,
                "total_results": len(results),
                "threshold": threshold
            }
        
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_path)
            except Exception as e:
                logger.warning(f"Failed to delete temp file {temp_path}: {e}")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Vector search error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Vector search failed: {str(e)}"
        )


@router.post("/store")
async def store_property_vector(payload: Dict[str, Any]):
    """
    Store property embedding in vector database
    
    Request body:
        {
            "property_id": "prop_123",
            "address": "123 Main St",
            "image_path": "/path/to/image.jpg",
            "metadata": {"price": 500000, "beds": 3}
        }
    """
    try:
        if not VECTOR_DB_AVAILABLE or vector_db is None:
            raise HTTPException(
                status_code=503, 
                detail="Vector database not available"
            )
        
        property_id = payload.get("property_id")
        address = payload.get("address")
        image_path = payload.get("image_path")
        metadata = payload.get("metadata")
        
        # Validate required fields
        if not all([property_id, address, image_path]):
            raise HTTPException(
                status_code=400,
                detail="property_id, address, and image_path are required"
            )
        
        # Validate image exists
        if not os.path.exists(image_path):
            raise HTTPException(
                status_code=404, 
                detail=f"Image not found: {image_path}"
            )

        # Store embedding
        success = vector_db.store_property_embedding(
            property_id=property_id,
            address=address,
            image_path=image_path,
            metadata=metadata
        )

        if not success:
            raise HTTPException(
                status_code=500, 
                detail="Failed to store embedding"
            )

        return {
            "success": True,
            "property_id": property_id,
            "message": "Property embedding stored",
            "timestamp": datetime.now().isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Vector store error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/property/{property_id}")
async def get_property_vector(property_id: str):
    """Get property vector by ID"""
    try:
        if not VECTOR_DB_AVAILABLE or vector_db is None:
            raise HTTPException(
                status_code=503, 
                detail="Vector database not available"
            )
        
        property_data = vector_db.get_property_by_id(property_id)
        
        if not property_data:
            raise HTTPException(
                status_code=404, 
                detail=f"Property {property_id} not found"
            )
        
        return {
            "property_id": property_data.get("property_id"),
            "address": property_data.get("address"),
            "metadata": property_data.get("metadata"),
            "has_embedding": bool(property_data.get("embedding")),
            "embedding_dimension": len(property_data.get("embedding", [])),
            "created_at": property_data.get("created_at")
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Vector get error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/property/{property_id}")
async def delete_property_vector(property_id: str):
    """Delete property vector"""
    try:
        if not VECTOR_DB_AVAILABLE or vector_db is None:
            raise HTTPException(
                status_code=503, 
                detail="Vector database not available"
            )
        
        success = vector_db.delete_property(property_id)
        
        if not success:
            raise HTTPException(
                status_code=404, 
                detail=f"Property {property_id} not found"
            )
        
        return {
            "success": True,
            "property_id": property_id,
            "message": "Property embedding deleted",
            "timestamp": datetime.now().isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Vector delete error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_vector_stats():
    """Get vector DB statistics"""
    try:
        if not VECTOR_DB_AVAILABLE or vector_db is None:
            raise HTTPException(
                status_code=503, 
                detail="Vector database not available"
            )
        
        stats = vector_db.get_statistics()
        return {
            **stats,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Vector stats error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch-store")
async def batch_store_vectors(
    background_tasks: BackgroundTasks,
    limit: int = Query(100, ge=1, le=1000)
):
    """Batch store property vectors"""
    try:
        if not VECTOR_DB_AVAILABLE or vector_db is None:
            raise HTTPException(
                status_code=503, 
                detail="Vector database not available"
            )
        
        from ..crud import property_crud
        
        # Get properties
        properties = await property_crud.get_all_properties(limit=limit)
        
        if not properties:
            return {
                "message": "No properties found",
                "processed": 0
            }
        
        # Start background task
        task_id = f"batch_vector_{int(datetime.now().timestamp())}"
        
        async def process_batch():
            """Background task to process properties"""
            processed = 0
            errors = 0
            
            for prop in properties:
                try:
                    property_id = prop.get('id')
                    address = prop.get('address')
                    
                    # Check for image
                    image_path = os.path.join("uploads", f"{property_id}.jpg")
                    
                    if os.path.exists(image_path):
                        success = vector_db.store_property_embedding(
                            property_id=property_id,
                            address=address,
                            image_path=image_path,
                            metadata={
                                "price": prop.get("price"),
                                "bedrooms": prop.get("bedrooms"),
                                "city": prop.get("city")
                            }
                        )
                        
                        if success:
                            processed += 1
                
                except Exception as e:
                    errors += 1
                    logger.error(f"Error processing {property_id}: {e}")
                    continue
            
            logger.info(f"Batch complete: {processed} processed, {errors} errors")
        
        background_tasks.add_task(process_batch)
        
        return {
            "task_id": task_id,
            "status": "processing",
            "total_properties": len(properties),
            "message": "Batch processing started",
            "timestamp": datetime.now().isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch vector error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))