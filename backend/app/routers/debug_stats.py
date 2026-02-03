"""
Debug & Stats Router
Extracted from main.py - System information and debugging endpoints
"""
from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import logging
from datetime import datetime
import os

logger = logging.getLogger(__name__)

router = APIRouter(tags=["debug-stats"])


@router.get("/api/stats")
async def get_stats():
    """
    Get system statistics
    
    Returns:
        {
            "total_properties": int,
            "total_analyses": int,
            "unique_cities": int,
            "average_price": float,
            "system_status": str,
            "uptime": str,
            "timestamp": str
        }
    """
    try:
        from ..crud import property_crud, get_analysis_count
        from ..database import Database
        
        analysis_count = await get_analysis_count()
        properties = await property_crud.get_all_properties(limit=1000)
        
        total_properties = len(properties)
        
        avg_price = 0
        cities = set()
        if properties:
            prices = [p.get('price', 0) for p in properties if p.get('price')]
            avg_price = sum(prices) / len(prices) if prices else 0
            cities = {p.get('city') for p in properties if p.get('city')}
        
        # Get uptime (would need to track startup_time globally)
        uptime = "N/A"
        
        db_connected = await Database.is_connected()
        system_status = "healthy" if db_connected else "degraded"

        return {
            "total_properties": total_properties,
            "total_analyses": analysis_count,
            "unique_cities": len(cities),
            "average_price": round(avg_price, 2),
            "system_status": system_status,
            "uptime": uptime,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to retrieve statistics: {str(e)}"
        )


@router.get("/api/debug/db_info/summary")
async def debug_db_info_summary():
    """
    Debug endpoint: Show DB connection, properties count and samples
    
    Useful for troubleshooting database connectivity issues
    """
    try:
        from ..database import Database
        
        connected = await Database.is_connected()
        db = await Database.get_database()

        # Count properties and fetch a few samples
        try:
            props_count = await db.properties.count_documents({})
        except Exception:
            props_count = None

        samples = []
        try:
            cursor = db.properties.find().limit(5)
            async for doc in cursor:
                # Convert ObjectId to string id
                if isinstance(doc.get('_id', None), object):
                    doc['id'] = str(doc['_id'])
                    del doc['_id']
                samples.append(doc)
        except Exception:
            samples = []

        return {
            "database": os.getenv('DATABASE_NAME', 'geoinsight_ai'),
            "connected": connected,
            "properties_count": props_count,
            "sample_properties": samples
        }
    except Exception as e:
        logger.error(f"Debug DB info failed: {e}")
        return {"error": str(e)}


@router.get("/api/debug/db_info")
async def debug_db_info():
    """
    Debug endpoint: Show which MongoDB server and properties the app sees
    
    Returns detailed server information
    """
    try:
        from ..database import Database
        
        db = await Database.get_database()
        
        # Get server info if available
        server_info = None
        try:
            server_info = await db.client.admin.command('ismaster')
        except Exception:
            try:
                server_info = await db.client.server_info()
            except Exception:
                server_info = {'info': 'unavailable'}

        total = await db.properties.count_documents({})
        sample = []
        cursor = db.properties.find().limit(10)
        async for doc in cursor:
            sample.append({
                'id': str(doc.get('_id')),
                'address': doc.get('address')
            })

        return {
            'server_info': server_info,
            'database': db.name,
            'total_properties': total,
            'sample': sample
        }
    except Exception as e:
        logger.error(f"Debug DB info failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/test/direct-properties")
async def test_direct_properties():
    """
    Test endpoint - direct property access
    
    Bypasses all middleware and directly queries the database
    Useful for debugging property endpoint issues
    """
    try:
        from ..crud import property_crud
        from ..database import get_database
        
        # Direct DB query
        db = await get_database()
        count = await db.properties.count_documents({})
        
        # CRUD query
        properties = await property_crud.get_all_properties(skip=0, limit=10)
        
        return {
            "db_count": count,
            "crud_returned": len(properties),
            "first_property": properties[0] if properties else None,
            "all_properties": properties
        }
    except Exception as e:
        logger.error(f"Test endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/debug/verify-imports")
async def verify_imports():
    """
    Verify what version of code is loaded
    
    Useful for debugging when changes don't seem to take effect
    """
    import inspect
    try:
        import backend.app.main as main_module
        
        # Check if we can get source
        try:
            source_snippet = "Source inspection not available"
            # Just return module info instead of source
        except:
            source_snippet = "Could not get source"
        
        return {
            "module_file": getattr(main_module, '__file__', 'Unknown'),
            "has_routers": "Using new router-based structure" if hasattr(main_module, 'app') else "Old structure",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"error": str(e)}