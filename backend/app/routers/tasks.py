"""
Tasks Router
Extracted from main.py - Handles async task status tracking
"""
from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/tasks", tags=["tasks"])

# Check Celery availability
CELERY_AVAILABLE = False
try:
    from celery.result import AsyncResult
    from celery_config import celery_app
    CELERY_AVAILABLE = True
    logger.info("✅ Celery available for task tracking")
except ImportError:
    logger.info("⚠️ Celery not available - using in-memory task store")

# Import CRUD for neighborhood analysis
try:
    from ..crud import get_neighborhood_analysis
except ImportError:
    async def get_neighborhood_analysis(analysis_id: str):
        return None


@router.get("/{task_id}")
async def get_task_status(task_id: str):
    """
    Get task status with proper fallback chain
    
    Supports:
    - Celery tasks (if available)
    - Background analysis tasks (analysis_*)
    - In-memory task store
    
    Returns:
        {
            "task_id": str,
            "status": str,  # pending, processing, completed, failed
            "progress": int,  # 0-100
            "message": str,
            "result": dict,  # if completed
            "error": str  # if failed
        }
    """
    logger.info(f"Checking status for task: {task_id}")
    
    # Strategy 1: Check if it's a background task (analysis_*)
    if task_id.startswith("analysis_"):
        analysis_id = task_id.replace("analysis_", "")
        logger.info(f"Background task detected, checking analysis: {analysis_id}")
        
        try:
            analysis = await get_neighborhood_analysis(analysis_id)
            if analysis:
                status = analysis.get('status', 'unknown')
                progress = analysis.get('progress', 0)
                
                # Map analysis status to task status
                task_status = status
                if status == 'processing':
                    task_status = 'processing'
                elif status == 'completed':
                    task_status = 'completed'
                elif status == 'failed':
                    task_status = 'failed'
                elif status == 'pending':
                    task_status = 'pending'
                
                return {
                    'task_id': task_id,
                    'analysis_id': analysis_id,
                    'status': task_status,
                    'progress': progress,
                    'message': analysis.get('message', f'Analysis {status}'),
                    'result': analysis if status == 'completed' else None,
                    'error': analysis.get('error'),
                    'address': analysis.get('address'),
                    'walk_score': analysis.get('walk_score'),
                    'total_amenities': analysis.get('total_amenities', 0)
                }
        except Exception as e:
            logger.error(f"Error fetching analysis {analysis_id}: {e}")
            # Don't return here, try Celery next
    
    # Strategy 2: Check Celery if available
    if CELERY_AVAILABLE:
        try:
            celery_task = AsyncResult(task_id, app=celery_app)
            state = celery_task.state
            
            logger.info(f"Celery task state: {state}")
            
            # Map Celery states to our status
            status_map = {
                'PENDING': 'pending',
                'STARTED': 'processing', 
                'PROGRESS': 'processing',
                'SUCCESS': 'completed',
                'FAILURE': 'failed',
                'RETRY': 'processing',
                'REVOKED': 'failed'
            }
            
            task_status = status_map.get(state, state.lower())
            
            # Get progress from Celery meta
            progress = 0
            if state == 'PROGRESS' and celery_task.info:
                progress = celery_task.info.get('progress', 50)
            elif state == 'SUCCESS':
                progress = 100
            
            result_data = None
            error_msg = None

            if state == 'SUCCESS':
                result_data = celery_task.result
            elif state == 'FAILURE':
                error_msg = str(celery_task.info) if celery_task.info else 'Task failed'

            return {
                'task_id': task_id,
                'status': task_status,
                'progress': progress,
                'message': str(celery_task.info) if celery_task.info else f'Task {state}',
                'result': result_data,
                'error': error_msg
            }
        except Exception as e:
            logger.error(f"Celery lookup failed for {task_id}: {e}")
            # Don't return here, check analysis by task_id next
    
    # Strategy 3: Check if task_id is actually an analysis_id
    try:
        analysis = await get_neighborhood_analysis(task_id)
        if analysis:
            logger.info(f"Found analysis using task_id as analysis_id")
            status = analysis.get('status', 'unknown')
            
            return {
                'task_id': task_id,
                'analysis_id': task_id,
                'status': status,
                'progress': analysis.get('progress', 0),
                'message': analysis.get('message', f'Analysis {status}'),
                'result': analysis if status == 'completed' else None,
                'error': analysis.get('error')
            }
    except Exception as e:
        logger.error(f"Error checking task_id as analysis_id: {e}")
    
    # Strategy 4: If nothing found, return helpful error
    logger.warning(f"Task {task_id} not found in any system")
    
    raise HTTPException(
        status_code=404,
        detail={
            "error": "Task not found",
            "task_id": task_id,
            "message": "Task may have expired or never existed",
            "troubleshooting": {
                "celery_available": CELERY_AVAILABLE,
                "suggestions": [
                    "Check if the task was created successfully",
                    "Task results expire after 1 hour",
                    "Check backend logs for errors"
                ]
            }
        }
    )