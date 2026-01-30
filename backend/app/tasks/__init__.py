"""
Celery tasks package
Task definitions for background job processing
"""

from .agent_tasks import process_agent_query_task
from .computer_vision_tasks import analyze_street_image_task
from .geospatial_tasks import analyze_neighborhood_task
from .maintenance_tasks import (
    cleanup_old_tasks, 
    update_analysis_results, 
    archive_old_results
)

__all__ = [
    'process_agent_query_task',
    'analyze_street_image_task',
    'analyze_neighborhood_task',
    'cleanup_old_tasks',
    'update_analysis_results',
    'archive_old_results'
]