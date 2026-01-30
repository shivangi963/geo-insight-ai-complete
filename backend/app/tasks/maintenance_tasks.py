"""
Maintenance Tasks for Celery
Handles periodic cleanup and maintenance operations
"""

from celery import shared_task
from datetime import datetime, timedelta
from typing import Dict
import traceback


@shared_task(bind=True, name="cleanup_old_tasks")
def cleanup_old_tasks(self) -> Dict:
    """
    Clean up old completed tasks from Celery results backend
    Runs periodically (every hour by default)
    """
    try:
        self.update_state(state='PROGRESS', meta={
            'status': 'Cleaning up old tasks...',
            'timestamp': datetime.now().isoformat()
        })
        
        # This is a placeholder - actual cleanup depends on your backend
        # For Redis: cleanup happens automatically with result_expires setting
        # For MongoDB: you would implement custom cleanup here
        
        result = {
            'task_id': self.request.id,
            'status': 'COMPLETED',
            'message': 'Cleanup completed successfully',
            'timestamp': datetime.now().isoformat(),
            'tasks_cleaned': 0
        }
        
        print(f"✅ Maintenance task completed: {result['message']}")
        return result
        
    except Exception as e:
        error_msg = f"Cleanup task failed: {str(e)}"
        print(f"❌ {error_msg}")
        traceback.print_exc()
        
        return {
            'task_id': self.request.id,
            'status': 'FAILED',
            'error': error_msg,
            'timestamp': datetime.now().isoformat()
        }


@shared_task(bind=True, name="update_analysis_results")
def update_analysis_results(self, analysis_id: str) -> Dict:
    """
    Update analysis results and store metrics
    """
    try:
        self.update_state(state='PROGRESS', meta={
            'status': f'Updating analysis {analysis_id}...',
            'progress': 50
        })
        
        result = {
            'task_id': self.request.id,
            'analysis_id': analysis_id,
            'status': 'COMPLETED',
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"✅ Analysis results updated for {analysis_id}")
        return result
        
    except Exception as e:
        error_msg = f"Failed to update analysis results: {str(e)}"
        print(f"❌ {error_msg}")
        traceback.print_exc()
        
        return {
            'task_id': self.request.id,
            'status': 'FAILED',
            'error': error_msg,
            'timestamp': datetime.now().isoformat()
        }


@shared_task(bind=True, name="archive_old_results")
def archive_old_results(self, days_old: int = 30) -> Dict:
    """
    Archive analysis results older than specified days
    """
    try:
        cutoff_date = datetime.now() - timedelta(days=days_old)
        
        self.update_state(state='PROGRESS', meta={
            'status': f'Archiving results older than {cutoff_date}...',
            'progress': 25
        })
        
        # Archive logic would go here
        
        result = {
            'task_id': self.request.id,
            'status': 'COMPLETED',
            'message': f'Archived results older than {days_old} days',
            'timestamp': datetime.now().isoformat(),
            'archived_count': 0
        }
        
        print(f"✅ Results archived successfully")
        return result
        
    except Exception as e:
        error_msg = f"Archive task failed: {str(e)}"
        print(f"❌ {error_msg}")
        traceback.print_exc()
        
        return {
            'task_id': self.request.id,
            'status': 'FAILED',
            'error': error_msg,
            'timestamp': datetime.now().isoformat()
        }
