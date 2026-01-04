import os
from celery import Celery
from dotenv import load_dotenv

load_dotenv()

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "geo_insight_tasks",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=[
        "app.tasks.computer_vision_tasks",
        "app.tasks.geospatial_tasks",
        "app.tasks.agent_tasks"
    ]
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    
 
    task_track_started=True,
    task_time_limit=30 * 60,  
    task_soft_time_limit=25 * 60,  
    
    result_expires=3600, 
    
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=100,
    
    beat_schedule={
        'cleanup-old-tasks': {
            'task': 'app.tasks.maintenance_tasks.cleanup_old_tasks',
            'schedule': 3600.0, 
        },
    }
)