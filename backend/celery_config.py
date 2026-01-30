import os
from celery import Celery
from kombu import Queue
from dotenv import load_dotenv

load_dotenv()

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# configure Celery app
celery_app = Celery(
    "geo_insight_tasks",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=[
        "app.tasks.computer_vision_tasks",
        "app.tasks.geospatial_tasks",
        "app.tasks.agent_tasks",
        "app.tasks.maintenance_tasks",
    ]
)

# Production-friendly default configuration (overridable via env)
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,

    # Task execution
    task_track_started=True,
    task_time_limit=int(os.getenv("CELERY_TASK_TIME_LIMIT", 30 * 60)),
    task_soft_time_limit=int(os.getenv("CELERY_TASK_SOFT_TIME_LIMIT", 25 * 60)),

    # Result backend
    result_expires=int(os.getenv("CELERY_RESULT_EXPIRES", 3600)),

    # Worker tuning
    worker_prefetch_multiplier=int(os.getenv("CELERY_PREFETCH_MULTIPLIER", 1)),
    worker_max_tasks_per_child=int(os.getenv("CELERY_MAX_TASKS_PER_CHILD", 100)),
    # Windows doesn't support prefork; use solo or threads instead
    worker_pool=os.getenv("CELERY_POOL", "solo"),  # solo, threads, or prefork (Linux only)
    worker_concurrency=int(os.getenv("CELERY_CONCURRENCY", 1)),
    worker_send_task_events=True,

    # Queues and routing
    task_default_queue=os.getenv("CELERY_DEFAULT_QUEUE", "default"),
    task_queues=(
        Queue("default", routing_key="task.default"),
        Queue("high_priority", routing_key="task.high"),
        Queue("cpu_bound", routing_key="task.cpu"),
        Queue("io_bound", routing_key="task.io"),
        Queue("maintenance", routing_key="task.maintenance"),
    ),
    task_routes={
        # route heavy image processing to cpu_bound
        "app.tasks.computer_vision_tasks.analyze_street_image": {"queue": "cpu_bound", "routing_key": "task.cpu"},
        "app.tasks.computer_vision_tasks.calculate_green_space": {"queue": "cpu_bound", "routing_key": "task.cpu"},
        # route geospatial analyses to io_bound
        "app.tasks.geospatial_tasks.analyze_neighborhood": {"queue": "io_bound", "routing_key": "task.io"},
        # agent tasks are lightweight - default
        "app.tasks.agent_tasks.process_agent_query": {"queue": "default", "routing_key": "task.default"},
        # maintenance tasks
        "app.tasks.maintenance_tasks.cleanup_old_tasks": {"queue": "maintenance", "routing_key": "task.maintenance"},
    },

    # Transport options for Redis
    broker_transport_options={
        "visibility_timeout": int(os.getenv("CELERY_VISIBILITY_TIMEOUT", 3600)),
    },

    # Annotations for per-task settings
    task_annotations={
        "*": {"rate_limit": None},
        "app.tasks.computer_vision_tasks.analyze_street_image": {"rate_limit": None, "acks_late": True},
    },

    beat_schedule={
        "cleanup-old-tasks": {
            "task": "app.tasks.maintenance_tasks.cleanup_old_tasks",
            "schedule": float(os.getenv("MAINTENANCE_SCHEDULE_SECONDS", 3600.0)),
            "options": {"queue": "maintenance"}
        },
    },
)


def get_celery_queues():
    """Return configured queue names"""
    return [q.name for q in celery_app.conf.task_queues]