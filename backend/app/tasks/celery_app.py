from celery import Celery
from app.config import settings

celery_app = Celery(
    "aitrace_tasks",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    # Explicitly register task modules so the worker imports them on startup
    include=[
        "app.tasks.vision_tasks",
        "app.tasks.text_tasks",
    ],
    # Partition workers: route heavy GPU tasks vs CPU math ops
    task_routes={
        "app.tasks.vision_tasks.*": {"queue": "queue_cpu"},
        "app.tasks.text_tasks.*": {"queue": "queue_cpu"},
    },
    task_track_started=True,
)