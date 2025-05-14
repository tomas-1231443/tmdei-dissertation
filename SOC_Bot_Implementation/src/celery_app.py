# celery_app.py

import os
from celery import Celery

# Configure the broker and backend to use a Redis server running locally.
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")

celery_app = Celery(__name__, broker=CELERY_BROKER_URL, backend=CELERY_RESULT_BACKEND)

# Optional: Configure Celery settings if needed, for example:
celery_app.conf.update(
    task_serializer='json',
    result_serializer='json',
    accept_content=['json']
)

celery_app.autodiscover_tasks(['src.queue'])