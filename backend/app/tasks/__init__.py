"""
Celery tasks package
"""

from celery import current_app as celery_app

__all__ = ['celery_app']