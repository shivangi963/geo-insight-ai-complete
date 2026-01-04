@echo off
echo Starting Celery worker...
echo.

REM Activate virtual environment
call .venv\Scripts\activate.bat

REM Start Celery worker
celery -A celery_config.celery_app worker --loglevel=info --pool=solo

pause