@echo off
echo Starting Celery beat scheduler...
echo.

REM Activate virtual environment
call .venv\Scripts\activate.bat

REM Start Celery beat
celery -A celery_config.celery_app beat --loglevel=info

pause