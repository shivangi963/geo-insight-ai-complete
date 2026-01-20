@echo off
echo ========================================
echo  GeoInsight AI - Complete Startup
echo ========================================
echo.
echo This will start:
echo  1. MongoDB (if not running)
echo  2. Redis (Docker container)
echo  3. Backend API (Port 8000)
echo  4. Frontend Dashboard (Port 8501)
echo.
echo Press any key to continue...
pause >nul

REM Start MongoDB
echo.
echo [1/4] Starting MongoDB...
sc query MongoDB | find "RUNNING" >nul
if %errorlevel% neq 0 (
    net start MongoDB
    if %errorlevel% neq 0 (
        echo [X] Failed to start MongoDB
        echo     Run as Administrator: net start MongoDB
        pause
        exit /b 1
    )
)
echo [OK] MongoDB is running

REM Start Redis using Docker
echo.
echo [2/4] Starting Redis (Docker)...
docker ps | find "geoinsight-redis" >nul
if %errorlevel% neq 0 (
    echo Starting new Redis container...
    docker run -d --name geoinsight-redis -p 6379:6379 redis:7-alpine
    if %errorlevel% neq 0 (
        echo [!] Failed to start Redis container
        echo     Make sure Docker Desktop is running
        pause
        exit /b 1
    )
) else (
    echo [OK] Redis container already running
)

REM Test Redis connection
timeout /t 2 /nobreak >nul
docker exec geoinsight-redis redis-cli ping >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] Redis is responding
) else (
    echo [!] Redis container started but not responding yet
)

REM Start Backend in new window
echo.
echo [3/4] Starting Backend API...
start "GeoInsight AI - Backend" cmd /k "call venv\Scripts\activate.bat && cd backend && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"

REM Wait for backend to start
echo Waiting for backend to start...
timeout /t 8 /nobreak >nul

REM Start Frontend in new window
echo.
echo [4/4] Starting Frontend Dashboard...
start "GeoInsight AI - Frontend" cmd /k "call venv\Scripts\activate.bat && cd ui && streamlit run streamlit_app.py"

echo.
echo ========================================
echo  GeoInsight AI Started Successfully!
echo ========================================
echo.
echo Services:
echo   MongoDB:      Running locally
echo   Redis:        http://localhost:6379 (Docker)
echo   Backend API:  http://localhost:8000
echo   API Docs:     http://localhost:8000/docs
echo   Dashboard:    http://localhost:8501
echo.
echo Two new windows opened:
echo   - Backend API (don't close)
echo   - Frontend Dashboard (don't close)
echo.
echo To stop:
echo   - Close terminal windows
echo   - Stop Redis: docker stop geoinsight-redis
echo   - Stop MongoDB: net stop MongoDB
echo ========================================
echo.
pause