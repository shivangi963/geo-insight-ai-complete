@echo off
REM ============================================
REM GeoInsight AI - Simple Startup
REM Run everything with ONE command
REM ============================================

echo.
echo ========================================
echo   GeoInsight AI - Starting
echo ========================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found
    echo Install Python 3.10+ from python.org
    pause
    exit /b 1
)

REM Check/Create venv
if not exist "venv\" (
    echo [1/5] Creating virtual environment...
    python -m venv venv
)
echo [1/5] Virtual environment ready

REM Activate venv
call venv\Scripts\activate.bat
echo [2/5] Environment activated

REM Install dependencies (if needed)
python -c "import fastapi" >nul 2>&1
if errorlevel 1 (
    echo [3/5] Installing dependencies...
    echo This takes 2-5 minutes (first time only)
    pip install --upgrade pip
    pip install fastapi uvicorn pydantic python-dotenv
    pip install motor pymongo aiosqlite
    pip install ultralytics opencv-python Pillow numpy
    pip install streamlit plotly pandas requests
    pip install geopy osmnx folium shapely
    pip install google-generativeai
    pip install celery redis flower
) else (
    echo [3/5] Dependencies already installed
)

REM Check .env
if not exist "backend\.env" (
    echo [4/5] Creating .env template...
    (
        echo # GeoInsight AI Configuration
        echo GEMINI_API_KEY=your_key_here
        echo MONGODB_URL=mongodb://localhost:27017
        echo DATABASE_NAME=geoinsight_ai
        echo REDIS_URL=redis://localhost:6379/0
    ) > backend\.env
    echo.
    echo WARNING: Edit backend\.env and add your GEMINI_API_KEY
    echo Get free key from: https://makersuite.google.com/app/apikey
    echo.
) else (
    echo [4/5] Configuration file found
)

REM Create directories
if not exist "backend\data" mkdir backend\data
if not exist "backend\results" mkdir backend\results
if not exist "backend\maps" mkdir backend\maps
if not exist "backend\temp" mkdir backend\temp
echo [5/5] Directories ready

REM Start Backend
echo.
echo Starting Backend API...
start "GeoInsight Backend" cmd /k "cd backend && ..\venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000"

REM Wait for backend
timeout /t 5 /nobreak >nul

REM Start Frontend
echo Starting Frontend Dashboard...
start "GeoInsight Frontend" cmd /k "..\venv\Scripts\python.exe -m streamlit run streamlit_app.py" --server.port 8501

echo.
echo ========================================
echo   GeoInsight AI Started!
echo ========================================
echo.
echo   Backend:   http://localhost:8000
echo   API Docs:  http://localhost:8000/docs
echo   Dashboard: http://localhost:8501
echo.
echo Two windows opened - don't close them!
echo.
echo Optional: Start Celery for async tasks
echo   cd backend
echo   celery -A celery_config.celery_app worker --loglevel=info --pool=solo
echo.
echo Press any key to exit this window
pause >nul