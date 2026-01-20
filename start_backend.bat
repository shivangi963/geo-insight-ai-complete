@echo off
echo ========================================
echo  GeoInsight AI - Starting Backend API
echo ========================================
echo.

REM Check if MongoDB is running
echo Checking MongoDB...
sc query MongoDB | find "RUNNING" >nul
if %errorlevel% neq 0 (
    echo [!] MongoDB is not running
    echo [*] Starting MongoDB...
    net start MongoDB
    if %errorlevel% neq 0 (
        echo [X] Failed to start MongoDB. Please start it manually.
        echo     Run as Administrator: net start MongoDB
        pause
        exit /b 1
    )
)
echo [OK] MongoDB is running
echo.

REM Activate virtual environment
if exist venv\Scripts\activate.bat (
    echo [*] Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo [!] Virtual environment not found. Using system Python.
)

REM Change to backend directory
cd backend

REM Test MongoDB connection
echo [*] Testing database connection...
python simple_mongodb_test.py
if %errorlevel% neq 0 (
    echo [X] Database connection failed. Please check MongoDB.
    pause
    exit /b 1
)

echo.
echo ========================================
echo  Starting FastAPI Server
echo ========================================
echo.
echo API will be available at:
echo   http://localhost:8000
echo   http://localhost:8000/docs
echo.
echo Press Ctrl+C to stop the server
echo ========================================
echo.

REM Start the API server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

pause