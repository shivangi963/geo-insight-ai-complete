@echo off
echo ========================================
echo  GeoInsight AI - Starting Frontend UI
echo ========================================
echo.

REM Activate virtual environment
if exist venv\Scripts\activate.bat (
    echo [*] Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo [!] Virtual environment not found. Using system Python.
)

REM Check if backend is running
echo [*] Checking if backend API is running...
curl -s http://localhost:8000/health >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo [!] WARNING: Backend API is not running!
    echo [!] Please start the backend first using start_backend.bat
    echo.
    echo Press any key to continue anyway or Ctrl+C to exit...
    pause >nul
)

echo.
echo ========================================
echo  Starting Streamlit Dashboard
echo ========================================
echo.
echo Dashboard will open automatically at:
echo   http://localhost:8501
echo.
echo Press Ctrl+C to stop the dashboard
echo ========================================
echo.

REM Change to ui directory
cd ui

REM Start Streamlit
streamlit run streamlit_app.py

pause