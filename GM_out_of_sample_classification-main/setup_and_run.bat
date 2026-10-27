@echo off
echo ============================================================
echo GM Out-of-Sample Classification - Python Setup and Run
echo ============================================================
echo.

echo Step 1: Installing Python dependencies...
pip install -r requirements.txt
echo.

if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

echo Step 2: Running example code...
echo.
python run_example.py

echo.
echo ============================================================
echo Done! Press any key to exit.
echo ============================================================
pause
