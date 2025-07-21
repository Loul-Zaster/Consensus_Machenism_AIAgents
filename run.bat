@echo off
echo.
echo ====================================================
echo      CONSENSUS MECHANISM AI AGENTS - DASHBOARD
echo ====================================================
echo.

REM Check if Python is available
where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [31mError: Python not found. Please install Python and try again.[0m
    pause
    exit /b 1
)

REM Check if dashboard.py exists
if not exist "dashboard.py" (
    echo [31mError: File dashboard.py not found.[0m
    pause
    exit /b 1
)

REM Run the dashboard
echo Launching Dashboard...
echo.
python run.py

REM Wait for user input before exiting
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [31mAn error occurred. Please check the error messages above.[0m
    pause
) 