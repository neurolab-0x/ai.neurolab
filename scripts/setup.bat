@echo off
REM NeuroLab Setup Script for Windows
REM This script sets up the NeuroLab EEG Analysis platform

echo ==========================================
echo NeuroLab EEG Analysis Platform Setup
echo ==========================================
echo.

REM Check if Python is installed
echo Checking prerequisites...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed. Please install Python 3.8 or higher.
    exit /b 1
)
echo [OK] Python is installed

REM Create virtual environment
echo.
echo Setting up virtual environment...
if not exist "venv" (
    python -m venv venv
    echo [OK] Virtual environment created
) else (
    echo [INFO] Virtual environment already exists
)

REM Activate virtual environment
call venv\Scripts\activate.bat
echo [OK] Virtual environment activated

REM Upgrade pip
echo.
echo Upgrading pip...
python -m pip install --upgrade pip setuptools wheel
echo [OK] Pip upgraded

REM Install dependencies
echo.
echo Installing dependencies...
pip install -r requirements.txt
echo [OK] Dependencies installed

REM Create necessary directories
echo.
echo Creating project directories...
if not exist "data" mkdir data
if not exist "processed" mkdir processed
if not exist "logs" mkdir logs
if not exist "test_data" mkdir test_data
if not exist "models" mkdir models
if not exist "utils" mkdir utils
if not exist "api" mkdir api
if not exist "config" mkdir config
if not exist "preprocessing" mkdir preprocessing
if not exist "tests" mkdir tests
echo [OK] Directories created

REM Copy environment template
echo.
if not exist ".env" (
    if exist ".env.example" (
        copy .env.example .env
        echo [OK] Environment file created from template
        echo [INFO] Please edit .env file with your configuration
    ) else (
        echo [INFO] No .env.example found, skipping environment file creation
    )
) else (
    echo [INFO] .env file already exists
)

REM Check if model exists
echo.
if exist "processed\trained_model.h5" (
    echo [OK] Trained model found
) else (
    echo [INFO] No trained model found. You can train one using: python train_model.py
)

REM Summary
echo.
echo ==========================================
echo Setup Complete!
echo ==========================================
echo.
echo Next steps:
echo 1. Edit .env file with your configuration
echo 2. Train a model: python train_model.py
echo 3. Start the API server: python main.py
echo    or with Docker: docker-compose up
echo.
echo Access the API at: http://localhost:8000
echo API Documentation: http://localhost:8000/docs
echo.
echo [OK] Setup completed successfully!
echo.
pause
