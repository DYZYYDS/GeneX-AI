@echo off
echo =======================================================
echo     XenoGenesis OpenSource Environment Installer (Windows)
echo =======================================================
echo.

python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Please install Python 3.10 or higher from https://www.python.org/
    pause
    exit /b
)

echo [INFO] Creating virtual environment...
python -m venv venv
if %errorlevel% neq 0 (
    echo [ERROR] Failed to create virtual environment.
    pause
    exit /b
)

echo [INFO] Activating virtual environment and installing dependencies...
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt

echo.
echo =======================================================
echo [SUCCESS] Environment setup complete!
echo.
echo To start the system, run:
echo   1. venv\Scripts\activate
echo   2. python run_gene_research.py
echo =======================================================
pause
