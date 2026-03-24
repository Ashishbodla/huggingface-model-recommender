@echo off
REM setup.bat — One-command setup for HuggingFace Model Recommender (Windows)
echo ==============================
echo  HF Model Recommender Setup
echo ==============================
echo.

REM Check Python
where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: python not found. Install Python 3.8+ from https://python.org
    echo Make sure to check "Add Python to PATH" during installation.
    exit /b 1
)

for /f "tokens=*" %%i in ('python --version') do set PYVER=%%i
echo Using Python: %PYVER%
echo.

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    echo Virtual environment created.
) else (
    echo Virtual environment already exists.
)

REM Activate
call venv\Scripts\activate.bat
echo Activated virtual environment.
echo.

REM Install dependencies
echo Installing dependencies...
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo Core dependencies installed.

REM Optional: try installing torch
echo.
echo Installing PyTorch (for GPU detection)...
pip install torch -q 2>nul && echo PyTorch installed. || echo PyTorch install failed (optional — tool will default to CPU mode).

echo.
echo ==============================
echo  Setup complete!
echo ==============================
echo.
echo To run the tool:
echo   venv\Scripts\activate
echo   python recommender.py
echo.
echo Or with filters:
echo   python recommender.py --quant gguf --top 20
echo.
echo See README.md for full instructions.
