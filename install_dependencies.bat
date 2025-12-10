@echo off
title VTU Results Automation - Dependency Installer

echo ======================================================
echo VTU Results Automation System - Dependency Installer
echo ======================================================
echo.
echo This script will install all required dependencies for the VTU Results Automation system.
echo You will be asked for permission before each dependency is installed.
echo.
echo Press any key to continue...
pause >nul
echo.

:: Check if Python is installed
echo Checking for Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH.
    echo Please install Python 3.7 or later from https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
) else (
    echo Python is installed.
    python --version
)
echo.

:: Check if pip is installed and update it
echo Checking for pip installation...
pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: pip is not installed.
    echo Please install pip or reinstall Python with pip option enabled.
    echo.
    pause
    exit /b 1
) else (
    echo pip is installed.
    pip --version
    echo.
    echo Updating pip to the latest version...
    python -m pip install --upgrade pip
    echo pip updated successfully.
)
echo.

echo Installing dependencies from requirements.txt...
echo The following packages will be installed:
echo ----------------------------------------
type requirements.txt
echo ----------------------------------------
echo.

:: Ask for permission to install Flask
echo 1. Installing Flask web framework...
choice /C YN /M "Do you want to install Flask"
if errorlevel 2 (
    echo Skipping Flask installation.
) else (
    echo Installing Flask...
    pip install flask
    if %errorlevel% neq 0 (
        echo Trying alternative Flask installation...
        pip install flask==2.3.3
        if %errorlevel% neq 0 (
            echo WARNING: Failed to install Flask. Continuing anyway...
        ) else (
            echo Flask installed successfully.
        )
    ) else (
        echo Flask installed successfully.
    )
)
echo.

:: Ask for permission to install Flask-SocketIO
echo 2. Installing Flask-SocketIO...
choice /C YN /M "Do you want to install Flask-SocketIO"
if errorlevel 2 (
    echo Skipping Flask-SocketIO installation.
) else (
    echo Installing Flask-SocketIO...
    pip install flask-socketio
    if %errorlevel% neq 0 (
        echo Trying alternative Flask-SocketIO installation...
        pip install flask-socketio==5.3.6
        if %errorlevel% neq 0 (
            echo WARNING: Failed to install Flask-SocketIO. Continuing anyway...
        ) else (
            echo Flask-SocketIO installed successfully.
        )
    ) else (
        echo Flask-SocketIO installed successfully.
    )
)
echo.

:: Ask for permission to install Playwright
echo 3. Installing Playwright browser automation...
choice /C YN /M "Do you want to install Playwright"
if errorlevel 2 (
    echo Skipping Playwright installation.
) else (
    echo Installing Playwright...
    pip install playwright
    if %errorlevel% neq 0 (
        echo Trying alternative Playwright installation...
        pip install playwright==1.40.0
        if %errorlevel% neq 0 (
            echo WARNING: Failed to install Playwright. Continuing anyway...
        ) else (
            echo Playwright installed successfully.
        )
    ) else (
        echo Playwright installed successfully.
    )
)
echo.

:: Ask for permission to install Pillow
echo 4. Installing Pillow (Python Imaging Library)...
choice /C YN /M "Do you want to install Pillow"
if errorlevel 2 (
    echo Skipping Pillow installation.
) else (
    echo Installing Pillow...
    pip install pillow
    if %errorlevel% neq 0 (
        echo Trying alternative Pillow installation...
        pip install pillow==10.1.0
        if %errorlevel% neq 0 (
            echo WARNING: Failed to install Pillow. Continuing anyway...
        ) else (
            echo Pillow installed successfully.
        )
    ) else (
        echo Pillow installed successfully.
    )
)
echo.

:: Ask for permission to install python-dotenv
echo 5. Installing python-dotenv...
choice /C YN /M "Do you want to install python-dotenv"
if errorlevel 2 (
    echo Skipping python-dotenv installation.
) else (
    echo Installing python-dotenv...
    pip install python-dotenv
    if %errorlevel% neq 0 (
        echo Trying alternative python-dotenv installation...
        pip install python-dotenv==1.0.0
        if %errorlevel% neq 0 (
            echo WARNING: Failed to install python-dotenv. Continuing anyway...
        ) else (
            echo python-dotenv installed successfully.
        )
    ) else (
        echo python-dotenv installed successfully.
    )
)
echo.

:: Ask for permission to install pathlib2
echo 6. Installing pathlib2...
choice /C YN /M "Do you want to install pathlib2"
if errorlevel 2 (
    echo Skipping pathlib2 installation.
) else (
    echo Installing pathlib2...
    pip install pathlib2
    if %errorlevel% neq 0 (
        echo Trying alternative pathlib2 installation...
        pip install pathlib2==2.3.7
        if %errorlevel% neq 0 (
            echo WARNING: Failed to install pathlib2. Continuing anyway...
        ) else (
            echo pathlib2 installed successfully.
        )
    ) else (
        echo pathlib2 installed successfully.
    )
)
echo.

:: Ask for permission to install requests
echo 7. Installing requests...
choice /C YN /M "Do you want to install requests"
if errorlevel 2 (
    echo Skipping requests installation.
) else (
    echo Installing requests...
    pip install requests
    if %errorlevel% neq 0 (
        echo Trying alternative requests installation...
        pip install requests==2.31.0
        if %errorlevel% neq 0 (
            echo WARNING: Failed to install requests. Continuing anyway...
        ) else (
            echo requests installed successfully.
        )
    ) else (
        echo requests installed successfully.
    )
)
echo.

:: Ask for permission to install PyTorch
echo 8. Installing PyTorch and TorchVision...
choice /C YN /M "Do you want to install PyTorch and TorchVision"
if errorlevel 2 (
    echo Skipping PyTorch installation.
) else (
    echo Installing PyTorch and TorchVision...
    pip install torch torchvision
    if %errorlevel% neq 0 (
        echo Trying alternative PyTorch installation...
        pip install torch==2.1.0 torchvision==0.16.0
        if %errorlevel% neq 0 (
            echo WARNING: Failed to install PyTorch. Continuing anyway...
        ) else (
            echo PyTorch and TorchVision installed successfully.
        )
    ) else (
        echo PyTorch and TorchVision installed successfully.
    )
)
echo.

:: Install Playwright browsers
echo Installing Playwright browsers (Chromium)...
choice /C YN /M "Do you want to install Playwright browsers (required for automation)"
if errorlevel 2 (
    echo Skipping Playwright browsers installation.
    echo NOTE: Without browsers, the automation will not work.
) else (
    echo Installing Playwright browsers...
    python -m playwright install-deps
    python -m playwright install chromium
    if %errorlevel% neq 0 (
        echo WARNING: Failed to install Playwright browsers. Continuing anyway...
    ) else (
        echo Playwright browsers installed successfully.
    )
)
echo.

echo ======================================================
echo Installation process completed!
echo ======================================================
echo.
echo NOTE: Some packages may have failed to install, but the process continued.
echo If you experience issues, try running this installer again.
echo.
echo To launch the application, double-click 'launch_app.bat'
echo Then open your browser and go to http://localhost:5000
echo.
pause