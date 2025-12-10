@echo off
title VTU Results Automation - Setup Wizard

echo ======================================================
echo VTU Results Automation System - Setup Wizard
echo ======================================================
echo.
echo Welcome to the VTU Results Automation System setup!
echo This wizard will guide you through the installation process.
echo.

choice /C YN /M "Do you want to begin the installation process"
if errorlevel 2 (
    echo Installation cancelled.
    pause
    exit /b 1
)
echo.

echo Step 1: Installing required dependencies...
echo ------------------------------------------
call install_dependencies.bat
echo.

echo Step 2: Creating desktop shortcuts...
echo -----------------------------------
call create_shortcuts.bat
echo.

echo ======================================================
echo Setup Process Completed!
echo ======================================================
echo.
echo The VTU Results Automation System setup process has completed!
echo.
echo To launch the application:
echo   1. Double-click the 'VTU Results Automation' icon on your desktop
echo   OR
echo   2. Double-click 'launch_app.bat' in this folder
echo.
echo Then open your browser and go to http://localhost:5000
echo.
echo If you experience any issues:
echo   - Try running 'install_dependencies.bat' again
echo   - Make sure you have Python installed
echo   - Check your internet connection
echo.
echo Thank you for using VTU Results Automation System!
echo.
pause