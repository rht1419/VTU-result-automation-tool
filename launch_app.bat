@echo off
title VTU Results Automation - Launcher

echo ======================================================
echo VTU Results Automation System - Application Launcher
echo ======================================================
echo.
echo Starting the VTU Results Automation web application...
echo.
echo Once the server is running, open your browser and go to:
echo    http://localhost:5000
echo.
echo To stop the application, close this window or press Ctrl+C
echo.

:: Change to the application directory
cd /d "%~dp0"

:: Start the web application
python vtu-automated-results.py

pause