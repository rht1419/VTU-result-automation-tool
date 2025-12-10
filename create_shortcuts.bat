@echo off
title VTU Results Automation - Desktop Shortcut Creator

echo ======================================================
echo VTU Results Automation System - Desktop Shortcut Creator
echo ======================================================
echo.
echo Creating desktop shortcuts for easy access...
echo.

:: Get the current directory
set "CURRENT_DIR=%~dp0"
set "DESKTOP=%USERPROFILE%\Desktop"

echo Current directory: %CURRENT_DIR%
echo Desktop directory: %DESKTOP%
echo.

:: Create shortcut for launcher
echo Creating application shortcut on desktop...
powershell.exe -Command "$WshShell = New-Object -ComObject WScript.Shell; $Shortcut = $WshShell.CreateShortcut('%DESKTOP%\VTU Results Automation.lnk'); $Shortcut.TargetPath = '%CURRENT_DIR%launch_app.bat'; $Shortcut.WorkingDirectory = '%CURRENT_DIR%'; $Shortcut.IconLocation = 'shell32.dll,14'; $Shortcut.Save()"
if %errorlevel% neq 0 (
    echo ERROR: Failed to create application shortcut
) else (
    echo Application shortcut created successfully.
)
echo.

:: Create shortcut for installer (in case they need to reinstall)
echo Creating installer shortcut on desktop...
powershell.exe -Command "$WshShell = New-Object -ComObject WScript.Shell; $Shortcut = $WshShell.CreateShortcut('%DESKTOP%\VTU Results Automation - Reinstall.lnk'); $Shortcut.TargetPath = '%CURRENT_DIR%install_dependencies.bat'; $Shortcut.WorkingDirectory = '%CURRENT_DIR%'; $Shortcut.IconLocation = 'shell32.dll,16'; $Shortcut.Save()"
if %errorlevel% neq 0 (
    echo ERROR: Failed to create installer shortcut
) else (
    echo Installer shortcut created successfully.
)
echo.

echo ======================================================
echo Desktop shortcuts created successfully!
echo ======================================================
echo.
echo You can now launch the application from your desktop.
echo Look for the 'VTU Results Automation' icon.
echo.
pause