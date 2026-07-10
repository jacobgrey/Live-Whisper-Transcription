@echo off
if "%~1"=="" (
    echo Drag a folder OR one-or-more .flac files onto this script.
    pause
    exit /b 1
)
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0combine-flacs.ps1" %*
echo.
pause
