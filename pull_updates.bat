@echo off
setlocal
cd /d "%~dp0"

git pull --ff-only origin master
if errorlevel 1 (
    echo.
    echo Pull failed. Resolve the Git issue above, then run this file again.
    exit /b 1
)

echo.
echo Updates pulled successfully.
endlocal
