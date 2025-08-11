@echo off
echo Starting Frontend Server...
echo.

cd frontend
if errorlevel 1 (
    echo ERROR: Could not navigate to frontend directory
    echo Make sure you are running this from the project root directory
    pause
    exit /b 1
)

if not exist node_modules (
    echo Installing dependencies...
    npm install
    if errorlevel 1 (
        echo ERROR: Failed to install dependencies
        pause
        exit /b 1
    )
)

echo Starting frontend server...
npm run windows:start
if errorlevel 1 (
    echo ERROR: Failed to start frontend server
    echo Try running: npm run windows:dev
    pause
    exit /b 1
)

pause
