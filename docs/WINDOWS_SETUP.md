# Windows 11 Home Setup Guide

## Quick Start Options

### Option 1: One-Click Start (Recommended)

Double-click `start-all.bat` to start all services automatically.

### Option 2: Individual Services

- **Redis:** Double-click `start-redis.bat`
- **Backend:** Double-click `start-backend.bat`
- **Frontend:** Double-click `start-frontend.bat`

### Option 3: PowerShell (Advanced)

Run `start-services.ps1` in PowerShell for advanced features.

## Manual Setup

### Prerequisites

- Python 3.8+ installed
- Node.js 18+ installed
- PowerShell (included with Windows 11)

### Backend Setup

```powershell
cd backend
python -m venv venv
.\venv\Scripts\Activate
pip install --upgrade pip
pip install -r requirements.txt
python -m uvicorn main:app --host localhost --port 8000
```

### Frontend Setup

```powershell
cd frontend
npm install
npm run windows:start
```

### Redis Setup

```powershell
cd redis-server
.\redis-server.exe --port 6379
```

## Important Notes

### PowerShell Commands

- **Never use `&&`** to chain commands in PowerShell
- Always run each command separately
- Use `cd` to change directories before running commands

### Ports Used

- **Frontend:** <http://localhost:3000>
- **Backend:** <http://localhost:8000>
- **Redis:** localhost:6379

### Troubleshooting

#### "Command not found" errors

- Make sure you're in the correct directory
- Check that files exist in the expected locations

#### Port already in use

- Close any existing instances of the service
- Use different ports if needed

#### Virtual environment issues

- Delete the `venv` folder and recreate it
- Make sure Python is installed and in PATH

## Windows-Specific Features

### Cross-Platform Scripts

- `rimraf` is used instead of `rm -rf` for Windows compatibility
- All scripts are tested on Windows 11 Home

### Auto-Installation

- Scripts automatically install dependencies if missing
- Virtual environment is created automatically if needed

### Service Management

- Each service runs in its own PowerShell window
- Easy to stop individual services (Ctrl+C in their window)

## Access Your Application

Once all services are running:

- **Frontend:** <http://localhost:3000>
- **Backend API:** <http://localhost:8000>
- **API Documentation:** <http://localhost:8000/docs>

## No Logic Removed

All business logic, trading algorithms, and application features remain completely intact. These changes only improve the Windows development experience.
