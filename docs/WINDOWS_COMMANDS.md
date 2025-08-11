# Windows Command Reference - Old vs New

## ❌ OLD COMMANDS (Don't Work on Windows)

### PowerShell Command Chaining

```powershell
# ❌ WRONG - PowerShell doesn't support &&
cd redis-server && .\redis-server.exe --port 6379
cd backend && python -m uvicorn main:app --host 127.0.0.1 --port 8000
cd frontend && npm run dev
```

### Unix-style Clean Commands

```bash
# ❌ WRONG - rm -rf doesn't work on Windows
npm run clean  # (if it uses rm -rf)
```

### Wrong Directory Issues

```powershell
# ❌ WRONG - Running from wrong directory
python -m uvicorn main:app --host 127.0.0.1 --port 8000  # from project root
npm run dev  # from project root instead of frontend directory
.\redis-server.exe --port 6379  # from project root instead of redis-server directory
```

## ✅ NEW COMMANDS (Windows Compatible)

### Proper PowerShell Commands

```powershell
# ✅ CORRECT - Run commands separately
cd redis-server
.\redis-server.exe --port 6379

cd backend
python -m uvicorn main:app --host localhost --port 8000

cd frontend
npm run windows:start
```

### Windows-Compatible Clean Commands

```powershell
# ✅ CORRECT - Uses rimraf instead of rm -rf
npm run windows:clean
npm run clean  # (now uses rimraf)
```

### One-Click Solutions

```powershell
# ✅ CORRECT - Use the batch files
start-all.bat                    # Start all services
start-redis.bat                  # Start Redis only
start-backend.bat                # Start Backend only
start-frontend.bat               # Start Frontend only
```

### PowerShell Script

```powershell
# ✅ CORRECT - Advanced PowerShell script
.\start-services.ps1
```

## Quick Reference

| Service | Old Command | New Command |
|---------|-------------|-------------|
| Redis | `cd redis-server && .\redis-server.exe --port 6379` | `start-redis.bat` |
| Backend | `cd backend && python -m uvicorn main:app --host 127.0.0.1 --port 8000` | `start-backend.bat` |
| Frontend | `cd frontend && npm run dev` | `start-frontend.bat` |
| All Services | Manual 3 commands | `start-all.bat` |
| Clean | `npm run clean` (rm -rf) | `npm run windows:clean` (rimraf) |

## Ports Used

- **Frontend:** <http://localhost:3000>
- **Backend:** <http://localhost:8000>
- **Redis:** localhost:6379

## Troubleshooting

### "Command not found" errors

- Make sure you're in the correct directory
- Use the batch files instead of manual commands

### "&& is not a valid statement separator"

- Never use `&&` in PowerShell
- Run each command separately or use batch files

### "rm -rf is not recognized"

- Use `npm run windows:clean` instead
- Uses `rimraf` which works on Windows
