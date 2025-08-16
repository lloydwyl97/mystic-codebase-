$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

# Env that both processes inherit
$env:PYTHONPATH = $root
$env:MYSTIC_BACKEND = "http://127.0.0.1:9000"
if (-not $env:COMPREHENSIVE_MAX_PER_EXCHANGE) { $env:COMPREHENSIVE_MAX_PER_EXCHANGE = "10" }

$py        = Join-Path $root "venv\Scripts\python.exe"
$streamlit = Join-Path $root "venv\Scripts\streamlit.exe"
$logs      = Join-Path $root "logs"
New-Item -ItemType Directory -Force -Path $logs | Out-Null

# Backend (no --reload for background)
$backendArgs = @("-m","uvicorn","backend.main:app","--host","127.0.0.1","--port","9000")
$bp = Start-Process -FilePath $py -ArgumentList $backendArgs `
  -WorkingDirectory $root -WindowStyle Hidden -PassThru `
  -RedirectStandardOutput (Join-Path $logs "backend.out.log") `
  -RedirectStandardError  (Join-Path $logs "backend.err.log")

Start-Sleep -Seconds 2

# UI
$uiArgs = @("run",(Join-Path $root "mystic_ui\app.py"),"--server.port","8501","--server.headless","true")
$sp = Start-Process -FilePath $streamlit -ArgumentList $uiArgs `
  -WorkingDirectory $root -WindowStyle Hidden -PassThru `
  -RedirectStandardOutput (Join-Path $logs "ui.out.log") `
  -RedirectStandardError  (Join-Path $logs "ui.err.log")

"backend=$($bp.Id)`nui=$($sp.Id)" | Set-Content (Join-Path $logs "pids.txt")
Write-Host "Started. Backend PID=$($bp.Id)  UI PID=$($sp.Id). Logs in $logs"
