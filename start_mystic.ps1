# start_mystic.ps1
$ErrorActionPreference = "Stop"
$ROOT = Split-Path -Parent $MyInvocation.MyCommand.Path
$base = "http://127.0.0.1:9000"

# Free stuck ports
foreach ($port in 9000,8501) {
  Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue |
    Select-Object -ExpandProperty OwningProcess -Unique |
    ForEach-Object { Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue }
}

# --- Backend ---
$backendCmd = @"
cd "$ROOT"; & "$ROOT\venv\Scripts\Activate.ps1";
`$env:PYTHONPATH="$ROOT";
`$env:COMPREHENSIVE_MAX_PER_EXCHANGE="10";
uvicorn backend.main:app --host 127.0.0.1 --port 9000
"@
Start-Process powershell -WindowStyle Minimized -ArgumentList @('-NoExit','-Command', $backendCmd)

# Wait briefly for backend
$ok = $false
for ($i=0; $i -lt 30; $i++) {
  try { Invoke-RestMethod "$base/openapi.json" -TimeoutSec 2 | Out-Null; $ok=$true; break } catch { Start-Sleep -Milliseconds 500 }
}

# --- UI ---
$uiCmd = @"
cd "$ROOT"; & "$ROOT\venv\Scripts\Activate.ps1";
`$env:PYTHONPATH="$ROOT";
`$env:MYSTIC_BACKEND="$base";
"$ROOT\venv\Scripts\streamlit.exe" run "$ROOT\mystic_ui\app.py" --server.port 8501 --server.headless true
"@
Start-Process powershell -WindowStyle Minimized -ArgumentList @('-NoExit','-Command', $uiCmd)

Start-Process "http://127.0.0.1:8501/"
Write-Host "Mystic started. Backend: $base  UI: http://127.0.0.1:8501/"
