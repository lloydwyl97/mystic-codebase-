Param(
  [string]$BaseUrl = "http://127.0.0.1:9000",
  [switch]$NoStreamlit
)

$ErrorActionPreference = "Stop"

function Write-Green($msg) { Write-Host $msg -ForegroundColor Green }
function Write-Red($msg) { Write-Host $msg -ForegroundColor Red }

# Prefer running in repo venv explicitly
$RepoRoot = Split-Path -Parent $PSScriptRoot
$VenvActivate = Join-Path $RepoRoot "venv\Scripts\Activate.ps1"
$VenvPython = Join-Path $RepoRoot "venv\Scripts\python.exe"
$HasVenv = Test-Path $VenvPython
if ($HasVenv) {
  & $VenvActivate | Out-Null
}
if (-not (Test-Path env:VIRTUAL_ENV)) {
  Write-Red "Python venv not active. Activate your venv first (expected $VenvPython)."
  exit 1
}

Write-Host "Using BASE_URL=$BaseUrl"
$env:BASE_URL = $BaseUrl

# Paths
$BackendDir = Join-Path $RepoRoot "backend"

Set-Location $BackendDir

# Start backend with uvicorn in background
$BackendLog = Join-Path $RepoRoot "backend_smoke.log"
if (Test-Path $BackendLog) { Remove-Item $BackendLog -Force }

# Start backend directly with uvicorn in repo backend dir using venv python
if (-not (Test-Path $BackendDir)) {
  Write-Red "Backend dir not found: $BackendDir"
  exit 2
}

Write-Host "Starting backend via uvicorn (venv: $env:VIRTUAL_ENV) in $BackendDir"
Start-Job -Name mystic-backend -ScriptBlock {
  param($activate, $backendDir, $venvPython, $log)
  $cmd = "& `"$activate`"; Set-Location `"$backendDir`"; & `"$venvPython`" -m uvicorn main:app --host 127.0.0.1 --port 9000 --reload 2>&1 | Tee-Object -FilePath `"$log`" -Append"
  powershell -NoProfile -ExecutionPolicy Bypass -Command $cmd | Out-Null
  Start-Sleep -Seconds 1
} -ArgumentList $VenvActivate, $BackendDir, $VenvPython, $BackendLog | Out-Null

# Wait for health-check up
$ok = $false
for ($i=0; $i -lt 30; $i++) {
  try {
    $r = Invoke-WebRequest -UseBasicParsing -Uri "$BaseUrl/api/system/health-check" -TimeoutSec 5
    if ($r.StatusCode -eq 200) { $ok = $true; break }
  } catch { }
  Start-Sleep -Milliseconds 800
}
if (-not $ok) {
  Write-Red "Backend failed to become healthy. Tail of log:";
  if (Test-Path $BackendLog) { Get-Content $BackendLog -Tail 50 }
  Get-Job mystic-backend | Stop-Job | Out-Null
  exit 2
}
Write-Green "Backend is up."

# Optionally start Streamlit (best-effort)
if (-not $NoStreamlit) {
  $streamlitScript = Join-Path $RepoRoot "streamlit\main.py"
  if (Test-Path $streamlitScript) {
    Write-Host "Starting Streamlit (optional)"
    Start-Job -Name mystic-streamlit -ScriptBlock {
      param($script)
      $psi = New-Object System.Diagnostics.ProcessStartInfo
      $psi.FileName = "powershell"
      $psi.Arguments = "-NoProfile -Command streamlit run `"$script`" --server.headless true --server.port 8501"
      $psi.UseShellExecute = $true
      [System.Diagnostics.Process]::Start($psi) | Out-Null
      Start-Sleep -Seconds 1
    } -ArgumentList $streamlitScript | Out-Null
  }
}

# Run pytest
Set-Location $RepoRoot
Write-Host "Running smoke tests with venv python..."
$pytestCmd = "`"$VenvPython`" -m pytest -q tests/smoke"
$last = ""
try {
  $p = Start-Process -FilePath "powershell" -ArgumentList "-NoProfile -Command $pytestCmd" -PassThru -NoNewWindow -Wait
  $code = $p.ExitCode
} catch {
  $code = 1
}

if ($code -eq 0) {
  Write-Host "`n`n====================="
  Write-Green "   SMOKE TESTS PASS   "
  Write-Host "====================="
} else {
  Write-Host "`n`n====================="
  Write-Red "   SMOKE TESTS FAIL   "
  Write-Host "====================="
  if (Test-Path $BackendLog) {
    Write-Host "Last 50 lines of backend log:"; Get-Content $BackendLog -Tail 50
  }
}

# Cleanup background jobs
Get-Job mystic-backend -ErrorAction SilentlyContinue | ForEach-Object {
  try { Stop-Job $_ | Out-Null } catch {}
  try { Remove-Job $_ | Out-Null } catch {}
}
Get-Job mystic-streamlit -ErrorAction SilentlyContinue | ForEach-Object {
  try { Stop-Job $_ -Force | Out-Null } catch {}
  try { Remove-Job $_ | Out-Null } catch {}
}

exit $code


