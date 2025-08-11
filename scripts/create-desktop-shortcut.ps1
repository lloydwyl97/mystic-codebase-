# Create Desktop Shortcuts for Mystic Trading Platform
Write-Host "Creating Desktop Shortcuts..." -ForegroundColor Cyan

# Get the project directory
$PROJECT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$DESKTOP_DIR = [Environment]::GetFolderPath("Desktop")

# Create shortcut for the main startup script
$WshShell = New-Object -comObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut("$DESKTOP_DIR\Mystic Trading Platform.lnk")
$Shortcut.TargetPath = "powershell.exe"
$Shortcut.Arguments = "-ExecutionPolicy Bypass -File `"$PROJECT_DIR\start-mystic-trading.ps1`""
$Shortcut.WorkingDirectory = $PROJECT_DIR
$Shortcut.Description = "Launch Mystic Trading Platform"
$Shortcut.IconLocation = "$PROJECT_DIR\frontend\public\favicon.ico"
$Shortcut.Save()

# Create shortcut for backend API docs
$Shortcut2 = $WshShell.CreateShortcut("$DESKTOP_DIR\Mystic API Docs.lnk")
$Shortcut2.TargetPath = "http://localhost:8000/docs"
$Shortcut2.Description = "Mystic Trading Platform API Documentation"
$Shortcut2.IconLocation = "$PROJECT_DIR\frontend\public\favicon.ico"
$Shortcut2.Save()

# Create shortcut for frontend
$Shortcut3 = $WshShell.CreateShortcut("$DESKTOP_DIR\Mystic Frontend.lnk")
$Shortcut3.TargetPath = "http://localhost:80"
$Shortcut3.Description = "Mystic Trading Platform Frontend"
$Shortcut3.IconLocation = "$PROJECT_DIR\frontend\public\favicon.ico"
$Shortcut3.Save()

# Create shortcut for health check
$Shortcut4 = $WshShell.CreateShortcut("$DESKTOP_DIR\Mystic Health Check.lnk")
$Shortcut4.TargetPath = "http://localhost:8000/health"
$Shortcut4.Description = "Mystic Trading Platform Health Status"
$Shortcut4.IconLocation = "$PROJECT_DIR\frontend\public\favicon.ico"
$Shortcut4.Save()

# Create shortcut for direct backend start
$Shortcut5 = $WshShell.CreateShortcut("$DESKTOP_DIR\Mystic Backend Start.lnk")
$Shortcut5.TargetPath = "powershell.exe"
$Shortcut5.Arguments = "-NoExit -Command `"cd '$PROJECT_DIR'; python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload`""
$Shortcut5.WorkingDirectory = $PROJECT_DIR
$Shortcut5.Description = "Start Mystic Backend Server"
$Shortcut5.IconLocation = "$PROJECT_DIR\frontend\public\favicon.ico"
$Shortcut5.Save()

Write-Host "✅ Desktop shortcuts created successfully!" -ForegroundColor Green
Write-Host "Shortcuts created on desktop:" -ForegroundColor White
Write-Host "  - Mystic Trading Platform.lnk" -ForegroundColor Gray
Write-Host "  - Mystic API Docs.lnk" -ForegroundColor Gray
Write-Host "  - Mystic Frontend.lnk" -ForegroundColor Gray
Write-Host "  - Mystic Health Check.lnk" -ForegroundColor Gray
Write-Host "  - Mystic Backend Start.lnk" -ForegroundColor Gray
Write-Host ""
Write-Host "You can now double-click any shortcut to launch the platform!" -ForegroundColor Yellow
