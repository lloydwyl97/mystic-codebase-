param(
  [string]$Venv="mystic-venv",
  [string]$Port="8501"
)
$env:MYSTIC_BACKEND="http://127.0.0.1:9000"
# If you ever want to override from env instead of config/coins.py, you can:
# $env:FEATURED_EXCHANGE="binanceus"
# $env:FEATURED_SYMBOLS="BTCUSDT,ETHUSDT"
& "$Venv\Scripts\python.exe" -m streamlit run "streamlit\main_single_exchange.py" --server.port $Port --server.address 0.0.0.0

# Optional desktop shortcut
$Shell = New-Object -ComObject WScript.Shell
$Shortcut = $Shell.CreateShortcut("$env:USERPROFILE\Desktop\Mystic (BinanceUS).lnk")
$Shortcut.TargetPath = "powershell.exe"
$Shortcut.Arguments  = "-ExecutionPolicy Bypass -File `"$PWD\start-dashboard-binanceus.ps1`""
$Shortcut.WorkingDirectory = "$PWD"
$Shortcut.IconLocation = "$PWD\assets\favicon.ico,0"
$Shortcut.Save()


