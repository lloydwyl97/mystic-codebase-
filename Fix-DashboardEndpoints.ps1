# Fix-DashboardEndpoints.ps1
# SCAN ONLY - NO REPAIRS WITHOUT APPROVAL

# Skip execution if none of these files exist
$sourceFiles = @(
    "frontend\mystic_super_dashboard.py",
    "frontend\unified_dashboard.py",
    "frontend\streamlit_dashboard.py",
    "frontend\app.py"
)

# Skip execution if none of these files exist
if (-not ($sourceFiles | Where-Object { Test-Path $_ })) {
    Write-Host "✅ No stub dashboard files exist. Nothing to fix."
    exit 0
}

$targetFile = "services/mystic_super_dashboard/app/main.py"
$logFile = "dashboard_scan_report.txt"

# Clear previous log
if (Test-Path $logFile) { Remove-Item $logFile }
New-Item -ItemType File -Path $logFile -Force | Out-Null

Write-Host "SCANNING FOR DASHBOARD ENDPOINTS IN WRONG FILES..." -ForegroundColor Yellow
Write-Host "==================================================" -ForegroundColor Yellow

Add-Content -Path $logFile -Value "DASHBOARD ENDPOINT SCAN REPORT - $(Get-Date)"
Add-Content -Path $logFile -Value "================================================"

$totalFunctionsFound = 0
$filesWithFunctions = @()

foreach ($sourceFile in $sourceFiles) {
    Write-Host "`nChecking: $sourceFile" -ForegroundColor Cyan
    
    if (Test-Path $sourceFile) {
        $content = Get-Content $sourceFile -Raw
        $lines = Get-Content $sourceFile
        
        # Find dashboard-related functions
        $endpointLines = $lines | Where-Object { 
            $_ -match "def\s+(show_|render_|fetch_|api_|endpoint_|get_|post_|put_|delete_)" -or
            $_ -match "@.*\.(route|get|post|put|delete)" -or
            $_ -match "st\.(page_config|title|subheader|metric|dataframe|chart)" -or
            $_ -match "PAGES\[" -or
            $_ -match "def main\(\):"
        }
        
        if ($endpointLines.Count -gt 0) {
            $totalFunctionsFound += $endpointLines.Count
            $filesWithFunctions += $sourceFile
            
            Write-Host "FOUND $($endpointLines.Count) DASHBOARD FUNCTIONS!" -ForegroundColor Red
            Add-Content -Path $logFile -Value "`nFILE: $sourceFile"
            Add-Content -Path $logFile -Value "   FOUND $($endpointLines.Count) DASHBOARD FUNCTIONS:"
            
            foreach ($line in $endpointLines) {
                $trimmedLine = $line.Trim()
                Write-Host "   -> $trimmedLine" -ForegroundColor Red
                Add-Content -Path $logFile -Value "   -> $trimmedLine"
            }
            
            # Check file size
            $fileSize = (Get-Item $sourceFile).Length
            Write-Host "   File size: $fileSize bytes" -ForegroundColor Yellow
            Add-Content -Path $logFile -Value "   File size: $fileSize bytes"
            
        } else {
            Write-Host "No dashboard functions found" -ForegroundColor Green
            Add-Content -Path $logFile -Value "`nFILE: $sourceFile - No dashboard functions found"
        }
        
        # Check if file is just a stub
        if ($lines.Count -lt 100) {
            Write-Host "File appears to be a stub (only $($lines.Count) lines)" -ForegroundColor Yellow
            Add-Content -Path $logFile -Value "   File appears to be a stub (only $($lines.Count) lines)"
        }
        
    } else {
        Write-Host "File not found: $sourceFile" -ForegroundColor Red
        Add-Content -Path $logFile -Value "`nFILE NOT FOUND: $sourceFile"
    }
}

# Check target file
Write-Host "`nChecking target file: $targetFile" -ForegroundColor Cyan
if (Test-Path $targetFile) {
    $targetContent = Get-Content $targetFile
    $targetSize = (Get-Item $targetFile).Length
    $targetLines = $targetContent.Count
    
    Write-Host "Target file exists" -ForegroundColor Green
    Write-Host "   Size: $targetSize bytes, Lines: $targetLines" -ForegroundColor Green
    Add-Content -Path $logFile -Value "`nTARGET FILE: $targetFile"
    Add-Content -Path $logFile -Value "   Size: $targetSize bytes, Lines: $targetLines"
    
    # Count functions in target
    $targetFunctions = $targetContent | Where-Object { $_ -match "def\s+" }
    Write-Host "   Functions: $($targetFunctions.Count)" -ForegroundColor Green
    Add-Content -Path $logFile -Value "   Functions: $($targetFunctions.Count)"
    
} else {
    Write-Host "Target file not found: $targetFile" -ForegroundColor Red
    Add-Content -Path $logFile -Value "`nTARGET FILE NOT FOUND: $targetFile"
}

# Summary
Write-Host "`nSCAN SUMMARY:" -ForegroundColor Yellow
Write-Host "=================" -ForegroundColor Yellow
Write-Host "Total dashboard functions found in wrong files: $totalFunctionsFound" -ForegroundColor Cyan
Write-Host "Files with dashboard functions: $($filesWithFunctions.Count)" -ForegroundColor Cyan

if ($filesWithFunctions.Count -gt 0) {
    Write-Host "`nFILES THAT NEED FIXING:" -ForegroundColor Red
    foreach ($file in $filesWithFunctions) {
        Write-Host "   -> $file" -ForegroundColor Red
    }
    
    Write-Host "`nRECOMMENDED ACTIONS:" -ForegroundColor Yellow
    Write-Host "   1. Move dashboard functions to: $targetFile" -ForegroundColor White
    Write-Host "   2. Clean up stub files" -ForegroundColor White
    Write-Host "   3. Verify target dashboard works" -ForegroundColor White
} else {
    Write-Host "`nNO ACTION NEEDED - All files are clean!" -ForegroundColor Green
}

Add-Content -Path $logFile -Value "`nSCAN SUMMARY:"
Add-Content -Path $logFile -Value "Total dashboard functions found in wrong files: $totalFunctionsFound"
Add-Content -Path $logFile -Value "Files with dashboard functions: $($filesWithFunctions.Count)"

Write-Host "`nFull report saved to: $logFile" -ForegroundColor Green
Write-Host "NO CHANGES MADE - SCAN ONLY MODE" -ForegroundColor Yellow 