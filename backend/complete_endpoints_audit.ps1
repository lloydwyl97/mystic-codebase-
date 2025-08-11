$outputFile = "complete_endpoints_audit.txt"
if (Test-Path $outputFile) { Remove-Item $outputFile }

$pyFiles = Get-ChildItem -Path . -Recurse -Include *.py

foreach ($file in $pyFiles) {
    $lines = Get-Content $file.FullName
    for ($i = 0; $i -lt $lines.Count; $i++) {
        $line = $lines[$i].Trim()

        if ($line -match '^\s*@.*route|^\s*@.*router\.(get|post|put|delete)') {
            "$($file.FullName):$($i+1): ACTIVE ROUTE DECORATOR -> $line" | Out-File -FilePath $outputFile -Append
        }
        elseif ($line -match '^\s*#.*@.*route|^\s*#.*@.*router\.(get|post|put|delete)') {
            "$($file.FullName):$($i+1): COMMENTED ROUTE DECORATOR -> $line" | Out-File -FilePath $outputFile -Append
        }
        elseif ($line -match 'TODO|endpoint|route|api|comment|planned|removed|deprecated') {
            "$($file.FullName):$($i+1): COMMENT/NOTE -> $line" | Out-File -FilePath $outputFile -Append
        }
    }
}

Write-Host "Complete endpoint audit saved to $outputFile" 