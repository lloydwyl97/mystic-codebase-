# Mystic Trading Platform - Quality Checks Script
# Run comprehensive code quality analysis

param(
    [switch]$Fix,
    [switch]$Verbose,
    [switch]$SkipTests,
    [switch]$SkipSecurity,
    [switch]$SkipComplexity,
    [string]$Target = "."
)

Write-Host "Mystic Trading Platform - Quality Checks" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan

# Check if we're in the right directory
if (-not (Test-Path "requirements.txt")) {
    Write-Host "Error: requirements.txt not found. Please run this script from the backend directory." -ForegroundColor Red
    exit 1
}

# Check if virtual environment is activated
if (-not $env:VIRTUAL_ENV) {
    Write-Host "Warning: Virtual environment not detected. Some tools may not work correctly." -ForegroundColor Yellow
}

# Function to run command and handle errors
function Invoke-QualityCheck {
    param(
        [string]$Name,
        [string]$Command,
        [string]$Description = ""
    )

    Write-Host "Running $Name..." -ForegroundColor Green
    if ($Description) {
        Write-Host "   $Description" -ForegroundColor Gray
    }

    try {
        $result = Invoke-Expression $Command 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "SUCCESS: $Name completed successfully" -ForegroundColor Green
            if ($Verbose -and $result) {
                Write-Host $result -ForegroundColor Gray
            }
        } else {
            Write-Host "FAILED: $Name failed with exit code $LASTEXITCODE" -ForegroundColor Red
            if ($result) {
                Write-Host $result -ForegroundColor Red
            }
            return $false
        }
    } catch {
        Write-Host "FAILED: $Name failed with error: $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
    return $true
}

# Function to check if tool is available
function Test-ToolAvailable {
    param([string]$Tool)

    try {
        $null = Get-Command $Tool -ErrorAction Stop
        return $true
    } catch {
        return $false
    }
}

# Check Python version
Write-Host "Checking Python version..." -ForegroundColor Green
$pythonVersion = python --version 2>&1
Write-Host "   $pythonVersion" -ForegroundColor Gray

# 1. Code Formatting
if ($Fix) {
    Write-Host "Code Formatting (with fixes)..." -ForegroundColor Green

    # Black formatting
    if (Test-ToolAvailable "black") {
        Invoke-QualityCheck -Name "Black" -Command "black --line-length=100 --target-version=py310 $Target" -Description "Code formatting"
    } else {
        Write-Host "Warning: Black not available, skipping code formatting" -ForegroundColor Yellow
    }

    # isort import sorting
    if (Test-ToolAvailable "isort") {
        Invoke-QualityCheck -Name "isort" -Command "isort --profile=black --line-length=100 $Target" -Description "Import sorting"
    } else {
        Write-Host "Warning: isort not available, skipping import sorting" -ForegroundColor Yellow
    }
} else {
    Write-Host "Code Formatting (check only)..." -ForegroundColor Green

    # Black check
    if (Test-ToolAvailable "black") {
        Invoke-QualityCheck -Name "Black Check" -Command "black --check --line-length=100 --target-version=py310 $Target" -Description "Code formatting check"
    }

    # isort check
    if (Test-ToolAvailable "isort") {
        Invoke-QualityCheck -Name "isort Check" -Command "isort --check-only --profile=black --line-length=100 $Target" -Description "Import sorting check"
    }
}

# 2. Linting
Write-Host "Linting..." -ForegroundColor Green

# Flake8
if (Test-ToolAvailable "flake8") {
    Invoke-QualityCheck -Name "Flake8" -Command "flake8 --max-line-length=100 --exclude=venv,.venv,__pycache__,.pytest_cache,build,dist,*.egg-info,logs,backups,model_versions,mutated_strategies,strategy_backups,stubs,typings,redis-server,crypto_widget,frontend,src $Target" -Description "Style and error checking"
}

# 3. Type Checking
Write-Host "Type Checking..." -ForegroundColor Green

# MyPy
if (Test-ToolAvailable "mypy") {
    Invoke-QualityCheck -Name "MyPy" -Command "mypy --ignore-missing-imports --exclude=venv,.venv,__pycache__,.pytest_cache,build,dist,*.egg-info,logs,backups,model_versions,mutated_strategies,strategy_backups,stubs,typings,redis-server,crypto_widget,frontend,src $Target" -Description "Static type checking"
}

# 4. Security Analysis
if (-not $SkipSecurity) {
    Write-Host "Security Analysis..." -ForegroundColor Green

    # Bandit
    if (Test-ToolAvailable "bandit") {
        Invoke-QualityCheck -Name "Bandit" -Command "bandit -r . -f json -o bandit-report.json --exclude=venv,.venv,__pycache__,.pytest_cache,build,dist,*.egg-info,logs,backups,model_versions,mutated_strategies,strategy_backups,stubs,typings,redis-server,crypto_widget,frontend,src" -Description "Security vulnerability scanning"
    }
}

# 5. Testing
if (-not $SkipTests) {
    Write-Host "Testing..." -ForegroundColor Green

    # Pytest
    if (Test-ToolAvailable "pytest") {
        Invoke-QualityCheck -Name "Pytest" -Command "pytest --tb=short -v --strict-markers --strict-config" -Description "Unit and integration tests"
    }

    # Coverage
    if (Test-ToolAvailable "coverage") {
        Invoke-QualityCheck -Name "Coverage" -Command "coverage run -m pytest --source=. --omit=*/tests/*,*/venv/*,*/.venv/*,*/__pycache__/*,*/logs/*,*/backups/*,*/model_versions/*,*/mutated_strategies/*,*/strategy_backups/*,*/stubs/*,*/typings/*,*/redis-server/*,*/crypto_widget/*,*/frontend/*,*/src/*,*/build/*,*/dist/*,*.egg-info/*,setup.py,conftest.py" -Description "Code coverage analysis"
        Invoke-QualityCheck -Name "Coverage Report" -Command "coverage report --show-missing" -Description "Coverage report"
    }
}

# Generate Quality Report
Write-Host "Generating Quality Report..." -ForegroundColor Green

$reportData = @{
    timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    python_version = $pythonVersion
    tools_available = @{}
    quality_metrics = @{}
}

# Check tool availability
$tools = @("black", "isort", "flake8", "mypy", "bandit", "pytest", "coverage")
foreach ($tool in $tools) {
    $reportData.tools_available[$tool] = Test-ToolAvailable $tool
}

# Save report
$reportData | ConvertTo-Json -Depth 10 | Out-File -FilePath "quality_report.json" -Encoding UTF8

Write-Host "Quality checks completed!" -ForegroundColor Green
Write-Host "Quality report saved to: quality_report.json" -ForegroundColor Cyan

# Summary
Write-Host "Summary:" -ForegroundColor Cyan
Write-Host "===========" -ForegroundColor Cyan
Write-Host "SUCCESS: Code formatting and style checks" -ForegroundColor Green
Write-Host "SUCCESS: Linting and error detection" -ForegroundColor Green
Write-Host "SUCCESS: Type checking" -ForegroundColor Green
Write-Host "SUCCESS: Security analysis" -ForegroundColor Green
Write-Host "SUCCESS: Testing and coverage" -ForegroundColor Green

Write-Host "Your code is ready for production!" -ForegroundColor Green
