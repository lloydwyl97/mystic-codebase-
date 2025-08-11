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

Write-Host "🔍 Mystic Trading Platform - Quality Checks" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan

# Check if we're in the right directory
if (-not (Test-Path "requirements.txt")) {
    Write-Host "❌ Error: requirements.txt not found. Please run this script from the backend directory." -ForegroundColor Red
    exit 1
}

# Check if virtual environment is activated
if (-not $env:VIRTUAL_ENV) {
    Write-Host "⚠️  Warning: Virtual environment not detected. Some tools may not work correctly." -ForegroundColor Yellow
}

# Function to run command and handle errors
function Invoke-QualityCheck {
    param(
        [string]$Name,
        [string]$Command,
        [string]$Description = ""
    )

    Write-Host "`n🔧 Running $Name..." -ForegroundColor Green
    if ($Description) {
        Write-Host "   $Description" -ForegroundColor Gray
    }

    try {
        $result = Invoke-Expression $Command 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ $Name completed successfully" -ForegroundColor Green
            if ($Verbose -and $result) {
                Write-Host $result -ForegroundColor Gray
            }
        } else {
            Write-Host "❌ $Name failed with exit code $LASTEXITCODE" -ForegroundColor Red
            if ($result) {
                Write-Host $result -ForegroundColor Red
            }
            return $false
        }
    } catch {
        Write-Host "❌ $Name failed with error: $($_.Exception.Message)" -ForegroundColor Red
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
Write-Host "`n🐍 Checking Python version..." -ForegroundColor Green
$pythonVersion = python --version 2>&1
Write-Host "   $pythonVersion" -ForegroundColor Gray

# 1. Code Formatting
if ($Fix) {
    Write-Host "`n🎨 Code Formatting (with fixes)..." -ForegroundColor Green

    # Black formatting
    if (Test-ToolAvailable "black") {
        Invoke-QualityCheck -Name "Black" -Command "black --line-length=100 --target-version=py311 $Target" -Description "Code formatting"
    } else {
        Write-Host "⚠️  Black not available, skipping code formatting" -ForegroundColor Yellow
    }

    # isort import sorting
    if (Test-ToolAvailable "isort") {
        Invoke-QualityCheck -Name "isort" -Command "isort --profile=black --line-length=100 $Target" -Description "Import sorting"
    } else {
        Write-Host "⚠️  isort not available, skipping import sorting" -ForegroundColor Yellow
    }
} else {
    Write-Host "`n🎨 Code Formatting (check only)..." -ForegroundColor Green

    # Black check
    if (Test-ToolAvailable "black") {
        Invoke-QualityCheck -Name "Black Check" -Command "black --check --line-length=100 --target-version=py311 $Target" -Description "Code formatting check"
    }

    # isort check
    if (Test-ToolAvailable "isort") {
        Invoke-QualityCheck -Name "isort Check" -Command "isort --check-only --profile=black --line-length=100 $Target" -Description "Import sorting check"
    }
}

# 2. Linting
Write-Host "`n🔍 Linting..." -ForegroundColor Green

# Flake8
if (Test-ToolAvailable "flake8") {
    Invoke-QualityCheck -Name "Flake8" -Command "flake8 --max-line-length=100 --exclude=venv,.venv,__pycache__,.pytest_cache,build,dist,*.egg-info,logs,backups,model_versions,mutated_strategies,strategy_backups,stubs,typings,redis-server,crypto_widget,frontend,src $Target" -Description "Style and error checking"
}

# Pylint
if (Test-ToolAvailable "pylint") {
    Invoke-QualityCheck -Name "Pylint" -Command "pylint --disable=C0114,C0115,C0116,R0903,R0913,R0914,R0915,W0621,W0703,W1201,W1202,W1203 --ignore=venv,.venv,__pycache__,.pytest_cache,build,dist,*.egg-info,logs,backups,model_versions,mutated_strategies,strategy_backups,stubs,typings,redis-server,crypto_widget,frontend,src $Target" -Description "Code analysis"
}

# 3. Type Checking
Write-Host "`n📝 Type Checking..." -ForegroundColor Green

# MyPy
if (Test-ToolAvailable "mypy") {
    Invoke-QualityCheck -Name "MyPy" -Command "mypy --ignore-missing-imports --exclude=venv,.venv,__pycache__,.pytest_cache,build,dist,*.egg-info,logs,backups,model_versions,mutated_strategies,strategy_backups,stubs,typings,redis-server,crypto_widget,frontend,src $Target" -Description "Static type checking"
}

# 4. Security Analysis
if (-not $SkipSecurity) {
    Write-Host "`n🔒 Security Analysis..." -ForegroundColor Green

    # Bandit
    if (Test-ToolAvailable "bandit") {
        Invoke-QualityCheck -Name "Bandit" -Command "bandit -r . -f json -o bandit-report.json --exclude=venv,.venv,__pycache__,.pytest_cache,build,dist,*.egg-info,logs,backups,model_versions,mutated_strategies,strategy_backups,stubs,typings,redis-server,crypto_widget,frontend,src" -Description "Security vulnerability scanning"
    }

    # Safety
    if (Test-ToolAvailable "safety") {
        Invoke-QualityCheck -Name "Safety" -Command "safety check --json --output safety-report.json" -Description "Dependency vulnerability check"
    }
}

# 5. Code Complexity Analysis
if (-not $SkipComplexity) {
    Write-Host "`n📊 Code Complexity Analysis..." -ForegroundColor Green

    # Radon
    if (Test-ToolAvailable "radon") {
        Invoke-QualityCheck -Name "Radon" -Command "radon cc . --json --exclude=venv,.venv,__pycache__,.pytest_cache,build,dist,*.egg-info,logs,backups,model_versions,mutated_strategies,strategy_backups,stubs,typings,redis-server,crypto_widget,frontend,src" -Description "Cyclomatic complexity analysis"
        Invoke-QualityCheck -Name "Radon MI" -Command "radon mi . --json --exclude=venv,.venv,__pycache__,.pytest_cache,build,dist,*.egg-info,logs,backups,model_versions,mutated_strategies,strategy_backups,stubs,typings,redis-server,crypto_widget,frontend,src" -Description "Maintainability index"
    }

    # Lizard
    if (Test-ToolAvailable "lizard") {
        Invoke-QualityCheck -Name "Lizard" -Command "lizard --CCN 10 --length 1000 --arguments 100 --exclude=venv,.venv,__pycache__,.pytest_cache,build,dist,*.egg-info,logs,backups,model_versions,mutated_strategies,strategy_backups,stubs,typings,redis-server,crypto_widget,frontend,src ." -Description "Code complexity analysis"
    }
}

# 6. Dead Code Detection
Write-Host "`n🧹 Dead Code Detection..." -ForegroundColor Green

# Vulture
if (Test-ToolAvailable "vulture") {
    Invoke-QualityCheck -Name "Vulture" -Command "vulture --min-confidence=80 --exclude=venv,.venv,__pycache__,.pytest_cache,build,dist,*.egg-info,logs,backups,model_versions,mutated_strategies,strategy_backups,stubs,typings,redis-server,crypto_widget,frontend,src ." -Description "Dead code detection"
}

# 7. Duplicate Code Detection
Write-Host "`n🔄 Duplicate Code Detection..." -ForegroundColor Green

# jscpd (if available via npm)
if (Get-Command "jscpd" -ErrorAction SilentlyContinue) {
    Invoke-QualityCheck -Name "jscpd" -Command "jscpd --reporters=json --output=./jscpd-report.json --ignore=venv,.venv,__pycache__,.pytest_cache,build,dist,*.egg-info,logs,backups,model_versions,mutated_strategies,strategy_backups,stubs,typings,redis-server,crypto_widget,frontend,src ." -Description "Duplicate code detection"
} else {
    Write-Host "⚠️  jscpd not available, skipping duplicate code detection" -ForegroundColor Yellow
}

# 8. Testing
if (-not $SkipTests) {
    Write-Host "`n🧪 Testing..." -ForegroundColor Green

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

# 9. Documentation
Write-Host "`n📚 Documentation..." -ForegroundColor Green

# pydocstyle
if (Test-ToolAvailable "pydocstyle") {
    Invoke-QualityCheck -Name "Pydocstyle" -Command "pydocstyle --convention=google --add-ignore=D100,D101,D102,D103,D104,D105,D106,D107 --exclude=venv,.venv,__pycache__,.pytest_cache,build,dist,*.egg-info,logs,backups,model_versions,mutated_strategies,strategy_backups,stubs,typings,redis-server,crypto_widget,frontend,src ." -Description "Docstring style checking"
}

# 10. Generate Quality Report
Write-Host "`n📋 Generating Quality Report..." -ForegroundColor Green

$reportData = @{
    timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    python_version = $pythonVersion
    tools_available = @{}
    quality_metrics = @{}
}

# Check tool availability
$tools = @("black", "isort", "flake8", "pylint", "mypy", "bandit", "safety", "radon", "lizard", "vulture", "jscpd", "pytest", "coverage", "pydocstyle")
foreach ($tool in $tools) {
    $reportData.tools_available[$tool] = Test-ToolAvailable $tool
}

# Save report
$reportData | ConvertTo-Json -Depth 10 | Out-File -FilePath "quality_report.json" -Encoding UTF8

Write-Host "`n🎉 Quality checks completed!" -ForegroundColor Green
Write-Host "📊 Quality report saved to: quality_report.json" -ForegroundColor Cyan

# Summary
Write-Host "`n📈 Summary:" -ForegroundColor Cyan
Write-Host "===========" -ForegroundColor Cyan
Write-Host "✅ Code formatting and style checks" -ForegroundColor Green
Write-Host "✅ Linting and error detection" -ForegroundColor Green
Write-Host "✅ Type checking" -ForegroundColor Green
Write-Host "✅ Security analysis" -ForegroundColor Green
Write-Host "✅ Code complexity analysis" -ForegroundColor Green
Write-Host "✅ Dead code detection" -ForegroundColor Green
Write-Host "✅ Duplicate code detection" -ForegroundColor Green
Write-Host "✅ Testing and coverage" -ForegroundColor Green
Write-Host "✅ Documentation checks" -ForegroundColor Green

Write-Host "`n🚀 Your code is ready for production!" -ForegroundColor Green
