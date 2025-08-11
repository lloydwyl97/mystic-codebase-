# ===== COMPREHENSIVE REQUIREMENTS INSTALLATION SCRIPT =====
# This script installs all missing packages for the entire Mystic Trading Platform

Write-Host "Starting comprehensive requirements installation..." -ForegroundColor Green

# Function to install packages with error handling
function Install-Packages {
    param([string[]]$Packages, [string]$Category)
    
    Write-Host "Installing $Category packages..." -ForegroundColor Yellow
    $success_count = 0
    $total_count = $Packages.Count
    
    foreach ($package in $Packages) {
        Write-Host "Installing $package..." -ForegroundColor Cyan
        pip install $package
        if ($LASTEXITCODE -eq 0) {
            $success_count++
            Write-Host "✓ Successfully installed $package" -ForegroundColor Green
        } else {
            Write-Host "✗ Failed to install $package" -ForegroundColor Red
        }
    }
    
    $color = if ($success_count -eq $total_count) { "Green" } else { "Yellow" }
    Write-Host "$Category`: $success_count/$total_count packages installed successfully" -ForegroundColor $color
    return $success_count
}

# Core AI/ML packages
$ai_packages = @(
    "ray==2.7.0",
    "mlflow==2.7.0", 
    "wandb==0.15.8",
    "langchain==0.0.267",
    "langchain-openai==0.0.2",
    "langchain-anthropic==0.0.2",
    "accelerate==0.24.0",
    "sentence-transformers==2.2.2",
    "optuna==3.4.0",
    "tensorflow==2.15.0",
    "keras==2.15.0"
)

# Advanced ML packages
$advanced_ml_packages = @(
    "xgboost==2.0.3",
    "lightgbm==4.1.0", 
    "catboost==1.2.2",
    "spacy==3.7.2",
    "nltk==3.8.1",
    "gensim==4.3.2",
    "prophet==1.1.4",
    "shap==0.44.0",
    "lime==0.2.0.1",
    "flaml==2.3.5",
    "h2o==3.44.0.3",
    "tpot==0.12.1"
)

# Distributed computing and GPU packages
$distributed_packages = @(
    "dask==2023.11.0",
    "vaex==4.17.0",
    "numba==0.58.1"
)

# Model serving packages
$serving_packages = @(
    "bentoml==1.0.20"
)

# Additional missing packages from requirements
$additional_packages = @(
    "fastapi-limiter==0.1.5",
    "asyncio-mqtt==0.16.1",
    "gradio==4.0.0",
    "dash==2.14.0",
    "huggingface-hub>=0.16.4,<0.18"
)

# Feature engineering packages
$feature_engineering_packages = @(
    "feature-engine==1.6.2",
    "category-encoders==2.6.3"
)

# Install all packages
$total_success = 0
$total_packages = 0

$total_success += Install-Packages -Packages $ai_packages -Category "AI/ML"
$total_packages += $ai_packages.Count

$total_success += Install-Packages -Packages $advanced_ml_packages -Category "Advanced ML"
$total_packages += $advanced_ml_packages.Count

$total_success += Install-Packages -Packages $distributed_packages -Category "Distributed Computing"
$total_packages += $distributed_packages.Count

$total_success += Install-Packages -Packages $serving_packages -Category "Model Serving"
$total_packages += $serving_packages.Count

$total_success += Install-Packages -Packages $additional_packages -Category "Additional"
$total_packages += $additional_packages.Count

$total_success += Install-Packages -Packages $feature_engineering_packages -Category "Feature Engineering"
$total_packages += $feature_engineering_packages.Count

# Summary
Write-Host "`nInstallation Summary:" -ForegroundColor Green
Write-Host "Total packages attempted: $total_packages" -ForegroundColor White
Write-Host "Successfully installed: $total_success" -ForegroundColor Green
$failed_color = if (($total_packages - $total_success) -eq 0) { "Green" } else { "Red" }
Write-Host "Failed installations: $($total_packages - $total_success)" -ForegroundColor $failed_color

# Verify installations
Write-Host "`nVerifying installations..." -ForegroundColor Yellow
python -c "import pkg_resources; installed = {pkg.key.lower(): pkg.version for pkg in pkg_resources.working_set}; missing = ['ray', 'mlflow', 'wandb', 'langchain', 'accelerate', 'sentence-transformers', 'optuna', 'tensorflow', 'keras', 'xgboost', 'lightgbm', 'catboost', 'spacy', 'gensim', 'prophet', 'shap', 'lime', 'flaml', 'h2o', 'tpot', 'dask', 'vaex', 'numba', 'bentoml']; still_missing = [pkg for pkg in missing if pkg not in installed]; print(f'Still missing: {len(still_missing)} packages'); [print(f'  - {pkg}') for pkg in still_missing]"

Write-Host "`nInstallation complete!" -ForegroundColor Green
