# ===== INSTALL MISSING PACKAGES SCRIPT =====
# This script installs all missing packages for the Mystic Trading Platform

Write-Host "Installing missing packages for Mystic Trading Platform..." -ForegroundColor Green

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

# Combine all packages
$all_packages = $ai_packages + $advanced_ml_packages + $distributed_packages + $serving_packages + $additional_packages

Write-Host "Installing AI/ML packages..." -ForegroundColor Yellow
foreach ($package in $ai_packages) {
    Write-Host "Installing $package..." -ForegroundColor Cyan
    pip install $package
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to install $package" -ForegroundColor Red
    }
}

Write-Host "Installing advanced ML packages..." -ForegroundColor Yellow
foreach ($package in $advanced_ml_packages) {
    Write-Host "Installing $package..." -ForegroundColor Cyan
    pip install $package
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to install $package" -ForegroundColor Red
    }
}

Write-Host "Installing distributed computing packages..." -ForegroundColor Yellow
foreach ($package in $distributed_packages) {
    Write-Host "Installing $package..." -ForegroundColor Cyan
    pip install $package
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to install $package" -ForegroundColor Red
    }
}

Write-Host "Installing model serving packages..." -ForegroundColor Yellow
foreach ($package in $serving_packages) {
    Write-Host "Installing $package..." -ForegroundColor Cyan
    pip install $package
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to install $package" -ForegroundColor Red
    }
}

Write-Host "Installing additional packages..." -ForegroundColor Yellow
foreach ($package in $additional_packages) {
    Write-Host "Installing $package..." -ForegroundColor Cyan
    pip install $package
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to install $package" -ForegroundColor Red
    }
}

Write-Host "Installation complete!" -ForegroundColor Green
Write-Host "Total packages to install: $($all_packages.Count)" -ForegroundColor Green

# Verify installations
Write-Host "Verifying installations..." -ForegroundColor Yellow
python -c "import pkg_resources; installed = {pkg.key.lower(): pkg.version for pkg in pkg_resources.working_set}; missing = ['ray', 'mlflow', 'wandb', 'langchain', 'accelerate', 'sentence-transformers', 'optuna', 'tensorflow', 'keras', 'xgboost', 'lightgbm', 'catboost', 'spacy', 'gensim', 'prophet', 'shap', 'lime', 'flaml', 'h2o', 'tpot', 'dask', 'vaex', 'numba', 'bentoml']; print('Still missing:'); [print(f'  - {pkg}') for pkg in missing if pkg not in installed]"
