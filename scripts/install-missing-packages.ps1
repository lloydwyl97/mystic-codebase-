# Install Missing Packages Script for Mystic Trading Platform - Python 3.10 Compatible
# This script installs all missing packages from requirements.txt

Write-Host "Installing Missing Packages for Mystic Trading Platform (Python 3.10 Compatible)..." -ForegroundColor Green

# Using global Python environment (dedicated laptop setup)
Write-Host "Using global Python environment..." -ForegroundColor Yellow
Write-Host "Python path: C:\Users\lloyd\AppData\Local\Programs\Python\Python310\python.exe" -ForegroundColor Cyan

# Core FastAPI and web framework packages - Python 3.11 compatible
Write-Host "Installing Core FastAPI packages..." -ForegroundColor Cyan
pip install fastapi==0.104.1 uvicorn[standard]==0.24.0 python-multipart==0.0.20 python-jose[cryptography]==3.3.0 passlib[bcrypt]==1.7.4 python-dotenv==1.0.1 pydantic==2.5.2 pydantic-settings==2.1.0

# Database dependencies - Python 3.11 compatible
Write-Host "Installing Database packages..." -ForegroundColor Cyan
pip install sqlalchemy==2.0.23 alembic==1.12.1 psycopg2-binary==2.9.7

# Data processing and analysis - Python 3.11 compatible
Write-Host "Installing Data processing packages..." -ForegroundColor Cyan
pip install pandas==2.0.3 numpy==1.24.3 matplotlib==3.7.2 plotly==5.17.0 scipy==1.11.1 scikit-learn==1.3.0

# Live data sources and trading - Python 3.11 compatible
Write-Host "Installing Trading and data source packages..." -ForegroundColor Cyan
pip install python-binance==1.0.19 coinbase==2.1.0 ccxt==4.1.77 aiohttp==3.9.1 requests==2.31.0 websockets==12.0 yfinance==0.2.28 alpha-vantage==2.3.1

# Caching and messaging - Python 3.11 compatible
Write-Host "Installing Caching and messaging packages..." -ForegroundColor Cyan
pip install redis==5.0.1 celery==5.3.4 flower==2.0.1

# Monitoring and logging - Python 3.11 compatible
Write-Host "Installing Monitoring and logging packages..." -ForegroundColor Cyan
pip install prometheus-client==0.19.0 structlog==23.2.0 python-json-logger==2.0.7 sentry-sdk[fastapi]==1.38.0

# HTTP and async - Python 3.11 compatible
Write-Host "Installing HTTP and async packages..." -ForegroundColor Cyan
pip install httpx==0.25.0 asyncio-mqtt==0.16.1

# Advanced AI and Machine Learning - Python 3.11 compatible
Write-Host "Installing Advanced AI packages..." -ForegroundColor Magenta
pip install openai==1.3.0 anthropic==0.7.0 torch==2.0.1 transformers==4.35.0

# Reinforcement Learning - Python 3.11 compatible
Write-Host "Installing Reinforcement Learning packages..." -ForegroundColor Magenta
pip install ray[rllib]==2.7.0 deepspeed==0.12.0 stable-baselines3==2.1.0 gymnasium==0.29.0 optuna==3.4.0 mlflow==2.7.0 wandb==0.16.0

# Strategy Evolution / Genetic Algorithms - Python 3.11 compatible
Write-Host "Installing Genetic Algorithm packages..." -ForegroundColor Magenta
pip install deap==1.3.3 neat-python==0.92 pygad==3.3.1 pymoo==0.6.0 platypus==1.0.4

# Web UI / Visualization - Python 3.11 compatible
Write-Host "Installing Visualization packages..." -ForegroundColor Magenta
pip install dash==2.14.0 streamlit==1.28.0 gradio==4.0.0 bokeh==3.3.0 altair==5.1.0

# Model Export / Sharing - Python 3.11 compatible
Write-Host "Installing Model export packages..." -ForegroundColor Magenta
pip install joblib==1.3.2 onnx==1.15.0 tensorflow==2.14.0 tensorflow-serving-api==2.14.0

# Utilities - Python 3.11 compatible
Write-Host "Installing Utility packages..." -ForegroundColor Cyan
pip install tenacity==8.2.3 backoff==2.2.1 ratelimit==2.2.1 python-dateutil==2.8.2 pytz==2023.3 tzlocal==5.0.1 click==8.1.7 rich==13.7.0

# Development and testing - Python 3.11 compatible
Write-Host "Installing Development packages..." -ForegroundColor Cyan
pip install pytest==7.4.3 pytest-asyncio==0.21.1 black==23.11.0 flake8==6.1.0 mypy==1.7.0

# Additional utilities - Python 3.11 compatible
Write-Host "Installing Additional utility packages..." -ForegroundColor Cyan
pip install aiofiles==23.2.1 fastapi-cache2==0.2.1 fastapi-limiter==0.1.5 gunicorn==21.2.0

# Notification services - Python 3.11 compatible
Write-Host "Installing Notification packages..." -ForegroundColor Cyan
pip install twilio==8.10.0 slack-sdk==3.26.1

# Data visualization - Python 3.11 compatible
Write-Host "Installing Data visualization packages..." -ForegroundColor Cyan
pip install seaborn==0.12.2

# Environment and configuration - Python 3.11 compatible
Write-Host "Installing Configuration packages..." -ForegroundColor Cyan
pip install pyyaml==6.0.1

# Rate limiting and caching - Python 3.11 compatible
Write-Host "Installing Rate limiting packages..." -ForegroundColor Cyan
pip install slowapi==0.1.9 cachetools==5.3.2

# Error handling and monitoring - Python 3.11 compatible
Write-Host "Installing Monitoring packages..." -ForegroundColor Cyan
pip install opentelemetry-api==1.21.0 opentelemetry-sdk==1.21.0

# Additional utilities - Python 3.11 compatible
Write-Host "Installing Additional utilities..." -ForegroundColor Cyan
pip install more-itertools==10.1.0 toolz==0.12.0 cytoolz==0.12.3

# System utilities - Python 3.11 compatible
Write-Host "Installing System utility packages..." -ForegroundColor Cyan
pip install psutil==5.9.6 pathlib2==2.3.7

# WebSocket and real-time - Python 3.11 compatible
Write-Host "Installing WebSocket packages..." -ForegroundColor Cyan
pip install python-socketio==5.9.0

# Security - Python 3.11 compatible
Write-Host "Installing Security packages..." -ForegroundColor Cyan
pip install cryptography==41.0.7

# Date and time - Python 3.11 compatible
Write-Host "Installing Date/time packages..." -ForegroundColor Cyan
pip install pendulum==2.1.2

# JSON handling - Python 3.11 compatible
Write-Host "Installing JSON handling packages..." -ForegroundColor Cyan
pip install ujson==5.8.0

# UUID handling - Python 3.11 compatible
Write-Host "Installing UUID packages..." -ForegroundColor Cyan
pip install uuid==1.30

# Environment variables - Python 3.11 compatible
Write-Host "Installing Environment packages..." -ForegroundColor Cyan
pip install python-decouple==3.8

# HTTP client - Python 3.11 compatible
Write-Host "Installing HTTP client packages..." -ForegroundColor Cyan
pip install urllib3==2.0.7

# SSL and certificates - Python 3.11 compatible
Write-Host "Installing SSL packages..." -ForegroundColor Cyan
pip install certifi==2023.11.17

# Compression - Python 3.11 compatible
Write-Host "Installing Compression packages..." -ForegroundColor Cyan
pip install brotli==1.1.0

# Image processing (for charts) - Python 3.11 compatible
Write-Host "Installing Image processing packages..." -ForegroundColor Cyan
pip install pillow==10.1.0

# Configuration management - Python 3.11 compatible
Write-Host "Installing Configuration management packages..." -ForegroundColor Cyan
pip install configparser==6.0.0

# Data validation - Python 3.11 compatible
Write-Host "Installing Data validation packages..." -ForegroundColor Cyan
pip install marshmallow==3.20.1

# API documentation - Python 3.11 compatible
Write-Host "Installing API documentation packages..." -ForegroundColor Cyan
pip install apispec==6.3.0

# Background tasks - Python 3.11 compatible
Write-Host "Installing Background task packages..." -ForegroundColor Cyan
pip install celery[redis]==5.3.4

# Health checks - Python 3.11 compatible
Write-Host "Installing Health check packages..." -ForegroundColor Cyan
pip install healthcheck==1.3.4

# Metrics collection - Python 3.11 compatible
Write-Host "Installing Metrics packages..." -ForegroundColor Cyan
pip install datadog==0.48.0

# Performance monitoring - Python 3.11 compatible
Write-Host "Installing Performance monitoring packages..." -ForegroundColor Cyan
pip install memory-profiler==0.61.0

# Code quality - Python 3.11 compatible
Write-Host "Installing Code quality packages..." -ForegroundColor Cyan
pip install bandit==1.7.5 pylint==3.0.3

# Type checking - Python 3.11 compatible
Write-Host "Installing Type checking packages..." -ForegroundColor Cyan
pip install types-requests==2.31.0.20231025 types-PyYAML==6.0.12.12

# Testing utilities - Python 3.11 compatible
Write-Host "Installing Testing utility packages..." -ForegroundColor Cyan
pip install factory-boy==3.3.0 faker==20.1.0 coverage==7.3.2

# Additional packages that might be missing - Python 3.11 compatible
Write-Host "Installing Additional packages..." -ForegroundColor Cyan
pip install orjson==3.9.10 starlette==0.27.0 python-consul==1.1.0 hvac==1.1.0 minio==7.2.0 influxdb-client==1.38.0

Write-Host "All packages installation completed!" -ForegroundColor Green
Write-Host "Checking for any installation errors..." -ForegroundColor Yellow

# Check for any failed installations
pip check

Write-Host "Package installation process completed!" -ForegroundColor Green
Write-Host "Your Mystic Trading Platform now has all features available with Python 3.11!" -ForegroundColor Green
