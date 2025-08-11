# ===== MYSTIC BASE IMAGE - SHARED ACROSS ALL SERVICES =====
FROM python:3.10

# Set working directory
WORKDIR /app

# Install system dependencies (common across all services)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install base requirements
COPY requirements/base.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r base.txt

# Set common environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Create common directories
RUN mkdir -p /app/logs /app/data /app/cache

# Base image is ready - services will inherit from this 