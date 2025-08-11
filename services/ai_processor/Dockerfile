FROM python:3.10

WORKDIR /app

# Copy AI processor files
COPY services/ai_processor/ai_strategy_generator.py .
COPY services/ai_processor/ai_genetic_algorithm.py .
COPY services/ai_processor/ai_model_versioning.py .
COPY services/ai_processor/ai_auto_retrain.py .

# Environment file will be mounted at runtime

# Create requirements file for AI processor
COPY services/ai_processor/requirements.txt .

# Install AI-specific dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create main entry point
COPY services/ai_processor/main.py .

# Expose port for AI processor service
EXPOSE 8002

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8002/health')" || exit 1

# Start AI processor service
CMD ["python", "main.py"] 