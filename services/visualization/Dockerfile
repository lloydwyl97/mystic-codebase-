FROM python:3.10


WORKDIR /app

# Copy visualization files
COPY services/visualization/chart_generator.py .
COPY services/visualization/dashboard.py .
COPY services/visualization/mutation_graph.py .
COPY services/visualization/main.py .

# Copy requirements file for visualization service
COPY services/visualization/requirements.txt .

# Install visualization-specific dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy environment file
COPY .env .

# Expose port for visualization service
EXPOSE 8003

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8003/health')" || exit 1

# Start visualization service
CMD ["python", "main.py"] 