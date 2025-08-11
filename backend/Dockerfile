FROM python:3.10

WORKDIR /app

# Copy requirements and install
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create ai directory structure
RUN mkdir -p ai/ai

# Copy the backend application code
COPY backend/ .

# Copy missing ai directory files from root
COPY ai/ai/persistent_cache.py ./ai/ai/persistent_cache.py
COPY ai/persistent_cache.py ./ai/persistent_cache.py

# Expose port
EXPOSE 9000

# Command to run the backend application
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9000"]
