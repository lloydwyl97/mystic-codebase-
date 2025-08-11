FROM mystic-backend-base:latest

# Copy requirements and install Python dependencies
COPY backend/ai_strategy_executor_requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy AI strategy execution files
COPY backend/ai_strategy_execution.py .
COPY .env .env

# Create logs directory
RUN mkdir -p logs

# Run the AI strategy execution system
CMD ["python", "ai_strategy_execution.py"]
