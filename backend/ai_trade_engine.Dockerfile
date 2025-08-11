FROM mystic-backend-base:latest

# Copy requirements and install Python dependencies
COPY backend/ai_trade_engine_requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy AI trade engine files
COPY backend/ai_trade_engine.py .
COPY .env .env

# Create logs directory
RUN mkdir -p logs

# Run the AI trade engine
CMD ["python", "ai_trade_engine.py"]
