FROM mystic-backend-base:latest

# Copy requirements and install Python dependencies
COPY backend/ai_leaderboard_executor_requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy AI leaderboard executor files
COPY backend/ai_leaderboard_executor.py .
COPY backend/ai_strategy_execution.py .
COPY backend/mutation_leaderboard.json .
COPY .env .env

# Create logs directory
RUN mkdir -p logs

# Run the AI leaderboard executor
CMD ["python", "ai_leaderboard_executor.py"]
