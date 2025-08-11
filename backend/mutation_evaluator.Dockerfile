FROM mystic-backend-base:latest

# Copy requirements and install Python dependencies
COPY backend/requirements-light.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy mutation evaluator files
COPY backend/mutation_evaluator.py .
COPY backend/mutations.json .
COPY backend/mutation_leaderboard.json .
COPY .env .env

# Create logs directory
RUN mkdir -p logs

# Run the mutation evaluator
CMD ["python", "mutation_evaluator.py"]
