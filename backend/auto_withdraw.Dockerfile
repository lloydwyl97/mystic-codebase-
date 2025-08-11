FROM mystic-backend-base:latest

# Copy requirements and install Python dependencies
COPY backend/auto_withdraw_requirements.txt auto_withdraw_requirements.txt
RUN pip install --no-cache-dir -r auto_withdraw_requirements.txt

# Copy auto-withdraw files
COPY backend/auto_withdraw.py .
COPY .env .env

# Create logs directory
RUN mkdir -p logs

# Run the auto-withdraw system
CMD ["python", "auto_withdraw.py"]
