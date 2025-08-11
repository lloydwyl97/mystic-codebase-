FROM python:3.10

WORKDIR /app

# Copy alerts-specific requirements and install
COPY alerts/alerts.txt .
RUN pip install --no-cache-dir -r alerts.txt

# Copy only specific alert files
COPY alerts/notify_bot.py .
COPY alerts/__init__.py .

# Copy environment file
COPY .env .

# Run alerts service
CMD ["python", "notify_bot.py"] 