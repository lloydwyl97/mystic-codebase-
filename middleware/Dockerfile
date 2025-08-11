FROM python:3.10

WORKDIR /app

# Copy middleware-specific requirements and install
COPY middleware/middleware.txt .
RUN pip install --no-cache-dir -r middleware.txt

# Copy only specific middleware files
COPY middleware/middleware_router.py .
COPY middleware/__init__.py .

# Copy environment file
COPY .env .

# Run middleware service
CMD ["python", "middleware_router.py"] 