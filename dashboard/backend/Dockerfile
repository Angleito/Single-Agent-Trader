FROM python:3.12-alpine

# Set working directory
WORKDIR /app

# Create a non-root user with UID 1000 (matching docker-compose)
# This prevents permission conflicts when docker-compose overrides the user
RUN addgroup -g 1000 backend && \
    adduser -u 1000 -G backend -s /bin/sh -D backend

# Install system dependencies
RUN apk add --no-cache \
    docker-cli \
    curl

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .
COPY tradingview_feed.py .
COPY log_parser.py .
COPY llm_log_parser.py .
COPY rate_limiter.py .
COPY start.sh .

# Make startup script executable
RUN chmod +x start.sh

# Change ownership of the entire /app directory to the non-root user
# This ensures all files are accessible when running as UID 1000
RUN chown -R backend:backend /app

# Switch to the non-root user
USER backend

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["python", "main.py"]
