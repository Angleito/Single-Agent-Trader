FROM python:3.12-alpine

WORKDIR /app

# Install minimal dependencies
RUN apk add --no-cache curl && \
    pip install --no-cache-dir fastapi uvicorn python-dotenv structlog requests

# Copy memory server code
COPY ../memory_server.py /app/memory_server.py

# Create data directory
RUN mkdir -p /app/data

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8765/health')" || exit 1

# Expose port
EXPOSE 8765

# Start the memory server
CMD ["python", "-u", "memory_server.py"]
