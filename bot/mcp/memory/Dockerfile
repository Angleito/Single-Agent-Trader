FROM python:3.12-slim

WORKDIR /app

# Install minimal dependencies
RUN pip install --no-cache-dir fastapi uvicorn python-dotenv structlog

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
