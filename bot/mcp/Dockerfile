# Dockerfile for MCP Memory Server
FROM python:3.12-alpine

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apk add --no-cache curl

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data and logs directories
RUN mkdir -p /app/data /app/logs

# Create non-root user for security
RUN addgroup -g 1000 memoryserver && \
    adduser -u 1000 -G memoryserver -s /bin/sh -D memoryserver && \
    chown -R memoryserver:memoryserver /app

# Switch to non-root user
USER memoryserver

# Expose MCP server port
EXPOSE 8765

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8765/health || exit 1

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV MCP_SERVER_PORT=8765

# Run the memory server
CMD ["python", "server_standalone.py"]
