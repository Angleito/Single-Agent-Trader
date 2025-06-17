# MCP-OmniSearch Server Dockerfile
FROM node:18-alpine

# Set working directory
WORKDIR /app

# Install pnpm and wget
RUN apk add --no-cache wget
RUN npm install -g pnpm

# Copy package files
COPY package*.json ./
COPY pnpm-lock.yaml ./

# Install dependencies
RUN pnpm install

# Copy only source files (not node_modules)
COPY tsconfig.json ./
COPY src ./src
COPY http_wrapper.js ./

# Build the TypeScript project
RUN pnpm run build

# Create non-root user
RUN addgroup -g 1001 -S nodejs
RUN adduser -S mcp -u 1001
RUN chown -R mcp:nodejs /app
USER mcp

# Expose port for health checks
EXPOSE 8767

# Health check using HTTP endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:8767/health || exit 1

# Create a simple wrapper script that starts the HTTP wrapper
RUN echo '#!/bin/sh' > /app/start-server.sh && \
    echo 'echo "MCP-OmniSearch HTTP Wrapper starting..."' >> /app/start-server.sh && \
    echo 'echo "API Keys status: TAVILY=$TAVILY_API_KEY PERPLEXITY=$PERPLEXITY_API_KEY"' >> /app/start-server.sh && \
    echo 'node http_wrapper.js' >> /app/start-server.sh && \
    chmod +x /app/start-server.sh

# Start the HTTP wrapper which will spawn the MCP server
CMD ["/app/start-server.sh"]
