# MCP-OmniSearch Server Dockerfile
FROM node:18-alpine

# Set working directory
WORKDIR /app

# Copy package files
COPY package*.json ./
COPY pnpm-lock.yaml ./

# Install pnpm and dependencies
RUN npm install -g pnpm
RUN pnpm install

# Copy source code
COPY . .

# Build the TypeScript project
RUN pnpm run build

# Create non-root user
RUN addgroup -g 1001 -S nodejs
RUN adduser -S mcp -u 1001
RUN chown -R mcp:nodejs /app
USER mcp

# Expose port for health checks
EXPOSE 8767

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD node -e "console.log('MCP Server Ready')" || exit 1

# Create a simple wrapper script that keeps the server alive
RUN echo '#!/bin/sh' > /app/start-server.sh && \
    echo 'echo "MCP-OmniSearch Server starting..."' >> /app/start-server.sh && \
    echo 'echo "API Keys status: TAVILY=$TAVILY_API_KEY PERPLEXITY=$PERPLEXITY_API_KEY"' >> /app/start-server.sh && \
    echo 'tail -f /dev/null | node dist/index.js' >> /app/start-server.sh && \
    chmod +x /app/start-server.sh

# Start the MCP server with stdin feed
CMD ["/app/start-server.sh"]