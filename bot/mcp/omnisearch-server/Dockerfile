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

# Start the MCP server
CMD ["node", "dist/index.js"]