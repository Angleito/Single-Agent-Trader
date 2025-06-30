# Production-optimized Dockerfile for AI Trading Bot Dashboard

# Build stage - Force x64 platform for consistent builds
FROM node:18-alpine AS builder
ARG TARGETPLATFORM
ARG TARGETARCH
ARG TARGETOS

WORKDIR /app

# Install minimal build dependencies
RUN apk add --no-cache python3 make g++ gcc libc-dev

# Copy package files and npm configuration
COPY package*.json .npmrc ./

# Install dependencies with platform specification
RUN TARGETPLATFORM=linux/amd64 TARGETARCH=amd64 npm ci --no-audit --no-fund && \
    npm rebuild esbuild --update-binary

# Copy source code and configuration
COPY src/ ./src/
COPY public/ ./public/
COPY index.html vite.config.ts tsconfig.json ./

# Build the application
RUN npm run build

# Remove development dependencies and source files to reduce size
RUN rm -rf node_modules src tsconfig.json vite.config.ts

# Production stage - Force x64 platform for consistent deployment
FROM nginx:alpine AS production

# Update nginx user to UID 1000 to match docker-compose expectations
# The nginx:alpine image already has nginx user, but we need to change its UID
RUN deluser nginx && \
    addgroup -g 1000 -S nginx && \
    adduser -S nginx -u 1000 -G nginx -h /var/cache/nginx -s /sbin/nologin

# Copy built assets from builder stage to a temporary location
COPY --from=builder /app/dist /app/dist-static

# Copy optimized nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy nginx main configuration
COPY nginx-main.conf /etc/nginx/nginx.conf

# Copy runtime environment injection script
COPY inject-env.sh /docker-entrypoint.d/inject-env.sh
RUN chmod +x /docker-entrypoint.d/inject-env.sh

# Copy startup script to copy assets to writable directory
COPY copy-assets.sh /docker-entrypoint.d/00-copy-assets.sh
RUN chmod +x /docker-entrypoint.d/00-copy-assets.sh

# Set proper permissions for static assets
RUN chown -R nginx:nginx /app/dist-static && \
    chmod -R 755 /app/dist-static

# Create and configure nginx directories with proper ownership (simplified for user 1000)
RUN mkdir -p /var/log/nginx && \
    chown -R nginx:nginx /var/log/nginx && \
    chmod -R 755 /var/log/nginx && \
    chown -R nginx:nginx /etc/nginx && \
    chmod -R 755 /etc/nginx

# Switch to non-root user
USER 1000

# Expose port
EXPOSE 8080

# Health check - using wget which is available in nginx:alpine
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:8080/ || exit 1

# Start nginx as non-root user
CMD ["nginx", "-g", "daemon off;"]
