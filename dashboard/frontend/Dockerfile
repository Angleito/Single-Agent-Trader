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

# Create a simple nginx.conf that uses /tmp for all temporary directories
RUN cat > /etc/nginx/nginx.conf << 'EOF'
user nginx;
worker_processes auto;
error_log /var/log/nginx/error.log warn;
pid /tmp/nginx.pid;

events {
    worker_connections 1024;
}

http {
    include /etc/nginx/mime.types;
    default_type application/octet-stream;

    # Use /tmp for all temporary files to avoid permission issues
    client_body_temp_path /tmp/nginx_client_body;
    proxy_temp_path /tmp/nginx_proxy;
    fastcgi_temp_path /tmp/nginx_fastcgi;
    uwsgi_temp_path /tmp/nginx_uwsgi;
    scgi_temp_path /tmp/nginx_scgi;

    log_format main '$remote_addr - $remote_user [$time_local] "$request" '
                    '$status $body_bytes_sent "$http_referer" '
                    '"$http_user_agent" "$http_x_forwarded_for"';

    access_log /var/log/nginx/access.log main;

    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    keepalive_timeout 65;
    types_hash_max_size 2048;

    # Disable server_tokens for security
    server_tokens off;

    # Include server configurations
    include /etc/nginx/conf.d/*.conf;
}
EOF

# Copy runtime environment injection script
COPY inject-env.sh /docker-entrypoint.d/inject-env.sh
RUN chmod +x /docker-entrypoint.d/inject-env.sh

# Create startup script to copy assets to writable directory
RUN cat > /docker-entrypoint.d/00-copy-assets.sh << 'EOF'
#!/bin/sh
set -e
echo "📦 Copying static assets to writable directory..."
HTML_DIR="/tmp/html"
echo "Creating ${HTML_DIR} directory..."
mkdir -p ${HTML_DIR}
echo "Copying files from /app/dist-static to ${HTML_DIR}..."
if [ -d "/app/dist-static" ] && [ "$(ls -A /app/dist-static)" ]; then
    cp -r /app/dist-static/* ${HTML_DIR}/
    echo "✅ Assets copied successfully"
else
    echo "❌ Source directory /app/dist-static is empty or doesn't exist"
    ls -la /app/
fi
EOF

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
