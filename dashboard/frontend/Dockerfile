# Production-optimized Dockerfile for AI Trading Bot Dashboard

# Build stage
FROM --platform=$BUILDPLATFORM node:18-alpine AS builder

WORKDIR /app

# Install minimal build dependencies
RUN apk add --no-cache python3 make g++ gcc libc-dev

# Copy package files and install dependencies
COPY package*.json ./
RUN npm ci --no-audit --no-fund && \
    npm rebuild esbuild --update-binary

# Copy source code and configuration
COPY src/ ./src/
COPY public/ ./public/
COPY index.html vite.config.ts tsconfig.json ./

# Build the application
RUN npm run build

# Remove development dependencies and source files to reduce size
RUN rm -rf node_modules src tsconfig.json vite.config.ts

# Production stage
FROM nginx:alpine AS production

# Create non-root user for security
RUN addgroup -g 1001 -S nodejs && \
    adduser -S nextjs -u 1001

# Copy built assets from builder stage
COPY --from=builder /app/dist /usr/share/nginx/html

# Copy optimized nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Set proper permissions
RUN chown -R nginx:nginx /usr/share/nginx/html && \
    chmod -R 755 /usr/share/nginx/html

# Create directories for nginx
RUN mkdir -p /var/cache/nginx /var/log/nginx /var/run/nginx && \
    chown -R nginx:nginx /var/cache/nginx /var/log/nginx /var/run/nginx

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:8080/ || exit 1

# Start nginx
CMD ["nginx", "-g", "daemon off;"]