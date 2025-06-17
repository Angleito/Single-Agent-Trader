#!/bin/sh

# Runtime environment variable injection for containerized frontend
# This script injects environment variables into the built frontend at runtime

set -e

# Default values
DEFAULT_API_URL=""
DEFAULT_WS_URL=""
DEFAULT_DOCKER_ENV="false"

# Get environment variables with defaults
API_URL="${VITE_API_URL:-$DEFAULT_API_URL}"
WS_URL="${VITE_WS_URL:-$DEFAULT_WS_URL}"
DOCKER_ENV="${VITE_DOCKER_ENV:-$DEFAULT_DOCKER_ENV}"

# Path to the main JavaScript bundle (adjust as needed)
MAIN_JS_PATH="/usr/share/nginx/html/assets"

echo "🚀 Injecting runtime environment variables..."
echo "   API_URL: ${API_URL}"
echo "   WS_URL: ${WS_URL}"
echo "   DOCKER_ENV: ${DOCKER_ENV}"

# Create a JavaScript snippet with environment variables
ENV_JS="
window.__RUNTIME_CONFIG__ = {
  API_URL: '${API_URL}',
  WS_URL: '${WS_URL}',
  DOCKER_ENV: '${DOCKER_ENV}'
};
"

# Inject into index.html
if [ -f "/usr/share/nginx/html/index.html" ]; then
  # Create the environment script
  echo "${ENV_JS}" > /usr/share/nginx/html/runtime-env.js
  
  # Inject script tag into index.html if not already present
  if ! grep -q "runtime-env.js" /usr/share/nginx/html/index.html; then
    sed -i 's|<head>|<head><script src="/runtime-env.js"></script>|' /usr/share/nginx/html/index.html
  fi
  
  echo "✅ Environment variables injected successfully"
else
  echo "❌ index.html not found, skipping injection"
fi

# Set global variables for the JavaScript to access
cat > /usr/share/nginx/html/env-config.js << EOF
// Runtime environment configuration
window.__API_URL__ = '${API_URL}';
window.__WS_URL__ = '${WS_URL}';
window.__DOCKER_ENV__ = '${DOCKER_ENV}';
EOF

echo "✅ Runtime environment injection completed"