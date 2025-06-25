#!/bin/sh

# Runtime environment variable injection for containerized frontend
# This script injects environment variables into the built frontend at runtime

set -e

# Default values for container environment
DEFAULT_API_URL="/api"
DEFAULT_WS_URL="/api/ws"
DEFAULT_DOCKER_ENV="true"

# Get environment variables with defaults (support both VITE_API_URL and VITE_API_BASE_URL)
API_URL="${VITE_API_URL:-${VITE_API_BASE_URL:-$DEFAULT_API_URL}}"
WS_URL="${VITE_WS_URL:-$DEFAULT_WS_URL}"
DOCKER_ENV="${VITE_DOCKER_ENV:-$DEFAULT_DOCKER_ENV}"

# Path to the main JavaScript bundle (adjust as needed)
MAIN_JS_PATH="/tmp/html/assets"
HTML_DIR="/tmp/html"

echo "🚀 Injecting runtime environment variables..."
echo "   API_URL: ${API_URL}"
echo "   WS_URL: ${WS_URL}"
echo "   DOCKER_ENV: ${DOCKER_ENV}"

# Create a JavaScript snippet with environment variables
ENV_JS="
window.__RUNTIME_CONFIG__ = {
  API_URL: '${API_URL}',
  API_BASE_URL: '${API_URL}',
  WS_URL: '${WS_URL}',
  DOCKER_ENV: '${DOCKER_ENV}'
};
"

# Inject into index.html
if [ -f "${HTML_DIR}/index.html" ]; then
  # Create the environment script
  echo "${ENV_JS}" > ${HTML_DIR}/runtime-env.js

  # Inject script tag into index.html if not already present
  if ! grep -q "runtime-env.js" ${HTML_DIR}/index.html; then
    sed -i 's|<head>|<head><script src="/runtime-env.js"></script>|' ${HTML_DIR}/index.html
  fi

  echo "✅ Environment variables injected successfully"
else
  echo "❌ index.html not found at ${HTML_DIR}/index.html, skipping injection"
fi

# Set global variables for the JavaScript to access (multiple naming patterns for compatibility)
# Ensure the directory exists before writing the file
mkdir -p ${HTML_DIR}
cat > ${HTML_DIR}/env-config.js << EOF
// Runtime environment configuration with multiple naming patterns
window.__API_URL__ = '${API_URL}';
window.__API_BASE_URL__ = '${API_URL}';
window.__WS_URL__ = '${WS_URL}';
window.__DOCKER_ENV__ = '${DOCKER_ENV}';

// Vite-style environment variables for compatibility
window.__VITE_API_URL__ = '${API_URL}';
window.__VITE_API_BASE_URL__ = '${API_URL}';
window.__VITE_WS_URL__ = '${WS_URL}';
window.__VITE_DOCKER_ENV__ = '${DOCKER_ENV}';
EOF

echo "✅ Runtime environment injection completed"
