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
