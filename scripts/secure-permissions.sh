#!/bin/bash

# Set secure file permissions for the AI Trading Bot project

echo "Setting secure file permissions..."

# Set restrictive permissions on sensitive files
if [ -f ".env" ]; then
    chmod 600 .env
    echo "✓ Set .env permissions to 600"
fi

if [ -f ".env.local" ]; then
    chmod 600 .env.local
    echo "✓ Set .env.local permissions to 600"
fi

# Set permissions on scripts
find scripts -name "*.sh" -exec chmod 755 {} \;
echo "✓ Set executable permissions on shell scripts"

# Set permissions on Python files
find . -name "*.py" -exec chmod 644 {} \;
echo "✓ Set read permissions on Python files"

# Create logs directory with proper permissions
mkdir -p logs
chmod 755 logs
echo "✓ Created/verified logs directory"

# Set permissions on configuration files
find config -name "*.json" -exec chmod 644 {} \;
echo "✓ Set read permissions on config files"

echo "File permissions secured!"