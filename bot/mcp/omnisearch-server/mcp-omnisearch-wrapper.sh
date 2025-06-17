#!/bin/bash

# MCP Omnisearch Wrapper Script (Non-Docker Version)
# This script provides a wrapper for the direct Node.js MCP server

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if node is available
if ! command -v node &> /dev/null; then
    echo "Error: Node.js is not installed or not in PATH" >&2
    exit 1
fi

# Check if dist/index.js exists
if [ ! -f "dist/index.js" ]; then
    echo "Error: dist/index.js not found. Running build..." >&2
    npm run build >&2
    if [ ! -f "dist/index.js" ]; then
        echo "Error: Build failed, dist/index.js still not found" >&2
        exit 1
    fi
fi

# Execute the MCP server
exec node dist/index.js