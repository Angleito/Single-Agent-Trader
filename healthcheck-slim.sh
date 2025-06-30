#!/bin/sh
# Minimal Health Check
if ! curl -f http://localhost:8080/health; then
  echo "Health check failed."
  exit 1
fi
exit 0
