#!/bin/bash

# Pre-commit Code Quality Check Script
# Runs essential code quality tools before committing

set -e  # Exit on any error

echo "🚀 Running Pre-commit Code Quality Checks..."

# Format code
echo "📐 Formatting code..."
poetry run black . --check --diff
poetry run ruff format . --check

# Lint code
echo "🔍 Linting code..."
poetry run ruff check .

# Security scanning (fail on high severity)
echo "🔒 Security scanning..."
poetry run bandit -r bot/ --configfile pyproject.toml --severity-level high

# Dead code detection (fail on high confidence)
echo "🧹 Dead code detection..."
poetry run vulture bot/ --min-confidence 95

echo "✅ Pre-commit checks passed!"
echo "🎯 Ready to commit your changes."