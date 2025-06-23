#!/bin/bash

# Code Quality Pipeline Script
# Runs all code quality tools following the guidelines in CLAUDE.md

set -e  # Exit on any error

echo "🔧 Starting Code Quality Pipeline..."

# Format code
echo "📐 Formatting code with Black..."
poetry run black .

echo "📐 Sorting imports with isort..."
poetry run isort .

# Lint and fix basic issues
echo "🔍 Linting with Ruff (with fixes)..."
poetry run ruff check . --fix

echo "📐 Formatting with Ruff..."
poetry run ruff format .

# Type checking
echo "🔬 Type checking with MyPy (strict mode)..."
poetry run mypy bot/ --config-file pyproject.toml || echo "⚠️  Type checking issues found (non-blocking)"

# Dead code detection
echo "🧹 Dead code detection with Vulture..."
poetry run vulture bot/ --min-confidence 95 || echo "⚠️  Dead code found (non-blocking)"

# Security scanning
echo "🔒 Security scanning with Bandit..."
poetry run bandit -r bot/ --configfile pyproject.toml || echo "⚠️  Security issues found (non-blocking)"

# Run tests
echo "🧪 Running tests with coverage..."
poetry run pytest --cov=bot || echo "⚠️  Test failures found (non-blocking)"

echo "✅ Code Quality Pipeline Complete!"
echo ""
echo "📋 Summary:"
echo "  ✅ Code formatted with Black and Ruff"
echo "  ✅ Imports sorted with isort"
echo "  ✅ Linting completed with Ruff"
echo "  ✅ Type checking run with MyPy"
echo "  ✅ Dead code detection run with Vulture"
echo "  ✅ Security scanning run with Bandit"
echo "  ✅ Tests run with coverage"
echo ""
echo "🎯 Next steps:"
echo "  1. Review any warnings or errors above"
echo "  2. Fix critical security issues from Bandit"
echo "  3. Remove dead code identified by Vulture"
echo "  4. Address type checking issues from MyPy"
echo "  5. Commit your changes when ready"
