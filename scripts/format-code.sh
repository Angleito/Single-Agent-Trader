#!/bin/bash
# Auto-format all Python code in the project

echo "🔧 Running ruff fixes..."
poetry run ruff check --fix .

echo "🎨 Running ruff formatter..."
poetry run ruff format .

echo "⚫ Running black formatter..."
poetry run black .

echo "✅ Code formatting complete!"