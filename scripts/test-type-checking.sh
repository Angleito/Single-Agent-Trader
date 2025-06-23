#!/bin/bash
# Test type checking configuration for the AI Trading Bot

set -e

echo "🔍 Testing Type Checking Configuration"
echo "======================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    local status=$1
    local message=$2
    if [ "$status" = "success" ]; then
        echo -e "${GREEN}✓${NC} $message"
    elif [ "$status" = "error" ]; then
        echo -e "${RED}✗${NC} $message"
    elif [ "$status" = "warning" ]; then
        echo -e "${YELLOW}⚠${NC} $message"
    elif [ "$status" = "info" ]; then
        echo -e "${BLUE}ℹ${NC} $message"
    fi
}

# Check if we're in the project root
if [ ! -f "pyproject.toml" ]; then
    print_status "error" "Must run from project root directory"
    exit 1
fi

# Install type checking dependencies
print_status "info" "Installing type checking dependencies..."
poetry install --only dev

# Create test directories if they don't exist
mkdir -p reports

# Run MyPy with strict configuration
print_status "info" "Running MyPy type checker..."
if poetry run mypy bot/ --config-file pyproject.toml --html-report reports/mypy-html --txt-report reports/mypy-txt; then
    print_status "success" "MyPy passed with strict configuration"
else
    print_status "warning" "MyPy found type errors (see reports/mypy-html/index.html)"
fi

# Run MyPy on specific critical modules with extra strictness
print_status "info" "Running MyPy on critical modules with maximum strictness..."
CRITICAL_MODULES=(
    "bot/config.py"
    "bot/trading_types.py"
    "bot/risk/types.py"
    "bot/exchange/base.py"
    "bot/validation/"
)

for module in "${CRITICAL_MODULES[@]}"; do
    if [ -e "$module" ]; then
        if poetry run mypy "$module" --disallow-any-expr --disallow-any-decorated --disallow-any-explicit; then
            print_status "success" "Strict type checking passed for $module"
        else
            print_status "warning" "Type issues in critical module: $module"
        fi
    fi
done

# Run Pyright if available
if command -v pyright &> /dev/null; then
    print_status "info" "Running Pyright type checker..."
    if pyright --project .; then
        print_status "success" "Pyright passed with strict configuration"
    else
        print_status "warning" "Pyright found type errors"
    fi
else
    print_status "warning" "Pyright not installed. Install with: npm install -g pyright"
fi

# Check for type stub files
print_status "info" "Checking type stub files..."
STUB_DIR="bot/types/stubs"
if [ -d "$STUB_DIR" ]; then
    stub_count=$(find "$STUB_DIR" -name "*.pyi" | wc -l)
    print_status "success" "Found $stub_count type stub files in $STUB_DIR"
    find "$STUB_DIR" -name "*.pyi" -exec basename {} \; | sort
else
    print_status "error" "Type stubs directory not found at $STUB_DIR"
fi

# Run pre-commit hooks for type checking
print_status "info" "Running pre-commit type checking hooks..."
if poetry run pre-commit run mypy --all-files; then
    print_status "success" "Pre-commit MyPy hook passed"
else
    print_status "warning" "Pre-commit MyPy hook found issues"
fi

# Generate type checking report
print_status "info" "Generating type checking summary..."
cat > reports/type-checking-summary.txt << EOF
Type Checking Configuration Summary
===================================
Generated: $(date)

MyPy Configuration:
- Python version: 3.12
- Strict mode: Enabled
- Disallow untyped defs: Yes
- Disallow incomplete defs: Yes
- Check untyped defs: Yes
- Warn return any: Yes
- Strict equality: Yes

Pyright Configuration:
- Type checking mode: strict
- Python version: 3.12
- Stub path: bot/types/stubs

Type Stubs Available:
$(find "$STUB_DIR" -name "*.pyi" -exec basename {} \; 2>/dev/null | sort | sed 's/^/- /')

Critical Modules with Strict Typing:
$(printf '%s\n' "${CRITICAL_MODULES[@]}" | sed 's/^/- /')

External Dependencies with Type Support:
- pandas (via pandas-stubs)
- numpy (built-in types)
- pydantic (built-in types)
- requests (via types-requests)
- aiofiles (via types-aiofiles)
- python-dateutil (via types-python-dateutil)

External Dependencies with Custom Stubs:
- websockets
- pandas_ta
- coinbase_advanced_py
- aiohttp
- docker
- psutil
- ccxt
EOF

print_status "success" "Type checking summary saved to reports/type-checking-summary.txt"

# Check for common type issues
print_status "info" "Checking for common type issues..."

# Check for Any imports
any_count=$(grep -r "from typing import.*Any" bot/ --include="*.py" | wc -l || echo 0)
if [ "$any_count" -gt 0 ]; then
    print_status "warning" "Found $any_count files importing 'Any' type"
fi

# Check for type: ignore comments
ignore_count=$(grep -r "type: ignore" bot/ --include="*.py" | wc -l || echo 0)
if [ "$ignore_count" -gt 0 ]; then
    print_status "warning" "Found $ignore_count 'type: ignore' comments"
fi

# Check for missing type annotations
print_status "info" "Checking for missing type annotations..."
poetry run mypy bot/ --disallow-untyped-defs --no-error-summary 2>&1 | grep -c "error: Function is missing a type annotation" || true

echo ""
print_status "info" "Type checking configuration test complete!"
print_status "info" "View detailed reports in the 'reports' directory"

# Exit with appropriate code
if [ -f "reports/mypy-txt/index.txt" ] && grep -q "Success: no issues found" "reports/mypy-txt/index.txt"; then
    exit 0
else
    exit 1
fi
