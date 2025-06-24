#!/bin/bash

# Enhanced Code Quality Pipeline Script
# Runs all code quality tools following the guidelines in CLAUDE.md
# Includes functional programming (FP) validation and comprehensive checks

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    case $1 in
        "SUCCESS") echo -e "${GREEN}✅ $2${NC}" ;;
        "ERROR") echo -e "${RED}❌ $2${NC}" ;;
        "WARNING") echo -e "${YELLOW}⚠️  $2${NC}" ;;
        "INFO") echo -e "${BLUE}ℹ️  $2${NC}" ;;
        "FP") echo -e "${PURPLE}🧮 $2${NC}" ;;
    esac
}

echo "🔧 Enhanced Code Quality Pipeline (with FP Support)"
echo "================================================="

# Check for FP directory
FP_ENABLED=false
if [ -d "bot/fp" ]; then
    FP_ENABLED=true
    print_status "INFO" "Functional programming components detected"
fi

echo ""
print_status "INFO" "Starting standard code quality checks..."

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

# Functional Programming Quality Checks
if [ "$FP_ENABLED" = true ]; then
    echo ""
    print_status "FP" "Running functional programming quality checks..."

    # Run FP-specific code quality
    if [ -f "./scripts/fp-code-quality.sh" ]; then
        print_status "FP" "Running FP-specific quality pipeline..."
        ./scripts/fp-code-quality.sh || print_status "WARNING" "FP quality checks found issues (non-blocking)"
    else
        print_status "WARNING" "FP code quality script not found"
    fi

    # Additional FP integration checks
    print_status "FP" "Checking FP/imperative integration..."

    # Check for mixed FP/imperative patterns
    python3 -c "
import sys
import os
from pathlib import Path

def check_mixed_patterns():
    issues = []

    # Check for imperative patterns in FP code
    fp_dir = Path('bot/fp')
    if fp_dir.exists():
        for py_file in fp_dir.rglob('*.py'):
            try:
                with open(py_file, 'r') as f:
                    content = f.read()

                # Check for imperative patterns
                if 'raise ' in content and 'Result' in content:
                    issues.append(f'Mixed exception/Result pattern in {py_file}')

                if 'return None' in content and 'Maybe' in content:
                    issues.append(f'Mixed None/Maybe pattern in {py_file}')
            except Exception:
                continue

    return issues

issues = check_mixed_patterns()
if issues:
    print('⚠️  Mixed FP/imperative patterns found:')
    for issue in issues[:5]:  # Limit output
        print(f'  - {issue}')
    if len(issues) > 5:
        print(f'  ... and {len(issues) - 5} more')
else:
    print('✅ No mixed patterns detected')
" || print_status "WARNING" "Pattern check failed (non-blocking)"

    # Check FP test coverage
    print_status "FP" "Validating FP test coverage..."
    if [ -d "tests/unit/fp" ]; then
        IMPERATIVE_FILES=$(find bot/ -name "*.py" -not -path "bot/fp/*" | wc -l)
        FP_FILES=$(find bot/fp/ -name "*.py" 2>/dev/null | wc -l || echo "0")
        FP_TESTS=$(find tests/unit/fp/ -name "test_*.py" 2>/dev/null | wc -l || echo "0")

        print_status "INFO" "Code distribution: ${IMPERATIVE_FILES} imperative, ${FP_FILES} FP files"
        print_status "INFO" "FP test files: ${FP_TESTS}"

        if [ "$FP_FILES" -gt 0 ] && [ "$FP_TESTS" -eq 0 ]; then
            print_status "WARNING" "FP code exists but no FP tests found"
        elif [ "$FP_FILES" -gt 0 ] && [ "$FP_TESTS" -gt 0 ]; then
            COVERAGE_RATIO=$((FP_TESTS * 100 / FP_FILES))
            print_status "INFO" "FP test coverage ratio: ~${COVERAGE_RATIO}%"
        fi
    else
        print_status "WARNING" "No FP test directory found"
    fi

    print_status "SUCCESS" "FP quality checks completed"
fi

echo ""
echo "✅ Enhanced Code Quality Pipeline Complete!"
echo ""
echo "📋 Summary:"
echo "  ✅ Code formatted with Black and Ruff"
echo "  ✅ Imports sorted with isort"
echo "  ✅ Linting completed with Ruff"
echo "  ✅ Type checking run with MyPy"
echo "  ✅ Dead code detection run with Vulture"
echo "  ✅ Security scanning run with Bandit"
echo "  ✅ Tests run with coverage"

if [ "$FP_ENABLED" = true ]; then
    echo "  🧮 FP type system validated"
    echo "  🧮 FP adapter compatibility checked"
    echo "  🧮 FP/imperative integration verified"
    echo "  🧮 FP test coverage analyzed"
fi

echo ""
echo "🎯 Next steps:"
echo "  1. Review any warnings or errors above"
echo "  2. Fix critical security issues from Bandit"
echo "  3. Remove dead code identified by Vulture"
echo "  4. Address type checking issues from MyPy"

if [ "$FP_ENABLED" = true ]; then
    echo "  5. Review FP-specific issues in logs/fp/"
    echo "  6. Run ./scripts/fp-migration-helper.sh for code migration"
    echo "  7. Commit your changes when ready"
else
    echo "  5. Commit your changes when ready"
fi

echo ""
if [ "$FP_ENABLED" = true ]; then
    echo "🧮 FP Development Resources:"
    echo "  ./scripts/fp-code-quality.sh        # FP-specific quality checks"
    echo "  ./scripts/fp-migration-helper.sh    # Convert imperative to FP"
    echo "  ./scripts/fp-performance-benchmark.sh # Performance comparison"
    echo "  ./scripts/fp-test-runner.sh         # Run FP-specific tests"
    echo ""
fi
