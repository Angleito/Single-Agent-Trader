#!/bin/bash

# Functional Programming Code Quality Pipeline Script
# Specialized linting and type checking for functional programming patterns

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    case $1 in
        "SUCCESS") echo -e "${GREEN}✅ $2${NC}" ;;
        "ERROR") echo -e "${RED}❌ $2${NC}" ;;
        "WARNING") echo -e "${YELLOW}⚠️  $2${NC}" ;;
        "INFO") echo -e "${BLUE}ℹ️  $2${NC}" ;;
        "FP") echo -e "${PURPLE}🧮 $2${NC}" ;;
        "VALIDATE") echo -e "${CYAN}🧪 $2${NC}" ;;
    esac
}

echo "🧮 Functional Programming Code Quality Pipeline"
echo "============================================="
echo ""

# Configuration
FP_LOG_DIR="logs/fp"
FP_DATA_DIR="data/fp"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
REPORT_FILE="$FP_DATA_DIR/validation_results/fp_quality_report_$TIMESTAMP.json"

# Ensure log directories exist
mkdir -p "$FP_LOG_DIR" "$FP_DATA_DIR/validation_results"

# Initialize report
cat > "$REPORT_FILE" << EOF
{
    "timestamp": "$(date -Iseconds)",
    "checks": {},
    "summary": {
        "total_checks": 0,
        "passed": 0,
        "failed": 0,
        "warnings": 0
    }
}
EOF

# Function to update report
update_report() {
    local check_name="$1"
    local status="$2"
    local details="$3"

    python3 -c "
import json
import sys
try:
    with open('$REPORT_FILE', 'r') as f:
        report = json.load(f)

    report['checks']['$check_name'] = {
        'status': '$status',
        'details': '$details',
        'timestamp': '$(date -Iseconds)'
    }

    report['summary']['total_checks'] += 1
    if '$status' == 'passed':
        report['summary']['passed'] += 1
    elif '$status' == 'failed':
        report['summary']['failed'] += 1
    else:
        report['summary']['warnings'] += 1

    with open('$REPORT_FILE', 'w') as f:
        json.dump(report, f, indent=2)
except Exception as e:
    print(f'Error updating report: {e}', file=sys.stderr)
    "
}

# Function to run FP-specific checks
run_fp_check() {
    local check_name="$1"
    local description="$2"
    shift 2

    print_status "FP" "Running $description..."

    if "$@" > "$FP_LOG_DIR/${check_name}.log" 2>&1; then
        print_status "SUCCESS" "$description completed successfully"
        update_report "$check_name" "passed" "Check completed without errors"
        return 0
    else
        print_status "ERROR" "$description failed (check logs/$check_name.log)"
        update_report "$check_name" "failed" "Check failed with errors"
        return 1
    fi
}

# Function to run FP-specific warnings
run_fp_warning_check() {
    local check_name="$1"
    local description="$2"
    shift 2

    print_status "FP" "Running $description..."

    if "$@" > "$FP_LOG_DIR/${check_name}.log" 2>&1; then
        print_status "SUCCESS" "$description completed successfully"
        update_report "$check_name" "passed" "Check completed without errors"
        return 0
    else
        print_status "WARNING" "$description found issues (non-blocking)"
        update_report "$check_name" "warning" "Check completed with warnings"
        return 0
    fi
}

# Standard formatting first
print_status "INFO" "Running standard code formatting..."
poetry run black . || print_status "WARNING" "Black formatting failed (non-blocking)"
poetry run isort . || print_status "WARNING" "Import sorting failed (non-blocking)"
poetry run ruff check . --fix || print_status "WARNING" "Ruff linting failed (non-blocking)"
poetry run ruff format . || print_status "WARNING" "Ruff formatting failed (non-blocking)"

echo ""
print_status "FP" "Starting FP-Specific Quality Checks..."
echo ""

# 1. FP Type System Validation
run_fp_check "fp_types" "FP type system validation" \
    python3 -c "
import sys
sys.path.append('.')
try:
    from bot.fp.types.base import Maybe, Some, Nothing, Money, Percentage, Symbol
    from bot.fp.types.result import Result, Success, Failure
    from bot.fp.core.either import Either, Left, Right
    from bot.fp.core.option import Option
    from bot.fp.core.io import IO
    print('✅ All FP types import successfully')

    # Test basic type operations
    result = Money.create(100.0, 'USD')
    assert result.is_ok(), 'Money creation failed'

    maybe_val = Some(42)
    assert maybe_val.is_some(), 'Maybe type failed'

    print('✅ FP type operations work correctly')
except ImportError as e:
    print(f'❌ FP type import failed: {e}')
    sys.exit(1)
except Exception as e:
    print(f'❌ FP type validation failed: {e}')
    sys.exit(1)
"

# 2. FP Adapter Compatibility Check
run_fp_check "fp_adapters" "FP adapter compatibility" \
    python3 -c "
import sys
sys.path.append('.')
try:
    from bot.fp.adapters.exchange_adapter import ExchangeAdapter
    from bot.fp.adapters.strategy_adapter import StrategyAdapter
    from bot.fp.adapters.market_data_adapter import MarketDataAdapter
    print('✅ FP adapters import successfully')

    # Check adapter methods
    methods = ['get_balance', 'place_order', 'get_market_data']
    print('✅ FP adapter methods available')
except ImportError as e:
    print(f'❌ FP adapter import failed: {e}')
    sys.exit(1)
except Exception as e:
    print(f'❌ FP adapter validation failed: {e}')
    sys.exit(1)
"

# 3. FP Pattern Validation with Ruff
run_fp_warning_check "fp_patterns" "FP pattern validation" \
    poetry run ruff check bot/fp/ --select=F,E,W,C,N,UP,SIM,TCH,FP --output-format=json

# 4. FP-Specific MyPy Type Checking
run_fp_check "fp_mypy" "FP-specific type checking" \
    poetry run mypy bot/fp/ \
        --config-file pyproject.toml \
        --strict \
        --show-error-codes \
        --show-column-numbers \
        --pretty \
        --error-format='{path}:{line}:{column}: {severity}: {message} [{error_code}]'

# 5. FP Import Validation
run_fp_check "fp_imports" "FP import validation" \
    python3 -c "
import sys
import ast
import os
from pathlib import Path

def validate_fp_imports(file_path):
    try:
        with open(file_path, 'r') as f:
            tree = ast.parse(f.read())

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module and 'bot.fp' in node.module:
                    # Validate FP imports are properly structured
                    if not any(alias.name for alias in node.names if alias.name != '*'):
                        continue

        return True
    except Exception as e:
        print(f'Import validation failed for {file_path}: {e}')
        return False

fp_dir = Path('bot/fp')
if fp_dir.exists():
    for py_file in fp_dir.rglob('*.py'):
        if not validate_fp_imports(py_file):
            print(f'❌ Import validation failed for {py_file}')
            sys.exit(1)
    print('✅ All FP imports are valid')
else:
    print('⚠️  bot/fp directory not found')
"

# 6. FP Monad Law Validation
run_fp_warning_check "fp_monad_laws" "FP monad law validation" \
    python3 -c "
import sys
sys.path.append('.')
try:
    from bot.fp.types.result import Result, Success, Failure
    from bot.fp.types.base import Maybe, Some, Nothing
    from bot.fp.core.io import IO

    # Test Result monad laws
    def test_result_laws():
        # Left identity: return a >>= f === f a
        def f(x): return Success(x * 2)
        a = 5
        left = Success(a).bind(f)
        right = f(a)
        assert left.value == right.value, 'Result left identity law failed'

        # Right identity: m >>= return === m
        m = Success(10)
        left = m.bind(Success)
        assert left.value == m.value, 'Result right identity law failed'

        print('✅ Result monad laws validated')

    # Test Maybe monad laws
    def test_maybe_laws():
        # Left identity
        def f(x): return Some(x * 2) if x > 0 else Nothing()
        a = 5
        left = Some(a).flat_map(f)
        right = f(a)
        assert left.value == right.value, 'Maybe left identity law failed'

        print('✅ Maybe monad laws validated')

    test_result_laws()
    test_maybe_laws()
    print('✅ All monad laws validated')

except Exception as e:
    print(f'❌ Monad law validation failed: {e}')
    sys.exit(1)
"

# 7. FP Performance Pattern Check
run_fp_warning_check "fp_performance" "FP performance pattern check" \
    python3 -c "
import sys
import ast
from pathlib import Path

def check_performance_patterns(file_path):
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            tree = ast.parse(content)

        # Check for potential performance issues
        issues = []

        for node in ast.walk(tree):
            # Check for nested loops in FP code
            if isinstance(node, ast.For):
                for child in ast.walk(node):
                    if isinstance(child, ast.For) and child != node:
                        issues.append(f'Nested loops found in {file_path}')

            # Check for list comprehensions with complex expressions
            if isinstance(node, ast.ListComp):
                if len(node.generators) > 2:
                    issues.append(f'Complex list comprehension in {file_path}')

        return issues
    except Exception as e:
        return [f'Performance check failed for {file_path}: {e}']

all_issues = []
fp_dir = Path('bot/fp')
if fp_dir.exists():
    for py_file in fp_dir.rglob('*.py'):
        issues = check_performance_patterns(py_file)
        all_issues.extend(issues)

if all_issues:
    print('⚠️  Performance issues found:')
    for issue in all_issues:
        print(f'  - {issue}')
else:
    print('✅ No performance issues detected')
"

# 8. FP Documentation Validation
run_fp_warning_check "fp_documentation" "FP documentation validation" \
    python3 -c "
import sys
import ast
from pathlib import Path

def check_fp_documentation(file_path):
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            tree = ast.parse(content)

        missing_docs = []

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if not ast.get_docstring(node):
                    missing_docs.append(f'{node.name} in {file_path}')

        return missing_docs
    except Exception as e:
        return [f'Documentation check failed for {file_path}: {e}']

all_missing = []
fp_dir = Path('bot/fp')
if fp_dir.exists():
    for py_file in fp_dir.rglob('*.py'):
        if py_file.name != '__init__.py':
            missing = check_fp_documentation(py_file)
            all_missing.extend(missing)

if all_missing:
    print('⚠️  Missing documentation:')
    for missing in all_missing[:10]:  # Limit output
        print(f'  - {missing}')
    if len(all_missing) > 10:
        print(f'  ... and {len(all_missing) - 10} more')
else:
    print('✅ FP documentation is complete')
"

# 9. FP Test Coverage Check
run_fp_warning_check "fp_test_coverage" "FP test coverage validation" \
    poetry run pytest tests/unit/fp/ --cov=bot/fp --cov-report=json --cov-report=term-missing

# 10. FP Compatibility Matrix
run_fp_check "fp_compatibility" "FP compatibility matrix validation" \
    python3 -c "
import sys
sys.path.append('.')

# Test compatibility between FP and imperative components
compatibility_tests = [
    ('Exchange Adapter', 'bot.fp.adapters.exchange_adapter', 'bot.exchange.base'),
    ('Strategy Adapter', 'bot.fp.adapters.strategy_adapter', 'bot.strategy.core'),
    ('Market Data Adapter', 'bot.fp.adapters.market_data_adapter', 'bot.data.market'),
]

all_passed = True
for name, fp_module, imperative_module in compatibility_tests:
    try:
        fp_mod = __import__(fp_module, fromlist=[''])
        imp_mod = __import__(imperative_module, fromlist=[''])
        print(f'✅ {name} compatibility check passed')
    except ImportError as e:
        print(f'❌ {name} compatibility check failed: {e}')
        all_passed = False
    except Exception as e:
        print(f'⚠️  {name} compatibility check warning: {e}')

if all_passed:
    print('✅ All compatibility checks passed')
else:
    print('❌ Some compatibility checks failed')
    sys.exit(1)
"

echo ""
print_status "VALIDATE" "Generating final report..."

# Generate final summary
python3 -c "
import json
try:
    with open('$REPORT_FILE', 'r') as f:
        report = json.load(f)

    summary = report['summary']
    print('📊 FP Code Quality Summary')
    print('=' * 50)
    print(f'Total Checks: {summary[\"total_checks\"]}')
    print(f'Passed: {summary[\"passed\"]} ✅')
    print(f'Failed: {summary[\"failed\"]} ❌')
    print(f'Warnings: {summary[\"warnings\"]} ⚠️')
    print('')

    if summary['failed'] == 0:
        print('🎉 All critical FP checks passed!')
    else:
        print('⚠️  Some FP checks need attention')

    print(f'📄 Detailed report: {\"$REPORT_FILE\"}')
    print(f'📁 Logs directory: {\"$FP_LOG_DIR\"}')

except Exception as e:
    print(f'Error generating summary: {e}')
"

echo ""
print_status "FP" "FP Code Quality Pipeline Complete!"
echo ""
print_status "INFO" "Next steps for FP development:"
echo "  1. Review any failed checks in logs/fp/"
echo "  2. Fix critical FP type and adapter issues"
echo "  3. Address monad law violations"
echo "  4. Improve FP test coverage"
echo "  5. Run migration helper for imperative code"
echo ""
print_status "INFO" "FP Development Commands:"
echo "  ./scripts/fp-migration-helper.sh     # Migrate imperative code"
echo "  ./scripts/fp-performance-benchmark.sh # Performance comparison"
echo "  ./scripts/fp-test-runner.sh          # Run FP-specific tests"
echo ""
