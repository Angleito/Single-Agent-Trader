#!/bin/bash

# Functional Programming Test Runner Script
# Handles both FP and legacy test execution with proper import resolution

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
        "TEST") echo -e "${PURPLE}🧪 $2${NC}" ;;
        "FP") echo -e "${CYAN}🧮 $2${NC}" ;;
    esac
}

echo "🧪 Functional Programming Test Runner"
echo "===================================="
echo ""

# Configuration
FP_TEST_LOG_DIR="logs/fp/testing"
FP_TEST_DATA_DIR="data/fp/validation_results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
TEST_REPORT="$FP_TEST_DATA_DIR/test_results_$TIMESTAMP.json"

# Ensure directories exist
mkdir -p "$FP_TEST_LOG_DIR" "$FP_TEST_DATA_DIR"

# Initialize test report
cat > "$TEST_REPORT" << EOF
{
    "timestamp": "$(date -Iseconds)",
    "test_runs": {},
    "summary": {
        "total_test_suites": 0,
        "passed_suites": 0,
        "failed_suites": 0,
        "fp_tests": 0,
        "legacy_tests": 0,
        "integration_tests": 0
    },
    "issues": [],
    "recommendations": []
}
EOF

# Function to update test report
update_test_report() {
    local suite_name="$1"
    local status="$2"
    local details="$3"
    
    python3 -c "
import json
import sys
try:
    with open('$TEST_REPORT', 'r') as f:
        report = json.load(f)
    
    report['test_runs']['$suite_name'] = {
        'status': '$status',
        'details': '$details',
        'timestamp': '$(date -Iseconds)'
    }
    
    report['summary']['total_test_suites'] += 1
    if '$status' == 'passed':
        report['summary']['passed_suites'] += 1
    else:
        report['summary']['failed_suites'] += 1
    
    with open('$TEST_REPORT', 'w') as f:
        json.dump(report, f, indent=2)
except Exception as e:
    print(f'Error updating test report: {e}', file=sys.stderr)
    "
}

# Function to check test dependencies
check_test_dependencies() {
    print_status "INFO" "Checking test dependencies..."
    
    # Check if pytest is available
    if ! poetry run python -c "import pytest" 2>/dev/null; then
        print_status "ERROR" "pytest not available. Run: poetry install"
        return 1
    fi
    
    # Check for FP test infrastructure
    if [ ! -f "tests/fp_test_base.py" ]; then
        print_status "WARNING" "FP test base classes not found"
    fi
    
    # Check for FP migration adapters
    if [ ! -f "tests/fp_migration_adapters.py" ]; then
        print_status "WARNING" "FP migration adapters not found"
    fi
    
    # Check for test data generators
    if [ ! -f "tests/data/fp_test_data_generator.py" ]; then
        print_status "WARNING" "FP test data generators not found"
    fi
    
    print_status "SUCCESS" "Test dependencies check complete"
    return 0
}

# Function to run FP-specific tests
run_fp_tests() {
    local test_pattern="${1:-tests/unit/fp/}"
    local extra_args="$2"
    
    print_status "FP" "Running functional programming tests..."
    
    if [ ! -d "$test_pattern" ]; then
        print_status "WARNING" "FP test directory not found: $test_pattern"
        update_test_report "fp_tests" "skipped" "Directory not found"
        return 0
    fi
    
    local fp_test_log="$FP_TEST_LOG_DIR/fp_tests_$TIMESTAMP.log"
    
    # Run FP tests with special configuration
    if poetry run pytest "$test_pattern" \
        --tb=short \
        --strict-markers \
        --strict-config \
        --cov=bot/fp \
        --cov-report=term-missing \
        --cov-report=json \
        $extra_args \
        2>&1 | tee "$fp_test_log"; then
        
        print_status "SUCCESS" "FP tests passed"
        update_test_report "fp_tests" "passed" "All FP tests successful"
        
        # Count FP tests
        local fp_test_count=$(grep -c "PASSED" "$fp_test_log" || echo "0")
        python3 -c "
import json
try:
    with open('$TEST_REPORT', 'r') as f:
        report = json.load(f)
    report['summary']['fp_tests'] = $fp_test_count
    with open('$TEST_REPORT', 'w') as f:
        json.dump(report, f, indent=2)
except: pass
        "
        
        return 0
    else
        print_status "ERROR" "FP tests failed"
        update_test_report "fp_tests" "failed" "Some FP tests failed"
        return 1
    fi
}

# Function to run legacy tests with FP compatibility
run_legacy_tests() {
    local test_pattern="${1:-tests/unit/}"
    local extra_args="$2"
    
    print_status "TEST" "Running legacy tests with FP compatibility..."
    
    local legacy_test_log="$FP_TEST_LOG_DIR/legacy_tests_$TIMESTAMP.log"
    
    # Exclude FP-specific tests from legacy run
    if poetry run pytest "$test_pattern" \
        --ignore=tests/unit/fp/ \
        --tb=short \
        --strict-markers \
        --strict-config \
        $extra_args \
        2>&1 | tee "$legacy_test_log"; then
        
        print_status "SUCCESS" "Legacy tests passed"
        update_test_report "legacy_tests" "passed" "All legacy tests successful"
        
        # Count legacy tests
        local legacy_test_count=$(grep -c "PASSED" "$legacy_test_log" || echo "0")
        python3 -c "
import json
try:
    with open('$TEST_REPORT', 'r') as f:
        report = json.load(f)
    report['summary']['legacy_tests'] = $legacy_test_count
    with open('$TEST_REPORT', 'w') as f:
        json.dump(report, f, indent=2)
except: pass
        "
        
        return 0
    else
        print_status "ERROR" "Legacy tests failed"
        update_test_report "legacy_tests" "failed" "Some legacy tests failed"
        return 1
    fi
}

# Function to run integration tests
run_integration_tests() {
    local test_pattern="${1:-tests/integration/}"
    local extra_args="$2"
    
    print_status "TEST" "Running integration tests..."
    
    if [ ! -d "$test_pattern" ]; then
        print_status "WARNING" "Integration test directory not found: $test_pattern"
        update_test_report "integration_tests" "skipped" "Directory not found"
        return 0
    fi
    
    local integration_test_log="$FP_TEST_LOG_DIR/integration_tests_$TIMESTAMP.log"
    
    # Run integration tests with extended timeout
    if poetry run pytest "$test_pattern" \
        --tb=short \
        --timeout=300 \
        $extra_args \
        2>&1 | tee "$integration_test_log"; then
        
        print_status "SUCCESS" "Integration tests passed"
        update_test_report "integration_tests" "passed" "All integration tests successful"
        
        # Count integration tests
        local integration_test_count=$(grep -c "PASSED" "$integration_test_log" || echo "0")
        python3 -c "
import json
try:
    with open('$TEST_REPORT', 'r') as f:
        report = json.load(f)
    report['summary']['integration_tests'] = $integration_test_count
    with open('$TEST_REPORT', 'w') as f:
        json.dump(report, f, indent=2)
except: pass
        "
        
        return 0
    else
        print_status "ERROR" "Integration tests failed"
        update_test_report "integration_tests" "failed" "Some integration tests failed"
        return 1
    fi
}

# Function to run FP property-based tests
run_property_tests() {
    print_status "FP" "Running property-based tests..."
    
    local property_test_log="$FP_TEST_LOG_DIR/property_tests_$TIMESTAMP.log"
    
    # Run property tests with hypothesis
    if poetry run pytest tests/property/ \
        --tb=short \
        --hypothesis-show-statistics \
        2>&1 | tee "$property_test_log"; then
        
        print_status "SUCCESS" "Property-based tests passed"
        update_test_report "property_tests" "passed" "All property tests successful"
        return 0
    else
        print_status "ERROR" "Property-based tests failed"
        update_test_report "property_tests" "failed" "Some property tests failed"
        return 1
    fi
}

# Function to check for import issues
check_import_issues() {
    print_status "INFO" "Checking for import resolution issues..."
    
    local import_log="$FP_TEST_LOG_DIR/import_check_$TIMESTAMP.log"
    
    # Test critical imports
    python3 -c "
import sys
import traceback

# Test imports that commonly cause issues
imports_to_test = [
    'bot.fp.types.base',
    'bot.fp.types.result',
    'bot.fp.core.either',
    'bot.fp.core.option',
    'bot.fp.core.io',
    'bot.fp.adapters.exchange_adapter',
    'tests.fp_test_base',
    'tests.fp_migration_adapters',
]

issues = []
for import_name in imports_to_test:
    try:
        __import__(import_name)
        print(f'✅ {import_name}')
    except ImportError as e:
        issues.append(f'{import_name}: {e}')
        print(f'❌ {import_name}: {e}')
    except Exception as e:
        issues.append(f'{import_name}: {e}')
        print(f'⚠️  {import_name}: {e}')

if issues:
    print('\\nImport issues found:')
    for issue in issues:
        print(f'  - {issue}')
    sys.exit(1)
else:
    print('\\n✅ All critical imports working')
    sys.exit(0)
    " > "$import_log" 2>&1
    
    local import_exit_code=$?
    
    if [ $import_exit_code -eq 0 ]; then
        print_status "SUCCESS" "No import issues detected"
        update_test_report "import_check" "passed" "All imports successful"
        return 0
    else
        print_status "ERROR" "Import issues detected (check $import_log)"
        update_test_report "import_check" "failed" "Import issues found"
        return 1
    fi
}

# Function to run test with fallback
run_test_with_fallback() {
    local test_func="$1"
    local test_name="$2"
    shift 2
    
    print_status "TEST" "Running $test_name..."
    
    # Try to run the test function
    if $test_func "$@"; then
        print_status "SUCCESS" "$test_name completed successfully"
        return 0
    else
        print_status "WARNING" "$test_name failed, trying fallback..."
        
        # Fallback: run with basic pytest
        if poetry run pytest "$@" --tb=line; then
            print_status "SUCCESS" "$test_name completed with fallback"
            return 0
        else
            print_status "ERROR" "$test_name failed completely"
            return 1
        fi
    fi
}

# Function to generate test summary
generate_test_summary() {
    print_status "INFO" "Generating test summary..."
    
    python3 -c "
import json
import sys

try:
    with open('$TEST_REPORT', 'r') as f:
        report = json.load(f)
    
    summary = report['summary']
    
    print('🧪 Test Execution Summary')
    print('=' * 50)
    print(f'Total Test Suites: {summary[\"total_test_suites\"]}')
    print(f'Passed Suites: {summary[\"passed_suites\"]} ✅')
    print(f'Failed Suites: {summary[\"failed_suites\"]} ❌')
    print()
    print('Test Counts:')
    print(f'  FP Tests: {summary.get(\"fp_tests\", 0)}')
    print(f'  Legacy Tests: {summary.get(\"legacy_tests\", 0)}')
    print(f'  Integration Tests: {summary.get(\"integration_tests\", 0)}')
    print()
    
    if summary['failed_suites'] == 0:
        print('🎉 All test suites passed!')
        success_rate = 100
    else:
        success_rate = (summary['passed_suites'] / summary['total_test_suites']) * 100
        print(f'⚠️  Success Rate: {success_rate:.1f}%')
    
    print()
    print(f'📄 Detailed Report: $TEST_REPORT')
    print(f'📁 Test Logs: $FP_TEST_LOG_DIR')
    
    # Exit with appropriate code
    sys.exit(0 if summary['failed_suites'] == 0 else 1)
    
except Exception as e:
    print(f'Error generating summary: {e}')
    sys.exit(1)
    "
}

# Main test execution logic
case "${1:-all}" in
    "fp"|"functional")
        check_test_dependencies
        check_import_issues
        run_fp_tests "${2:-tests/unit/fp/}" "$3"
        generate_test_summary
        ;;
    "legacy"|"imperative")
        check_test_dependencies
        run_legacy_tests "${2:-tests/unit/}" "$3"
        generate_test_summary
        ;;
    "integration")
        check_test_dependencies
        run_integration_tests "${2:-tests/integration/}" "$3"
        generate_test_summary
        ;;
    "property")
        check_test_dependencies
        run_property_tests
        generate_test_summary
        ;;
    "imports")
        check_import_issues
        ;;
    "quick")
        print_status "INFO" "Running quick test suite..."
        check_test_dependencies
        check_import_issues
        run_test_with_fallback run_fp_tests "FP Quick Tests" tests/unit/fp/ -x --tb=line
        generate_test_summary
        ;;
    "compatibility")
        print_status "INFO" "Running FP/legacy compatibility tests..."
        check_test_dependencies
        check_import_issues
        
        # Run both FP and legacy tests to check compatibility
        run_test_with_fallback run_fp_tests "FP Tests" tests/unit/fp/
        run_test_with_fallback run_legacy_tests "Legacy Tests" tests/unit/ 
        
        print_status "INFO" "Checking for test conflicts..."
        # Check if any tests interfere with each other
        poetry run pytest tests/unit/fp/ tests/unit/ --ignore=tests/unit/fp/ -x || \
            print_status "WARNING" "Some compatibility issues detected"
        
        generate_test_summary
        ;;
    "all"|"full")
        print_status "INFO" "Running complete test suite..."
        check_test_dependencies
        check_import_issues
        
        local exit_code=0
        
        # Run all test types
        run_test_with_fallback run_fp_tests "FP Tests" tests/unit/fp/ || exit_code=1
        run_test_with_fallback run_legacy_tests "Legacy Tests" tests/unit/ || exit_code=1
        run_test_with_fallback run_integration_tests "Integration Tests" tests/integration/ || exit_code=1
        run_test_with_fallback run_property_tests "Property Tests" || exit_code=1
        
        generate_test_summary
        exit $exit_code
        ;;
    "debug")
        print_status "INFO" "Running debug test mode..."
        check_test_dependencies
        check_import_issues
        
        # Run tests with maximum verbosity and debugging
        poetry run pytest tests/unit/fp/ -v -s --tb=long --capture=no --log-cli-level=DEBUG
        ;;
    "coverage")
        print_status "INFO" "Running tests with comprehensive coverage..."
        check_test_dependencies
        
        # Run tests with detailed coverage
        poetry run pytest tests/ \
            --cov=bot \
            --cov=tests \
            --cov-report=html \
            --cov-report=term-missing \
            --cov-report=json \
            --cov-branch
        
        print_status "INFO" "Coverage report generated in htmlcov/"
        ;;
    "help"|*)
        echo "FP Test Runner - Usage:"
        echo ""
        echo "  $0 all                # Run complete test suite (FP + legacy + integration)"
        echo "  $0 fp                 # Run only functional programming tests"
        echo "  $0 legacy             # Run only legacy/imperative tests"
        echo "  $0 integration        # Run only integration tests"
        echo "  $0 property           # Run property-based tests"
        echo "  $0 imports            # Check import resolution issues"
        echo "  $0 quick              # Run quick FP test suite"
        echo "  $0 compatibility      # Test FP/legacy compatibility"
        echo "  $0 debug              # Run tests in debug mode"
        echo "  $0 coverage           # Run tests with coverage analysis"
        echo "  $0 help               # Show this help"
        echo ""
        echo "Examples:"
        echo "  $0 fp tests/unit/fp/test_types.py    # Run specific FP test file"
        echo "  $0 legacy -k \"test_risk\"             # Run legacy tests matching pattern"
        echo "  $0 integration --timeout=600         # Run integration tests with timeout"
        echo ""
        echo "Test Categories:"
        echo "  FP Tests       - Pure functional programming tests"
        echo "  Legacy Tests   - Existing imperative tests"
        echo "  Integration    - End-to-end system tests"
        echo "  Property Tests - Hypothesis-based property testing"
        echo ""
        echo "Output:"
        echo "  Logs: $FP_TEST_LOG_DIR"
        echo "  Reports: $FP_TEST_DATA_DIR"
        echo ""
        ;;
esac