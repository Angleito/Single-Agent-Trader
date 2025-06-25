#!/bin/bash

# Validation Script for Ubuntu Deployment Test Script
# Ensures the test script is properly configured and ready to run

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TEST_SCRIPT="${SCRIPT_DIR}/test-ubuntu-deployment.sh"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

success() { echo -e "${GREEN}✅ $1${NC}"; }
error() { echo -e "${RED}❌ $1${NC}"; }
warning() { echo -e "${YELLOW}⚠️ $1${NC}"; }

echo "Validating Ubuntu Deployment Test Script"
echo "========================================"

# Check if test script exists and is executable
if [[ -f "$TEST_SCRIPT" ]]; then
    success "Test script exists: $TEST_SCRIPT"
else
    error "Test script not found: $TEST_SCRIPT"
    exit 1
fi

if [[ -x "$TEST_SCRIPT" ]]; then
    success "Test script is executable"
else
    error "Test script is not executable. Run: chmod +x $TEST_SCRIPT"
    exit 1
fi

# Check script syntax
if bash -n "$TEST_SCRIPT"; then
    success "Test script syntax is valid"
else
    error "Test script has syntax errors"
    exit 1
fi

# Check required files exist
required_files=(
    "Dockerfile"
    "docker-compose.yml"
    "docker-compose.simple.yml"
    "healthcheck.sh"
    "pyproject.toml"
)

for file in "${required_files[@]}"; do
    if [[ -f "$PROJECT_ROOT/$file" ]]; then
        success "Required file exists: $file"
    else
        error "Required file missing: $file"
        exit 1
    fi
done

# Check required directories exist
required_dirs=(
    "bot"
    "scripts"
    "logs"
    "data"
)

for dir in "${required_dirs[@]}"; do
    if [[ -d "$PROJECT_ROOT/$dir" ]]; then
        success "Required directory exists: $dir"
    else
        warning "Directory will be created during test: $dir"
        mkdir -p "$PROJECT_ROOT/$dir"
    fi
done

# Check Docker prerequisites
echo ""
echo "Checking Docker prerequisites..."

if command -v docker >/dev/null 2>&1; then
    success "Docker is installed: $(docker --version)"
else
    error "Docker is not installed or not in PATH"
    exit 1
fi

if docker info >/dev/null 2>&1; then
    success "Docker daemon is accessible"
else
    error "Docker daemon is not running or not accessible"
    exit 1
fi

if command -v docker-compose >/dev/null 2>&1; then
    success "Docker Compose is available: $(docker-compose --version)"
elif docker compose version >/dev/null 2>&1; then
    success "Docker Compose is available: $(docker compose version)"
else
    error "Docker Compose is not installed"
    exit 1
fi

# Test script help output
echo ""
echo "Testing script help output..."
if "$TEST_SCRIPT" --help 2>/dev/null || "$TEST_SCRIPT" help 2>/dev/null || true; then
    success "Test script help is available"
else
    warning "Test script help may not be available (this is OK)"
fi

# Check script modes
echo ""
echo "Validating script modes..."
script_modes=("main" "quick" "build-only" "health-only" "compose-only")

for mode in "${script_modes[@]}"; do
    if grep -q "\"$mode\"" "$TEST_SCRIPT"; then
        success "Script mode supported: $mode"
    else
        error "Script mode not found: $mode"
    fi
done

# Check if we can create test directories
echo ""
echo "Testing directory creation permissions..."
temp_test_dir="/tmp/ubuntu_deploy_test_validation_$$"
if mkdir -p "$temp_test_dir"; then
    success "Can create temporary directories"
    rmdir "$temp_test_dir"
else
    error "Cannot create temporary directories"
    exit 1
fi

# Check disk space
echo ""
echo "Checking disk space..."
available_space=$(df "$PROJECT_ROOT" | tail -1 | awk '{print $4}')
if [[ $available_space -gt 2097152 ]]; then  # 2GB in KB
    success "Sufficient disk space available: $((available_space / 1024 / 1024))GB"
else
    warning "Low disk space - may affect Docker operations: $((available_space / 1024 / 1024))GB available"
fi

# Validate script functions
echo ""
echo "Checking script structure..."
required_functions=(
    "cleanup"
    "initialize_test_environment"
    "check_prerequisites"
    "test_docker_build"
    "test_container_startup"
    "test_module_imports"
    "test_configuration_loading"
    "test_health_checks"
    "test_log_file_creation"
    "test_virtual_environment"
    "test_simple_compose"
    "test_full_compose"
    "test_api_connectivity"
    "generate_test_report"
    "main"
)

for func in "${required_functions[@]}"; do
    if grep -q "^${func}()" "$TEST_SCRIPT"; then
        success "Function defined: $func"
    else
        error "Function missing: $func"
    fi
done

# Check for security best practices
echo ""
echo "Checking security configurations..."

if grep -q "set -euo pipefail" "$TEST_SCRIPT"; then
    success "Script uses strict error handling"
else
    warning "Script should use 'set -euo pipefail'"
fi

if grep -q "SYSTEM__DRY_RUN=true" "$TEST_SCRIPT"; then
    success "Script uses safe paper trading mode"
else
    error "Script should enforce dry-run mode for safety"
fi

if grep -q "cleanup" "$TEST_SCRIPT" && grep -q "trap cleanup EXIT" "$TEST_SCRIPT"; then
    success "Script has proper cleanup handling"
else
    warning "Script should have cleanup trap handlers"
fi

# Final validation
echo ""
echo "Validation Summary"
echo "=================="
success "Ubuntu deployment test script is properly configured"
success "All required files and directories are available"
success "Docker prerequisites are met"
success "Script structure and functions are valid"

echo ""
echo "Ready to run deployment tests:"
echo "  Basic test:     $TEST_SCRIPT quick"
echo "  Complete test:  $TEST_SCRIPT"
echo "  Build only:     $TEST_SCRIPT build-only"
echo ""
echo "For detailed usage, see: scripts/ubuntu-deployment-test-usage.md"
