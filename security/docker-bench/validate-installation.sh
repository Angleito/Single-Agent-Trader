#!/bin/bash

# Docker Bench Security Installation Validation
# Validates the complete Docker security automation installation

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Counters
TESTS_PASSED=0
TESTS_FAILED=0
WARNINGS=0

# Logging
log() {
    local level="$1"
    local message="$2"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "${timestamp} [${level}] ${message}"
}

info() { log "INFO" "${BLUE}$1${NC}"; }
success() { log "SUCCESS" "${GREEN}$1${NC}"; TESTS_PASSED=$((TESTS_PASSED + 1)); }
warning() { log "WARNING" "${YELLOW}$1${NC}"; WARNINGS=$((WARNINGS + 1)); }
error() { log "ERROR" "${RED}$1${NC}"; TESTS_FAILED=$((TESTS_FAILED + 1)); }

# Test functions
test_directory_structure() {
    info "Testing directory structure..."
    
    local required_dirs=(
        "scripts"
        "remediation"
        "monitoring"
        "cicd"
        "config"
        "custom-checks"
        "templates"
    )
    
    for dir in "${required_dirs[@]}"; do
        if [ -d "${SCRIPT_DIR}/${dir}" ]; then
            success "Directory exists: $dir"
        else
            error "Missing directory: $dir"
        fi
    done
}

test_script_files() {
    info "Testing script files..."
    
    local required_scripts=(
        "install-docker-bench.sh"
        "scripts/run-security-scan.sh"
        "scripts/security-automation.sh"
        "scripts/compliance-reporting.sh"
        "remediation/auto-remediate.sh"
        "monitoring/security-monitor.sh"
        "cicd/security-gate.sh"
    )
    
    for script in "${required_scripts[@]}"; do
        local script_path="${SCRIPT_DIR}/${script}"
        if [ -f "$script_path" ]; then
            if [ -x "$script_path" ]; then
                success "Script exists and executable: $script"
            else
                error "Script not executable: $script"
            fi
        else
            error "Missing script: $script"
        fi
    done
}

test_configuration_files() {
    info "Testing configuration files..."
    
    local config_files=(
        "config/security-automation.conf"
        "config/security-monitor.conf"
        "config/remediation.conf"
    )
    
    for config in "${config_files[@]}"; do
        local config_path="${SCRIPT_DIR}/${config}"
        if [ -f "$config_path" ]; then
            success "Configuration file exists: $config"
        else
            error "Missing configuration file: $config"
        fi
    done
}

test_custom_checks() {
    info "Testing custom security checks..."
    
    local custom_checks=(
        "custom-checks/8_1_crypto_security.sh"
        "custom-checks/8_2_trading_network.sh"
        "custom-checks/8_3_data_persistence.sh"
    )
    
    for check in "${custom_checks[@]}"; do
        local check_path="${SCRIPT_DIR}/${check}"
        if [ -f "$check_path" ]; then
            if [ -x "$check_path" ]; then
                success "Custom check exists and executable: $(basename "$check")"
            else
                error "Custom check not executable: $(basename "$check")"
            fi
        else
            error "Missing custom check: $(basename "$check")"
        fi
    done
}

test_system_dependencies() {
    info "Testing system dependencies..."
    
    local required_tools=("docker" "git" "curl" "jq" "awk" "grep")
    
    for tool in "${required_tools[@]}"; do
        if command -v "$tool" &> /dev/null; then
            success "System dependency available: $tool"
        else
            error "Missing system dependency: $tool"
        fi
    done
    
    # Test Docker access
    if docker info &> /dev/null; then
        success "Docker daemon accessible"
    else
        error "Docker daemon not accessible"
    fi
}

test_github_actions_workflow() {
    info "Testing GitHub Actions workflow..."
    
    local workflow_file="${PROJECT_ROOT}/.github/workflows/security-gate.yml"
    if [ -f "$workflow_file" ]; then
        success "GitHub Actions security workflow exists"
        
        # Basic syntax validation
        if grep -q "name: Security Gate" "$workflow_file"; then
            success "Workflow has correct name"
        else
            error "Workflow missing or incorrect name"
        fi
        
        if grep -q "docker-bench-security" "$workflow_file"; then
            success "Workflow references Docker Bench Security"
        else
            warning "Workflow may not reference Docker Bench Security"
        fi
    else
        warning "GitHub Actions workflow not found (may not be using GitHub)"
    fi
}

test_integration_with_trading_bot() {
    info "Testing integration with trading bot..."
    
    # Check if trading bot docker-compose file exists
    local compose_file="${PROJECT_ROOT}/docker-compose.yml"
    if [ -f "$compose_file" ]; then
        success "Docker Compose file found"
        
        # Check for trading bot services
        if grep -q "ai-trading-bot" "$compose_file"; then
            success "AI trading bot service found in compose file"
        else
            warning "AI trading bot service not found in compose file"
        fi
        
        # Check for security configurations
        if grep -q "security_opt" "$compose_file"; then
            success "Security options configured in compose file"
        else
            warning "No security options found in compose file"
        fi
        
        if grep -q "read_only:" "$compose_file"; then
            success "Read-only filesystem configurations found"
        else
            warning "No read-only filesystem configurations found"
        fi
    else
        error "Docker Compose file not found"
    fi
}

test_script_syntax() {
    info "Testing script syntax..."
    
    # Find all shell scripts and test syntax
    find "$SCRIPT_DIR" -name "*.sh" | while read -r script; do
        if bash -n "$script" 2>/dev/null; then
            success "Script syntax valid: $(basename "$script")"
        else
            error "Script syntax error: $(basename "$script")"
        fi
    done
}

run_basic_functionality_test() {
    info "Running basic functionality tests..."
    
    # Test configuration loading
    local test_config="${SCRIPT_DIR}/config/security-automation.conf"
    if [ -f "$test_config" ]; then
        if source "$test_config" 2>/dev/null; then
            success "Configuration file loads without errors"
        else
            error "Configuration file has syntax errors"
        fi
    fi
    
    # Test help outputs
    local scripts_to_test=(
        "scripts/security-automation.sh"
        "monitoring/security-monitor.sh"
        "cicd/security-gate.sh"
    )
    
    for script in "${scripts_to_test[@]}"; do
        local script_path="${SCRIPT_DIR}/${script}"
        if [ -f "$script_path" ]; then
            # Try to run with invalid argument to see help
            if "$script_path" --help &>/dev/null || "$script_path" help &>/dev/null || "$script_path" invalid-command &>/dev/null; then
                success "Script responds to help/invalid commands: $(basename "$script")"
            else
                warning "Script may not have proper help functionality: $(basename "$script")"
            fi
        fi
    done
}

test_permissions() {
    info "Testing file permissions..."
    
    # Check script permissions
    find "$SCRIPT_DIR" -name "*.sh" | while read -r script; do
        if [ -x "$script" ]; then
            success "Script executable: $(basename "$script")"
        else
            error "Script not executable: $(basename "$script")"
        fi
    done
    
    # Check directory permissions
    local required_dirs=("logs" "reports" "metrics" "alerts" "backups")
    for dir in "${required_dirs[@]}"; do
        local dir_path="${SCRIPT_DIR}/${dir}"
        if [ -d "$dir_path" ]; then
            if [ -w "$dir_path" ]; then
                success "Directory writable: $dir"
            else
                error "Directory not writable: $dir"
            fi
        else
            # Create directory if it doesn't exist
            if mkdir -p "$dir_path" 2>/dev/null; then
                success "Created directory: $dir"
            else
                error "Cannot create directory: $dir"
            fi
        fi
    done
}

# Main validation function
main() {
    echo "🔒 Docker Bench Security Installation Validation"
    echo "=================================================="
    echo
    
    info "Starting validation of Docker Bench Security automation installation..."
    echo
    
    # Run all tests
    test_directory_structure
    echo
    
    test_script_files
    echo
    
    test_configuration_files
    echo
    
    test_custom_checks
    echo
    
    test_system_dependencies
    echo
    
    test_github_actions_workflow
    echo
    
    test_integration_with_trading_bot
    echo
    
    test_script_syntax
    echo
    
    run_basic_functionality_test
    echo
    
    test_permissions
    echo
    
    # Summary
    echo "=================================================="
    echo "🔒 VALIDATION SUMMARY"
    echo "=================================================="
    echo
    
    if [ $TESTS_FAILED -eq 0 ]; then
        success "✅ ALL TESTS PASSED!"
        echo
        success "   Tests passed: $TESTS_PASSED"
        if [ $WARNINGS -gt 0 ]; then
            warning "   Warnings: $WARNINGS"
        fi
        echo
        info "🚀 Docker Bench Security automation is ready for use!"
        echo
        info "Next steps:"
        info "1. Run initial security scan: ./scripts/run-security-scan.sh"
        info "2. Start monitoring: ./monitoring/security-monitor.sh monitor"
        info "3. Generate compliance report: ./scripts/compliance-reporting.sh generate"
        info "4. Test security gate: ./cicd/security-gate.sh test"
        echo
    else
        error "❌ VALIDATION FAILED!"
        echo
        error "   Tests failed: $TESTS_FAILED"
        success "   Tests passed: $TESTS_PASSED"
        warning "   Warnings: $WARNINGS"
        echo
        error "🛠️  Please fix the failed tests before using the security automation system."
        echo
        exit 1
    fi
}

# Run validation
main "$@"