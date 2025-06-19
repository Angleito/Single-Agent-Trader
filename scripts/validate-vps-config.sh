#!/bin/bash

# VPS Configuration Validation Script
# Validates VPS deployment configuration before deployment

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Counters
ERRORS=0
WARNINGS=0
CHECKS=0

# Logging functions
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] ✓ $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] ⚠ WARNING: $1${NC}"
    ((WARNINGS++))
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ✗ ERROR: $1${NC}"
    ((ERRORS++))
}

info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] ℹ INFO: $1${NC}"
}

check() {
    ((CHECKS++))
}

# Validate required files exist
validate_files() {
    info "Validating required files..."
    
    local required_files=(
        "docker-compose.vps.yml"
        "scripts/vps-deploy.sh"
        "scripts/vps-healthcheck.sh"
        "services/Dockerfile.bluefin"
        "config/vps_production.json"
        ".env.vps.template"
        "docs/VPS_DEPLOYMENT_GUIDE.md"
        "docs/VPS_TROUBLESHOOTING.md"
    )
    
    for file in "${required_files[@]}"; do
        check
        if [ -f "$PROJECT_ROOT/$file" ]; then
            log "Required file exists: $file"
        else
            error "Required file missing: $file"
        fi
    done
    
    # Check if scripts are executable
    local executable_scripts=(
        "scripts/vps-deploy.sh"
        "scripts/vps-healthcheck.sh"
    )
    
    for script in "${executable_scripts[@]}"; do
        check
        if [ -x "$PROJECT_ROOT/$script" ]; then
            log "Script is executable: $script"
        else
            error "Script is not executable: $script"
        fi
    done
}

# Validate Docker Compose configuration
validate_docker_compose() {
    info "Validating Docker Compose configuration..."
    
    cd "$PROJECT_ROOT"
    
    # Check if docker-compose.vps.yml is valid
    check
    if docker-compose -f docker-compose.vps.yml config >/dev/null 2>&1; then
        log "docker-compose.vps.yml syntax is valid"
    else
        error "docker-compose.vps.yml syntax is invalid"
        docker-compose -f docker-compose.vps.yml config
    fi
    
    # Check for required services
    local required_services=(
        "bluefin-service"
        "ai-trading-bot"
        "mcp-memory"
        "monitoring-dashboard"
        "log-aggregator"
    )
    
    for service in "${required_services[@]}"; do
        check
        if docker-compose -f docker-compose.vps.yml config | grep -q "^  $service:"; then
            log "Required service defined: $service"
        else
            error "Required service missing: $service"
        fi
    done
    
    # Check for VPS-specific configurations
    check
    if docker-compose -f docker-compose.vps.yml config | grep -q "VPS_DEPLOYMENT=true"; then
        log "VPS deployment flag is set"
    else
        warn "VPS deployment flag may not be set correctly"
    fi
    
    # Check for security configurations
    check
    if docker-compose -f docker-compose.vps.yml config | grep -q "read_only: true"; then
        log "Read-only filesystem security is configured"
    else
        warn "Read-only filesystem security may not be configured"
    fi
    
    # Check for resource limits
    check
    if docker-compose -f docker-compose.vps.yml config | grep -q "resources:"; then
        log "Resource limits are configured"
    else
        warn "Resource limits may not be configured"
    fi
}

# Validate environment template
validate_env_template() {
    info "Validating environment template..."
    
    check
    if [ -f "$PROJECT_ROOT/.env.vps.template" ]; then
        log "Environment template exists"
        
        # Check for required environment variables
        local required_vars=(
            "EXCHANGE__BLUEFIN_PRIVATE_KEY"
            "BLUEFIN_SERVICE_API_KEY"
            "LLM__OPENAI_API_KEY"
            "VPS_DEPLOYMENT"
            "GEOGRAPHIC_REGION"
            "MONITORING__ENABLED"
        )
        
        for var in "${required_vars[@]}"; do
            check
            if grep -q "^$var=" "$PROJECT_ROOT/.env.vps.template"; then
                log "Required environment variable in template: $var"
            else
                error "Required environment variable missing from template: $var"
            fi
        done
        
        # Check for security-related variables
        local security_vars=(
            "PROXY_ENABLED"
            "ALERT_WEBHOOK_URL"
            "BACKUP_ENABLED"
            "RATE_LIMIT_ENABLED"
        )
        
        for var in "${security_vars[@]}"; do
            check
            if grep -q "^$var=" "$PROJECT_ROOT/.env.vps.template"; then
                log "Security environment variable in template: $var"
            else
                warn "Security environment variable missing from template: $var"
            fi
        done
        
    else
        error "Environment template does not exist"
    fi
}

# Validate configuration files
validate_config_files() {
    info "Validating configuration files..."
    
    # Check VPS production config
    check
    if [ -f "$PROJECT_ROOT/config/vps_production.json" ]; then
        log "VPS production config exists"
        
        # Validate JSON syntax
        check
        if jq empty "$PROJECT_ROOT/config/vps_production.json" >/dev/null 2>&1; then
            log "VPS production config has valid JSON syntax"
        else
            error "VPS production config has invalid JSON syntax"
        fi
        
        # Check for required sections
        local required_sections=(
            "system"
            "exchange"
            "trading"
            "risk"
            "logging"
            "monitoring"
            "network"
            "security"
        )
        
        for section in "${required_sections[@]}"; do
            check
            if jq -e ".$section" "$PROJECT_ROOT/config/vps_production.json" >/dev/null 2>&1; then
                log "Required config section exists: $section"
            else
                error "Required config section missing: $section"
            fi
        done
        
    else
        error "VPS production config does not exist"
    fi
}

# Validate Dockerfile modifications
validate_dockerfile() {
    info "Validating Dockerfile modifications..."
    
    check
    if [ -f "$PROJECT_ROOT/services/Dockerfile.bluefin" ]; then
        log "Bluefin Dockerfile exists"
        
        # Check for VPS-specific additions
        local vps_features=(
            "VPS_DEPLOYMENT"
            "GEOGRAPHIC_REGION"
            "vps-healthcheck.sh"
            "prometheus-client"
            "structlog"
        )
        
        for feature in "${vps_features[@]}"; do
            check
            if grep -q "$feature" "$PROJECT_ROOT/services/Dockerfile.bluefin"; then
                log "VPS feature in Dockerfile: $feature"
            else
                warn "VPS feature may be missing from Dockerfile: $feature"
            fi
        done
        
        # Check for security enhancements
        check
        if grep -q "USER bluefin" "$PROJECT_ROOT/services/Dockerfile.bluefin"; then
            log "Non-root user configured in Dockerfile"
        else
            error "Non-root user not configured in Dockerfile"
        fi
        
    else
        error "Bluefin Dockerfile does not exist"
    fi
}

# Validate scripts
validate_scripts() {
    info "Validating deployment scripts..."
    
    # Check deployment script
    check
    if [ -f "$PROJECT_ROOT/scripts/vps-deploy.sh" ]; then
        log "VPS deployment script exists"
        
        # Check for required functions
        local required_functions=(
            "check_system_requirements"
            "validate_environment"
            "setup_vps_system"
            "deploy_services"
            "verify_deployment"
        )
        
        for func in "${required_functions[@]}"; do
            check
            if grep -q "^$func()" "$PROJECT_ROOT/scripts/vps-deploy.sh"; then
                log "Required function in deployment script: $func"
            else
                error "Required function missing from deployment script: $func"
            fi
        done
        
    else
        error "VPS deployment script does not exist"
    fi
    
    # Check health check script
    check
    if [ -f "$PROJECT_ROOT/scripts/vps-healthcheck.sh" ]; then
        log "VPS health check script exists"
        
        # Check for service-specific checks
        local service_checks=(
            "check_bluefin_service_health"
            "check_trading_bot_health"
            "check_mcp_memory_health"
        )
        
        for check_func in "${service_checks[@]}"; do
            check
            if grep -q "$check_func" "$PROJECT_ROOT/scripts/vps-healthcheck.sh"; then
                log "Service health check function exists: $check_func"
            else
                warn "Service health check function missing: $check_func"
            fi
        done
        
    else
        error "VPS health check script does not exist"
    fi
}

# Validate documentation
validate_documentation() {
    info "Validating documentation..."
    
    # Check deployment guide
    check
    if [ -f "$PROJECT_ROOT/docs/VPS_DEPLOYMENT_GUIDE.md" ]; then
        log "VPS deployment guide exists"
        
        # Check for required sections
        local required_sections=(
            "Prerequisites"
            "Quick Start"
            "Regional Restrictions"
            "Monitoring"
            "Troubleshooting"
        )
        
        for section in "${required_sections[@]}"; do
            check
            if grep -q "## $section" "$PROJECT_ROOT/docs/VPS_DEPLOYMENT_GUIDE.md"; then
                log "Required documentation section exists: $section"
            else
                warn "Required documentation section missing: $section"
            fi
        done
        
    else
        error "VPS deployment guide does not exist"
    fi
    
    # Check troubleshooting guide
    check
    if [ -f "$PROJECT_ROOT/docs/VPS_TROUBLESHOOTING.md" ]; then
        log "VPS troubleshooting guide exists"
    else
        error "VPS troubleshooting guide does not exist"
    fi
}

# Validate directory structure
validate_directory_structure() {
    info "Validating directory structure..."
    
    # Check if we're in the right directory
    check
    if [ -f "$PROJECT_ROOT/docker-compose.yml" ] && [ -f "$PROJECT_ROOT/pyproject.toml" ]; then
        log "Project root directory structure is correct"
    else
        error "Not in correct project root directory"
    fi
    
    # Check for required directories
    local required_dirs=(
        "scripts"
        "config"
        "services"
        "docs"
        "bot"
    )
    
    for dir in "${required_dirs[@]}"; do
        check
        if [ -d "$PROJECT_ROOT/$dir" ]; then
            log "Required directory exists: $dir"
        else
            error "Required directory missing: $dir"
        fi
    done
}

# Generate validation report
generate_report() {
    echo ""
    echo "========================================"
    echo "VPS CONFIGURATION VALIDATION REPORT"
    echo "========================================"
    echo "Date: $(date)"
    echo "Project: AI Trading Bot VPS Deployment"
    echo ""
    echo "Validation Summary:"
    echo "  Total Checks: $CHECKS"
    echo "  Errors: $ERRORS"
    echo "  Warnings: $WARNINGS"
    echo "  Success Rate: $(( (CHECKS - ERRORS) * 100 / CHECKS ))%"
    echo ""
    
    if [ $ERRORS -eq 0 ]; then
        echo -e "${GREEN}✓ VALIDATION PASSED${NC}"
        echo "VPS deployment configuration is ready for deployment."
        echo ""
        echo "Next steps:"
        echo "1. Copy .env.vps.template to .env and configure with your values"
        echo "2. Run: ./scripts/vps-deploy.sh"
        echo "3. Monitor deployment with: docker-compose -f docker-compose.vps.yml logs -f"
        return 0
    else
        echo -e "${RED}✗ VALIDATION FAILED${NC}"
        echo "Please fix the $ERRORS error(s) before proceeding with deployment."
        
        if [ $WARNINGS -gt 0 ]; then
            echo ""
            echo -e "${YELLOW}Note: $WARNINGS warning(s) found. These are not critical but should be reviewed.${NC}"
        fi
        
        return 1
    fi
}

# Main validation function
main() {
    echo "Starting VPS configuration validation..."
    echo ""
    
    validate_directory_structure
    validate_files
    validate_docker_compose
    validate_env_template
    validate_config_files
    validate_dockerfile
    validate_scripts
    validate_documentation
    
    generate_report
}

# Run validation
main "$@"