#!/bin/bash
# AI Trading Bot Docker Entrypoint Script
# Requires bash 4.0+ for associative arrays
# Handles directory permissions, setup, and initialization before starting the main application
#
# FIXED VERSION - Handles permission failures gracefully:
# - Removed 'set -e' to prevent exits on permission errors
# - Simplified directory verification (relies on Dockerfile pre-creation)
# - Streamlined fallback mechanism using tmpfs
# - All setup steps are resilient and won't crash the container
# - Focus on verification rather than modification for better Docker compatibility

set -uo pipefail  # Exit on undefined variable or pipe failure (graceful error handling)

# Script metadata
SCRIPT_NAME="docker-entrypoint.sh"
SCRIPT_VERSION="1.0.0"
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

# Colors for output formatting
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly CYAN='\033[0;36m'
readonly NC='\033[0m' # No Color

# Configuration (get actual runtime user)
readonly APP_USER="botuser"
readonly APP_GROUP="botuser"
readonly APP_UID=$(id -u)
readonly APP_GID=$(id -g)

# Python environment configuration
readonly PYTHON_PATH="/app"
readonly BOT_MODULE="bot"

# Directory configuration (path:permissions pairs)
readonly REQUIRED_DIRS=(
    "/app/logs:775"
    "/app/data:775"
    "/app/tmp:775"
    "/app/config:755"
    "/app/prompts:755"
    "/app/data/mcp_memory:775"
    "/app/logs/mcp:775"
    "/app/logs/bluefin:775"
)

# Fallback directories (original:fallback pairs)
readonly FALLBACK_DIRS=(
    "/app/logs:/tmp/app-logs"
    "/app/data:/tmp/app-data"
    "/app/tmp:/tmp/app-tmp"
)

# Environment detection
readonly ENVIRONMENT="${SYSTEM__ENVIRONMENT:-development}"
readonly DRY_RUN="${SYSTEM__DRY_RUN:-true}"
readonly EXCHANGE_TYPE="${EXCHANGE__EXCHANGE_TYPE:-coinbase}"

# Logging functions
log_info() {
    echo -e "${CYAN}[INFO]${NC} ${TIMESTAMP} [${SCRIPT_NAME}] $1" >&2
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} ${TIMESTAMP} [${SCRIPT_NAME}] $1" >&2
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} ${TIMESTAMP} [${SCRIPT_NAME}] $1" >&2
}

log_error() {
    echo -e "${RED}[ERROR]${NC} ${TIMESTAMP} [${SCRIPT_NAME}] $1" >&2
}

log_debug() {
    if [[ "${DEBUG:-false}" == "true" ]]; then
        echo -e "${BLUE}[DEBUG]${NC} ${TIMESTAMP} [${SCRIPT_NAME}] $1" >&2
    fi
}

# Display startup banner
display_banner() {
    log_info "=================================="
    log_info "AI Trading Bot Container Initialization"
    log_info "Version: ${SCRIPT_VERSION}"
    log_info "Environment: ${ENVIRONMENT}"
    log_info "Exchange: ${EXCHANGE_TYPE}"
    log_info "Dry Run: ${DRY_RUN}"
    log_info "User: ${APP_USER}:${APP_GROUP} (${APP_UID}:${APP_GID})"
    log_info "=================================="
}

# Check if running as correct user
check_user() {
    local current_uid=$(id -u)
    local current_gid=$(id -g)
    local current_user=$(whoami 2>/dev/null || echo "uid_${current_uid}")

    log_info "Running as user ${current_user} (${current_uid}:${current_gid})"

    if [[ "${current_uid}" != "${APP_UID}" ]] || [[ "${current_gid}" != "${APP_GID}" ]]; then
        log_info "Running as UID:GID ${current_uid}:${current_gid}, expected ${APP_UID}:${APP_GID}"
        log_info "This is normal when using Docker Compose user mapping"
    else
        log_success "Running as correct user: ${APP_USER}:${APP_GROUP} (${APP_UID}:${APP_GID})"
    fi

    # Always succeed - user check is informational only
    return 0
}

# Verify and prepare directory (simplified approach)
verify_directory() {
    local dir_path="$1"
    local permissions="$2"

    log_debug "Verifying directory: ${dir_path}"

    # Check if directory exists and is accessible
    if [[ -d "${dir_path}" ]] && [[ -r "${dir_path}" ]]; then
        log_debug "Directory exists and is accessible: ${dir_path}"

        # Try to set permissions if possible (but don't fail if it doesn't work)
        if chmod "${permissions}" "${dir_path}" 2>/dev/null; then
            log_debug "Set permissions ${permissions} on ${dir_path}"
        else
            log_debug "Cannot set permissions on ${dir_path} (may be read-only filesystem)"
        fi

        return 0
    else
        log_warning "Directory ${dir_path} is not accessible or does not exist"
        return 1
    fi
}

# Test write permissions in a directory
test_write_permission() {
    local dir_path="$1"
    local test_file="${dir_path}/.write-test-${RANDOM}"

    log_debug "Testing write permissions in: ${dir_path}"

    if touch "${test_file}" 2>/dev/null; then
        rm -f "${test_file}" 2>/dev/null
        log_debug "Write permission confirmed for: ${dir_path}"
        return 0
    else
        log_warning "Write permission test failed for: ${dir_path}"
        return 1
    fi
}

# Setup fallback directory (simplified)
setup_fallback() {
    local original_dir="$1"
    local fallback_dir="$2"

    log_info "Setting up tmpfs fallback for: ${original_dir}"

    # Create fallback in tmpfs
    local tmpfs_fallback="/tmp/$(basename "${original_dir}")-$$"

    if mkdir -p "${tmpfs_fallback}" 2>/dev/null; then
        log_success "Created tmpfs fallback: ${tmpfs_fallback}"

        # Set permissions (but don't fail if it doesn't work)
        chmod 755 "${tmpfs_fallback}" 2>/dev/null || true

        # Set environment variable for application to use
        local env_var_name="FALLBACK_$(basename "${original_dir}" | tr '[:lower:]' '[:upper:]')_DIR"
        export "${env_var_name}=${tmpfs_fallback}"
        log_info "Set environment variable: ${env_var_name}=${tmpfs_fallback}"

        return 0
    else
        log_warning "Failed to create tmpfs fallback for ${original_dir}"
        return 1
    fi
}

# Get fallback directory for a given original directory
get_fallback_dir() {
    local original_dir="$1"
    for fallback_pair in "${FALLBACK_DIRS[@]}"; do
        local original="${fallback_pair%:*}"
        local fallback="${fallback_pair#*:}"
        if [[ "${original}" == "${original_dir}" ]]; then
            echo "${fallback}"
            return 0
        fi
    done
    return 1
}

# Setup all required directories
setup_directories() {
    log_info "Verifying required directories..."

    local failed_dirs=()
    local success_count=0
    local total_dirs=${#REQUIRED_DIRS[@]}

    # Verify directories (they should already exist from Dockerfile)
    for dir_config in "${REQUIRED_DIRS[@]}"; do
        local dir_path="${dir_config%:*}"
        local permissions="${dir_config#*:}"

        if verify_directory "${dir_path}" "${permissions}"; then
            if test_write_permission "${dir_path}"; then
                ((success_count++))
                log_success "Directory ready: ${dir_path}"
            else
                log_warning "Directory exists but not writable: ${dir_path}"
                failed_dirs+=("${dir_path}")
            fi
        else
            log_warning "Directory verification failed: ${dir_path}"
            failed_dirs+=("${dir_path}")
        fi
    done

    # Setup fallbacks for failed directories
    if [[ ${#failed_dirs[@]} -gt 0 ]]; then
        log_info "Setting up fallbacks for ${#failed_dirs[@]} directories"
        for failed_dir in "${failed_dirs[@]}"; do
            local fallback_dir
            if fallback_dir=$(get_fallback_dir "${failed_dir}"); then
                if setup_fallback "${failed_dir}" "${fallback_dir}"; then
                    ((success_count++))
                    log_success "Fallback ready for: ${failed_dir}"
                else
                    log_warning "Fallback setup failed for: ${failed_dir}"
                fi
            else
                # Use tmpfs fallback as last resort
                if setup_fallback "${failed_dir}" "/tmp/fallback"; then
                    ((success_count++))
                    log_success "Tmpfs fallback ready for: ${failed_dir}"
                else
                    log_warning "All fallback options failed for: ${failed_dir}"
                fi
            fi
        done
    fi

    log_info "Directory setup complete: ${success_count}/${total_dirs} directories available"

    # Always succeed - the application can handle missing directories
    if [[ "${success_count}" -gt 0 ]]; then
        log_success "Directories are ready for application startup"
    else
        log_warning "No directories available, but application can use default locations"
    fi

    return 0
}

# Setup Python environment variables
setup_python_environment() {
    log_info "Setting up Python environment..."

    # Set Python environment variables
    export PYTHONPATH="${PYTHON_PATH}"
    export PYTHONUNBUFFERED=1
    export PYTHONDONTWRITEBYTECODE=1

    log_success "Python environment variables set:"
    log_info "  PYTHONPATH=${PYTHONPATH}"
    log_info "  PYTHONUNBUFFERED=${PYTHONUNBUFFERED}"
    log_info "  PYTHONDONTWRITEBYTECODE=${PYTHONDONTWRITEBYTECODE}"

    return 0
}

# Verify Python installation and environment
verify_python_environment() {
    log_info "Verifying Python environment..."

    # Check Python version
    if ! python --version >/dev/null 2>&1; then
        log_error "Python not available in PATH"
        log_error "Please ensure Python is installed and available"
        return 1
    fi

    local python_version=$(python --version 2>&1 || echo "unknown")
    log_success "Python available: ${python_version}"

    # Check Python version is 3.12+
    local python_major=$(python -c "import sys; print(sys.version_info.major)" 2>/dev/null || echo "0")
    local python_minor=$(python -c "import sys; print(sys.version_info.minor)" 2>/dev/null || echo "0")

    if [[ "${python_major}" -lt 3 ]] || [[ "${python_major}" -eq 3 && "${python_minor}" -lt 12 ]]; then
        log_warning "Python ${python_major}.${python_minor} detected, but Python 3.12+ is recommended"
    else
        log_success "Python version is compatible (${python_major}.${python_minor})"
    fi

    # Check virtual environment
    if [[ -n "${VIRTUAL_ENV:-}" ]]; then
        log_success "Virtual environment active: ${VIRTUAL_ENV}"
    elif [[ -f "/app/.venv/bin/python" ]]; then
        log_success "Virtual environment available: /app/.venv"
    else
        log_debug "No virtual environment detected (using system Python)"
    fi

    # Verify working directory
    if [[ ! -d "${PYTHON_PATH}" ]]; then
        log_error "Python path directory does not exist: ${PYTHON_PATH}"
        return 1
    fi

    if [[ ! -r "${PYTHON_PATH}" ]]; then
        log_error "Python path directory is not readable: ${PYTHON_PATH}"
        return 1
    fi

    log_success "Python environment verification completed"
    return 0
}

# Check critical Python dependencies
check_python_dependencies() {
    log_info "Checking critical Python dependencies..."

    local critical_packages=(
        "pydantic:Configuration validation"
        "click:CLI interface"
        "pandas:Data analysis"
        "numpy:Numerical computations"
        "aiohttp:HTTP client"
        "websockets:WebSocket connections"
        "langchain:LLM integration"
        "openai:OpenAI API client"
    )

    local missing_packages=()
    local available_count=0

    for package_info in "${critical_packages[@]}"; do
        local package="${package_info%:*}"
        local description="${package_info#*:}"

        if python -c "import ${package}" 2>/dev/null; then
            log_debug "✓ ${package} (${description})"
            ((available_count++))
        else
            log_warning "✗ ${package} (${description}) - NOT AVAILABLE"
            missing_packages+=("${package}")
        fi
    done

    local total_packages=${#critical_packages[@]}
    log_info "Dependencies check: ${available_count}/${total_packages} critical packages available"

    if [[ ${#missing_packages[@]} -gt 0 ]]; then
        log_warning "Missing packages: ${missing_packages[*]}"
        log_warning "The bot may not function correctly without these dependencies"
        return 1
    else
        log_success "All critical dependencies are available"
        return 0
    fi
}

# Validate bot module can be imported
validate_bot_module() {
    log_info "Validating bot module can be imported..."

    # Check if bot module directory exists
    if [[ ! -d "${PYTHON_PATH}/${BOT_MODULE}" ]]; then
        log_error "Bot module directory not found: ${PYTHON_PATH}/${BOT_MODULE}"
        log_error "Expected directory structure:"
        log_error "  ${PYTHON_PATH}/"
        log_error "  ├── ${BOT_MODULE}/"
        log_error "  │   ├── __init__.py"
        log_error "  │   └── main.py"
        return 1
    fi

    # Check if __init__.py exists
    if [[ ! -f "${PYTHON_PATH}/${BOT_MODULE}/__init__.py" ]]; then
        log_error "Bot module __init__.py not found: ${PYTHON_PATH}/${BOT_MODULE}/__init__.py"
        return 1
    fi

    # Try to import the bot module
    if ! python -c "import ${BOT_MODULE}" 2>/dev/null; then
        log_error "Failed to import bot module"
        log_error "Python import error details:"
        python -c "import ${BOT_MODULE}" 2>&1 | while IFS= read -r line; do
            log_error "  ${line}"
        done
        return 1
    fi

    log_success "Bot module imported successfully"

    # Check if main module exists and can be imported
    if python -c "import ${BOT_MODULE}.main" 2>/dev/null; then
        log_success "Bot main module validated"
    else
        log_warning "Bot main module could not be imported"
        log_warning "This may cause issues when starting the application"
    fi

    return 0
}

# Setup environment-specific configuration
setup_environment() {
    log_info "Setting up environment-specific configuration..."

    case "${ENVIRONMENT}" in
        "development")
            log_info "Development environment setup"
            export PYTHONOPTIMIZE=0
            export DEBUG=true
            ;;
        "production")
            log_info "Production environment setup"
            export PYTHONOPTIMIZE=1
            export DEBUG=false
            ;;
        "testing")
            log_info "Testing environment setup"
            export TESTING=true
            export DEBUG=false
            export PYTHONOPTIMIZE=0
            ;;
        *)
            log_warning "Unknown environment: ${ENVIRONMENT}, using defaults"
            export PYTHONOPTIMIZE=0
            ;;
    esac

    # Ensure dry run mode for safety
    if [[ "${DRY_RUN}" == "true" ]]; then
        log_success "Paper trading mode enabled (SYSTEM__DRY_RUN=true)"
    else
        log_warning "LIVE TRADING MODE DETECTED (SYSTEM__DRY_RUN=false)"
        log_warning "This will place real trades with real money!"
    fi

    return 0
}

# Enhanced startup validation
perform_startup_validation() {
    log_info "Performing comprehensive startup validation..."

    local validation_errors=()
    local validation_warnings=()

    # Check if we can create a simple Python script and run it
    local test_script="/tmp/python_test_$$"
    cat > "${test_script}" << 'EOF'
import sys
import os
print(f"Python {sys.version}")
print(f"Working directory: {os.getcwd()}")
print(f"Python path: {sys.path}")
EOF

    if python "${test_script}" >/dev/null 2>&1; then
        log_success "Python execution test passed"
        rm -f "${test_script}"
    else
        log_error "Python execution test failed"
        validation_errors+=("Python execution test failed")
        rm -f "${test_script}"
    fi

    # Validate environment variables
    local required_env_vars=("PYTHONPATH" "PYTHONUNBUFFERED" "PYTHONDONTWRITEBYTECODE")
    for env_var in "${required_env_vars[@]}"; do
        if [[ -n "${!env_var:-}" ]]; then
            log_debug "✓ ${env_var}=${!env_var}"
        else
            log_warning "Environment variable not set: ${env_var}"
            validation_warnings+=("Missing environment variable: ${env_var}")
        fi
    done

    # Check for configuration files
    local config_files=("config/development.json" "config/production.json" "prompts")
    for config_path in "${config_files[@]}"; do
        local full_path="${PYTHON_PATH}/${config_path}"
        if [[ -e "${full_path}" ]]; then
            log_debug "✓ Configuration exists: ${config_path}"
        else
            log_warning "Configuration missing: ${config_path}"
            validation_warnings+=("Missing configuration: ${config_path}")
        fi
    done

    # Summary
    local error_count=${#validation_errors[@]}
    local warning_count=${#validation_warnings[@]}

    if [[ ${error_count} -gt 0 ]]; then
        log_error "Startup validation failed with ${error_count} error(s):"
        for error in "${validation_errors[@]}"; do
            log_error "  - ${error}"
        done
        return 1
    fi

    if [[ ${warning_count} -gt 0 ]]; then
        log_warning "Startup validation completed with ${warning_count} warning(s):"
        for warning in "${validation_warnings[@]}"; do
            log_warning "  - ${warning}"
        done
    else
        log_success "Startup validation completed with no issues"
    fi

    return 0
}

# Perform health checks
perform_health_checks() {
    log_info "Performing health checks..."

    # Check disk space (graceful handling)
    if command -v df >/dev/null 2>&1; then
        local available_space=$(df /app 2>/dev/null | awk 'NR==2 {print $4}' || echo "0")
        if [[ "${available_space}" -gt 1048576 ]]; then  # 1GB in KB
            log_success "Sufficient disk space available: ${available_space}KB"
        else
            log_warning "Limited disk space: ${available_space}KB available"
        fi
    else
        log_debug "Cannot check disk space (df not available)"
    fi

    # Check memory (if available)
    if command -v free >/dev/null 2>&1; then
        local available_memory=$(free -m 2>/dev/null | awk 'NR==2{printf "%.0f", $7}' || echo "0")
        if [[ "${available_memory}" -gt 500 ]]; then
            log_success "Sufficient memory available: ${available_memory}MB"
        else
            log_warning "Limited memory: ${available_memory}MB available"
        fi
    else
        log_debug "Cannot check memory (free not available)"
    fi

    # Check network connectivity (basic, non-blocking)
    if command -v curl >/dev/null 2>&1; then
        if curl -s --connect-timeout 3 --max-time 5 https://api.coinbase.com/v2/time >/dev/null 2>&1; then
            log_success "Network connectivity verified"
        else
            log_debug "Network connectivity test failed (may be normal in some environments)"
        fi
    else
        log_debug "Cannot test network connectivity (curl not available)"
    fi

    # Always succeed - health checks are informational
    return 0
}

# Generate troubleshooting report
generate_troubleshooting_report() {
    log_error "==================== TROUBLESHOOTING REPORT ===================="
    log_error "Container startup failed. Here's diagnostic information:"
    log_error ""

    # System information
    log_error "System Information:"
    log_error "  Timestamp: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
    log_error "  Container ID: ${HOSTNAME:-unknown}"
    log_error "  User: $(whoami 2>/dev/null || echo 'unknown') ($(id -u):$(id -g))"
    log_error "  Shell: ${SHELL:-unknown}"
    log_error "  Working Directory: $(pwd 2>/dev/null || echo 'unknown')"
    log_error ""

    # Environment variables
    log_error "Critical Environment Variables:"
    local env_vars=("PYTHONPATH" "PYTHONUNBUFFERED" "PYTHONDONTWRITEBYTECODE" "SYSTEM__ENVIRONMENT" "SYSTEM__DRY_RUN" "EXCHANGE__EXCHANGE_TYPE")
    for var in "${env_vars[@]}"; do
        log_error "  ${var}=${!var:-'<not set>'}"
    done
    log_error ""

    # Python information
    log_error "Python Environment:"
    if command -v python >/dev/null 2>&1; then
        log_error "  Python Path: $(command -v python)"
        log_error "  Python Version: $(python --version 2>&1 || echo 'unknown')"
        log_error "  Python Executable: $(python -c 'import sys; print(sys.executable)' 2>/dev/null || echo 'unknown')"
        log_error "  Python Path: $(python -c 'import sys; print(sys.path)' 2>/dev/null || echo 'unknown')"
    else
        log_error "  Python: NOT FOUND"
    fi
    log_error ""

    # Directory structure
    log_error "Directory Structure:"
    if [[ -d "/app" ]]; then
        log_error "  /app directory exists"
        log_error "  /app contents:"
        ls -la /app 2>/dev/null | head -20 | while IFS= read -r line; do
            log_error "    ${line}"
        done

        if [[ -d "/app/bot" ]]; then
            log_error "  /app/bot directory exists"
            log_error "  /app/bot contents:"
            ls -la /app/bot 2>/dev/null | head -10 | while IFS= read -r line; do
                log_error "    ${line}"
            done
        else
            log_error "  /app/bot directory: NOT FOUND"
        fi
    else
        log_error "  /app directory: NOT FOUND"
    fi
    log_error ""

    # Permissions
    log_error "Permission Issues:"
    local test_dirs=("/app" "/app/logs" "/app/data" "/app/bot")
    for dir in "${test_dirs[@]}"; do
        if [[ -d "${dir}" ]]; then
            local perms=$(ls -ld "${dir}" 2>/dev/null | awk '{print $1,$3,$4}' || echo "unknown")
            log_error "  ${dir}: ${perms}"
        else
            log_error "  ${dir}: NOT FOUND"
        fi
    done
    log_error ""

    # Common solutions
    log_error "Common Solutions:"
    log_error "  1. Ensure the container has the correct user permissions:"
    log_error "     ./setup-docker-permissions.sh"
    log_error "  2. Check that all required directories exist in the container:"
    log_error "     docker-compose build --no-cache"
    log_error "  3. Verify Python and dependencies are installed:"
    log_error "     poetry install"
    log_error "  4. Check the Dockerfile builds correctly:"
    log_error "     docker build -t ai-trading-bot ."
    log_error "  5. Run the container with debug enabled:"
    log_error "     docker run -e DEBUG=true ai-trading-bot"
    log_error ""
    log_error "=============================================================="
}

# Cleanup function for graceful shutdown
cleanup() {
    local exit_code=$?
    if [[ ${exit_code} -ne 0 ]]; then
        log_error "Container initialization failed with exit code: ${exit_code}"
        generate_troubleshooting_report
    else
        log_success "Container initialization completed successfully"
    fi
    exit ${exit_code}
}

# Main initialization function
main() {
    # Set up signal handlers
    trap cleanup EXIT
    trap 'log_warning "Received SIGTERM, cleaning up..."; exit 0' TERM
    trap 'log_warning "Received SIGINT, cleaning up..."; exit 0' INT

    # Display startup information
    display_banner

    # Run initialization steps
    log_info "Starting container initialization..."

    # Step 1: Check user (always succeeds)
    check_user

    # Step 2: Setup directories (always succeeds with fallbacks)
    setup_directories

    # Step 3: Setup Python environment variables (required)
    if ! setup_python_environment; then
        log_error "Failed to setup Python environment variables"
        return 1
    fi

    # Step 4: Verify Python installation (critical)
    if ! verify_python_environment; then
        log_error "Python environment verification failed"
        log_error "Cannot proceed without a working Python installation"
        return 1
    fi

    # Step 5: Check Python dependencies (warning if missing)
    if ! check_python_dependencies; then
        log_warning "Some Python dependencies are missing, but continuing..."
        log_warning "The bot may not function correctly without all dependencies"
    fi

    # Step 6: Validate bot module import (critical)
    if ! validate_bot_module; then
        log_error "Bot module validation failed"
        log_error "Cannot start the application without the bot module"
        return 1
    fi

    # Step 7: Setup environment-specific configuration
    if ! setup_environment; then
        log_warning "Environment setup issues detected, but continuing..."
    fi

    # Step 8: Perform comprehensive startup validation
    if ! perform_startup_validation; then
        log_error "Startup validation failed"
        log_error "Critical issues detected that prevent safe startup"
        return 1
    fi

    # Step 9: Health checks (informational)
    if ! perform_health_checks; then
        log_warning "Health checks reported issues, but continuing..."
    fi

    log_success "Container initialization completed successfully"
    log_info "Starting application with command: $*"
    log_info "Working directory: $(pwd)"
    log_info "Python path: ${PYTHONPATH}"

    # Final pre-execution check
    if [[ $# -eq 0 ]]; then
        log_error "No command provided to execute"
        log_error "Usage: docker run <image> <command> [args...]"
        return 1
    fi

    # Log the final command being executed
    log_info "Executing: $*"

    # Execute the original command
    exec "$@"
}

# Execute main function with all arguments
main "$@"
