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

# Verify Python environment
verify_python_environment() {
    log_info "Verifying Python environment..."

    # Check Python version
    if python --version >/dev/null 2>&1; then
        local python_version=$(python --version 2>&1 || echo "unknown")
        log_success "Python available: ${python_version}"
    else
        log_warning "Python not available in PATH"
        # Don't return error - container might still work
    fi

    # Check virtual environment
    if [[ -n "${VIRTUAL_ENV:-}" ]]; then
        log_success "Virtual environment active: ${VIRTUAL_ENV}"
    elif [[ -f "/app/.venv/bin/python" ]]; then
        log_success "Virtual environment available: /app/.venv"
    else
        log_debug "No virtual environment detected (may be system Python)"
    fi

    # Check critical Python packages (informational only)
    local critical_packages=("pydantic" "click" "pandas")
    for package in "${critical_packages[@]}"; do
        if python -c "import ${package}" 2>/dev/null; then
            log_debug "Package available: ${package}"
        else
            log_debug "Package not immediately available: ${package}"
        fi
    done

    # Always succeed - Python issues will be caught at runtime
    return 0
}

# Setup environment-specific configuration
setup_environment() {
    log_info "Setting up environment-specific configuration..."

    case "${ENVIRONMENT}" in
        "development")
            log_info "Development environment setup"
            export PYTHONDONTWRITEBYTECODE=1
            export PYTHONUNBUFFERED=1
            ;;
        "production")
            log_info "Production environment setup"
            export PYTHONDONTWRITEBYTECODE=1
            export PYTHONUNBUFFERED=1
            export PYTHONOPTIMIZE=1
            ;;
        "testing")
            log_info "Testing environment setup"
            export TESTING=true
            export DEBUG=false
            ;;
        *)
            log_warning "Unknown environment: ${ENVIRONMENT}, using defaults"
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

# Cleanup function for graceful shutdown
cleanup() {
    local exit_code=$?
    if [[ ${exit_code} -ne 0 ]]; then
        log_error "Container initialization failed with exit code: ${exit_code}"
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

    check_user  # Always succeeds now

    setup_directories  # Always succeeds now

    # These steps are now resilient and won't fail the container startup
    if ! verify_python_environment; then
        log_warning "Python environment issues detected, but continuing..."
    fi

    if ! setup_environment; then
        log_warning "Environment setup issues detected, but continuing..."
    fi

    if ! perform_health_checks; then
        log_warning "Health checks reported issues, but continuing..."
    fi

    log_success "Container initialization completed successfully"
    log_info "Starting application with command: $*"

    # Execute the original command
    exec "$@"
}

# Execute main function with all arguments
main "$@"
