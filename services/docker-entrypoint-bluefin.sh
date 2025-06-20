#!/bin/bash
# Bluefin SDK Service Docker Entrypoint Script
# Handles directory permissions, setup, and initialization before starting the Bluefin service
#
# This script runs before the Bluefin service to ensure:
# - All required directories exist and have proper permissions
# - Write permissions are verified
# - Fallback locations are available if needed
# - Environment-specific setup is performed
# - Clear error reporting for setup failures

set -euo pipefail  # Exit on any error, undefined variable, or pipe failure

# Script metadata
SCRIPT_NAME="docker-entrypoint-bluefin.sh"
SCRIPT_VERSION="1.0.0"
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

# Colors for output formatting
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly CYAN='\033[0;36m'
readonly NC='\033[0m' # No Color

# Configuration
readonly APP_USER="bluefin"
readonly APP_GROUP="bluefin"

# Directory configuration (path:permissions pairs)
readonly REQUIRED_DIRS=(
    "/app/logs:775"
    "/app/data:775"
    "/app/tmp:775"
)

# Fallback directories (original:fallback pairs)
readonly FALLBACK_DIRS=(
    "/app/logs:/tmp/bluefin-logs"
    "/app/data:/tmp/bluefin-data"
    "/app/tmp:/tmp/bluefin-tmp"
)

# Environment detection
readonly ENVIRONMENT="${NODE_ENV:-production}"
readonly BLUEFIN_NETWORK="${BLUEFIN_NETWORK:-mainnet}"

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
    log_info "Bluefin SDK Service Container Initialization"
    log_info "Version: ${SCRIPT_VERSION}"
    log_info "Environment: ${ENVIRONMENT}"
    log_info "Network: ${BLUEFIN_NETWORK}"
    log_info "User: ${APP_USER}:${APP_GROUP}"
    log_info "=================================="
}

# Check if running as correct user
check_user() {
    local current_user=$(whoami)

    if [[ "${current_user}" != "${APP_USER}" ]]; then
        log_warning "Running as user ${current_user}, expected ${APP_USER}"
        log_warning "This may indicate a container configuration issue"
    else
        log_success "Running as correct user: ${APP_USER}"
    fi
}

# Create directory with proper permissions
create_directory() {
    local dir_path="$1"
    local permissions="$2"

    log_debug "Creating directory: ${dir_path} with permissions ${permissions}"

    # Create directory if it doesn't exist
    if [[ ! -d "${dir_path}" ]]; then
        if mkdir -p "${dir_path}" 2>/dev/null; then
            log_success "Created directory: ${dir_path}"
        else
            log_error "Failed to create directory: ${dir_path}"
            return 1
        fi
    else
        log_debug "Directory already exists: ${dir_path}"
    fi

    # Set permissions
    if chmod "${permissions}" "${dir_path}" 2>/dev/null; then
        log_debug "Set permissions ${permissions} on ${dir_path}"
    else
        log_warning "Failed to set permissions ${permissions} on ${dir_path}"
        return 1
    fi

    return 0
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

# Setup fallback directory
setup_fallback() {
    local original_dir="$1"
    local fallback_dir="$2"

    log_warning "Setting up fallback directory: ${fallback_dir} for ${original_dir}"

    # Create fallback directory
    if mkdir -p "${fallback_dir}" 2>/dev/null && chmod 775 "${fallback_dir}" 2>/dev/null; then
        log_success "Fallback directory ready: ${fallback_dir}"

        # Test write permission
        if test_write_permission "${fallback_dir}"; then
            # Create symlink if original directory exists but isn't writable
            if [[ -d "${original_dir}" ]] && [[ ! -L "${original_dir}" ]]; then
                log_warning "Creating backup of original directory: ${original_dir}.backup"
                mv "${original_dir}" "${original_dir}.backup" 2>/dev/null || true
            fi

            # Create symlink to fallback
            if ln -sf "${fallback_dir}" "${original_dir}" 2>/dev/null; then
                log_success "Created symlink: ${original_dir} -> ${fallback_dir}"
                return 0
            else
                log_error "Failed to create symlink to fallback directory"
                return 1
            fi
        else
            log_error "Fallback directory is not writable: ${fallback_dir}"
            return 1
        fi
    else
        log_error "Failed to create fallback directory: ${fallback_dir}"
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
    log_info "Setting up required directories..."

    local failed_dirs=()
    local success_count=0
    local total_dirs=${#REQUIRED_DIRS[@]}

    # Create required directories
    for dir_config in "${REQUIRED_DIRS[@]}"; do
        local dir_path="${dir_config%:*}"
        local permissions="${dir_config#*:}"

        if create_directory "${dir_path}" "${permissions}"; then
            if test_write_permission "${dir_path}"; then
                ((success_count++))
                log_success "Directory ready: ${dir_path}"
            else
                log_warning "Directory created but not writable: ${dir_path}"
                failed_dirs+=("${dir_path}")
            fi
        else
            log_error "Failed to setup directory: ${dir_path}"
            failed_dirs+=("${dir_path}")
        fi
    done

    # Handle failed directories with fallbacks
    if [[ ${#failed_dirs[@]} -gt 0 ]]; then
        for failed_dir in "${failed_dirs[@]}"; do
            local fallback_dir
            if fallback_dir=$(get_fallback_dir "${failed_dir}"); then
                if setup_fallback "${failed_dir}" "${fallback_dir}"; then
                    ((success_count++))
                    log_success "Fallback setup successful for: ${failed_dir}"
                else
                    log_error "Fallback setup failed for: ${failed_dir}"
                fi
            else
                log_error "No fallback available for: ${failed_dir}"
            fi
        done
    fi

    log_info "Directory setup complete: ${success_count}/${total_dirs} directories ready"

    # Fail if critical directories are not available
    if [[ ${#failed_dirs[@]} -gt 0 ]] && [[ "${success_count}" -lt 2 ]]; then
        log_error "Too many critical directories failed setup. Cannot continue safely."
        return 1
    fi

    return 0
}

# Verify Python environment
verify_python_environment() {
    log_info "Verifying Python environment..."

    # Check Python version
    if python --version >/dev/null 2>&1; then
        local python_version=$(python --version 2>&1)
        log_success "Python available: ${python_version}"
    else
        log_error "Python not available"
        return 1
    fi

    # Check critical Python packages for Bluefin service
    local critical_packages=("fastapi" "uvicorn" "pydantic" "aiohttp")
    for package in "${critical_packages[@]}"; do
        if python -c "import ${package}" 2>/dev/null; then
            log_debug "Package available: ${package}"
        else
            log_warning "Package not available: ${package}"
        fi
    done

    # Check Bluefin SDK specifically
    if python -c "import bluefinApi" 2>/dev/null; then
        log_success "Bluefin SDK available"
    else
        log_warning "Bluefin SDK not available - some features may not work"
    fi

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
        *)
            log_warning "Unknown environment: ${ENVIRONMENT}, using defaults"
            ;;
    esac

    # Setup Bluefin network configuration
    case "${BLUEFIN_NETWORK}" in
        "mainnet")
            log_success "Bluefin mainnet configuration"
            ;;
        "testnet")
            log_info "Bluefin testnet configuration"
            ;;
        *)
            log_warning "Unknown Bluefin network: ${BLUEFIN_NETWORK}"
            ;;
    esac

    return 0
}

# Perform health checks
perform_health_checks() {
    log_info "Performing health checks..."

    # Check disk space
    local available_space=$(df /app | awk 'NR==2 {print $4}')
    if [[ "${available_space}" -gt 1048576 ]]; then  # 1GB in KB
        log_success "Sufficient disk space available"
    else
        log_warning "Low disk space detected: ${available_space}KB available"
    fi

    # Check network connectivity (basic)
    if command -v curl >/dev/null 2>&1; then
        if curl -s --connect-timeout 5 https://api.bluefin.io/health >/dev/null 2>&1; then
            log_success "Bluefin API connectivity verified"
        else
            log_warning "Bluefin API connectivity issues detected"
        fi
    fi

    return 0
}

# Cleanup function for graceful shutdown
cleanup() {
    local exit_code=$?
    if [[ ${exit_code} -ne 0 ]]; then
        log_error "Bluefin service initialization failed with exit code: ${exit_code}"
    else
        log_success "Bluefin service initialization completed successfully"
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
    log_info "Starting Bluefin service initialization..."

    check_user || {
        log_warning "User check failed, but continuing..."
    }

    setup_directories || {
        log_error "Directory setup failed"
        exit 1
    }

    verify_python_environment || {
        log_error "Python environment verification failed"
        exit 1
    }

    setup_environment || {
        log_error "Environment setup failed"
        exit 1
    }

    perform_health_checks || {
        log_warning "Health checks reported issues, but continuing..."
    }

    log_success "Bluefin service initialization completed successfully"
    log_info "Starting service with command: $*"

    # Execute the original command
    exec "$@"
}

# Execute main function with all arguments
main "$@"
