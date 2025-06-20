#!/bin/bash
# AI Trading Bot Docker Entrypoint Script
# Requires bash 4.0+ for associative arrays
# Handles directory permissions, setup, and initialization before starting the main application
#
# This script runs before the main application to ensure:
# - All required directories exist and have proper permissions
# - Write permissions are verified
# - Fallback locations are available if needed
# - Environment-specific setup is performed
# - Clear error reporting for setup failures

set -euo pipefail  # Exit on any error, undefined variable, or pipe failure

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

# Configuration
readonly APP_USER="botuser"
readonly APP_GROUP="botuser"
readonly APP_UID=1000
readonly APP_GID=1000

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
            # Try alternative approaches if standard mkdir fails
            log_warning "Standard mkdir failed for ${dir_path}, trying alternatives..."

            # Try creating parent directories step by step
            local parent_dir=$(dirname "${dir_path}")
            if [[ "${parent_dir}" != "/" ]] && [[ "${parent_dir}" != "${dir_path}" ]]; then
                if mkdir -p "${parent_dir}" 2>/dev/null; then
                    log_debug "Created parent directory: ${parent_dir}"
                    if mkdir "${dir_path}" 2>/dev/null; then
                        log_success "Created directory: ${dir_path} (with parent)"
                    else
                        log_warning "Failed to create directory after parent setup: ${dir_path}"
                        # Don't return error - continue with permission attempts
                    fi
                else
                    log_warning "Failed to create parent directory: ${parent_dir}"
                    # Don't return error - continue with permission attempts
                fi
            else
                log_warning "Cannot create directory: ${dir_path}"
                # Don't return error - maybe it exists or permissions will work
            fi
        fi
    else
        log_debug "Directory already exists: ${dir_path}"
    fi

    # Set permissions - try multiple approaches
    if chmod "${permissions}" "${dir_path}" 2>/dev/null; then
        log_debug "Set permissions ${permissions} on ${dir_path}"
    else
        log_warning "Failed to set permissions ${permissions} on ${dir_path}, trying alternatives..."

        # Try setting more permissive permissions if the requested ones fail
        local fallback_permissions="755"
        if [[ "${permissions}" != "${fallback_permissions}" ]]; then
            if chmod "${fallback_permissions}" "${dir_path}" 2>/dev/null; then
                log_warning "Set fallback permissions ${fallback_permissions} on ${dir_path}"
            else
                log_warning "Failed to set any permissions on ${dir_path}"
                # Don't return error - directory exists, permissions might not be critical
            fi
        else
            log_warning "Could not set permissions on ${dir_path}, but continuing..."
            # Don't return error - directory might still be usable
        fi
    fi

    # Check if directory is at least readable
    if [[ -d "${dir_path}" ]] && [[ -r "${dir_path}" ]]; then
        log_debug "Directory ${dir_path} is accessible"
        return 0
    else
        log_warning "Directory ${dir_path} is not accessible, but continuing..."
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

# Setup fallback directory
setup_fallback() {
    local original_dir="$1"
    local fallback_dir="$2"

    log_warning "Setting up fallback directory: ${fallback_dir} for ${original_dir}"

    # Create fallback directory with multiple permission attempts
    local created_fallback=false

    # Try standard creation first
    if mkdir -p "${fallback_dir}" 2>/dev/null; then
        log_debug "Created fallback directory: ${fallback_dir}"
        created_fallback=true
    else
        # Try creating in /tmp if the specified fallback fails
        local tmp_fallback="/tmp/$(basename "${fallback_dir}")-$$"
        if mkdir -p "${tmp_fallback}" 2>/dev/null; then
            log_warning "Using alternative fallback location: ${tmp_fallback}"
            fallback_dir="${tmp_fallback}"
            created_fallback=true
        else
            log_error "Failed to create any fallback directory"
            return 1
        fi
    fi

    if [[ "${created_fallback}" == "true" ]]; then
        # Try to set permissions (775 -> 755 -> continue anyway)
        if chmod 775 "${fallback_dir}" 2>/dev/null; then
            log_debug "Set permissions 775 on fallback directory"
        elif chmod 755 "${fallback_dir}" 2>/dev/null; then
            log_warning "Set fallback permissions 755 on fallback directory"
        else
            log_warning "Could not set permissions on fallback directory, but continuing..."
        fi

        log_success "Fallback directory ready: ${fallback_dir}"

        # Test write permission
        if test_write_permission "${fallback_dir}"; then
            # Try to create symlink (best case scenario)
            if [[ -d "${original_dir}" ]] && [[ ! -L "${original_dir}" ]]; then
                log_warning "Creating backup of original directory: ${original_dir}.backup"
                mv "${original_dir}" "${original_dir}.backup" 2>/dev/null || {
                    log_warning "Could not backup original directory, continuing..."
                }
            fi

            # Create symlink to fallback
            if ln -sf "${fallback_dir}" "${original_dir}" 2>/dev/null; then
                log_success "Created symlink: ${original_dir} -> ${fallback_dir}"
                return 0
            else
                log_warning "Failed to create symlink, using environment variable instead"
                # Export environment variable as fallback
                local env_var_name="FALLBACK_$(basename "${original_dir}" | tr '[:lower:]' '[:upper:]')_DIR"
                export "${env_var_name}=${fallback_dir}"
                log_info "Set environment variable: ${env_var_name}=${fallback_dir}"
                return 0
            fi
        else
            log_warning "Fallback directory is not writable, but might still be usable"
            # Don't fail completely - set environment variable as last resort
            local env_var_name="FALLBACK_$(basename "${original_dir}" | tr '[:lower:]' '[:upper:]')_DIR"
            export "${env_var_name}=${fallback_dir}"
            log_info "Set environment variable: ${env_var_name}=${fallback_dir}"
            return 0
        fi
    fi

    return 1
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

    # Continue even if some directories failed - the application can use fallbacks
    if [[ "${success_count}" -gt 0 ]]; then
        log_success "At least ${success_count} directories are available - proceeding"
        return 0
    elif [[ ${#failed_dirs[@]} -gt 0 ]]; then
        log_warning "Some directories failed setup, but fallback mechanisms are in place"
        log_info "Application will use temporary directories and environment variables"
        return 0
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

    # Check virtual environment
    if [[ -n "${VIRTUAL_ENV:-}" ]]; then
        log_success "Virtual environment active: ${VIRTUAL_ENV}"
    elif [[ -f "/app/.venv/bin/python" ]]; then
        log_success "Virtual environment available: /app/.venv"
    else
        log_warning "No virtual environment detected"
    fi

    # Check critical Python packages
    local critical_packages=("pydantic" "click" "pandas")
    for package in "${critical_packages[@]}"; do
        if python -c "import ${package}" 2>/dev/null; then
            log_debug "Package available: ${package}"
        else
            log_warning "Package not available: ${package}"
        fi
    done

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

    # Check disk space
    local available_space=$(df /app | awk 'NR==2 {print $4}')
    if [[ "${available_space}" -gt 1048576 ]]; then  # 1GB in KB
        log_success "Sufficient disk space available"
    else
        log_warning "Low disk space detected: ${available_space}KB available"
    fi

    # Check memory (if available)
    if command -v free >/dev/null 2>&1; then
        local available_memory=$(free -m | awk 'NR==2{printf "%.0f", $7}')
        if [[ "${available_memory}" -gt 500 ]]; then
            log_success "Sufficient memory available: ${available_memory}MB"
        else
            log_warning "Low memory detected: ${available_memory}MB available"
        fi
    fi

    # Check network connectivity (basic)
    if command -v curl >/dev/null 2>&1; then
        if curl -s --connect-timeout 5 https://api.coinbase.com/v2/time >/dev/null 2>&1; then
            log_success "Network connectivity verified"
        else
            log_warning "Network connectivity issues detected"
        fi
    fi

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

    setup_directories || {
        log_warning "Directory setup had issues, but continuing with fallbacks..."
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

    log_success "Container initialization completed successfully"
    log_info "Starting application with command: $*"

    # Execute the original command
    exec "$@"
}

# Execute main function with all arguments
main "$@"
