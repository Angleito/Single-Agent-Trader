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
# - Robust user mapping handling (named users, UIDs, anonymous users)
# - Container environment detection and adaptation
# - Graceful fallback for permission and directory creation issues

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

# Configuration - detect actual user and group
readonly CURRENT_USER=$(whoami)
readonly CURRENT_UID=$(id -u)
readonly CURRENT_GID=$(id -g)
readonly CURRENT_GROUP=$(id -gn)

# Expected user (can be overridden by environment)
readonly APP_USER="${EXPECTED_USER:-bluefin}"
readonly APP_GROUP="${EXPECTED_GROUP:-bluefin}"

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

# Detect container environment and capabilities
detect_container_environment() {
    # Check if we're in a container
    if [[ -f /.dockerenv ]] || grep -q 'docker\|lxc' /proc/1/cgroup 2>/dev/null; then
        log_debug "Running in container environment"
        return 0
    else
        log_debug "Not running in container environment"
        return 1
    fi
}

# Check if we have root privileges
has_root_privileges() {
    if [[ $(id -u) -eq 0 ]]; then
        return 0
    else
        return 1
    fi
}

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
    log_info "Expected User: ${APP_USER}:${APP_GROUP}"
    log_info "=================================="
}

# Check if running as correct user
check_user() {
    # Check if we're running as expected user
    if [[ "${CURRENT_USER}" == "${APP_USER}" ]]; then
        log_success "Running as expected user: ${APP_USER}"
        log_info "Using built-in user scenario"
    else
        log_info "Running as user ${CURRENT_USER} (${CURRENT_UID}:${CURRENT_GID}), expected ${APP_USER}"
        log_info "This is normal when using Docker Compose user mapping"
        log_info "Using host user mapping scenario"
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
                        log_error "Failed to create directory: ${dir_path} even with parent creation"
                        return 1
                    fi
                else
                    log_error "Failed to create parent directory: ${parent_dir}"
                    return 1
                fi
            else
                log_error "Failed to create directory: ${dir_path}"
                return 1
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
            log_warning "Failed to set permissions on ${dir_path}"
            # Don't return error - directory exists, permissions might not be critical
        fi
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
            log_warning "Using temporary fallback: ${tmp_fallback} instead of ${fallback_dir}"
            fallback_dir="${tmp_fallback}"
            created_fallback=true
        else
            log_error "Failed to create any fallback directory"
            return 1
        fi
    fi

    if [[ "${created_fallback}" == "true" ]]; then
        # Try to set permissions, but don't fail if it doesn't work
        if chmod 775 "${fallback_dir}" 2>/dev/null; then
            log_debug "Set permissions 775 on fallback: ${fallback_dir}"
        elif chmod 755 "${fallback_dir}" 2>/dev/null; then
            log_warning "Set fallback permissions 755 on: ${fallback_dir}"
        else
            log_warning "Could not set specific permissions on: ${fallback_dir}"
        fi

        log_success "Fallback directory ready: ${fallback_dir}"

        # Test write permission
        if test_write_permission "${fallback_dir}"; then
            # Handle original directory backup/removal
            if [[ -d "${original_dir}" ]] && [[ ! -L "${original_dir}" ]]; then
                local backup_name="${original_dir}.backup.$(date +%s)"
                if mv "${original_dir}" "${backup_name}" 2>/dev/null; then
                    log_info "Created backup of original directory: ${backup_name}"
                else
                    # Try removing if move fails
                    if rm -rf "${original_dir}" 2>/dev/null; then
                        log_warning "Removed original directory: ${original_dir}"
                    else
                        log_warning "Could not backup or remove original directory: ${original_dir}"
                    fi
                fi
            fi

            # Create symlink to fallback
            if ln -sf "${fallback_dir}" "${original_dir}" 2>/dev/null; then
                log_success "Created symlink: ${original_dir} -> ${fallback_dir}"
                return 0
            else
                log_warning "Failed to create symlink, trying direct path export"
                # Export the fallback path as an environment variable for the application to use
                export "FALLBACK_$(basename "${original_dir}" | tr '[:lower:]' '[:upper:]')_DIR=${fallback_dir}"
                log_info "Exported fallback path as environment variable"
                return 0
            fi
        else
            log_error "Fallback directory is not writable: ${fallback_dir}"
            return 1
        fi
    else
        log_error "Failed to create fallback directory"
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

# Ensure proper permissions for current user
ensure_user_permissions() {
    log_info "Ensuring proper permissions for current user (${CURRENT_USER})"

    # List of directories that need to be accessible
    local key_dirs=("/app/logs" "/app/data" "/app/tmp")

    for dir in "${key_dirs[@]}"; do
        if [[ -d "${dir}" ]]; then
            # Check if directory is writable by current user
            if [[ -w "${dir}" ]]; then
                log_debug "Directory ${dir} is writable by current user"
            else
                log_warning "Directory ${dir} is not writable by current user"
                # Try to fix permissions if possible
                if chmod g+w "${dir}" 2>/dev/null; then
                    log_success "Fixed permissions for ${dir}"
                else
                    log_warning "Could not fix permissions for ${dir} - will use fallback"
                fi
            fi
        fi
    done

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

    # Environment detection
    if detect_container_environment; then
        log_info "Container environment detected"
    else
        log_warning "Not running in container - some features may behave differently"
    fi

    if has_root_privileges; then
        log_warning "Running with root privileges - consider using a non-root user for security"
    else
        log_info "Running with non-root privileges (recommended)"
    fi

    # User check is now always informational and never fails
    check_user

    # Ensure directories have proper permissions for current user
    ensure_user_permissions

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
