#!/bin/bash

# Docker Bench Security Installation Script
# Automates installation and configuration of Docker Bench Security for AI Trading Bot

set -euo pipefail

# Script Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKER_BENCH_DIR="${SCRIPT_DIR}/docker-bench-security"
LOG_FILE="${SCRIPT_DIR}/logs/install.log"
CONFIG_DIR="${SCRIPT_DIR}/config"
CUSTOM_CHECKS_DIR="${SCRIPT_DIR}/custom-checks"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "${timestamp} [${level}] ${message}" | tee -a "${LOG_FILE}"
}

# Error handling
error_exit() {
    log "ERROR" "$1"
    exit 1
}

# Success message
success() {
    log "INFO" "${GREEN}$1${NC}"
}

# Warning message
warning() {
    log "WARN" "${YELLOW}$1${NC}"
}

# Info message
info() {
    log "INFO" "${BLUE}$1${NC}"
}

# Check if running as root
check_root() {
    if [[ $EUID -eq 0 ]]; then
        error_exit "This script should not be run as root for security reasons"
    fi
}

# Create directory structure
create_directories() {
    info "Creating directory structure..."

    local dirs=(
        "${SCRIPT_DIR}/logs"
        "${SCRIPT_DIR}/config"
        "${SCRIPT_DIR}/custom-checks"
        "${SCRIPT_DIR}/reports"
        "${SCRIPT_DIR}/remediation"
        "${SCRIPT_DIR}/scripts"
        "${SCRIPT_DIR}/templates"
    )

    for dir in "${dirs[@]}"; do
        mkdir -p "$dir"
        info "Created directory: $dir"
    done
}

# Check system requirements
check_requirements() {
    info "Checking system requirements..."

    local required_tools=("docker" "git" "curl" "jq" "awk" "grep")
    local missing_tools=()

    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            missing_tools+=("$tool")
        fi
    done

    if [ ${#missing_tools[@]} -ne 0 ]; then
        error_exit "Missing required tools: ${missing_tools[*]}"
    fi

    # Check Docker daemon
    if ! docker info &> /dev/null; then
        error_exit "Docker daemon is not running or not accessible"
    fi

    success "All system requirements met"
}

# Install Docker Bench Security
install_docker_bench() {
    info "Installing Docker Bench Security..."

    if [ -d "$DOCKER_BENCH_DIR" ]; then
        warning "Docker Bench Security already exists. Updating..."
        cd "$DOCKER_BENCH_DIR"
        git pull origin main || error_exit "Failed to update Docker Bench Security"
    else
        git clone https://github.com/docker/docker-bench-security.git "$DOCKER_BENCH_DIR" \
            || error_exit "Failed to clone Docker Bench Security"
    fi

    # Make scripts executable
    chmod +x "${DOCKER_BENCH_DIR}/docker-bench-security.sh"

    success "Docker Bench Security installed successfully"
}

# Create custom configuration for trading bot
create_custom_config() {
    info "Creating custom Docker Bench configuration for trading bot..."

    cat > "${CONFIG_DIR}/docker-bench.conf" << 'EOF'
# Docker Bench Security Configuration for AI Trading Bot
# Custom configuration tailored for cryptocurrency trading applications

# Logging configuration
logdir="/app/security/docker-bench/logs"
logger="logger -t docker-bench-security"

# Output formats
create_logfile="true"
logfile="docker-bench-security.log"
outputfile="docker-bench-security-report.json"

# Check configuration
include_check_4_1="true"    # Ensure a user for the container has been created
include_check_4_2="true"    # Ensure that containers use only trusted base images
include_check_4_3="true"    # Ensure that unnecessary packages are not installed in the container
include_check_4_4="true"    # Ensure images are scanned and rebuilt to include security patches
include_check_4_5="true"    # Ensure Content trust for Docker is Enabled
include_check_4_6="true"    # Ensure that HEALTHCHECK instructions have been added to container images
include_check_4_7="true"    # Ensure update instructions are not used alone in the Dockerfile
include_check_4_8="true"    # Ensure setuid and setgid permissions are removed before making any user as root
include_check_4_9="true"    # Ensure that COPY is used instead of ADD in Dockerfiles
include_check_4_10="true"   # Ensure secrets are not stored in Dockerfiles

# Runtime security checks
include_check_5_1="true"    # Ensure that, if applicable, an AppArmor Profile is enabled
include_check_5_2="true"    # Ensure that, if applicable, SELinux security options are set
include_check_5_3="true"    # Ensure that Linux kernel capabilities are restricted within containers
include_check_5_4="true"    # Ensure that privileged containers are not used
include_check_5_5="true"    # Ensure sensitive host system directories are not mounted on containers
include_check_5_6="true"    # Ensure sshd is not run within containers
include_check_5_7="true"    # Ensure privileged ports are not mapped within containers
include_check_5_8="true"    # Ensure that only needed ports are open on the container
include_check_5_9="true"    # Ensure that the host's network namespace is not shared
include_check_5_10="true"   # Ensure that the memory usage for containers is limited
include_check_5_11="true"   # Ensure that CPU priority is set appropriately on containers
include_check_5_12="true"   # Ensure that the container's root filesystem is mounted as read only
include_check_5_13="true"   # Ensure that incoming container traffic is bound to a specific host interface
include_check_5_14="true"   # Ensure that the 'on-failure' container restart policy is set to '5'
include_check_5_15="true"   # Ensure that the host's process namespace is not shared
include_check_5_16="true"   # Ensure that the host's IPC namespace is not shared
include_check_5_17="true"   # Ensure that host devices are not directly exposed to containers
include_check_5_18="true"   # Ensure that the default ulimit is overwritten at runtime if needed
include_check_5_19="true"   # Ensure mount propagation mode is not set to shared
include_check_5_20="true"   # Ensure that the host's UTS namespace is not shared
include_check_5_21="true"   # Ensure the default seccomp profile is not Disabled
include_check_5_25="true"   # Ensure that the container is restricted from acquiring additional privileges
include_check_5_26="true"   # Ensure that container health is checked at runtime
include_check_5_27="true"   # Ensure that Docker commands always make use of the latest version of their image
include_check_5_28="true"   # Ensure that the PIDs cgroup limit is used
include_check_5_29="true"   # Ensure that Docker's default bridge 'docker0' is not used
include_check_5_30="true"   # Ensure that the host's user namespaces are not shared
include_check_5_31="true"   # Ensure that the Docker socket is not mounted inside any containers

# Docker Swarm configuration (if applicable)
include_check_7_1="false"   # Disable swarm mode checks for standalone deployment
include_check_7_2="false"
include_check_7_3="false"
include_check_7_4="false"
include_check_7_5="false"
include_check_7_6="false"
include_check_7_7="false"
include_check_7_8="false"
include_check_7_9="false"
include_check_7_10="false"

# Trading bot specific configurations
TRADING_BOT_CONTAINERS="ai-trading-bot bluefin-service dashboard-backend mcp-memory mcp-omnisearch"
TRADING_BOT_IMAGES="ai-trading-bot bluefin-sdk-service dashboard-backend mcp-memory-server mcp-omnisearch-server"
TRADING_BOT_NETWORKS="trading-network"

# Security thresholds
MAX_CRITICAL_ISSUES=0
MAX_HIGH_ISSUES=2
MAX_MEDIUM_ISSUES=5
MAX_LOW_ISSUES=10

# Remediation settings
AUTO_REMEDIATION_ENABLED="true"
AUTO_REMEDIATION_DRY_RUN="false"
BACKUP_BEFORE_REMEDIATION="true"

# Notification settings
NOTIFY_ON_CRITICAL="true"
NOTIFY_ON_HIGH="true"
NOTIFY_ON_MEDIUM="false"
NOTIFY_ON_LOW="false"

# Report retention
REPORT_RETENTION_DAYS=30
CLEANUP_OLD_REPORTS="true"
EOF

    success "Custom Docker Bench configuration created"
}

# Create custom security checks for trading bot
create_custom_checks() {
    info "Creating custom security checks for AI trading bot..."

    # Check for cryptocurrency-specific security issues
    cat > "${CUSTOM_CHECKS_DIR}/8_1_crypto_security.sh" << 'EOF'
#!/bin/bash
# Custom Check 8.1: Cryptocurrency Security Checks
# Validates security measures specific to cryptocurrency trading applications

check_8_1() {
    local id="8.1"
    local desc="Ensure cryptocurrency keys are not exposed in environment variables"
    local remediation="Remove sensitive keys from environment variables and use Docker secrets or external key management"
    local remediationImpact="High - Prevents unauthorized access to trading accounts"

    local totalChecks=0
    local totalFailed=0

    # Check for exposed private keys in environment variables
    for container in $(docker ps --format "{{.Names}}" | grep -E "(trading|bluefin|coinbase)"); do
        totalChecks=$((totalChecks + 1))

        # Check environment variables for sensitive patterns
        if docker exec "$container" env 2>/dev/null | grep -iE "(private_key|secret_key|api_key|mnemonic)" | grep -v "\\*\\*\\*"; then
            warn "$id     * Exposed cryptocurrency keys found in container: $container"
            logjson "WARN" "$id" "$desc" "$container" "Keys may be exposed in environment variables"
            totalFailed=$((totalFailed + 1))
        fi
    done

    if [ $totalFailed -eq 0 ]; then
        pass "$id     * No exposed cryptocurrency keys found in containers"
        logjson "PASS" "$id" "$desc"
    fi
}
EOF

    cat > "${CUSTOM_CHECKS_DIR}/8_2_trading_network.sh" << 'EOF'
#!/bin/bash
# Custom Check 8.2: Trading Network Security
# Validates network security for trading bot communications

check_8_2() {
    local id="8.2"
    local desc="Ensure trading bot network communications are secure"
    local remediation="Configure proper network isolation and encryption for trading communications"
    local remediationImpact="High - Prevents network-based attacks on trading systems"

    local totalChecks=0
    local totalFailed=0

    # Check for trading network isolation
    if docker network ls | grep -q "trading-network"; then
        totalChecks=$((totalChecks + 1))

        # Check network configuration
        network_info=$(docker network inspect trading-network)

        # Verify network is not using default bridge
        if echo "$network_info" | jq -r '.[0].Driver' | grep -q "bridge"; then
            # Check for custom bridge configuration
            if echo "$network_info" | jq -r '.[0].Name' | grep -q "bridge"; then
                warn "$id     * Trading network using default bridge - security risk"
                logjson "WARN" "$id" "$desc" "trading-network" "Using potentially insecure default bridge"
                totalFailed=$((totalFailed + 1))
            fi
        fi

        # Check for external connectivity restrictions
        if echo "$network_info" | jq -r '.[0].Internal' | grep -q "false"; then
            info "$id     * Trading network allows external connectivity"
        fi
    else
        warn "$id     * Trading network not found - containers may be using default network"
        logjson "WARN" "$id" "$desc" "system" "No custom trading network configured"
        totalFailed=$((totalFailed + 1))
    fi

    if [ $totalFailed -eq 0 ]; then
        pass "$id     * Trading network security configuration verified"
        logjson "PASS" "$id" "$desc"
    fi
}
EOF

    cat > "${CUSTOM_CHECKS_DIR}/8_3_data_persistence.sh" << 'EOF'
#!/bin/bash
# Custom Check 8.3: Trading Data Security
# Validates security of persistent trading data

check_8_3() {
    local id="8.3"
    local desc="Ensure trading data volumes have appropriate security permissions"
    local remediation="Set proper file permissions and ownership on trading data directories"
    local remediationImpact="Medium - Prevents unauthorized access to trading data and logs"

    local totalChecks=0
    local totalFailed=0

    # Check trading data directories
    local data_dirs=("./data" "./logs" "./config")

    for dir in "${data_dirs[@]}"; do
        if [ -d "$dir" ]; then
            totalChecks=$((totalChecks + 1))

            # Check permissions (should not be world-writable)
            if [ "$(stat -c %a "$dir" | cut -c3)" = "7" ]; then
                warn "$id     * Directory $dir is world-writable - security risk"
                logjson "WARN" "$id" "$desc" "$dir" "World-writable permissions"
                totalFailed=$((totalFailed + 1))
            fi

            # Check for sensitive files with wrong permissions
            find "$dir" -name "*.json" -o -name "*.log" -o -name "*.key" | while read -r file; do
                if [ -f "$file" ] && [ "$(stat -c %a "$file" | cut -c2-3)" = "77" ]; then
                    warn "$id     * Sensitive file $file has permissive permissions"
                    logjson "WARN" "$id" "$desc" "$file" "Permissive file permissions"
                    totalFailed=$((totalFailed + 1))
                fi
            done
        fi
    done

    if [ $totalFailed -eq 0 ]; then
        pass "$id     * Trading data directories have appropriate security permissions"
        logjson "PASS" "$id" "$desc"
    fi
}
EOF

    # Make custom checks executable
    chmod +x "${CUSTOM_CHECKS_DIR}"/*.sh

    success "Custom security checks created"
}

# Create wrapper script for Docker Bench
create_wrapper_script() {
    info "Creating Docker Bench wrapper script..."

    # Check if script already exists
    if [ -f "${SCRIPT_DIR}/scripts/run-security-scan.sh" ]; then
        success "Docker Bench wrapper script already exists"
    else
        warning "Docker Bench wrapper script not found - should be created separately"
    fi

    # Ensure script is executable
    if [ -f "${SCRIPT_DIR}/scripts/run-security-scan.sh" ]; then
        chmod +x "${SCRIPT_DIR}/scripts/run-security-scan.sh"
        success "Docker Bench wrapper script permissions set"
    fi
}

# Create systemd service for scheduled scans
create_systemd_service() {
    info "Creating systemd service for scheduled security scans..."

    # Create service file
    cat > "${SCRIPT_DIR}/templates/docker-security-scan.service" << EOF
[Unit]
Description=Docker Security Scan for AI Trading Bot
After=docker.service
Requires=docker.service

[Service]
Type=oneshot
User=${USER}
Group=${USER}
WorkingDirectory=${SCRIPT_DIR}
ExecStart=${SCRIPT_DIR}/scripts/run-security-scan.sh
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

    # Create timer file
    cat > "${SCRIPT_DIR}/templates/docker-security-scan.timer" << EOF
[Unit]
Description=Run Docker Security Scan Daily
Requires=docker-security-scan.service

[Timer]
OnCalendar=daily
RandomizedDelaySec=1800
Persistent=true

[Install]
WantedBy=timers.target
EOF

    success "Systemd service templates created"
    warning "To enable scheduled scans, copy service files to /etc/systemd/system/ and run:"
    warning "  sudo systemctl enable docker-security-scan.timer"
    warning "  sudo systemctl start docker-security-scan.timer"
}

# Main installation process
main() {
    info "Starting Docker Bench Security installation for AI Trading Bot"

    check_root
    create_directories
    check_requirements
    install_docker_bench
    create_custom_config
    create_custom_checks
    create_wrapper_script
    create_systemd_service

    success "Docker Bench Security installation completed successfully!"
    echo
    info "Next steps:"
    info "1. Review configuration in ${CONFIG_DIR}/docker-bench.conf"
    info "2. Run initial scan: ${SCRIPT_DIR}/scripts/run-security-scan.sh"
    info "3. Set up automated remediation (optional)"
    info "4. Configure monitoring and alerting"
    echo
    info "For VPS deployment, consider setting up the systemd timer for daily scans."
}

# Run main function
main "$@"
