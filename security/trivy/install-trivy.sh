#!/bin/bash

# Trivy Installation Script for AI Trading Bot Security Scanning
# This script installs Trivy on various Linux distributions and macOS

set -euo pipefail

# Configuration
TRIVY_VERSION="0.50.1"
INSTALL_DIR="/usr/local/bin"
CACHE_DIR="/tmp/trivy-cache"
CONFIG_DIR="/etc/trivy"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to detect OS
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if command -v apt-get >/dev/null 2>&1; then
            echo "ubuntu"
        elif command -v yum >/dev/null 2>&1; then
            echo "centos"
        elif command -v apk >/dev/null 2>&1; then
            echo "alpine"
        else
            echo "linux"
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    else
        echo "unknown"
    fi
}

# Function to check if running as root
check_root() {
    if [[ $EUID -eq 0 ]]; then
        log_warning "Running as root. This is not recommended for security reasons."
        return 0
    else
        return 1
    fi
}

# Function to install Trivy on Ubuntu/Debian
install_ubuntu() {
    log_info "Installing Trivy on Ubuntu/Debian..."
    
    # Install dependencies
    sudo apt-get update
    sudo apt-get install -y wget apt-transport-https gnupg lsb-release
    
    # Add Trivy repository
    wget -qO - https://aquasecurity.github.io/trivy-repo/deb/public.key | sudo apt-key add -
    echo "deb https://aquasecurity.github.io/trivy-repo/deb $(lsb_release -sc) main" | sudo tee -a /etc/apt/sources.list.d/trivy.list
    
    # Install Trivy
    sudo apt-get update
    sudo apt-get install -y trivy
    
    log_success "Trivy installed successfully via APT"
}

# Function to install Trivy on CentOS/RHEL
install_centos() {
    log_info "Installing Trivy on CentOS/RHEL..."
    
    # Add Trivy repository
    sudo tee /etc/yum.repos.d/trivy.repo <<EOF
[trivy]
name=Trivy repository
baseurl=https://aquasecurity.github.io/trivy-repo/rpm/releases/\$basearch/
gpgcheck=1
enabled=1
gpgkey=https://aquasecurity.github.io/trivy-repo/rpm/public.key
EOF
    
    # Install Trivy
    sudo yum -y update
    sudo yum -y install trivy
    
    log_success "Trivy installed successfully via YUM"
}

# Function to install Trivy on Alpine
install_alpine() {
    log_info "Installing Trivy on Alpine..."
    
    # Install dependencies
    apk add --no-cache wget ca-certificates
    
    # Download and install Trivy binary
    install_binary
}

# Function to install Trivy on macOS
install_macos() {
    log_info "Installing Trivy on macOS..."
    
    if command -v brew >/dev/null 2>&1; then
        # Install via Homebrew
        brew install trivy
        log_success "Trivy installed successfully via Homebrew"
    else
        log_warning "Homebrew not found. Installing binary directly..."
        install_binary
    fi
}

# Function to install Trivy binary directly
install_binary() {
    log_info "Installing Trivy binary directly..."
    
    # Detect architecture
    ARCH=$(uname -m)
    case $ARCH in
        x86_64) ARCH="64bit" ;;
        aarch64|arm64) ARCH="ARM64" ;;
        armv7l) ARCH="ARM" ;;
        *) log_error "Unsupported architecture: $ARCH"; exit 1 ;;
    esac
    
    # Detect OS for binary
    OS=$(uname -s)
    case $OS in
        Linux) OS="Linux" ;;
        Darwin) OS="macOS" ;;
        *) log_error "Unsupported OS: $OS"; exit 1 ;;
    esac
    
    # Download URL
    DOWNLOAD_URL="https://github.com/aquasecurity/trivy/releases/download/v${TRIVY_VERSION}/trivy_${TRIVY_VERSION}_${OS}-${ARCH}.tar.gz"
    
    log_info "Downloading Trivy from: $DOWNLOAD_URL"
    
    # Create temporary directory
    TEMP_DIR=$(mktemp -d)
    cd "$TEMP_DIR"
    
    # Download and extract
    wget -q "$DOWNLOAD_URL" -O trivy.tar.gz
    tar -xzf trivy.tar.gz
    
    # Install binary
    if check_root; then
        cp trivy "$INSTALL_DIR/"
        chmod +x "$INSTALL_DIR/trivy"
    else
        sudo cp trivy "$INSTALL_DIR/"
        sudo chmod +x "$INSTALL_DIR/trivy"
    fi
    
    # Cleanup
    cd /
    rm -rf "$TEMP_DIR"
    
    log_success "Trivy binary installed successfully"
}

# Function to create configuration directories
setup_directories() {
    log_info "Setting up Trivy directories..."
    
    # Create cache directory
    mkdir -p "$CACHE_DIR"
    chmod 755 "$CACHE_DIR"
    
    # Create config directory
    if check_root; then
        mkdir -p "$CONFIG_DIR"
        chmod 755 "$CONFIG_DIR"
    else
        sudo mkdir -p "$CONFIG_DIR"
        sudo chmod 755 "$CONFIG_DIR"
    fi
    
    log_success "Directories created successfully"
}

# Function to copy configuration files
setup_config() {
    log_info "Setting up Trivy configuration..."
    
    # Get script directory
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    
    # Copy configuration file
    if [[ -f "$SCRIPT_DIR/trivy-config.yaml" ]]; then
        if check_root; then
            cp "$SCRIPT_DIR/trivy-config.yaml" "$CONFIG_DIR/"
        else
            sudo cp "$SCRIPT_DIR/trivy-config.yaml" "$CONFIG_DIR/"
        fi
        log_success "Configuration file copied to $CONFIG_DIR/"
    else
        log_warning "Configuration file not found at $SCRIPT_DIR/trivy-config.yaml"
    fi
}

# Function to update Trivy database
update_database() {
    log_info "Updating Trivy vulnerability database..."
    
    # Update vulnerability database
    trivy image --download-db-only
    
    log_success "Vulnerability database updated successfully"
}

# Function to verify installation
verify_installation() {
    log_info "Verifying Trivy installation..."
    
    # Check if Trivy is installed
    if command -v trivy >/dev/null 2>&1; then
        VERSION=$(trivy --version | head -n1)
        log_success "Trivy installed successfully: $VERSION"
        
        # Test with a simple scan
        log_info "Running test scan..."
        if trivy image --quiet --format table hello-world >/dev/null 2>&1; then
            log_success "Test scan completed successfully"
        else
            log_warning "Test scan failed, but Trivy is installed"
        fi
    else
        log_error "Trivy installation failed"
        exit 1
    fi
}

# Function to create systemd service for Trivy server (optional)
setup_service() {
    if [[ "$1" == "--server" ]]; then
        log_info "Setting up Trivy server service..."
        
        # Create systemd service file
        sudo tee /etc/systemd/system/trivy-server.service >/dev/null <<EOF
[Unit]
Description=Trivy Vulnerability Scanner Server
After=network.target

[Service]
Type=simple
User=trivy
Group=trivy
ExecStart=/usr/local/bin/trivy server --listen 0.0.0.0:8080 --cache-dir /var/cache/trivy
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=trivy-server

# Security settings
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/var/cache/trivy
CapabilityBoundingSet=CAP_NET_BIND_SERVICE
AmbientCapabilities=CAP_NET_BIND_SERVICE

[Install]
WantedBy=multi-user.target
EOF
        
        # Create trivy user
        sudo useradd -r -s /bin/false -d /var/cache/trivy trivy 2>/dev/null || true
        sudo mkdir -p /var/cache/trivy
        sudo chown trivy:trivy /var/cache/trivy
        
        # Enable and start service
        sudo systemctl daemon-reload
        sudo systemctl enable trivy-server
        
        log_success "Trivy server service configured"
        log_info "Start with: sudo systemctl start trivy-server"
    fi
}

# Function to display help
show_help() {
    cat <<EOF
Trivy Installation Script for AI Trading Bot

Usage: $0 [OPTIONS]

Options:
    --server        Install Trivy as a server service (Linux only)
    --help, -h      Show this help message

Examples:
    $0                    # Install Trivy client
    $0 --server          # Install Trivy with server service
    $0 --help            # Show help

The script will:
1. Detect your operating system
2. Install Trivy using the appropriate method
3. Set up configuration directories
4. Copy configuration files
5. Update the vulnerability database
6. Verify the installation

Configuration files will be placed in:
- System config: $CONFIG_DIR/
- Cache directory: $CACHE_DIR

After installation, you can use the provided scanning scripts:
- scan-images.sh     # Scan Docker images
- scan-filesystem.sh # Scan filesystem
- scan-configs.sh    # Scan configurations
EOF
}

# Main installation function
main() {
    log_info "Starting Trivy installation for AI Trading Bot..."
    
    # Parse arguments
    SERVER_MODE=false
    while [[ $# -gt 0 ]]; do
        case $1 in
            --server)
                SERVER_MODE=true
                shift
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # Detect operating system
    OS=$(detect_os)
    log_info "Detected OS: $OS"
    
    # Install Trivy based on OS
    case $OS in
        ubuntu)
            install_ubuntu
            ;;
        centos)
            install_centos
            ;;
        alpine)
            install_alpine
            ;;
        macos)
            install_macos
            ;;
        *)
            log_warning "Unknown OS, attempting binary installation..."
            install_binary
            ;;
    esac
    
    # Setup directories and configuration
    setup_directories
    setup_config
    
    # Update database
    update_database
    
    # Verify installation
    verify_installation
    
    # Setup server if requested
    if [[ "$SERVER_MODE" == true ]]; then
        if [[ "$OS" != "ubuntu" && "$OS" != "centos" ]]; then
            log_warning "Server mode only supported on Ubuntu/CentOS"
        else
            setup_service --server
        fi
    fi
    
    log_success "Trivy installation completed successfully!"
    log_info "Next steps:"
    log_info "1. Run: ./scan-images.sh to scan Docker images"
    log_info "2. Run: ./scan-filesystem.sh to scan filesystem"
    log_info "3. Run: ./scan-configs.sh to scan configurations"
    log_info "4. Check the reports/ directory for detailed results"
}

# Run main function if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi