#!/bin/bash
# Docker Secrets Management Script
# This script helps create, update, and manage Docker secrets securely

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Secret name prefix
PREFIX="ai_trading_bot"

# Function to print colored output
print_message() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Function to check if Docker is running
check_docker() {
    if ! docker info >/dev/null 2>&1; then
        print_message "$RED" "Error: Docker is not running"
        exit 1
    fi
}

# Function to create a secret from stdin
create_secret_from_stdin() {
    local secret_name=$1
    local description=$2

    print_message "$YELLOW" "Creating secret: ${PREFIX}_${secret_name}"
    print_message "$GREEN" "Description: $description"
    echo -n "Enter value (input will be hidden): "

    # Read secret without echoing to terminal
    read -s secret_value
    echo # New line after hidden input

    # Create the secret
    echo -n "$secret_value" | docker secret create "${PREFIX}_${secret_name}" - >/dev/null 2>&1

    if [ $? -eq 0 ]; then
        print_message "$GREEN" "✓ Secret created successfully"
    else
        print_message "$RED" "✗ Failed to create secret"
        return 1
    fi
}

# Function to create a secret from file
create_secret_from_file() {
    local secret_name=$1
    local file_path=$2

    if [ ! -f "$file_path" ]; then
        print_message "$RED" "Error: File not found: $file_path"
        return 1
    fi

    print_message "$YELLOW" "Creating secret from file: ${PREFIX}_${secret_name}"

    docker secret create "${PREFIX}_${secret_name}" "$file_path" >/dev/null 2>&1

    if [ $? -eq 0 ]; then
        print_message "$GREEN" "✓ Secret created successfully"
        # Securely delete the source file if requested
        echo -n "Delete source file? (y/N): "
        read -r delete_confirm
        if [[ "$delete_confirm" =~ ^[Yy]$ ]]; then
            shred -vfz -n 3 "$file_path"
            print_message "$GREEN" "✓ Source file securely deleted"
        fi
    else
        print_message "$RED" "✗ Failed to create secret"
        return 1
    fi
}

# Function to update a secret
update_secret() {
    local secret_name=$1

    # Check if secret exists
    if ! docker secret inspect "${PREFIX}_${secret_name}" >/dev/null 2>&1; then
        print_message "$RED" "Error: Secret ${PREFIX}_${secret_name} does not exist"
        return 1
    fi

    # Remove old secret
    docker secret rm "${PREFIX}_${secret_name}" >/dev/null 2>&1

    # Create new secret
    echo -n "Enter new value for ${secret_name} (input will be hidden): "
    read -s new_value
    echo

    echo -n "$new_value" | docker secret create "${PREFIX}_${secret_name}" - >/dev/null 2>&1

    if [ $? -eq 0 ]; then
        print_message "$GREEN" "✓ Secret updated successfully"
    else
        print_message "$RED" "✗ Failed to update secret"
        return 1
    fi
}

# Function to list all secrets
list_secrets() {
    print_message "$GREEN" "Current Docker secrets:"
    docker secret ls --filter name="${PREFIX}" --format "table {{.Name}}\t{{.CreatedAt}}"
}

# Function to verify secrets are properly created
verify_secrets() {
    local required_secrets=(
        "openai_key"
        "coinbase_api_key"
        "coinbase_private_key"
        "bluefin_key"
    )

    print_message "$YELLOW" "Verifying required secrets..."

    local missing=0
    for secret in "${required_secrets[@]}"; do
        if docker secret inspect "${PREFIX}_${secret}" >/dev/null 2>&1; then
            print_message "$GREEN" "✓ ${PREFIX}_${secret}"
        else
            print_message "$RED" "✗ ${PREFIX}_${secret} (missing)"
            missing=$((missing + 1))
        fi
    done

    if [ $missing -eq 0 ]; then
        print_message "$GREEN" "\nAll required secrets are present!"
    else
        print_message "$RED" "\n$missing required secrets are missing"
        return 1
    fi
}

# Function to create all required secrets
create_all_secrets() {
    print_message "$YELLOW" "Creating all required secrets for AI Trading Bot\n"

    create_secret_from_stdin "openai_key" "OpenAI API Key for LLM trading decisions"
    echo

    create_secret_from_stdin "coinbase_api_key" "Coinbase API Key"
    echo

    print_message "$YELLOW" "For Coinbase Private Key, you can:"
    echo "1. Enter the key directly"
    echo "2. Provide a file path"
    echo -n "Choose option (1 or 2): "
    read -r option

    if [ "$option" = "2" ]; then
        echo -n "Enter file path: "
        read -r file_path
        create_secret_from_file "coinbase_private_key" "$file_path"
    else
        create_secret_from_stdin "coinbase_private_key" "Coinbase Private Key (PEM format)"
    fi
    echo

    create_secret_from_stdin "bluefin_key" "Bluefin/Sui Wallet Private Key"
    echo

    # Optional secrets
    echo -n "Create optional secrets (database_password, jwt_secret)? (y/N): "
    read -r create_optional
    if [[ "$create_optional" =~ ^[Yy]$ ]]; then
        create_secret_from_stdin "db_password" "Database Password"
        echo
        create_secret_from_stdin "jwt_secret" "JWT Secret for authentication"
    fi
}

# Function to export secrets to encrypted backup
backup_secrets() {
    local backup_dir="./secrets_backup_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$backup_dir"

    print_message "$YELLOW" "Backing up secrets to $backup_dir"

    # Export each secret
    for secret in $(docker secret ls --filter name="${PREFIX}" --format "{{.Name}}"); do
        # Note: Docker doesn't allow direct export of secrets
        # This is a placeholder for backup logic
        print_message "$YELLOW" "Backup of secrets must be done manually for security"
    done

    print_message "$YELLOW" "\nTo backup secrets securely:"
    echo "1. Use docker secret inspect to view secret metadata"
    echo "2. Store actual secret values in a password manager"
    echo "3. Never store unencrypted secrets in files"
}

# Main menu
main_menu() {
    while true; do
        echo
        print_message "$GREEN" "=== Docker Secrets Manager for AI Trading Bot ==="
        echo "1. Create all required secrets"
        echo "2. Create single secret"
        echo "3. Update existing secret"
        echo "4. List all secrets"
        echo "5. Verify required secrets"
        echo "6. Backup secrets (instructions)"
        echo "7. Exit"
        echo
        echo -n "Choose an option: "
        read -r choice

        case $choice in
            1)
                create_all_secrets
                ;;
            2)
                echo -n "Enter secret name (without prefix): "
                read -r name
                echo -n "Enter description: "
                read -r desc
                create_secret_from_stdin "$name" "$desc"
                ;;
            3)
                echo -n "Enter secret name to update (without prefix): "
                read -r name
                update_secret "$name"
                ;;
            4)
                list_secrets
                ;;
            5)
                verify_secrets
                ;;
            6)
                backup_secrets
                ;;
            7)
                print_message "$GREEN" "Goodbye!"
                exit 0
                ;;
            *)
                print_message "$RED" "Invalid option"
                ;;
        esac
    done
}

# Check if running in Docker Swarm mode
check_swarm_mode() {
    if ! docker info 2>/dev/null | grep -q "Swarm: active"; then
        print_message "$YELLOW" "Warning: Docker is not in Swarm mode"
        print_message "$YELLOW" "Docker secrets require Swarm mode. Initialize with:"
        echo "  docker swarm init"
        echo
        echo -n "Initialize Swarm mode now? (y/N): "
        read -r init_swarm
        if [[ "$init_swarm" =~ ^[Yy]$ ]]; then
            docker swarm init
            print_message "$GREEN" "✓ Swarm mode initialized"
        else
            print_message "$RED" "Docker secrets require Swarm mode. Exiting."
            exit 1
        fi
    fi
}

# Main execution
main() {
    print_message "$GREEN" "Docker Secrets Management Tool"
    print_message "$GREEN" "=============================="

    # Check prerequisites
    check_docker
    check_swarm_mode

    # Run main menu
    main_menu
}

# Run main function
main
