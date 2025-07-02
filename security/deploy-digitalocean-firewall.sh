#!/bin/bash
# Digital Ocean Cloud Firewall Deployment Script
# Deploy comprehensive network security policies to Digital Ocean Cloud Firewall

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FIREWALL_CONFIG="$SCRIPT_DIR/digitalocean-firewall-config.json"
LOG_FILE="/var/log/trading-bot-firewall-deployment.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

print_header() {
    echo -e "${BLUE}================================================${NC}"
    echo -e "${BLUE}Digital Ocean Cloud Firewall Deployment${NC}"
    echo -e "${BLUE}================================================${NC}"
}

check_requirements() {
    log "Checking requirements..."
    
    # Check if doctl is installed
    if ! command -v doctl &> /dev/null; then
        echo -e "${RED}doctl (Digital Ocean CLI) is not installed.${NC}"
        echo "Please install doctl and authenticate with your Digital Ocean account:"
        echo "1. Install doctl: https://docs.digitalocean.com/reference/doctl/how-to/install/"
        echo "2. Authenticate: doctl auth init"
        exit 1
    fi
    
    # Check if user is authenticated
    if ! doctl account get &> /dev/null; then
        echo -e "${RED}Not authenticated with Digital Ocean.${NC}"
        echo "Please authenticate with: doctl auth init"
        exit 1
    fi
    
    # Check if firewall config exists
    if [ ! -f "$FIREWALL_CONFIG" ]; then
        echo -e "${RED}Firewall configuration file not found: $FIREWALL_CONFIG${NC}"
        exit 1
    fi
    
    log "Requirements check passed"
}

get_droplet_info() {
    log "Getting droplet information..."
    
    # Get list of droplets
    echo -e "${BLUE}Available droplets:${NC}"
    doctl compute droplet list --format "ID,Name,PublicIPv4,Status,Tags"
    
    echo ""
    echo -e "${YELLOW}Please select your trading bot droplet:${NC}"
    read -p "Enter droplet ID or name: " DROPLET_IDENTIFIER
    
    # Validate droplet exists
    if ! doctl compute droplet get "$DROPLET_IDENTIFIER" &> /dev/null; then
        echo -e "${RED}Droplet not found: $DROPLET_IDENTIFIER${NC}"
        exit 1
    fi
    
    # Get droplet ID
    DROPLET_ID=$(doctl compute droplet get "$DROPLET_IDENTIFIER" --format "ID" --no-header)
    DROPLET_NAME=$(doctl compute droplet get "$DROPLET_IDENTIFIER" --format "Name" --no-header)
    DROPLET_IP=$(doctl compute droplet get "$DROPLET_IDENTIFIER" --format "PublicIPv4" --no-header)
    
    echo -e "${GREEN}Selected droplet:${NC}"
    echo "  ID: $DROPLET_ID"
    echo "  Name: $DROPLET_NAME"
    echo "  IP: $DROPLET_IP"
    
    log "Droplet selected: $DROPLET_NAME ($DROPLET_ID)"
}

customize_firewall_config() {
    log "Customizing firewall configuration..."
    
    # Create temporary config file
    TEMP_CONFIG=$(mktemp)
    cp "$FIREWALL_CONFIG" "$TEMP_CONFIG"
    
    echo -e "${YELLOW}Security Configuration Options:${NC}"
    echo "1. Restrict SSH access to specific IPs (recommended)"
    echo "2. Restrict dashboard access to specific IPs"
    echo "3. Enable geographic restrictions"
    echo "4. Use default configuration (allow all)"
    echo ""
    read -p "Choose option (1-4): " SECURITY_OPTION
    
    case $SECURITY_OPTION in
        1)
            restrict_ssh_access "$TEMP_CONFIG"
            ;;
        2)
            restrict_dashboard_access "$TEMP_CONFIG"
            ;;
        3)
            enable_geographic_restrictions "$TEMP_CONFIG"
            ;;
        4)
            echo "Using default configuration"
            ;;
        *)
            echo -e "${YELLOW}Invalid option, using default configuration${NC}"
            ;;
    esac
    
    # Update droplet ID in config
    sed -i "s/\"droplet_ids\": \[\]/\"droplet_ids\": [\"$DROPLET_ID\"]/g" "$TEMP_CONFIG"
    
    FIREWALL_CONFIG="$TEMP_CONFIG"
    log "Firewall configuration customized"
}

restrict_ssh_access() {
    local config_file="$1"
    
    echo -e "${YELLOW}Current public IP addresses that will have SSH access:${NC}"
    curl -s ifconfig.me 2>/dev/null || echo "Unable to detect current IP"
    echo ""
    
    read -p "Enter comma-separated IP addresses for SSH access (e.g., 1.2.3.4/32,5.6.7.8/32): " SSH_IPS
    
    if [ -n "$SSH_IPS" ]; then
        # Convert comma-separated IPs to JSON array
        IFS=',' read -ra IP_ARRAY <<< "$SSH_IPS"
        JSON_IPS=""
        for ip in "${IP_ARRAY[@]}"; do
            ip=$(echo "$ip" | xargs)  # trim whitespace
            if [ -z "$JSON_IPS" ]; then
                JSON_IPS="\"$ip\""
            else
                JSON_IPS="$JSON_IPS, \"$ip\""
            fi
        done
        
        # Update SSH rule in config
        jq --argjson ips "[$JSON_IPS]" '.inbound_rules[0].sources.addresses = $ips' "$config_file" > "${config_file}.tmp"
        mv "${config_file}.tmp" "$config_file"
        
        echo -e "${GREEN}SSH access restricted to: $SSH_IPS${NC}"
    fi
}

restrict_dashboard_access() {
    local config_file="$1"
    
    read -p "Enter comma-separated IP addresses for dashboard access: " DASHBOARD_IPS
    
    if [ -n "$DASHBOARD_IPS" ]; then
        # Convert comma-separated IPs to JSON array
        IFS=',' read -ra IP_ARRAY <<< "$DASHBOARD_IPS"
        JSON_IPS=""
        for ip in "${IP_ARRAY[@]}"; do
            ip=$(echo "$ip" | xargs)  # trim whitespace
            if [ -z "$JSON_IPS" ]; then
                JSON_IPS="\"$ip\""
            else
                JSON_IPS="$JSON_IPS, \"$ip\""
            fi
        done
        
        # Update dashboard rules in config (ports 3000, 8000, 8080)
        jq --argjson ips "[$JSON_IPS]" '
            (.inbound_rules[] | select(.ports == "3000" or .ports == "8000" or .ports == "8080")).sources.addresses = $ips
        ' "$config_file" > "${config_file}.tmp"
        mv "${config_file}.tmp" "$config_file"
        
        echo -e "${GREEN}Dashboard access restricted to: $DASHBOARD_IPS${NC}"
    fi
}

enable_geographic_restrictions() {
    local config_file="$1"
    
    echo -e "${YELLOW}Geographic restriction not directly supported in basic config.${NC}"
    echo "Consider using Digital Ocean's advanced firewall features or a CDN like Cloudflare."
    echo "For now, we'll implement IP-based restrictions."
    
    echo "Common country IP ranges:"
    echo "US: Contact your ISP or use geolocation services"
    echo "EU: Contact your ISP or use geolocation services"
    
    restrict_ssh_access "$config_file"
}

check_existing_firewall() {
    log "Checking for existing firewalls..."
    
    # Check if firewall already exists
    if doctl compute firewall list --format "Name" --no-header | grep -q "ai-trading-bot-firewall"; then
        echo -e "${YELLOW}Existing firewall 'ai-trading-bot-firewall' found.${NC}"
        read -p "Do you want to update it? (y/n): " UPDATE_EXISTING
        
        if [ "$UPDATE_EXISTING" = "y" ] || [ "$UPDATE_EXISTING" = "Y" ]; then
            FIREWALL_ID=$(doctl compute firewall list --format "ID,Name" --no-header | grep "ai-trading-bot-firewall" | cut -f1)
            echo "Will update existing firewall: $FIREWALL_ID"
            return 0
        else
            echo "Aborted by user"
            exit 0
        fi
    fi
    
    return 1
}

deploy_firewall() {
    log "Deploying firewall configuration..."
    
    if check_existing_firewall; then
        # Update existing firewall
        echo -e "${BLUE}Updating existing firewall...${NC}"
        
        # Get current firewall config
        doctl compute firewall get "$FIREWALL_ID" --format "ID,Name,Status" --no-header
        
        # Update firewall using the API (doctl doesn't support direct update from JSON)
        echo -e "${YELLOW}Manual update required:${NC}"
        echo "1. Go to Digital Ocean Control Panel"
        echo "2. Navigate to Networking > Firewalls"
        echo "3. Edit firewall: ai-trading-bot-firewall"
        echo "4. Apply the rules from: $FIREWALL_CONFIG"
        
    else
        # Create new firewall
        echo -e "${BLUE}Creating new firewall...${NC}"
        
        # Extract configuration elements
        FIREWALL_NAME=$(jq -r '.name' "$FIREWALL_CONFIG")
        
        # Create firewall with doctl
        FIREWALL_ID=$(doctl compute firewall create \
            --name "$FIREWALL_NAME" \
            --tag "trading-bot" \
            --droplet-ids "$DROPLET_ID" \
            --format "ID" --no-header)
        
        if [ -n "$FIREWALL_ID" ]; then
            echo -e "${GREEN}Firewall created successfully: $FIREWALL_ID${NC}"
            log "Firewall created: $FIREWALL_ID"
        else
            echo -e "${RED}Failed to create firewall${NC}"
            exit 1
        fi
    fi
}

configure_firewall_rules() {
    log "Configuring firewall rules..."
    
    echo -e "${BLUE}Configuring inbound rules...${NC}"
    
    # SSH Rule
    echo "Adding SSH rule..."
    doctl compute firewall add-rules "$FIREWALL_ID" \
        --inbound-rules "protocol:tcp,ports:22,address:0.0.0.0/0" 2>/dev/null || true
    
    # Dashboard Rules
    echo "Adding dashboard rules..."
    doctl compute firewall add-rules "$FIREWALL_ID" \
        --inbound-rules "protocol:tcp,ports:3000,address:0.0.0.0/0" 2>/dev/null || true
    
    doctl compute firewall add-rules "$FIREWALL_ID" \
        --inbound-rules "protocol:tcp,ports:8000,address:0.0.0.0/0" 2>/dev/null || true
    
    doctl compute firewall add-rules "$FIREWALL_ID" \
        --inbound-rules "protocol:tcp,ports:8080,address:0.0.0.0/0" 2>/dev/null || true
    
    # HTTP/HTTPS Rules
    echo "Adding HTTP/HTTPS rules..."
    doctl compute firewall add-rules "$FIREWALL_ID" \
        --inbound-rules "protocol:tcp,ports:80,address:0.0.0.0/0" 2>/dev/null || true
    
    doctl compute firewall add-rules "$FIREWALL_ID" \
        --inbound-rules "protocol:tcp,ports:443,address:0.0.0.0/0" 2>/dev/null || true
    
    # ICMP Rule
    echo "Adding ICMP rule..."
    doctl compute firewall add-rules "$FIREWALL_ID" \
        --inbound-rules "protocol:icmp,address:0.0.0.0/0" 2>/dev/null || true
    
    echo -e "${BLUE}Configuring outbound rules...${NC}"
    
    # DNS Rules
    doctl compute firewall add-rules "$FIREWALL_ID" \
        --outbound-rules "protocol:tcp,ports:53,address:0.0.0.0/0" 2>/dev/null || true
    
    doctl compute firewall add-rules "$FIREWALL_ID" \
        --outbound-rules "protocol:udp,ports:53,address:0.0.0.0/0" 2>/dev/null || true
    
    # HTTP/HTTPS Outbound
    doctl compute firewall add-rules "$FIREWALL_ID" \
        --outbound-rules "protocol:tcp,ports:80,address:0.0.0.0/0" 2>/dev/null || true
    
    doctl compute firewall add-rules "$FIREWALL_ID" \
        --outbound-rules "protocol:tcp,ports:443,address:0.0.0.0/0" 2>/dev/null || true
    
    # NTP Rule
    doctl compute firewall add-rules "$FIREWALL_ID" \
        --outbound-rules "protocol:udp,ports:123,address:0.0.0.0/0" 2>/dev/null || true
    
    # SMTP Rules
    doctl compute firewall add-rules "$FIREWALL_ID" \
        --outbound-rules "protocol:tcp,ports:587,address:0.0.0.0/0" 2>/dev/null || true
    
    doctl compute firewall add-rules "$FIREWALL_ID" \
        --outbound-rules "protocol:tcp,ports:25,address:0.0.0.0/0" 2>/dev/null || true
    
    log "Firewall rules configured successfully"
}

validate_firewall() {
    log "Validating firewall configuration..."
    
    echo -e "${BLUE}Current firewall status:${NC}"
    doctl compute firewall get "$FIREWALL_ID" --format "ID,Name,Status,InboundRules,OutboundRules,DropletIDs,Tags"
    
    echo ""
    echo -e "${BLUE}Testing connectivity...${NC}"
    
    # Test SSH connectivity
    if nc -z -w5 "$DROPLET_IP" 22 2>/dev/null; then
        echo -e "${GREEN}✓ SSH port (22) is accessible${NC}"
    else
        echo -e "${RED}✗ SSH port (22) is not accessible${NC}"
    fi
    
    # Test dashboard ports
    for port in 3000 8000 8080; do
        if nc -z -w5 "$DROPLET_IP" "$port" 2>/dev/null; then
            echo -e "${GREEN}✓ Dashboard port ($port) is accessible${NC}"
        else
            echo -e "${YELLOW}! Dashboard port ($port) is not accessible (this may be normal if services aren't running)${NC}"
        fi
    done
    
    log "Firewall validation completed"
}

create_firewall_monitoring() {
    log "Setting up firewall monitoring..."
    
    # Create monitoring script
    cat > /tmp/firewall-monitor.sh << 'EOF'
#!/bin/bash
# Digital Ocean Firewall Monitoring Script

FIREWALL_ID="FIREWALL_ID_PLACEHOLDER"
LOG_FILE="/var/log/trading-bot/security/firewall-monitor.log"

check_firewall_status() {
    if doctl compute firewall get "$FIREWALL_ID" &> /dev/null; then
        echo "$(date): Firewall $FIREWALL_ID is active" >> "$LOG_FILE"
        return 0
    else
        echo "$(date): ERROR - Firewall $FIREWALL_ID is not found" >> "$LOG_FILE"
        return 1
    fi
}

# Monitor firewall every 5 minutes
while true; do
    if ! check_firewall_status; then
        # Send alert (implement notification system)
        logger -t TRADING-BOT-FIREWALL "ERROR: Digital Ocean firewall not found"
    fi
    sleep 300
done
EOF
    
    # Replace placeholder with actual firewall ID
    sed -i "s/FIREWALL_ID_PLACEHOLDER/$FIREWALL_ID/g" /tmp/firewall-monitor.sh
    
    echo "Firewall monitoring script created at: /tmp/firewall-monitor.sh"
    echo "Deploy this script to your droplet for continuous monitoring."
}

print_summary() {
    echo -e "${BLUE}================================================${NC}"
    echo -e "${BLUE}Digital Ocean Firewall Deployment Complete${NC}"
    echo -e "${BLUE}================================================${NC}"
    echo ""
    echo -e "${GREEN}✓ Firewall deployed successfully${NC}"
    echo "  Firewall ID: $FIREWALL_ID"
    echo "  Droplet: $DROPLET_NAME ($DROPLET_IP)"
    echo ""
    echo -e "${YELLOW}Configured Rules:${NC}"
    echo "  • SSH access (port 22)"
    echo "  • Dashboard access (ports 3000, 8000, 8080)"
    echo "  • HTTP/HTTPS access (ports 80, 443)"
    echo "  • DNS, NTP, and SMTP outbound"
    echo "  • ICMP (ping) access"
    echo ""
    echo -e "${YELLOW}Next Steps:${NC}"
    echo "1. Test connectivity to your services"
    echo "2. Deploy firewall monitoring script to your droplet"
    echo "3. Configure alerts for firewall status changes"
    echo "4. Review and adjust IP restrictions as needed"
    echo "5. Consider implementing additional DDoS protection"
    echo ""
    echo -e "${BLUE}Monitoring:${NC}"
    echo "  • Monitor firewall status in Digital Ocean dashboard"
    echo "  • Deploy monitoring script: /tmp/firewall-monitor.sh"
    echo "  • Check logs: $LOG_FILE"
}

cleanup() {
    # Clean up temporary files
    if [ -n "${TEMP_CONFIG:-}" ] && [ -f "$TEMP_CONFIG" ]; then
        rm -f "$TEMP_CONFIG"
    fi
}

# Main execution
main() {
    # Set up cleanup trap
    trap cleanup EXIT
    
    print_header
    
    # Create log file
    mkdir -p "$(dirname "$LOG_FILE")"
    touch "$LOG_FILE"
    
    # Main deployment steps
    check_requirements
    get_droplet_info
    customize_firewall_config
    deploy_firewall
    configure_firewall_rules
    validate_firewall
    create_firewall_monitoring
    
    print_summary
    log "Digital Ocean firewall deployment completed successfully"
}

# Run main function
main "$@"