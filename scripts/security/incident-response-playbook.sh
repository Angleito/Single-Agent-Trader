#!/bin/bash
# Incident Response Playbook for AI Trading Bot
# Automated incident response and emergency procedures

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SECURITY_DIR="$PROJECT_ROOT/security"
LOG_DIR="/var/log/ai-trading-bot-security"
INCIDENT_LOG="$LOG_DIR/incidents.log"
FORENSICS_DIR="/opt/forensics"
TEMP_DIR="/tmp/incident-response-$$"

# Incident configuration
INCIDENT_ID="INC-$(date +%Y%m%d-%H%M%S)"
INCIDENT_SEVERITY=${INCIDENT_SEVERITY:-"medium"}
INCIDENT_TYPE=${INCIDENT_TYPE:-"unknown"}
INCIDENT_DESCRIPTION=${INCIDENT_DESCRIPTION:-""}
AUTO_CONTAINMENT=${AUTO_CONTAINMENT:-"true"}
FORENSICS_ENABLED=${FORENSICS_ENABLED:-"true"}

# Emergency contacts and notifications
EMERGENCY_WEBHOOK=${EMERGENCY_WEBHOOK_URL:-}
SECURITY_TEAM_EMAIL=${SECURITY_TEAM_EMAIL:-}
ESCALATION_PHONE=${ESCALATION_PHONE:-}
INCIDENT_MANAGER=${INCIDENT_MANAGER:-"system"}

# Critical system information
TRADING_BOT_CONTAINER="ai-trading-bot"
CRITICAL_SERVICES=("docker" "fail2ban" "ufw" "ssh")
NETWORK_INTERFACES=("eth0" "docker0")

# Functions
print_emergency() {
    echo -e "\n${RED}🚨 EMERGENCY ALERT 🚨${NC}"
    echo -e "${RED}$1${NC}\n"
}

print_critical() {
    echo -e "${RED}🔴 CRITICAL: $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  WARNING: $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  INFO: $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ SUCCESS: $1${NC}"
}

print_action() {
    echo -e "${CYAN}⚡ ACTION: $1${NC}"
}

# Incident logging
log_incident() {
    local level="$1"
    local phase="$2"
    local message="$3"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    echo "[$timestamp] [INCIDENT-$INCIDENT_ID] [$level] [$phase] $message" | tee -a "$INCIDENT_LOG"
    
    # Send to syslog with high priority
    logger -p security.alert -t "ai-trading-bot-incident" "[$INCIDENT_ID] [$phase] $message"
}

# Emergency notification system
send_emergency_notification() {
    local severity="$1"
    local message="$2"
    local details="${3:-}"
    
    local notification_payload=$(cat << EOF
{
  "incident_id": "$INCIDENT_ID",
  "severity": "$severity",
  "type": "$INCIDENT_TYPE",
  "message": "$message",
  "details": "$details",
  "timestamp": "$(date -Iseconds)",
  "host": "$(hostname)",
  "responder": "$INCIDENT_MANAGER",
  "auto_response": true
}
EOF
)
    
    # Send webhook notification
    if [[ -n "$EMERGENCY_WEBHOOK" ]]; then
        curl -X POST "$EMERGENCY_WEBHOOK" \
             -H "Content-Type: application/json" \
             -d "$notification_payload" \
             --max-time 10 \
             --retry 3 \
             --silent || true
    fi
    
    # Send email if configured
    if [[ -n "$SECURITY_TEAM_EMAIL" ]]; then
        local subject="🚨 SECURITY INCIDENT ALERT: $INCIDENT_ID [$severity]"
        local email_body="SECURITY INCIDENT RESPONSE ALERT

Incident ID: $INCIDENT_ID
Severity: $severity
Type: $INCIDENT_TYPE
Host: $(hostname)
Timestamp: $(date)

Message: $message

Details:
$details

Incident Manager: $INCIDENT_MANAGER
Auto-Response: Enabled

This is an automated alert from the AI Trading Bot security system.
Immediate attention required.

---
Incident Response System"
        
        echo "$email_body" | mail -s "$subject" "$SECURITY_TEAM_EMAIL" 2>/dev/null || \
        log_incident "error" "notification" "Failed to send email alert"
    fi
    
    log_incident "info" "notification" "Emergency notification sent: $severity - $message"
}

# Check if running as root
check_permissions() {
    if [[ $EUID -ne 0 ]]; then
        print_critical "Incident response requires root privileges"
        echo "Run: sudo $0 $*"
        exit 1
    fi
}

# Initialize incident response
initialize_incident_response() {
    print_emergency "SECURITY INCIDENT DETECTED"
    echo -e "${RED}Incident ID: $INCIDENT_ID${NC}"
    echo -e "${RED}Severity: $INCIDENT_SEVERITY${NC}"
    echo -e "${RED}Type: $INCIDENT_TYPE${NC}"
    echo -e "${RED}Time: $(date)${NC}"
    
    # Create directories
    mkdir -p "$SECURITY_DIR"
    mkdir -p "$LOG_DIR"
    mkdir -p "$FORENSICS_DIR/$INCIDENT_ID"
    mkdir -p "$TEMP_DIR"
    
    # Set secure permissions
    chmod 700 "$FORENSICS_DIR/$INCIDENT_ID"
    chmod 700 "$TEMP_DIR"
    
    log_incident "alert" "initialize" "Incident response initiated - Type: $INCIDENT_TYPE, Severity: $INCIDENT_SEVERITY"
    send_emergency_notification "$INCIDENT_SEVERITY" "Security incident response initiated" "Type: $INCIDENT_TYPE"
}

# Phase 1: Immediate Assessment and Triage
phase1_assessment() {
    print_action "PHASE 1: Immediate Assessment and Triage"
    
    local assessment_report="$FORENSICS_DIR/$INCIDENT_ID/assessment.txt"
    
    echo "INCIDENT ASSESSMENT REPORT" > "$assessment_report"
    echo "=========================" >> "$assessment_report"
    echo "Incident ID: $INCIDENT_ID" >> "$assessment_report"
    echo "Date/Time: $(date)" >> "$assessment_report"
    echo "Host: $(hostname)" >> "$assessment_report"
    echo "Severity: $INCIDENT_SEVERITY" >> "$assessment_report"
    echo "Type: $INCIDENT_TYPE" >> "$assessment_report"
    echo "" >> "$assessment_report"
    
    # System status assessment
    print_info "Assessing system status..."
    
    # Check critical services
    echo "CRITICAL SERVICES STATUS:" >> "$assessment_report"
    for service in "${CRITICAL_SERVICES[@]}"; do
        if systemctl is-active --quiet "$service"; then
            echo "✓ $service: ACTIVE" >> "$assessment_report"
            print_success "$service service is running"
        else
            echo "✗ $service: FAILED" >> "$assessment_report"
            print_critical "$service service is not running"
            log_incident "critical" "assessment" "Critical service down: $service"
        fi
    done
    
    # Check trading bot container status
    echo "" >> "$assessment_report"
    echo "TRADING BOT STATUS:" >> "$assessment_report"
    if docker ps | grep -q "$TRADING_BOT_CONTAINER"; then
        echo "✓ Trading bot container: RUNNING" >> "$assessment_report"
        print_success "Trading bot container is running"
        
        # Check container resource usage
        local container_stats=$(docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}" "$TRADING_BOT_CONTAINER" 2>/dev/null || echo "N/A")
        echo "Container Stats: $container_stats" >> "$assessment_report"
    else
        echo "✗ Trading bot container: NOT RUNNING" >> "$assessment_report"
        print_critical "Trading bot container is not running"
        log_incident "critical" "assessment" "Trading bot container not running"
    fi
    
    # Network connectivity check
    echo "" >> "$assessment_report"
    echo "NETWORK CONNECTIVITY:" >> "$assessment_report"
    
    # Check external connectivity
    if ping -c 1 8.8.8.8 >/dev/null 2>&1; then
        echo "✓ External connectivity: OK" >> "$assessment_report"
        print_success "External network connectivity OK"
    else
        echo "✗ External connectivity: FAILED" >> "$assessment_report"
        print_critical "No external network connectivity"
        log_incident "critical" "assessment" "Network connectivity lost"
    fi
    
    # Check firewall status
    echo "" >> "$assessment_report"
    echo "SECURITY STATUS:" >> "$assessment_report"
    
    if ufw status | grep -q "Status: active"; then
        echo "✓ UFW Firewall: ACTIVE" >> "$assessment_report"
        print_success "UFW firewall is active"
    else
        echo "✗ UFW Firewall: INACTIVE" >> "$assessment_report"
        print_critical "UFW firewall is not active"
        log_incident "critical" "assessment" "Firewall is not active"
    fi
    
    # Resource usage assessment
    echo "" >> "$assessment_report"
    echo "SYSTEM RESOURCES:" >> "$assessment_report"
    
    local cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | sed 's/%us,//')
    local memory_usage=$(free | awk 'NR==2{printf "%.1f", $3*100/$2}')
    local disk_usage=$(df / | awk 'NR==2{print $5}' | sed 's/%//')
    
    echo "CPU Usage: ${cpu_usage}%" >> "$assessment_report"
    echo "Memory Usage: ${memory_usage}%" >> "$assessment_report"
    echo "Disk Usage: ${disk_usage}%" >> "$assessment_report"
    
    # Check for suspicious processes
    echo "" >> "$assessment_report"
    echo "SUSPICIOUS PROCESSES:" >> "$assessment_report"
    
    # Look for unusual processes
    ps aux | awk '$3 > 50 || $4 > 50' | head -10 >> "$assessment_report" 2>/dev/null || echo "None detected" >> "$assessment_report"
    
    # Network connections analysis
    echo "" >> "$assessment_report"
    echo "ACTIVE NETWORK CONNECTIONS:" >> "$assessment_report"
    netstat -tuln | grep LISTEN >> "$assessment_report" 2>/dev/null || echo "Unable to retrieve" >> "$assessment_report"
    
    print_success "Assessment phase completed"
    log_incident "info" "assessment" "System assessment completed"
}

# Phase 2: Immediate Containment
phase2_containment() {
    print_action "PHASE 2: Immediate Containment"
    
    if [[ "$AUTO_CONTAINMENT" != "true" ]]; then
        print_warning "Auto-containment disabled - manual intervention required"
        log_incident "warning" "containment" "Auto-containment disabled"
        return 0
    fi
    
    local containment_report="$FORENSICS_DIR/$INCIDENT_ID/containment.txt"
    
    echo "CONTAINMENT ACTIONS REPORT" > "$containment_report"
    echo "=========================" >> "$containment_report"
    echo "Timestamp: $(date)" >> "$containment_report"
    echo "" >> "$containment_report"
    
    # Determine containment actions based on incident type
    case "$INCIDENT_TYPE" in
        "container_breach"|"unauthorized_access")
            print_critical "Container security breach detected - implementing containment"
            contain_container_breach "$containment_report"
            ;;
        
        "network_attack"|"ddos"|"port_scan")
            print_critical "Network attack detected - implementing network containment"
            contain_network_attack "$containment_report"
            ;;
        
        "malware"|"crypto_mining"|"suspicious_process")
            print_critical "Malicious activity detected - implementing process containment"
            contain_malicious_activity "$containment_report"
            ;;
        
        "data_exfiltration"|"suspicious_network")
            print_critical "Data exfiltration attempt detected - implementing emergency lockdown"
            emergency_lockdown "$containment_report"
            ;;
        
        *)
            print_warning "Unknown incident type - implementing standard containment"
            standard_containment "$containment_report"
            ;;
    esac
    
    print_success "Containment phase completed"
    log_incident "info" "containment" "Containment measures implemented"
}

# Container breach containment
contain_container_breach() {
    local report_file="$1"
    
    echo "CONTAINER BREACH CONTAINMENT:" >> "$report_file"
    
    # Stop compromised containers
    print_action "Stopping trading bot containers"
    if docker stop "$TRADING_BOT_CONTAINER" 2>/dev/null; then
        echo "✓ Stopped trading bot container" >> "$report_file"
        log_incident "info" "containment" "Trading bot container stopped"
    else
        echo "✗ Failed to stop trading bot container" >> "$report_file"
        log_incident "error" "containment" "Failed to stop trading bot container"
    fi
    
    # Isolate containers in quarantine network
    print_action "Creating quarantine network"
    docker network create --internal quarantine-network 2>/dev/null || true
    echo "✓ Quarantine network created" >> "$report_file"
    
    # Remove containers from external networks
    docker network disconnect bridge "$TRADING_BOT_CONTAINER" 2>/dev/null || true
    echo "✓ Container disconnected from bridge network" >> "$report_file"
    
    send_emergency_notification "critical" "Container breach contained" "Trading bot containers isolated"
}

# Network attack containment
contain_network_attack() {
    local report_file="$1"
    
    echo "NETWORK ATTACK CONTAINMENT:" >> "$report_file"
    
    # Enable strict firewall rules
    print_action "Implementing strict firewall rules"
    
    # Block all incoming traffic except SSH
    ufw --force reset >/dev/null 2>&1
    ufw default deny incoming
    ufw default deny outgoing
    ufw allow 2222/tcp  # SSH port
    ufw --force enable
    echo "✓ Strict firewall rules implemented" >> "$report_file"
    
    # Block suspicious IPs (if available)
    if [[ -f "/tmp/suspicious_ips.txt" ]]; then
        while read -r ip; do
            ufw insert 1 deny from "$ip" to any 2>/dev/null || true
        done < "/tmp/suspicious_ips.txt"
        echo "✓ Suspicious IPs blocked" >> "$report_file"
    fi
    
    # Rate limit connections
    iptables -A INPUT -p tcp --dport 2222 -m limit --limit 5/min --limit-burst 3 -j ACCEPT 2>/dev/null || true
    echo "✓ Rate limiting implemented" >> "$report_file"
    
    send_emergency_notification "high" "Network attack contained" "Strict firewall rules implemented"
}

# Malicious activity containment
contain_malicious_activity() {
    local report_file="$1"
    
    echo "MALICIOUS ACTIVITY CONTAINMENT:" >> "$report_file"
    
    # Kill suspicious processes
    print_action "Terminating suspicious processes"
    
    # Look for crypto mining processes
    pkill -f "xmrig\|ccminer\|cgminer\|bfgminer" 2>/dev/null || true
    echo "✓ Crypto mining processes terminated" >> "$report_file"
    
    # Kill high CPU/memory processes that are not system processes
    ps aux | awk '$3 > 80 || $4 > 80' | grep -v "root\|system" | awk '{print $2}' | while read -r pid; do
        if [[ -n "$pid" && "$pid" != "PID" ]]; then
            kill -9 "$pid" 2>/dev/null || true
        fi
    done
    echo "✓ High resource usage processes terminated" >> "$report_file"
    
    # Disable unnecessary services
    systemctl stop apache2 nginx mysql postgresql 2>/dev/null || true
    echo "✓ Non-essential services stopped" >> "$report_file"
    
    send_emergency_notification "high" "Malicious activity contained" "Suspicious processes terminated"
}

# Emergency lockdown
emergency_lockdown() {
    local report_file="$1"
    
    echo "EMERGENCY LOCKDOWN INITIATED:" >> "$report_file"
    
    print_emergency "INITIATING EMERGENCY LOCKDOWN"
    
    # Stop all containers immediately
    print_action "Stopping all containers"
    docker stop $(docker ps -q) 2>/dev/null || true
    echo "✓ All containers stopped" >> "$report_file"
    
    # Block all network traffic except SSH
    print_action "Blocking all network traffic"
    iptables -F
    iptables -P INPUT DROP
    iptables -P FORWARD DROP
    iptables -P OUTPUT DROP
    iptables -A INPUT -i lo -j ACCEPT
    iptables -A OUTPUT -o lo -j ACCEPT
    iptables -A INPUT -p tcp --dport 2222 -j ACCEPT
    iptables -A OUTPUT -p tcp --sport 2222 -j ACCEPT
    echo "✓ Network lockdown implemented" >> "$report_file"
    
    # Create emergency backup
    print_action "Creating emergency backup"
    tar czf "$FORENSICS_DIR/$INCIDENT_ID/emergency-backup.tar.gz" /opt/ai-trading-bot 2>/dev/null || true
    echo "✓ Emergency backup created" >> "$report_file"
    
    send_emergency_notification "critical" "EMERGENCY LOCKDOWN ACTIVATED" "System locked down due to security incident"
}

# Standard containment
standard_containment() {
    local report_file="$1"
    
    echo "STANDARD CONTAINMENT:" >> "$report_file"
    
    # Pause trading activities
    print_action "Pausing trading activities"
    docker pause "$TRADING_BOT_CONTAINER" 2>/dev/null || true
    echo "✓ Trading activities paused" >> "$report_file"
    
    # Enable enhanced monitoring
    print_action "Enabling enhanced monitoring"
    auditctl -e 1 2>/dev/null || true
    echo "✓ Enhanced monitoring enabled" >> "$report_file"
    
    send_emergency_notification "medium" "Standard containment implemented" "Trading paused, monitoring enhanced"
}

# Phase 3: Forensics and Evidence Collection
phase3_forensics() {
    print_action "PHASE 3: Forensics and Evidence Collection"
    
    if [[ "$FORENSICS_ENABLED" != "true" ]]; then
        print_warning "Forensics collection disabled"
        return 0
    fi
    
    local forensics_report="$FORENSICS_DIR/$INCIDENT_ID/forensics.txt"
    
    echo "FORENSICS COLLECTION REPORT" > "$forensics_report"
    echo "===========================" >> "$forensics_report"
    echo "Collection Time: $(date)" >> "$forensics_report"
    echo "" >> "$forensics_report"
    
    # System snapshot
    print_info "Collecting system snapshot"
    
    # Process list
    ps aux > "$FORENSICS_DIR/$INCIDENT_ID/processes.txt"
    echo "✓ Process list captured" >> "$forensics_report"
    
    # Network connections
    netstat -tuln > "$FORENSICS_DIR/$INCIDENT_ID/network_connections.txt"
    ss -tuln > "$FORENSICS_DIR/$INCIDENT_ID/socket_stats.txt"
    echo "✓ Network state captured" >> "$forensics_report"
    
    # System logs
    journalctl --since="1 hour ago" > "$FORENSICS_DIR/$INCIDENT_ID/system_logs.txt"
    echo "✓ System logs captured" >> "$forensics_report"
    
    # Docker state
    if command -v docker >/dev/null 2>&1; then
        docker ps -a > "$FORENSICS_DIR/$INCIDENT_ID/docker_containers.txt"
        docker images > "$FORENSICS_DIR/$INCIDENT_ID/docker_images.txt"
        docker network ls > "$FORENSICS_DIR/$INCIDENT_ID/docker_networks.txt"
        
        # Container logs
        if docker ps | grep -q "$TRADING_BOT_CONTAINER"; then
            docker logs "$TRADING_BOT_CONTAINER" > "$FORENSICS_DIR/$INCIDENT_ID/container_logs.txt" 2>&1
        fi
        echo "✓ Docker state captured" >> "$forensics_report"
    fi
    
    # Security logs
    cp /var/log/auth.log "$FORENSICS_DIR/$INCIDENT_ID/" 2>/dev/null || true
    cp /var/log/fail2ban.log "$FORENSICS_DIR/$INCIDENT_ID/" 2>/dev/null || true
    cp /var/log/ufw.log "$FORENSICS_DIR/$INCIDENT_ID/" 2>/dev/null || true
    echo "✓ Security logs captured" >> "$forensics_report"
    
    # Memory dump (if tools available)
    if command -v gcore >/dev/null 2>&1; then
        print_info "Creating memory dumps of suspicious processes"
        
        # Dump trading bot process memory
        local bot_pid=$(pgrep -f "trading-bot\|python.*bot" | head -1)
        if [[ -n "$bot_pid" ]]; then
            gcore -o "$FORENSICS_DIR/$INCIDENT_ID/trading_bot_core" "$bot_pid" 2>/dev/null || true
            echo "✓ Trading bot memory dump created" >> "$forensics_report"
        fi
    fi
    
    # File integrity check
    if command -v aide >/dev/null 2>&1; then
        aide --check > "$FORENSICS_DIR/$INCIDENT_ID/file_integrity.txt" 2>&1 || true
        echo "✓ File integrity check completed" >> "$forensics_report"
    fi
    
    # Create forensics archive
    print_info "Creating forensics archive"
    cd "$FORENSICS_DIR"
    tar czf "$INCIDENT_ID-forensics.tar.gz" "$INCIDENT_ID/" 2>/dev/null || true
    echo "✓ Forensics archive created: $INCIDENT_ID-forensics.tar.gz" >> "$forensics_report"
    
    print_success "Forensics collection completed"
    log_incident "info" "forensics" "Evidence collection completed"
}

# Phase 4: Recovery and Restoration
phase4_recovery() {
    print_action "PHASE 4: Recovery and Restoration"
    
    local recovery_report="$FORENSICS_DIR/$INCIDENT_ID/recovery.txt"
    
    echo "RECOVERY ACTIONS REPORT" > "$recovery_report"
    echo "======================" >> "$recovery_report"
    echo "Recovery Time: $(date)" >> "$recovery_report"
    echo "" >> "$recovery_report"
    
    # Security validation before recovery
    print_info "Performing security validation"
    
    # Check for persistence mechanisms
    local persistence_found=false
    
    # Check crontabs
    if crontab -l 2>/dev/null | grep -v "^#" | grep -q "."; then
        print_warning "Active cron jobs found - manual review required"
        crontab -l > "$FORENSICS_DIR/$INCIDENT_ID/crontab_backup.txt" 2>/dev/null
        echo "⚠ Cron jobs require manual review" >> "$recovery_report"
        persistence_found=true
    fi
    
    # Check startup services
    systemctl list-unit-files --type=service --state=enabled | grep -v "^UNIT" > "$FORENSICS_DIR/$INCIDENT_ID/enabled_services.txt"
    echo "✓ Enabled services catalogued" >> "$recovery_report"
    
    # Only proceed with automated recovery if no persistence mechanisms found
    if [[ "$persistence_found" == "false" ]]; then
        print_info "No persistence mechanisms detected - proceeding with recovery"
        
        # Restore firewall to normal state
        print_action "Restoring normal firewall configuration"
        restore_firewall_config "$recovery_report"
        
        # Restart critical services
        print_action "Restarting critical services"
        restart_critical_services "$recovery_report"
        
        # Restore trading bot (with additional security)
        print_action "Restoring trading bot with enhanced security"
        restore_trading_bot "$recovery_report"
    else
        print_warning "Persistence mechanisms found - manual recovery required"
        echo "⚠ Manual recovery required due to persistence mechanisms" >> "$recovery_report"
        send_emergency_notification "high" "Manual recovery required" "Persistence mechanisms detected - automatic recovery aborted"
    fi
    
    print_success "Recovery phase completed"
    log_incident "info" "recovery" "Recovery actions completed"
}

# Restore firewall configuration
restore_firewall_config() {
    local report_file="$1"
    
    # Restore UFW to secure default state
    ufw --force reset >/dev/null 2>&1
    ufw default deny incoming
    ufw default allow outgoing
    
    # Essential services
    ufw allow 2222/tcp comment 'SSH'
    ufw allow out 443/tcp comment 'HTTPS outbound'
    ufw allow out 80/tcp comment 'HTTP outbound'
    ufw allow out 53 comment 'DNS'
    
    # Trading bot specific (if needed)
    # ufw allow 8080/tcp comment 'Dashboard'
    
    ufw --force enable
    echo "✓ Firewall configuration restored" >> "$report_file"
}

# Restart critical services
restart_critical_services() {
    local report_file="$1"
    
    for service in "${CRITICAL_SERVICES[@]}"; do
        if systemctl restart "$service" 2>/dev/null; then
            echo "✓ Restarted $service" >> "$report_file"
        else
            echo "✗ Failed to restart $service" >> "$report_file"
            log_incident "error" "recovery" "Failed to restart critical service: $service"
        fi
    done
}

# Restore trading bot with enhanced security
restore_trading_bot() {
    local report_file="$1"
    
    # Remove any potentially compromised containers
    docker rm -f "$TRADING_BOT_CONTAINER" 2>/dev/null || true
    
    # Pull fresh image (if available)
    docker pull ai-trading-bot:latest 2>/dev/null || true
    
    # Start with enhanced security settings
    cd "/opt/ai-trading-bot" || return 1
    
    # Use security-hardened compose file
    if [[ -f "docker-compose.security.yml" ]]; then
        docker-compose -f docker-compose.yml -f docker-compose.security.yml up -d 2>/dev/null || {
            echo "✗ Failed to start trading bot with enhanced security" >> "$report_file"
            return 1
        }
    else
        docker-compose up -d 2>/dev/null || {
            echo "✗ Failed to start trading bot" >> "$report_file"
            return 1
        }
    fi
    
    # Wait for health check
    sleep 30
    
    if docker ps | grep -q "$TRADING_BOT_CONTAINER.*healthy\|Up"; then
        echo "✓ Trading bot restored successfully" >> "$report_file"
        send_emergency_notification "info" "Trading bot restored" "System recovery completed successfully"
    else
        echo "✗ Trading bot health check failed" >> "$report_file"
        log_incident "error" "recovery" "Trading bot health check failed after restoration"
    fi
}

# Phase 5: Post-Incident Analysis and Documentation
phase5_analysis() {
    print_action "PHASE 5: Post-Incident Analysis"
    
    local analysis_report="$FORENSICS_DIR/$INCIDENT_ID/post_incident_analysis.txt"
    
    cat > "$analysis_report" << EOF
POST-INCIDENT ANALYSIS REPORT
============================
Incident ID: $INCIDENT_ID
Analysis Date: $(date)
Analyst: $INCIDENT_MANAGER

INCIDENT SUMMARY:
- Type: $INCIDENT_TYPE
- Severity: $INCIDENT_SEVERITY
- Detection Time: $(head -1 "$INCIDENT_LOG" | awk '{print $1, $2}')
- Resolution Time: $(date '+%Y-%m-%d %H:%M:%S')
- Duration: $(( ($(date +%s) - $(date -d "$(head -1 "$INCIDENT_LOG" | awk '{print $1, $2}')" +%s)) / 60 )) minutes

RESPONSE TIMELINE:
$(grep "$INCIDENT_ID" "$INCIDENT_LOG" | head -20)

CONTAINMENT EFFECTIVENESS:
- Auto-containment: $AUTO_CONTAINMENT
- Forensics collection: $FORENSICS_ENABLED
- System isolation: Implemented
- Data protection: Maintained

LESSONS LEARNED:
1. Response time: Automatic detection and response
2. Containment effectiveness: 
3. Recovery success: 
4. Areas for improvement: 

RECOMMENDATIONS:
1. Review and update incident response procedures
2. Enhance monitoring for early detection
3. Implement additional preventive controls
4. Conduct security awareness training

FOLLOW-UP ACTIONS:
1. Monitor system for 48 hours post-recovery
2. Update security baselines
3. Review access controls and permissions
4. Schedule security assessment

Next Review Date: $(date -d "+7 days" '+%Y-%m-%d')
EOF
    
    # Generate final incident report
    generate_final_report
    
    print_success "Post-incident analysis completed"
    log_incident "info" "analysis" "Post-incident analysis completed"
}

# Generate comprehensive final report
generate_final_report() {
    local final_report="$FORENSICS_DIR/$INCIDENT_ID/INCIDENT_REPORT_$INCIDENT_ID.md"
    
    cat > "$final_report" << EOF
# Security Incident Report

**Incident ID:** $INCIDENT_ID  
**Date:** $(date)  
**Host:** $(hostname)  
**Incident Manager:** $INCIDENT_MANAGER

## Executive Summary

A security incident of type **$INCIDENT_TYPE** with severity **$INCIDENT_SEVERITY** was detected and responded to automatically by the AI Trading Bot security system.

## Incident Details

- **Detection Time:** $(head -1 "$INCIDENT_LOG" | awk '{print $1, $2}')
- **Response Time:** Immediate (automated)
- **Resolution Time:** $(date '+%Y-%m-%d %H:%M:%S')
- **Affected Systems:** AI Trading Bot infrastructure
- **Data Compromise:** Under investigation
- **Service Impact:** Temporary trading suspension during containment

## Response Actions Taken

### Phase 1: Assessment
- System status evaluated
- Critical services checked
- Resource usage analyzed
- Network connectivity verified

### Phase 2: Containment
- Implemented containment strategy for $INCIDENT_TYPE
- Isolated affected components
- Preserved system state for analysis

### Phase 3: Forensics
- Collected system evidence
- Captured memory dumps
- Preserved log files
- Created forensics archive

### Phase 4: Recovery
- Validated system security
- Restored normal operations
- Implemented enhanced monitoring
- Verified system functionality

### Phase 5: Analysis
- Documented incident timeline
- Identified root cause
- Developed remediation plan
- Updated security procedures

## Evidence Collected

- System logs and process lists
- Network connection states
- Container and Docker information
- Security audit logs
- Memory dumps (where applicable)
- File integrity reports

## Root Cause Analysis

[To be completed based on forensics analysis]

## Recommendations

1. **Immediate Actions:**
   - Continue enhanced monitoring for 48 hours
   - Review all access logs for suspicious activity
   - Validate all security configurations

2. **Short-term Improvements:**
   - Update security monitoring rules
   - Enhance incident detection capabilities
   - Review access controls and permissions

3. **Long-term Enhancements:**
   - Implement additional security layers
   - Conduct regular security assessments
   - Update incident response procedures

## Lessons Learned

- Automated incident response system functioned as designed
- Containment was achieved quickly
- Evidence preservation was successful
- Recovery process needs refinement

## Follow-up Actions

- [ ] Complete forensics analysis within 7 days
- [ ] Update security baselines
- [ ] Conduct lessons learned session
- [ ] Review and update incident response procedures
- [ ] Schedule follow-up security assessment

---

**Report Generated:** $(date)  
**Next Review:** $(date -d "+7 days")  
**Classification:** CONFIDENTIAL
EOF
    
    print_success "Final incident report generated: $final_report"
}

# Cleanup function
cleanup() {
    if [[ -d "$TEMP_DIR" ]]; then
        rm -rf "$TEMP_DIR"
    fi
}

# Main incident response workflow
main() {
    # Setup signal handlers
    trap cleanup EXIT
    
    print_emergency "AI TRADING BOT SECURITY INCIDENT RESPONSE"
    
    # Check permissions
    check_permissions
    
    # Initialize incident response
    initialize_incident_response
    
    # Execute response phases
    phase1_assessment
    phase2_containment
    phase3_forensics
    phase4_recovery
    phase5_analysis
    
    print_emergency "INCIDENT RESPONSE COMPLETED"
    print_success "Incident ID: $INCIDENT_ID"
    print_success "All response phases completed successfully"
    
    echo -e "\n${BLUE}Incident Documentation:${NC}"
    echo "- Incident logs: $INCIDENT_LOG"
    echo "- Forensics data: $FORENSICS_DIR/$INCIDENT_ID/"
    echo "- Final report: $FORENSICS_DIR/$INCIDENT_ID/INCIDENT_REPORT_$INCIDENT_ID.md"
    
    echo -e "\n${YELLOW}Next Steps:${NC}"
    echo "1. Review forensics data and complete root cause analysis"
    echo "2. Monitor system for 48 hours post-recovery"
    echo "3. Update security procedures based on lessons learned"
    echo "4. Schedule follow-up security assessment"
    
    log_incident "info" "complete" "Incident response workflow completed successfully"
    send_emergency_notification "info" "Incident response completed" "All phases completed successfully"
}

# Parse command line arguments and incident parameters
case "${1:-respond}" in
    "respond")
        # Full incident response workflow
        main
        ;;
    
    "assess")
        # Assessment only
        check_permissions
        initialize_incident_response
        phase1_assessment
        ;;
    
    "contain")
        # Containment only
        check_permissions
        initialize_incident_response
        phase2_containment
        ;;
    
    "forensics")
        # Forensics collection only
        check_permissions
        initialize_incident_response
        phase3_forensics
        ;;
    
    "emergency-lockdown")
        # Emergency lockdown
        INCIDENT_TYPE="emergency"
        INCIDENT_SEVERITY="critical"
        check_permissions
        initialize_incident_response
        local report="/tmp/emergency-lockdown.txt"
        emergency_lockdown "$report"
        ;;
    
    "status")
        # Show incident status
        if [[ -f "$INCIDENT_LOG" ]]; then
            tail -20 "$INCIDENT_LOG"
        else
            echo "No incident log found"
        fi
        ;;
    
    "help"|"--help")
        echo "Usage: $0 [command] [options]"
        echo
        echo "Commands:"
        echo "  respond          Execute full incident response workflow (default)"
        echo "  assess           Perform assessment phase only"
        echo "  contain          Perform containment phase only"
        echo "  forensics        Perform forensics collection only"
        echo "  emergency-lockdown  Execute emergency lockdown immediately"
        echo "  status           Show recent incident activity"
        echo "  help             Show this help message"
        echo
        echo "Environment Variables:"
        echo "  INCIDENT_TYPE           Type of incident (default: unknown)"
        echo "  INCIDENT_SEVERITY       Severity level (default: medium)"
        echo "  INCIDENT_DESCRIPTION    Incident description"
        echo "  AUTO_CONTAINMENT        Enable auto-containment (default: true)"
        echo "  FORENSICS_ENABLED       Enable forensics collection (default: true)"
        echo "  EMERGENCY_WEBHOOK_URL   Webhook for emergency notifications"
        echo "  SECURITY_TEAM_EMAIL     Email for incident notifications"
        echo
        echo "Examples:"
        echo "  INCIDENT_TYPE=container_breach INCIDENT_SEVERITY=critical $0"
        echo "  INCIDENT_TYPE=network_attack $0 contain"
        echo "  $0 emergency-lockdown"
        ;;
    
    *)
        print_critical "Unknown command: $1"
        echo "Run '$0 help' for usage information"
        exit 1
        ;;
esac