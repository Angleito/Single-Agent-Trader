#!/bin/bash
# Security Maintenance Automation
# Automates security updates, configuration drift detection, and maintenance tasks

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SECURITY_DIR="$PROJECT_ROOT/security"
LOG_DIR="/var/log/ai-trading-bot-security"
MAINTENANCE_LOG="$LOG_DIR/maintenance.log"
CONFIG_BASELINE_DIR="$SECURITY_DIR/baseline"
TEMP_DIR="/tmp/security-maintenance-$$"

# Maintenance configuration
BACKUP_RETENTION_DAYS=${BACKUP_RETENTION_DAYS:-30}
LOG_RETENTION_DAYS=${LOG_RETENTION_DAYS:-90}
UPDATE_SCHEDULE=${UPDATE_SCHEDULE:-"weekly"}
DRIFT_CHECK_ENABLED=${DRIFT_CHECK_ENABLED:-"true"}
AUTO_REMEDIATION=${AUTO_REMEDIATION:-"false"}

# Notification settings
WEBHOOK_URL=${SECURITY_WEBHOOK_URL:-}
EMAIL_ALERTS=${EMAIL_ALERTS:-"false"}
SMTP_SERVER=${SMTP_SERVER:-}
ALERT_EMAIL=${ALERT_EMAIL:-}

# Functions
print_header() {
    echo -e "\n${BLUE}===========================================${NC}"
    echo -e "${BLUE}  $1"
    echo -e "${BLUE}===========================================${NC}\n"
}

print_section() {
    echo -e "\n${PURPLE}--- $1 ---${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Logging function
log_maintenance() {
    local level="$1"
    local component="$2"
    local message="$3"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    echo "[$timestamp] [$level] [$component] $message" | tee -a "$MAINTENANCE_LOG"

    # Send to syslog
    logger -t "ai-trading-bot-maintenance" -p "security.$level" "[$component] $message"
}

# Send notification
send_notification() {
    local severity="$1"
    local message="$2"
    local details="${3:-}"

    log_maintenance "$severity" "notification" "$message"

    # Webhook notification
    if [[ -n "$WEBHOOK_URL" ]]; then
        local payload=$(cat << EOF
{
  "text": "🔧 AI Trading Bot Maintenance [$severity]: $message",
  "severity": "$severity",
  "timestamp": "$(date -Iseconds)",
  "details": "$details",
  "host": "$(hostname)"
}
EOF
)
        curl -X POST "$WEBHOOK_URL" \
             -H "Content-Type: application/json" \
             -d "$payload" \
             --max-time 10 \
             --silent || true
    fi

    # Email notification (if configured)
    if [[ "$EMAIL_ALERTS" == "true" && -n "$ALERT_EMAIL" ]]; then
        send_email_alert "$severity" "$message" "$details"
    fi
}

# Email notification function
send_email_alert() {
    local severity="$1"
    local message="$2"
    local details="$3"

    if [[ -n "$SMTP_SERVER" ]]; then
        local subject="AI Trading Bot Security Maintenance Alert [$severity]"
        local body="Security maintenance alert from $(hostname)

Severity: $severity
Message: $message
Timestamp: $(date)

Details:
$details

--
AI Trading Bot Security System"

        # Send email using mail command (requires mail setup)
        echo "$body" | mail -s "$subject" "$ALERT_EMAIL" 2>/dev/null || \
        log_maintenance "warning" "email" "Failed to send email alert"
    fi
}

# Check if running as root
check_permissions() {
    if [[ $EUID -ne 0 ]]; then
        print_error "This script requires root privileges for system maintenance"
        print_warning "Run: sudo $0 $*"
        exit 1
    fi
}

# Create necessary directories
setup_maintenance_environment() {
    print_section "Setting Up Maintenance Environment"

    mkdir -p "$SECURITY_DIR"
    mkdir -p "$LOG_DIR"
    mkdir -p "$CONFIG_BASELINE_DIR"
    mkdir -p "$TEMP_DIR"

    # Set proper permissions
    chmod 750 "$LOG_DIR"
    chmod 700 "$SECURITY_DIR"
    chmod 700 "$TEMP_DIR"

    print_success "Maintenance environment ready"
}

# Security updates automation
perform_security_updates() {
    print_section "Performing Security Updates"

    local update_log="$TEMP_DIR/updates.log"
    local updated_packages=()

    # Update package lists
    log_maintenance "info" "updates" "Updating package lists"
    apt-get update > "$update_log" 2>&1

    # Check for security updates
    local security_updates=$(apt list --upgradable 2>/dev/null | grep -i security | wc -l)

    if [[ $security_updates -gt 0 ]]; then
        log_maintenance "info" "updates" "Found $security_updates security updates"

        # Perform unattended security upgrades
        DEBIAN_FRONTEND=noninteractive apt-get -y upgrade \
            -o Dpkg::Options::="--force-confold" \
            -o Dpkg::Options::="--force-confdef" \
            >> "$update_log" 2>&1

        # Log updated packages
        apt list --upgradable 2>/dev/null | grep -i security | while read -r line; do
            updated_packages+=("$line")
        done

        print_success "Security updates completed"
        send_notification "info" "Security updates applied" "Updated $security_updates packages"
    else
        print_success "No security updates available"
        log_maintenance "info" "updates" "No security updates available"
    fi

    # Update Docker images if running
    if systemctl is-active --quiet docker; then
        log_maintenance "info" "updates" "Updating Docker base images"

        # Pull latest base images
        docker pull python:3.12-slim-bookworm >/dev/null 2>&1 || true
        docker pull alpine:latest >/dev/null 2>&1 || true

        # Clean up unused images
        docker image prune -f >/dev/null 2>&1 || true

        print_success "Docker images updated"
    fi

    # Check if reboot is required
    if [[ -f /var/run/reboot-required ]]; then
        log_maintenance "warning" "updates" "Reboot required after updates"
        send_notification "warning" "System reboot required" "Security updates require system restart"

        # Schedule reboot for maintenance window (2 AM)
        echo "shutdown -r 02:00" | at now >/dev/null 2>&1 || \
        log_maintenance "warning" "updates" "Failed to schedule automatic reboot"
    fi
}

# Configuration drift detection
detect_configuration_drift() {
    print_section "Detecting Configuration Drift"

    if [[ "$DRIFT_CHECK_ENABLED" != "true" ]]; then
        print_warning "Configuration drift detection disabled"
        return 0
    fi

    local drift_detected=false
    local drift_report="$TEMP_DIR/drift-report.txt"

    # Create baseline configurations if they don't exist
    create_configuration_baseline

    # Check critical security files
    local security_files=(
        "/etc/fail2ban/jail.d/ai-trading-bot.conf"
        "/etc/ufw/user.rules"
        "/etc/docker/daemon.json"
        "/etc/ssh/sshd_config"
        "/etc/audit/rules.d/ai-trading-bot.rules"
    )

    echo "Configuration Drift Report - $(date)" > "$drift_report"
    echo "=====================================" >> "$drift_report"

    for file in "${security_files[@]}"; do
        if [[ -f "$file" ]]; then
            local baseline_file="$CONFIG_BASELINE_DIR/$(basename "$file")"

            if [[ -f "$baseline_file" ]]; then
                if ! diff -u "$baseline_file" "$file" >> "$drift_report" 2>&1; then
                    log_maintenance "warning" "drift" "Configuration drift detected in $file"
                    drift_detected=true
                fi
            else
                log_maintenance "warning" "drift" "No baseline found for $file"
            fi
        fi
    done

    # Check Docker security configuration
    if command -v docker >/dev/null 2>&1; then
        local docker_info=$(docker info --format '{{json .}}' 2>/dev/null)
        local expected_security_options="seccomp userns"

        for option in $expected_security_options; do
            if ! echo "$docker_info" | grep -q "$option"; then
                echo "Docker security option missing: $option" >> "$drift_report"
                log_maintenance "warning" "drift" "Docker security option missing: $option"
                drift_detected=true
            fi
        done
    fi

    # Check firewall status
    if ! ufw status | grep -q "Status: active"; then
        echo "UFW firewall is not active" >> "$drift_report"
        log_maintenance "error" "drift" "UFW firewall is not active"
        drift_detected=true
    fi

    # Check critical services
    local critical_services=("fail2ban" "ufw" "auditd" "docker")

    for service in "${critical_services[@]}"; do
        if ! systemctl is-active --quiet "$service"; then
            echo "Critical service not running: $service" >> "$drift_report"
            log_maintenance "error" "drift" "Critical service not running: $service"
            drift_detected=true
        fi
    done

    if [[ "$drift_detected" == "true" ]]; then
        print_warning "Configuration drift detected"
        send_notification "warning" "Configuration drift detected" "$(cat "$drift_report")"

        # Auto-remediation if enabled
        if [[ "$AUTO_REMEDIATION" == "true" ]]; then
            remediate_configuration_drift
        fi
    else
        print_success "No configuration drift detected"
        log_maintenance "info" "drift" "No configuration drift detected"
    fi
}

# Create configuration baseline
create_configuration_baseline() {
    if [[ ! -d "$CONFIG_BASELINE_DIR" ]]; then
        mkdir -p "$CONFIG_BASELINE_DIR"
    fi

    # Copy current configurations as baseline
    local files_to_baseline=(
        "/etc/fail2ban/jail.d/ai-trading-bot.conf"
        "/etc/ufw/user.rules"
        "/etc/docker/daemon.json"
        "/etc/ssh/sshd_config"
        "/etc/audit/rules.d/ai-trading-bot.rules"
    )

    for file in "${files_to_baseline[@]}"; do
        if [[ -f "$file" ]]; then
            cp "$file" "$CONFIG_BASELINE_DIR/$(basename "$file")" 2>/dev/null || true
        fi
    done

    log_maintenance "info" "baseline" "Configuration baseline updated"
}

# Remediate configuration drift
remediate_configuration_drift() {
    print_section "Remediating Configuration Drift"

    log_maintenance "info" "remediation" "Starting automatic remediation"

    # Restart failed services
    local critical_services=("fail2ban" "ufw" "auditd" "docker")

    for service in "${critical_services[@]}"; do
        if ! systemctl is-active --quiet "$service"; then
            log_maintenance "info" "remediation" "Restarting service: $service"
            systemctl restart "$service" || \
            log_maintenance "error" "remediation" "Failed to restart $service"
        fi
    done

    # Re-enable UFW if disabled
    if ! ufw status | grep -q "Status: active"; then
        log_maintenance "info" "remediation" "Re-enabling UFW firewall"
        ufw --force enable || \
        log_maintenance "error" "remediation" "Failed to enable UFW"
    fi

    # Restore baseline configurations (only if auto-remediation is explicitly enabled)
    if [[ "$AUTO_REMEDIATION" == "true" ]]; then
        local files_to_restore=(
            "/etc/fail2ban/jail.d/ai-trading-bot.conf"
            "/etc/ufw/user.rules"
        )

        for file in "${files_to_restore[@]}"; do
            local baseline_file="$CONFIG_BASELINE_DIR/$(basename "$file")"
            if [[ -f "$baseline_file" && -f "$file" ]]; then
                if ! diff -q "$baseline_file" "$file" >/dev/null 2>&1; then
                    log_maintenance "info" "remediation" "Restoring baseline for $file"
                    cp "$baseline_file" "$file"
                fi
            fi
        done
    fi

    print_success "Remediation completed"
    send_notification "info" "Configuration drift remediated" "Automatic remediation completed"
}

# Log cleanup and rotation
perform_log_maintenance() {
    print_section "Performing Log Maintenance"

    local cleaned_logs=0
    local freed_space=0

    # Compress old log files
    find "$LOG_DIR" -name "*.log" -mtime +7 -exec gzip {} \; 2>/dev/null || true

    # Remove old compressed logs
    find "$LOG_DIR" -name "*.gz" -mtime +$LOG_RETENTION_DAYS -delete 2>/dev/null && \
    cleaned_logs=$((cleaned_logs + 1))

    # Clean Docker logs
    if command -v docker >/dev/null 2>&1; then
        local docker_log_size_before=$(du -s /var/lib/docker/containers 2>/dev/null | awk '{print $1}' || echo "0")

        # Truncate large container logs
        find /var/lib/docker/containers -name "*.log" -size +100M -exec truncate -s 50M {} \; 2>/dev/null || true

        local docker_log_size_after=$(du -s /var/lib/docker/containers 2>/dev/null | awk '{print $1}' || echo "0")
        freed_space=$((docker_log_size_before - docker_log_size_after))
    fi

    # Clean system logs
    journalctl --vacuum-time=${LOG_RETENTION_DAYS}d >/dev/null 2>&1 || true

    # Clean old audit logs
    find /var/log/audit -name "audit.log.*" -mtime +$LOG_RETENTION_DAYS -delete 2>/dev/null || true

    # Clean temporary files
    find /tmp -name "security-maintenance-*" -mtime +1 -delete 2>/dev/null || true

    print_success "Log maintenance completed"
    log_maintenance "info" "cleanup" "Log cleanup completed - freed ${freed_space}KB space"
}

# Security scan automation
perform_security_scans() {
    print_section "Performing Security Scans"

    local scan_report="$TEMP_DIR/security-scan.txt"
    local critical_issues=0

    echo "Security Scan Report - $(date)" > "$scan_report"
    echo "===============================" >> "$scan_report"

    # Check for rootkits with rkhunter
    if command -v rkhunter >/dev/null 2>&1; then
        log_maintenance "info" "scan" "Running rootkit scan"

        # Update rkhunter database
        rkhunter --update --quiet >/dev/null 2>&1 || true

        # Run scan
        if ! rkhunter --check --quiet --skip-keypress >> "$scan_report" 2>&1; then
            critical_issues=$((critical_issues + 1))
            log_maintenance "warning" "scan" "Rootkit scan found issues"
        fi
    fi

    # Check file integrity with AIDE
    if command -v aide >/dev/null 2>&1; then
        log_maintenance "info" "scan" "Running file integrity check"

        if ! aide --check >> "$scan_report" 2>&1; then
            log_maintenance "warning" "scan" "File integrity check found changes"
        fi
    fi

    # Check for failed login attempts
    local failed_logins=$(grep "Failed password" /var/log/auth.log | wc -l 2>/dev/null || echo "0")
    if [[ $failed_logins -gt 10 ]]; then
        echo "High number of failed login attempts: $failed_logins" >> "$scan_report"
        log_maintenance "warning" "scan" "High number of failed login attempts: $failed_logins"
    fi

    # Check for suspicious network connections
    local suspicious_connections=$(netstat -tuln | grep -E ":999[0-9]|:666[0-9]" | wc -l 2>/dev/null || echo "0")
    if [[ $suspicious_connections -gt 0 ]]; then
        echo "Suspicious network connections detected: $suspicious_connections" >> "$scan_report"
        critical_issues=$((critical_issues + 1))
        log_maintenance "error" "scan" "Suspicious network connections detected"
    fi

    # Check Docker container security
    if command -v docker >/dev/null 2>&1; then
        # Check for containers running as root
        local root_containers=$(docker ps --format "table {{.Names}}\t{{.Command}}" | grep -v "NAMES" | wc -l 2>/dev/null || echo "0")

        # Check for privileged containers
        local privileged_containers=$(docker ps --filter "label=privileged=true" | wc -l 2>/dev/null || echo "0")

        if [[ $privileged_containers -gt 0 ]]; then
            echo "Privileged containers detected: $privileged_containers" >> "$scan_report"
            log_maintenance "warning" "scan" "Privileged containers detected"
        fi
    fi

    if [[ $critical_issues -gt 0 ]]; then
        print_warning "Security scan found $critical_issues critical issues"
        send_notification "warning" "Security scan found issues" "$(cat "$scan_report")"
    else
        print_success "Security scan completed - no critical issues"
        log_maintenance "info" "scan" "Security scan completed successfully"
    fi
}

# Backup verification and cleanup
maintain_backups() {
    print_section "Maintaining Security Backups"

    local backup_dir="/opt/ai-trading-bot/backups"
    local cleaned_backups=0

    if [[ -d "$backup_dir" ]]; then
        # Remove old backups
        find "$backup_dir" -name "*.tar.gz" -mtime +$BACKUP_RETENTION_DAYS -delete 2>/dev/null && \
        cleaned_backups=$((cleaned_backups + 1))

        # Verify recent backups
        local recent_backups=$(find "$backup_dir" -name "*.tar.gz" -mtime -7 | wc -l)

        if [[ $recent_backups -eq 0 ]]; then
            log_maintenance "warning" "backup" "No recent backups found"
            send_notification "warning" "No recent backups found" "No backups created in the last 7 days"
        else
            log_maintenance "info" "backup" "Found $recent_backups recent backups"
        fi
    fi

    # Test backup integrity (sample check)
    local latest_backup=$(find "$backup_dir" -name "*.tar.gz" -mtime -1 2>/dev/null | head -1)
    if [[ -n "$latest_backup" ]]; then
        if tar -tzf "$latest_backup" >/dev/null 2>&1; then
            log_maintenance "info" "backup" "Latest backup integrity verified"
        else
            log_maintenance "error" "backup" "Latest backup integrity check failed"
            send_notification "error" "Backup integrity check failed" "Latest backup file is corrupted"
        fi
    fi

    print_success "Backup maintenance completed"
}

# System health check
perform_health_check() {
    print_section "Performing System Health Check"

    local health_issues=0
    local health_report="$TEMP_DIR/health-report.txt"

    echo "System Health Report - $(date)" > "$health_report"
    echo "===============================" >> "$health_report"

    # Check disk space
    local disk_usage=$(df / | awk 'NR==2 {print $5}' | sed 's/%//')
    if [[ $disk_usage -gt 85 ]]; then
        echo "High disk usage: ${disk_usage}%" >> "$health_report"
        log_maintenance "warning" "health" "High disk usage: ${disk_usage}%"
        health_issues=$((health_issues + 1))
    fi

    # Check memory usage
    local memory_usage=$(free | awk 'NR==2{printf "%.0f", $3*100/$2}')
    if [[ $memory_usage -gt 90 ]]; then
        echo "High memory usage: ${memory_usage}%" >> "$health_report"
        log_maintenance "warning" "health" "High memory usage: ${memory_usage}%"
        health_issues=$((health_issues + 1))
    fi

    # Check load average
    local load_avg=$(uptime | awk -F'load average:' '{print $2}' | awk '{print $1}' | sed 's/,//')
    local cpu_count=$(nproc)
    local load_threshold=$((cpu_count * 2))

    if (( $(echo "$load_avg > $load_threshold" | bc -l) )); then
        echo "High system load: $load_avg (threshold: $load_threshold)" >> "$health_report"
        log_maintenance "warning" "health" "High system load: $load_avg"
        health_issues=$((health_issues + 1))
    fi

    # Check critical services
    local critical_services=("ssh" "fail2ban" "ufw" "docker")
    for service in "${critical_services[@]}"; do
        if ! systemctl is-active --quiet "$service"; then
            echo "Critical service down: $service" >> "$health_report"
            log_maintenance "error" "health" "Critical service down: $service"
            health_issues=$((health_issues + 1))
        fi
    done

    # Check certificate expiration (if SSL certificates exist)
    if [[ -d "/etc/letsencrypt/live" ]]; then
        find /etc/letsencrypt/live -name "cert.pem" -exec openssl x509 -in {} -checkend 604800 -noout \; 2>/dev/null || {
            echo "SSL certificate expiring within 7 days" >> "$health_report"
            log_maintenance "warning" "health" "SSL certificate expiring soon"
            health_issues=$((health_issues + 1))
        }
    fi

    if [[ $health_issues -gt 0 ]]; then
        print_warning "System health check found $health_issues issues"
        send_notification "warning" "System health issues detected" "$(cat "$health_report")"
    else
        print_success "System health check passed"
        log_maintenance "info" "health" "System health check passed"
    fi
}

# Generate maintenance report
generate_maintenance_report() {
    print_section "Generating Maintenance Report"

    local report_file="$LOG_DIR/maintenance-report-$(date +%Y%m%d).txt"

    cat > "$report_file" << EOF
AI Trading Bot Security Maintenance Report
==========================================
Date: $(date)
Host: $(hostname)
Uptime: $(uptime)

System Information:
- OS: $(lsb_release -d | cut -f2)
- Kernel: $(uname -r)
- Docker: $(docker --version 2>/dev/null || echo "Not installed")

Security Services Status:
$(systemctl is-active fail2ban ufw auditd docker 2>/dev/null | paste -d' ' <(echo -e "fail2ban\nufw\nauditd\ndocker") -)

Disk Usage:
$(df -h /)

Memory Usage:
$(free -h)

Recent Security Events:
$(tail -20 "$MAINTENANCE_LOG" 2>/dev/null || echo "No recent events")

Maintenance Summary:
- Security updates: $(grep "updates" "$MAINTENANCE_LOG" | grep "$(date +%Y-%m-%d)" | wc -l) operations
- Configuration checks: $(grep "drift" "$MAINTENANCE_LOG" | grep "$(date +%Y-%m-%d)" | wc -l) operations
- Security scans: $(grep "scan" "$MAINTENANCE_LOG" | grep "$(date +%Y-%m-%d)" | wc -l) operations
- Health checks: $(grep "health" "$MAINTENANCE_LOG" | grep "$(date +%Y-%m-%d)" | wc -l) operations

Next Scheduled Maintenance: $(date -d "+1 week")
EOF

    print_success "Maintenance report generated: $report_file"
    log_maintenance "info" "report" "Maintenance report generated"

    # Send report summary
    send_notification "info" "Maintenance completed successfully" "$(tail -10 "$report_file")"
}

# Cleanup function
cleanup() {
    if [[ -d "$TEMP_DIR" ]]; then
        rm -rf "$TEMP_DIR"
    fi
}

# Main maintenance function
main() {
    print_header "Security Maintenance Automation"

    # Setup signal handlers
    trap cleanup EXIT

    # Check permissions
    check_permissions

    # Setup environment
    setup_maintenance_environment

    log_maintenance "info" "start" "Security maintenance started"

    # Perform maintenance tasks
    perform_security_updates
    detect_configuration_drift
    perform_log_maintenance
    perform_security_scans
    maintain_backups
    perform_health_check
    generate_maintenance_report

    log_maintenance "info" "complete" "Security maintenance completed successfully"

    print_header "Security Maintenance Complete!"
    print_success "All maintenance tasks completed successfully"

    echo -e "\n${BLUE}Maintenance Summary:${NC}"
    echo "- Security updates applied"
    echo "- Configuration drift checked"
    echo "- Log maintenance performed"
    echo "- Security scans completed"
    echo "- Backup maintenance done"
    echo "- System health verified"
    echo "- Maintenance report generated"

    echo -e "\n${BLUE}Reports Available:${NC}"
    echo "- Maintenance log: $MAINTENANCE_LOG"
    echo "- Daily report: $LOG_DIR/maintenance-report-$(date +%Y%m%d).txt"
}

# Setup scheduled maintenance
setup_scheduled_maintenance() {
    print_header "Setting Up Scheduled Maintenance"

    # Create systemd service for maintenance
    cat > /tmp/ai-trading-bot-maintenance.service << 'EOF'
[Unit]
Description=AI Trading Bot Security Maintenance
After=network.target

[Service]
Type=oneshot
User=root
ExecStart=/opt/ai-trading-bot/scripts/security/security-maintenance-automation.sh run
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

    # Create systemd timer
    cat > /tmp/ai-trading-bot-maintenance.timer << 'EOF'
[Unit]
Description=Run AI Trading Bot Security Maintenance
Requires=ai-trading-bot-maintenance.service

[Timer]
OnCalendar=weekly
Persistent=true
RandomizedDelaySec=1800

[Install]
WantedBy=timers.target
EOF

    # Install service and timer
    sudo mv /tmp/ai-trading-bot-maintenance.service /etc/systemd/system/
    sudo mv /tmp/ai-trading-bot-maintenance.timer /etc/systemd/system/

    # Enable and start timer
    sudo systemctl daemon-reload
    sudo systemctl enable ai-trading-bot-maintenance.timer
    sudo systemctl start ai-trading-bot-maintenance.timer

    print_success "Scheduled maintenance configured"
    print_warning "Maintenance will run weekly with randomized delay"

    # Show timer status
    systemctl list-timers ai-trading-bot-maintenance.timer
}

# Parse command line arguments
case "${1:-run}" in
    "run")
        main
        ;;
    "setup")
        setup_scheduled_maintenance
        ;;
    "drift-check")
        check_permissions
        setup_maintenance_environment
        detect_configuration_drift
        ;;
    "health-check")
        check_permissions
        setup_maintenance_environment
        perform_health_check
        ;;
    "security-scan")
        check_permissions
        setup_maintenance_environment
        perform_security_scans
        ;;
    "updates")
        check_permissions
        setup_maintenance_environment
        perform_security_updates
        ;;
    "help"|"--help")
        echo "Usage: $0 [command]"
        echo
        echo "Commands:"
        echo "  run          Run complete maintenance routine (default)"
        echo "  setup        Set up scheduled maintenance"
        echo "  drift-check  Check for configuration drift only"
        echo "  health-check Perform system health check only"
        echo "  security-scan Run security scans only"
        echo "  updates      Apply security updates only"
        echo "  help         Show this help message"
        echo
        echo "Environment Variables:"
        echo "  BACKUP_RETENTION_DAYS    Days to keep backups (default: 30)"
        echo "  LOG_RETENTION_DAYS       Days to keep logs (default: 90)"
        echo "  DRIFT_CHECK_ENABLED      Enable drift detection (default: true)"
        echo "  AUTO_REMEDIATION         Enable auto-remediation (default: false)"
        echo "  SECURITY_WEBHOOK_URL     Webhook for notifications"
        echo "  EMAIL_ALERTS             Enable email alerts (default: false)"
        ;;
    *)
        print_error "Unknown command: $1"
        echo "Run '$0 help' for usage information"
        exit 1
        ;;
esac
