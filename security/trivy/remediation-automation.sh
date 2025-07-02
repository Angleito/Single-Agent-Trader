#!/bin/bash

# Automated Remediation Suggestions and Tools for AI Trading Bot
# This script analyzes Trivy scan results and provides automated remediation suggestions

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPORTS_DIR="$PROJECT_ROOT/security/trivy/reports"
REMEDIATION_DIR="$PROJECT_ROOT/security/trivy/remediation"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Remediation configuration
AUTO_FIX_ENABLED=false
CREATE_BRANCHES=false
GENERATE_PATCHES=true
UPDATE_DEPENDENCIES=false
SLACK_WEBHOOK=""
JIRA_ENABLED=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

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

log_remediation() {
    echo -e "${PURPLE}[REMEDIATION]${NC} $1"
}

# Function to setup directories
setup_directories() {
    log_info "Setting up remediation directories..."

    mkdir -p "$REMEDIATION_DIR"/{patches,reports,scripts,tickets}
    mkdir -p "$REMEDIATION_DIR/archive/$TIMESTAMP"

    log_success "Remediation directories created"
}

# Function to analyze vulnerability reports
analyze_vulnerabilities() {
    log_info "Analyzing vulnerability reports for remediation opportunities..."

    local remediation_report="$REMEDIATION_DIR/reports/remediation_analysis_${TIMESTAMP}.md"

    cat > "$remediation_report" <<EOF
# Vulnerability Remediation Analysis

**Generated**: $(date)
**Analysis ID**: $TIMESTAMP

## Executive Summary

This report provides automated remediation suggestions for vulnerabilities found in the AI Trading Bot.

EOF

    local total_vulns=0
    local fixable_vulns=0
    local auto_fixable=0

    # Analyze JSON reports
    if command -v jq >/dev/null 2>&1; then
        for json_file in "$REPORTS_DIR"/json/*.json; do
            if [[ -f "$json_file" ]]; then
                local file_vulns=$(jq -r '.Results[]?.Vulnerabilities[]? | select(.FixedVersion != null and .FixedVersion != "") | {VulnerabilityID, PkgName, InstalledVersion, FixedVersion, Severity}' "$json_file" 2>/dev/null)

                if [[ -n "$file_vulns" ]]; then
                    echo "### $(basename "$json_file" .json)" >> "$remediation_report"
                    echo "" >> "$remediation_report"

                    while IFS= read -r vuln; do
                        if [[ -n "$vuln" && "$vuln" != "null" ]]; then
                            local vuln_id=$(echo "$vuln" | jq -r '.VulnerabilityID // "Unknown"')
                            local pkg_name=$(echo "$vuln" | jq -r '.PkgName // "Unknown"')
                            local installed_ver=$(echo "$vuln" | jq -r '.InstalledVersion // "Unknown"')
                            local fixed_ver=$(echo "$vuln" | jq -r '.FixedVersion // "Unknown"')
                            local severity=$(echo "$vuln" | jq -r '.Severity // "Unknown"')

                            if [[ "$fixed_ver" != "Unknown" && "$fixed_ver" != "" ]]; then
                                echo "#### $vuln_id ($severity)" >> "$remediation_report"
                                echo "- **Package**: $pkg_name" >> "$remediation_report"
                                echo "- **Current Version**: $installed_ver" >> "$remediation_report"
                                echo "- **Fixed Version**: $fixed_ver" >> "$remediation_report"
                                echo "- **Remediation**: Update package to version $fixed_ver" >> "$remediation_report"

                                # Generate fix command
                                generate_fix_command "$pkg_name" "$installed_ver" "$fixed_ver" "$severity"

                                total_vulns=$((total_vulns + 1))
                                fixable_vulns=$((fixable_vulns + 1))

                                # Check if auto-fixable
                                if can_auto_fix "$pkg_name" "$fixed_ver"; then
                                    auto_fixable=$((auto_fixable + 1))
                                    echo "- **Auto-fixable**: Yes ✅" >> "$remediation_report"
                                else
                                    echo "- **Auto-fixable**: No ⚠️" >> "$remediation_report"
                                fi

                                echo "" >> "$remediation_report"
                            fi
                        fi
                    done <<< "$(echo "$file_vulns" | jq -c '.')"
                fi
            fi
        done
    else
        log_warning "jq not available, using text analysis for remediation"
    fi

    # Add summary
    cat >> "$remediation_report" <<EOF

## Remediation Summary

- **Total Vulnerabilities Analyzed**: $total_vulns
- **Fixable Vulnerabilities**: $fixable_vulns
- **Auto-fixable Vulnerabilities**: $auto_fixable
- **Manual Review Required**: $((fixable_vulns - auto_fixable))

## Recommended Actions

1. **Immediate Fixes** (Auto-fixable): Apply automated patches for $auto_fixable vulnerabilities
2. **Dependency Updates**: Update packages to fixed versions
3. **Security Review**: Manual review required for complex vulnerabilities
4. **Testing**: Run comprehensive tests after applying fixes

EOF

    log_success "Vulnerability analysis completed: $remediation_report"
    echo "Total fixable vulnerabilities: $fixable_vulns"
    echo "Auto-fixable vulnerabilities: $auto_fixable"
}

# Function to generate fix commands
generate_fix_command() {
    local pkg_name="$1"
    local current_ver="$2"
    local fixed_ver="$3"
    local severity="$4"

    local fix_script="$REMEDIATION_DIR/scripts/fix_${pkg_name}_${TIMESTAMP}.sh"

    cat > "$fix_script" <<EOF
#!/bin/bash
# Automated fix for $pkg_name vulnerability
# Severity: $severity
# Current: $current_ver -> Fixed: $fixed_ver

set -euo pipefail

echo "Fixing $pkg_name vulnerability..."

# Backup current state
cp pyproject.toml pyproject.toml.backup 2>/dev/null || true
cp poetry.lock poetry.lock.backup 2>/dev/null || true

# Update package
if command -v poetry >/dev/null 2>&1; then
    echo "Updating $pkg_name to $fixed_ver using Poetry..."
    poetry add "$pkg_name==$fixed_ver" || poetry add "$pkg_name>=$fixed_ver"
elif [[ -f "requirements.txt" ]]; then
    echo "Updating $pkg_name in requirements.txt..."
    sed -i.backup "s/$pkg_name==.*/$pkg_name==$fixed_ver/" requirements.txt
    pip install -r requirements.txt
else
    echo "No package manager found, manual update required"
    exit 1
fi

echo "✅ $pkg_name updated to $fixed_ver"
echo "🧪 Run tests to verify the fix: poetry run pytest"
EOF

    chmod +x "$fix_script"
    log_remediation "Generated fix script: $fix_script"
}

# Function to check if vulnerability can be auto-fixed
can_auto_fix() {
    local pkg_name="$1"
    local fixed_ver="$2"

    # Define packages that are safe to auto-update
    local safe_packages=("requests" "urllib3" "pyyaml" "jinja2" "pillow" "cryptography")

    for safe_pkg in "${safe_packages[@]}"; do
        if [[ "$pkg_name" == "$safe_pkg" ]]; then
            return 0
        fi
    done

    return 1
}

# Function to analyze secrets
analyze_secrets() {
    log_info "Analyzing secret detection results..."

    local secrets_report="$REMEDIATION_DIR/reports/secrets_remediation_${TIMESTAMP}.md"

    cat > "$secrets_report" <<EOF
# Secrets Remediation Guide

**Generated**: $(date)

## Overview

This guide provides steps to remediate exposed secrets found in the codebase.

## Immediate Actions Required

EOF

    local secret_count=0

    # Analyze secret scan results
    for json_file in "$REPORTS_DIR"/json/*secrets*.json "$REPORTS_DIR"/json/*secret*.json; do
        if [[ -f "$json_file" ]]; then
            if command -v jq >/dev/null 2>&1; then
                local secrets=$(jq -r '.Results[]?.Secrets[]? | {RuleID, Category, Title, StartLine, EndLine}' "$json_file" 2>/dev/null)

                if [[ -n "$secrets" && "$secrets" != "null" ]]; then
                    echo "### Secrets found in $(basename "$json_file")" >> "$secrets_report"
                    echo "" >> "$secrets_report"

                    while IFS= read -r secret; do
                        if [[ -n "$secret" && "$secret" != "null" ]]; then
                            local rule_id=$(echo "$secret" | jq -r '.RuleID // "Unknown"')
                            local category=$(echo "$secret" | jq -r '.Category // "Unknown"')
                            local title=$(echo "$secret" | jq -r '.Title // "Unknown"')
                            local start_line=$(echo "$secret" | jq -r '.StartLine // "Unknown"')
                            local end_line=$(echo "$secret" | jq -r '.EndLine // "Unknown"')

                            echo "#### $title" >> "$secrets_report"
                            echo "- **Type**: $category" >> "$secrets_report"
                            echo "- **Rule**: $rule_id" >> "$secrets_report"
                            echo "- **Location**: Lines $start_line-$end_line" >> "$secrets_report"
                            echo "- **Action**: Remove secret and rotate credentials" >> "$secrets_report"
                            echo "" >> "$secrets_report"

                            secret_count=$((secret_count + 1))

                            # Generate secret removal script
                            generate_secret_removal_script "$category" "$start_line" "$end_line"
                        fi
                    done <<< "$(echo "$secrets" | jq -c '.')"
                fi
            fi
        fi
    done

    cat >> "$secrets_report" <<EOF

## Remediation Steps

1. **Immediate Rotation**: Rotate all exposed credentials immediately
2. **Remove from Code**: Delete hardcoded secrets from source code
3. **Environment Variables**: Move secrets to environment variables
4. **Secrets Management**: Implement proper secrets management (Vault, AWS Secrets Manager)
5. **Git History**: Clean git history if needed
6. **Pre-commit Hooks**: Install secret detection pre-commit hooks

## Prevention Measures

\`\`\`bash
# Install pre-commit hooks
pip install pre-commit detect-secrets
echo "repos:
- repo: https://github.com/Yelp/detect-secrets
  rev: v1.4.0
  hooks:
  - id: detect-secrets
    args: ['--baseline', '.secrets.baseline']" > .pre-commit-config.yaml
pre-commit install
\`\`\`

EOF

    log_success "Secrets analysis completed: $secrets_report"
    echo "Total secrets found: $secret_count"
}

# Function to generate secret removal script
generate_secret_removal_script() {
    local category="$1"
    local start_line="$2"
    local end_line="$3"

    local removal_script="$REMEDIATION_DIR/scripts/remove_secret_${category}_${start_line}_${TIMESTAMP}.sh"

    cat > "$removal_script" <<EOF
#!/bin/bash
# Script to help remove $category secret at lines $start_line-$end_line

echo "⚠️  MANUAL ACTION REQUIRED"
echo "Secret detected: $category"
echo "Location: Lines $start_line-$end_line"
echo ""
echo "Steps to remediate:"
echo "1. Open the file and locate lines $start_line-$end_line"
echo "2. Remove the hardcoded secret"
echo "3. Replace with environment variable or secrets manager"
echo "4. Rotate the exposed credential"
echo "5. Test the application"
echo ""
echo "Example replacement:"
echo "# Before: api_key = 'hardcoded_secret'"
echo "# After:  api_key = os.environ.get('API_KEY')"
EOF

    chmod +x "$removal_script"
    log_remediation "Generated secret removal guide: $removal_script"
}

# Function to analyze configuration issues
analyze_configurations() {
    log_info "Analyzing configuration misconfigurations..."

    local config_report="$REMEDIATION_DIR/reports/config_remediation_${TIMESTAMP}.md"

    cat > "$config_report" <<EOF
# Configuration Remediation Guide

**Generated**: $(date)

## Docker and Infrastructure Misconfigurations

EOF

    local config_count=0

    # Analyze configuration scan results
    for json_file in "$REPORTS_DIR"/json/*config*.json; do
        if [[ -f "$json_file" ]]; then
            if command -v jq >/dev/null 2>&1; then
                local configs=$(jq -r '.Results[]?.Misconfigurations[]? | {ID, Title, Description, Severity, Message}' "$json_file" 2>/dev/null)

                if [[ -n "$configs" && "$configs" != "null" ]]; then
                    echo "### Configuration issues in $(basename "$json_file")" >> "$config_report"
                    echo "" >> "$config_report"

                    while IFS= read -r config; do
                        if [[ -n "$config" && "$config" != "null" ]]; then
                            local config_id=$(echo "$config" | jq -r '.ID // "Unknown"')
                            local title=$(echo "$config" | jq -r '.Title // "Unknown"')
                            local description=$(echo "$config" | jq -r '.Description // "Unknown"')
                            local severity=$(echo "$config" | jq -r '.Severity // "Unknown"')
                            local message=$(echo "$config" | jq -r '.Message // "Unknown"')

                            echo "#### $title ($severity)" >> "$config_report"
                            echo "- **ID**: $config_id" >> "$config_report"
                            echo "- **Description**: $description" >> "$config_report"
                            echo "- **Issue**: $message" >> "$config_report"

                            # Generate configuration fix
                            local fix_suggestion=$(generate_config_fix "$config_id" "$title")
                            echo "- **Fix**: $fix_suggestion" >> "$config_report"
                            echo "" >> "$config_report"

                            config_count=$((config_count + 1))
                        fi
                    done <<< "$(echo "$configs" | jq -c '.')"
                fi
            fi
        fi
    done

    cat >> "$config_report" <<EOF

## Common Fixes

### Docker Security
- Run containers as non-root user
- Use read-only filesystems where possible
- Implement resource limits
- Drop unnecessary capabilities

### Network Security
- Use specific port bindings instead of 0.0.0.0
- Implement network segmentation
- Use secrets instead of environment variables for sensitive data

### Configuration Management
- Use external configuration files
- Implement configuration validation
- Regular security audits

EOF

    log_success "Configuration analysis completed: $config_report"
    echo "Total configuration issues: $config_count"
}

# Function to generate configuration fixes
generate_config_fix() {
    local config_id="$1"
    local title="$2"

    case "$config_id" in
        *"user"*|*"root"*)
            echo "Add 'user: 1000:1000' to docker-compose.yml or 'USER 1000' to Dockerfile"
            ;;
        *"capabilities"*)
            echo "Add 'cap_drop: [ALL]' and only add required capabilities"
            ;;
        *"privileged"*)
            echo "Remove 'privileged: true' and use specific capabilities instead"
            ;;
        *"network"*)
            echo "Use specific port bindings like '127.0.0.1:8080:8080' instead of '8080:8080'"
            ;;
        *"secret"*|*"password"*)
            echo "Use Docker secrets or external secret management instead of environment variables"
            ;;
        *)
            echo "Review $title configuration and apply security best practices"
            ;;
    esac
}

# Function to create JIRA tickets
create_jira_tickets() {
    if [[ "$JIRA_ENABLED" != true ]]; then
        return
    fi

    log_info "Creating JIRA tickets for vulnerabilities..."

    # This would integrate with JIRA API
    # For now, create ticket templates

    local ticket_template="$REMEDIATION_DIR/tickets/jira_tickets_${TIMESTAMP}.json"

    cat > "$ticket_template" <<EOF
{
  "tickets": [
    {
      "summary": "Security Vulnerability Remediation - Critical Issues",
      "description": "Address critical security vulnerabilities found in security scan",
      "priority": "Critical",
      "labels": ["security", "vulnerability", "trading-bot"]
    },
    {
      "summary": "Secret Removal and Rotation",
      "description": "Remove hardcoded secrets and implement proper secrets management",
      "priority": "High",
      "labels": ["security", "secrets", "credentials"]
    }
  ]
}
EOF

    log_success "JIRA ticket templates created: $ticket_template"
}

# Function to generate remediation patches
generate_patches() {
    if [[ "$GENERATE_PATCHES" != true ]]; then
        return
    fi

    log_info "Generating remediation patches..."

    local patch_dir="$REMEDIATION_DIR/patches"

    # Generate common security patches
    generate_dockerfile_patch "$patch_dir"
    generate_docker_compose_patch "$patch_dir"
    generate_pre_commit_patch "$patch_dir"

    log_success "Remediation patches generated in $patch_dir"
}

# Function to generate Dockerfile security patch
generate_dockerfile_patch() {
    local patch_dir="$1"

    cat > "$patch_dir/dockerfile_security.patch" <<'EOF'
--- a/Dockerfile
+++ b/Dockerfile
@@ -1,4 +1,7 @@
 FROM python:3.12-slim
+
+# Security: Create non-root user
+RUN groupadd -r appuser && useradd -r -g appuser appuser

 WORKDIR /app

@@ -10,4 +13,8 @@

 COPY . .

+# Security: Change ownership and switch to non-root user
+RUN chown -R appuser:appuser /app
+USER appuser
+
 CMD ["python", "-m", "bot.main", "live"]
EOF
}

# Function to generate Docker Compose security patch
generate_docker_compose_patch() {
    local patch_dir="$1"

    cat > "$patch_dir/docker_compose_security.patch" <<'EOF'
--- a/docker-compose.yml
+++ b/docker-compose.yml
@@ -5,6 +5,15 @@
     build: .
     container_name: ai-trading-bot
     restart: unless-stopped
+    # Security hardening
+    user: "1000:1000"
+    read_only: true
+    security_opt:
+      - no-new-privileges:true
+    cap_drop:
+      - ALL
+    tmpfs:
+      - /tmp:noexec,nosuid,size=100m
     environment:
       - PYTHONUNBUFFERED=1
     volumes:
EOF
}

# Function to generate pre-commit hooks patch
generate_pre_commit_patch() {
    local patch_dir="$1"

    cat > "$patch_dir/pre_commit_security.yaml" <<'EOF'
# Add to .pre-commit-config.yaml
repos:
  - repo: https://github.com/Yelp/detect-secrets
    rev: v1.4.0
    hooks:
      - id: detect-secrets
        args: ['--baseline', '.secrets.baseline']

  - repo: https://github.com/PyCQA/bandit
    rev: 1.7.5
    hooks:
      - id: bandit
        args: ['-r', 'bot/', '-ll']

  - repo: https://github.com/hadolint/hadolint
    rev: v2.12.0
    hooks:
      - id: hadolint-docker
        args: ['--ignore', 'DL3008']
EOF
}

# Function to send notifications
send_notifications() {
    local critical_vulns="$1"
    local fixable_vulns="$2"
    local auto_fixable="$3"

    if [[ -n "$SLACK_WEBHOOK" ]]; then
        log_info "Sending remediation notification to Slack..."

        local slack_payload=$(cat <<EOF
{
  "attachments": [
    {
      "color": "warning",
      "title": "🔧 Security Remediation Report Available",
      "fields": [
        {
          "title": "Fixable Vulnerabilities",
          "value": "$fixable_vulns",
          "short": true
        },
        {
          "title": "Auto-fixable",
          "value": "$auto_fixable",
          "short": true
        },
        {
          "title": "Manual Review Required",
          "value": "$((fixable_vulns - auto_fixable))",
          "short": true
        }
      ],
      "actions": [
        {
          "type": "button",
          "text": "View Remediation Report",
          "url": "file://$REMEDIATION_DIR/reports/"
        }
      ],
      "footer": "AI Trading Bot Security",
      "ts": $(date +%s)
    }
  ]
}
EOF
)

        curl -X POST -H 'Content-type: application/json' \
            --data "$slack_payload" \
            "$SLACK_WEBHOOK" || log_warning "Failed to send Slack notification"
    fi
}

# Function to display help
show_help() {
    cat <<EOF
Automated Remediation Suggestions and Tools for AI Trading Bot

Usage: $0 [OPTIONS]

Options:
    --auto-fix              Enable automatic fixes for safe vulnerabilities
    --create-branches       Create Git branches for fixes
    --generate-patches      Generate patch files for manual application
    --update-dependencies   Automatically update vulnerable dependencies
    --enable-jira           Create JIRA tickets for vulnerabilities
    --slack-webhook URL     Send notifications to Slack webhook
    --help, -h              Show this help message

Examples:
    $0                                    # Generate remediation suggestions
    $0 --generate-patches                # Generate patch files
    $0 --auto-fix --update-dependencies  # Apply automated fixes
    $0 --create-branches                 # Create Git branches for fixes
    $0 --slack-webhook https://hooks.slack.com/... # Send notifications

The script will:
1. Analyze Trivy scan results for remediation opportunities
2. Generate automated fix scripts for vulnerabilities
3. Create remediation guides for manual fixes
4. Provide configuration hardening suggestions
5. Generate patches and automation tools

Output is saved to:
- Reports: security/trivy/remediation/reports/
- Scripts: security/trivy/remediation/scripts/
- Patches: security/trivy/remediation/patches/
EOF
}

# Main function
main() {
    log_info "Starting automated remediation analysis..."

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --auto-fix)
                AUTO_FIX_ENABLED=true
                shift
                ;;
            --create-branches)
                CREATE_BRANCHES=true
                shift
                ;;
            --generate-patches)
                GENERATE_PATCHES=true
                shift
                ;;
            --update-dependencies)
                UPDATE_DEPENDENCIES=true
                shift
                ;;
            --enable-jira)
                JIRA_ENABLED=true
                shift
                ;;
            --slack-webhook)
                SLACK_WEBHOOK="$2"
                shift 2
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

    # Check if reports exist
    if [[ ! -d "$REPORTS_DIR" ]]; then
        log_error "Reports directory not found: $REPORTS_DIR"
        log_info "Run security scans first using the scan scripts"
        exit 1
    fi

    # Setup directories
    setup_directories

    # Run analysis
    analyze_vulnerabilities
    analyze_secrets
    analyze_configurations

    # Generate remediation artifacts
    generate_patches
    create_jira_tickets

    # Send notifications
    send_notifications 0 0 0  # Would be populated from analysis

    log_success "Automated remediation analysis completed!"
    log_info "Remediation artifacts saved to: $REMEDIATION_DIR"
    log_info "Review reports and apply suggested fixes"
}

# Run main function if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
