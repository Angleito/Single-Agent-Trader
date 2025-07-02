#!/bin/bash
# Local Security Scanning Script
# Run comprehensive security scans on the codebase and containers

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCAN_DIR="${1:-.}"
REPORT_DIR="./security-reports"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create report directory
mkdir -p "$REPORT_DIR"

# Function to print colored output
print_message() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Function to print section header
print_section() {
    echo
    print_message "$BLUE" "═══════════════════════════════════════════════════════════════"
    print_message "$BLUE" "  $1"
    print_message "$BLUE" "═══════════════════════════════════════════════════════════════"
    echo
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Main security scanning function
run_security_scans() {
    local failed_scans=0

    print_message "$GREEN" "Starting Security Scan Suite"
    print_message "$GREEN" "Scan Directory: $SCAN_DIR"
    print_message "$GREEN" "Report Directory: $REPORT_DIR"
    echo

    # 1. Python dependency vulnerabilities
    print_section "Python Dependency Security Scan"
    if command_exists poetry; then
        print_message "$YELLOW" "Running Safety check for known vulnerabilities..."
        poetry run pip install safety >/dev/null 2>&1 || true
        if poetry run safety check --json --output "$REPORT_DIR/safety-report-$TIMESTAMP.json" 2>/dev/null; then
            print_message "$GREEN" "✓ No known vulnerabilities found in dependencies"
        else
            print_message "$RED" "⚠ Vulnerabilities found in dependencies (see report)"
            ((failed_scans++))
        fi
    fi

    # 2. Static code security analysis
    print_section "Static Code Security Analysis (Bandit)"
    if command_exists bandit || poetry run bandit --version >/dev/null 2>&1; then
        print_message "$YELLOW" "Running Bandit security linter..."
        if poetry run bandit -r bot/ -f json -o "$REPORT_DIR/bandit-report-$TIMESTAMP.json" -ll 2>/dev/null; then
            print_message "$GREEN" "✓ No security issues found by Bandit"
        else
            print_message "$YELLOW" "⚠ Potential security issues found (see report)"
            poetry run bandit -r bot/ -ll 2>&1 | grep "Severity:" || true
        fi
    fi

    # 3. Secret detection
    print_section "Secret Detection Scan"

    # GitLeaks
    if command_exists gitleaks; then
        print_message "$YELLOW" "Running GitLeaks secret detection..."
        if gitleaks detect --source="$SCAN_DIR" --report-path="$REPORT_DIR/gitleaks-report-$TIMESTAMP.json" --report-format=json --no-git; then
            print_message "$GREEN" "✓ No secrets detected by GitLeaks"
        else
            print_message "$RED" "⚠ Potential secrets detected (see report)"
            ((failed_scans++))
        fi
    else
        print_message "$YELLOW" "GitLeaks not installed. Install with: brew install gitleaks"
    fi

    # detect-secrets
    if command_exists detect-secrets; then
        print_message "$YELLOW" "Running detect-secrets..."
        detect-secrets scan --baseline "$REPORT_DIR/secrets-baseline-$TIMESTAMP.json" >/dev/null 2>&1
        if detect-secrets audit "$REPORT_DIR/secrets-baseline-$TIMESTAMP.json" --report > "$REPORT_DIR/secrets-audit-$TIMESTAMP.txt" 2>&1; then
            print_message "$GREEN" "✓ No secrets detected by detect-secrets"
        else
            print_message "$YELLOW" "⚠ Review potential secrets in audit report"
        fi
    else
        print_message "$YELLOW" "detect-secrets not installed. Install with: pip install detect-secrets"
    fi

    # 4. Docker image scanning
    print_section "Container Security Scan"

    # Build images if not exists
    if [[ "$(docker images -q ai-trading-bot:latest 2> /dev/null)" == "" ]]; then
        print_message "$YELLOW" "Building Docker image for scanning..."
        docker build -t ai-trading-bot:latest . >/dev/null 2>&1
    fi

    # Trivy scan
    if command_exists trivy; then
        print_message "$YELLOW" "Running Trivy container vulnerability scan..."
        trivy image --format json --output "$REPORT_DIR/trivy-report-$TIMESTAMP.json" ai-trading-bot:latest >/dev/null 2>&1

        # Show summary
        CRITICAL=$(trivy image --severity CRITICAL --format json ai-trading-bot:latest 2>/dev/null | jq '.Results[].Vulnerabilities | length' | awk '{s+=$1} END {print s}')
        HIGH=$(trivy image --severity HIGH --format json ai-trading-bot:latest 2>/dev/null | jq '.Results[].Vulnerabilities | length' | awk '{s+=$1} END {print s}')

        if [[ "$CRITICAL" -gt 0 ]]; then
            print_message "$RED" "⚠ Found $CRITICAL CRITICAL vulnerabilities"
            ((failed_scans++))
        elif [[ "$HIGH" -gt 0 ]]; then
            print_message "$YELLOW" "⚠ Found $HIGH HIGH vulnerabilities"
        else
            print_message "$GREEN" "✓ No critical or high vulnerabilities found"
        fi
    else
        print_message "$YELLOW" "Trivy not installed. Install with: brew install aquasecurity/trivy/trivy"
    fi

    # 5. Dockerfile linting
    print_section "Dockerfile Security Linting"
    if command_exists hadolint; then
        print_message "$YELLOW" "Running Hadolint on Dockerfiles..."
        for dockerfile in $(find . -name "Dockerfile*" -type f); do
            if hadolint "$dockerfile" > "$REPORT_DIR/hadolint-$(basename $dockerfile)-$TIMESTAMP.txt" 2>&1; then
                print_message "$GREEN" "✓ $dockerfile passes security linting"
            else
                print_message "$YELLOW" "⚠ $dockerfile has linting warnings (see report)"
            fi
        done
    else
        print_message "$YELLOW" "Hadolint not installed. Install with: brew install hadolint"
    fi

    # 6. SAST with Semgrep
    print_section "SAST Analysis (Semgrep)"
    if command_exists semgrep; then
        print_message "$YELLOW" "Running Semgrep security patterns..."
        if semgrep --config=auto --json --output="$REPORT_DIR/semgrep-report-$TIMESTAMP.json" "$SCAN_DIR" 2>/dev/null; then
            print_message "$GREEN" "✓ Semgrep analysis complete"
        else
            print_message "$YELLOW" "⚠ Semgrep found potential issues (see report)"
        fi
    else
        print_message "$YELLOW" "Semgrep not installed. Install with: brew install semgrep"
    fi

    # 7. License compliance
    print_section "License Compliance Check"
    print_message "$YELLOW" "Checking license compliance..."
    poetry run pip-licenses --format=json --output-file="$REPORT_DIR/licenses-$TIMESTAMP.json" 2>/dev/null || true
    print_message "$GREEN" "✓ License report generated"

    # 8. Security configuration check
    print_section "Security Configuration Audit"
    print_message "$YELLOW" "Checking security configurations..."

    # Check for secure defaults
    local security_checks_passed=0
    local security_checks_total=0

    # Check 1: .env file should not exist in repo
    ((security_checks_total++))
    if [[ ! -f ".env" ]] || grep -q "^\.env$" .gitignore; then
        print_message "$GREEN" "✓ .env file properly gitignored"
        ((security_checks_passed++))
    else
        print_message "$RED" "✗ .env file not in .gitignore!"
    fi

    # Check 2: No hardcoded secrets in code
    ((security_checks_total++))
    if ! grep -r "sk-[a-zA-Z0-9]\{48\}" bot/ 2>/dev/null | grep -v "test" | grep -v "example"; then
        print_message "$GREEN" "✓ No hardcoded API keys found"
        ((security_checks_passed++))
    else
        print_message "$RED" "✗ Potential hardcoded API keys found!"
    fi

    # Check 3: Docker security options
    ((security_checks_total++))
    if grep -q "read_only: true" docker-compose*.yml; then
        print_message "$GREEN" "✓ Read-only root filesystem configured"
        ((security_checks_passed++))
    else
        print_message "$YELLOW" "⚠ Consider enabling read-only root filesystem"
    fi

    # Check 4: Non-root user in Dockerfile
    ((security_checks_total++))
    if grep -q "USER" Dockerfile; then
        print_message "$GREEN" "✓ Non-root user configured in Dockerfile"
        ((security_checks_passed++))
    else
        print_message "$RED" "✗ Running as root in container!"
    fi

    print_message "$BLUE" "\nSecurity configuration: $security_checks_passed/$security_checks_total checks passed"

    # Generate summary report
    print_section "Security Scan Summary"

    cat > "$REPORT_DIR/security-summary-$TIMESTAMP.md" << EOF
# Security Scan Report
**Date**: $(date)
**Directory**: $SCAN_DIR

## Scan Results Summary

| Scan Type | Status | Details |
|-----------|--------|---------|
| Dependency Vulnerabilities | $(if [[ -f "$REPORT_DIR/safety-report-$TIMESTAMP.json" ]]; then echo "✓ Completed"; else echo "⚠ Skipped"; fi) | See safety-report-$TIMESTAMP.json |
| Static Code Analysis | $(if [[ -f "$REPORT_DIR/bandit-report-$TIMESTAMP.json" ]]; then echo "✓ Completed"; else echo "⚠ Skipped"; fi) | See bandit-report-$TIMESTAMP.json |
| Secret Detection | $(if [[ -f "$REPORT_DIR/gitleaks-report-$TIMESTAMP.json" ]]; then echo "✓ Completed"; else echo "⚠ Skipped"; fi) | See gitleaks-report-$TIMESTAMP.json |
| Container Vulnerabilities | $(if [[ -f "$REPORT_DIR/trivy-report-$TIMESTAMP.json" ]]; then echo "✓ Completed"; else echo "⚠ Skipped"; fi) | See trivy-report-$TIMESTAMP.json |
| Dockerfile Linting | $(if ls "$REPORT_DIR"/hadolint-*-$TIMESTAMP.txt >/dev/null 2>&1; then echo "✓ Completed"; else echo "⚠ Skipped"; fi) | See hadolint-*-$TIMESTAMP.txt |
| SAST Analysis | $(if [[ -f "$REPORT_DIR/semgrep-report-$TIMESTAMP.json" ]]; then echo "✓ Completed"; else echo "⚠ Skipped"; fi) | See semgrep-report-$TIMESTAMP.json |
| License Compliance | $(if [[ -f "$REPORT_DIR/licenses-$TIMESTAMP.json" ]]; then echo "✓ Completed"; else echo "⚠ Skipped"; fi) | See licenses-$TIMESTAMP.json |

## Security Configuration
- Security checks passed: $security_checks_passed/$security_checks_total
- Failed vulnerability scans: $failed_scans

## Recommendations
1. Review all reports in $REPORT_DIR
2. Address any CRITICAL and HIGH vulnerabilities immediately
3. Update dependencies regularly
4. Enable all security features in production
5. Run this scan before each deployment

## Report Files
All detailed reports are available in: $REPORT_DIR
EOF

    print_message "$GREEN" "\n✓ Security scan complete!"
    print_message "$GREEN" "Summary report: $REPORT_DIR/security-summary-$TIMESTAMP.md"

    if [[ $failed_scans -gt 0 ]]; then
        print_message "$RED" "\n⚠ WARNING: $failed_scans security scans found issues!"
        return 1
    else
        print_message "$GREEN" "\n✓ All security scans passed!"
        return 0
    fi
}

# Function to install missing tools
install_security_tools() {
    print_section "Security Tools Installation"

    print_message "$YELLOW" "Checking for required security tools..."

    local tools_to_install=()

    # Check each tool
    command_exists gitleaks || tools_to_install+=("gitleaks")
    command_exists trivy || tools_to_install+=("aquasecurity/trivy/trivy")
    command_exists hadolint || tools_to_install+=("hadolint")
    command_exists semgrep || tools_to_install+=("semgrep")

    if [[ ${#tools_to_install[@]} -eq 0 ]]; then
        print_message "$GREEN" "✓ All security tools are installed"
    else
        print_message "$YELLOW" "Missing tools: ${tools_to_install[*]}"
        echo
        echo "Install with Homebrew (macOS/Linux):"
        for tool in "${tools_to_install[@]}"; do
            echo "  brew install $tool"
        done
        echo
        echo "Or install with package manager of your choice"
    fi

    # Python tools
    print_message "$YELLOW" "\nChecking Python security tools..."
    poetry run pip install safety detect-secrets bandit pip-licenses >/dev/null 2>&1
    print_message "$GREEN" "✓ Python security tools installed"
}

# Main execution
main() {
    print_message "$GREEN" "AI Trading Bot Security Scanner"
    print_message "$GREEN" "==============================="

    # Parse arguments
    case "${1:-scan}" in
        install)
            install_security_tools
            ;;
        scan|*)
            run_security_scans
            ;;
    esac
}

# Run main function
main "$@"
