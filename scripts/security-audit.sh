#!/bin/bash

# Security Audit Script for AI Trading Bot
# Run this script to check for common security issues

set -euo pipefail

echo "🔒 AI Trading Bot Security Audit"
echo "================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Counters
ISSUES_FOUND=0
WARNINGS_FOUND=0

# Function to report issues
report_issue() {
    echo -e "${RED}[ISSUE]${NC} $1"
    ((ISSUES_FOUND++))
}

# Function to report warnings
report_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
    ((WARNINGS_FOUND++))
}

# Function to report success
report_success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

# 1. Check for .env file in git
echo "1. Checking for sensitive files in git..."
if git ls-files | grep -E "^\.env$" > /dev/null 2>&1; then
    report_issue ".env file is tracked in git!"
else
    report_success ".env file is not tracked in git"
fi

# 2. Check for hardcoded secrets in code
echo ""
echo "2. Scanning for hardcoded secrets..."
SENSITIVE_PATTERNS=(
    "sk-[A-Za-z0-9]{48,}"
    "api[_-]?key.*=.*['\"][A-Za-z0-9]{20,}['\"]"
    "private[_-]?key.*=.*['\"][A-Za-z0-9]{32,}['\"]"
    "password.*=.*['\"][^'\"]+['\"]"
    "-----BEGIN.*PRIVATE KEY-----"
)

for pattern in "${SENSITIVE_PATTERNS[@]}"; do
    if grep -r -E "$pattern" --include="*.py" --include="*.yml" --include="*.yaml" --exclude-dir=".git" --exclude-dir="venv" --exclude-dir="__pycache__" . 2>/dev/null | grep -v "example\|template\|test\|mock"; then
        report_issue "Found potential hardcoded secret matching pattern: $pattern"
    fi
done

if [ $ISSUES_FOUND -eq 0 ]; then
    report_success "No hardcoded secrets detected"
fi

# 3. Check file permissions
echo ""
echo "3. Checking file permissions..."
if [ -f ".env" ]; then
    PERM=$(stat -f "%A" .env 2>/dev/null || stat -c "%a" .env 2>/dev/null)
    if [ "$PERM" != "600" ]; then
        report_warning ".env file has permissions $PERM (should be 600)"
    else
        report_success ".env file has secure permissions (600)"
    fi
fi

# 4. Check Docker security
echo ""
echo "4. Checking Docker configuration..."
if grep -q "/var/run/docker.sock" docker-compose*.yml 2>/dev/null | grep -v "#"; then
    report_issue "Docker socket is mounted in containers (security risk)"
else
    report_success "Docker socket is not mounted"
fi

if grep -q "privileged.*true" docker-compose*.yml 2>/dev/null; then
    report_issue "Privileged containers detected"
else
    report_success "No privileged containers"
fi

# 5. Check for debug mode
echo ""
echo "5. Checking for debug mode..."
if grep -r "DEBUG.*=.*True" --include="*.py" . 2>/dev/null | grep -v "test\|example"; then
    report_warning "Debug mode might be enabled in production code"
else
    report_success "No hardcoded debug mode found"
fi

# 6. Check dependencies for vulnerabilities
echo ""
echo "6. Checking dependencies..."
if command -v poetry &> /dev/null; then
    echo "Running safety check..."
    poetry run pip list --format freeze | poetry run safety check --stdin || report_warning "Some dependencies have known vulnerabilities"
else
    report_warning "Poetry not found, skipping dependency check"
fi

# 7. Check for exposed ports
echo ""
echo "7. Checking exposed ports..."
EXPOSED_PORTS=$(grep -h "ports:" -A 5 docker-compose*.yml 2>/dev/null | grep -E "^\s*-.*:" | grep -v "#" || true)
if [ -n "$EXPOSED_PORTS" ]; then
    echo "Exposed ports found:"
    echo "$EXPOSED_PORTS"
    report_warning "Review exposed ports for production deployment"
fi

# 8. Check logging configuration
echo ""
echo "8. Checking logging configuration..."
if grep -r "logger.*\(api[_-]?key\|password\|secret\)" --include="*.py" . 2>/dev/null | grep -v "test\|secure_logging"; then
    report_issue "Potential sensitive data in logs"
else
    report_success "No obvious sensitive data logging detected"
fi

# Summary
echo ""
echo "================================"
echo "Security Audit Summary:"
echo "--------------------------------"
echo -e "Issues found: ${RED}${ISSUES_FOUND}${NC}"
echo -e "Warnings found: ${YELLOW}${WARNINGS_FOUND}${NC}"

if [ $ISSUES_FOUND -eq 0 ] && [ $WARNINGS_FOUND -eq 0 ]; then
    echo -e "${GREEN}✅ No security issues found!${NC}"
    exit 0
else
    echo -e "${RED}⚠️  Please address the issues above before deployment${NC}"
    exit 1
fi