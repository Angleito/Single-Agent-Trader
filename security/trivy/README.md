# Trivy Security Scanning Pipeline for AI Trading Bot

This directory contains a comprehensive Trivy-based security scanning pipeline designed specifically for the AI Trading Bot project. The pipeline provides automated vulnerability scanning, security gates, reporting, and remediation guidance.

## 🎯 Overview

The Trivy security pipeline includes:

- **Multi-layered vulnerability scanning** (Docker images, filesystem, configurations)
- **Software Bill of Materials (SBOM)** generation for compliance
- **Secret detection** in source code and containers
- **Configuration security** analysis (Docker, Kubernetes, etc.)
- **CI/CD integration** with security gates and automated blocking
- **Comprehensive reporting** with visual dashboards
- **Automated remediation** suggestions and tools

## 📁 Directory Structure

```
security/trivy/
├── README.md                     # This file
├── trivy-config.yaml            # Trivy configuration
├── install-trivy.sh             # Trivy installation script
├── scan-images.sh               # Docker image vulnerability scanner
├── scan-filesystem.sh           # Filesystem and source code scanner
├── ci-cd-security-gate.sh       # CI/CD integration and security gates
├── security-dashboard.py        # Security metrics dashboard generator
├── remediation-automation.sh    # Automated remediation suggestions
├── run-complete-scan.sh         # Master script to run all scans
└── reports/                     # Scan results and reports
    ├── images/                  # Docker image scan results
    ├── filesystem/              # Filesystem scan results
    ├── secrets/                 # Secret detection results
    ├── configs/                 # Configuration scan results
    ├── licenses/                # License compliance results
    ├── json/                    # Machine-readable JSON reports
    ├── sarif/                   # SARIF format for CI/CD integration
    ├── html/                    # Human-readable HTML reports
    ├── sbom/                    # Software Bill of Materials
    ├── ci/                      # CI/CD specific reports
    ├── dashboard/               # Generated dashboards
    └── remediation/             # Remediation guides and scripts
```

## 🚀 Quick Start

### 1. Install Trivy

```bash
# Install Trivy on your system
./install-trivy.sh

# Or install with server mode (Linux only)
./install-trivy.sh --server
```

### 2. Run Complete Security Scan

```bash
# Run all security scans
./run-complete-scan.sh

# Run specific scan types
./scan-images.sh --all                    # Docker images only
./scan-filesystem.sh                      # Filesystem only
./scan-images.sh --format json           # JSON output
```

### 3. Generate Security Dashboard

```bash
# Generate interactive dashboard
python3 security-dashboard.py

# Or specify custom paths
python3 security-dashboard.py --reports-dir ./reports --output-dir ./dashboard
```

### 4. Get Remediation Suggestions

```bash
# Analyze results and get remediation guidance
./remediation-automation.sh

# Generate patches and automated fixes
./remediation-automation.sh --generate-patches --auto-fix
```

## 🔧 Individual Tools

### Docker Image Scanning

Scans Docker images for vulnerabilities, secrets, and misconfigurations:

```bash
# Scan all trading bot images
./scan-images.sh --all

# Scan specific image with SBOM generation
./scan-images.sh ai-trading-bot:latest --format json

# Scan with custom severity levels
./scan-images.sh --severity CRITICAL,HIGH --exit-on-vuln

# Send notifications
./scan-images.sh --slack-webhook https://hooks.slack.com/services/...
```

**Features:**
- Multi-format output (table, JSON, SARIF, HTML)
- SBOM generation (CycloneDX format)
- Secret detection in images
- Configuration analysis
- License scanning
- Automatic vulnerability database updates

### Filesystem Scanning

Scans source code, dependencies, and configuration files:

```bash
# Complete filesystem scan
./scan-filesystem.sh

# Scan specific types only
./scan-filesystem.sh --no-licenses --no-configs

# Custom exclusions
./scan-filesystem.sh --exclude-dirs ".git,venv,node_modules"

# Exit on vulnerabilities found
./scan-filesystem.sh --severity CRITICAL,HIGH --exit-on-vuln
```

**Features:**
- Dependency vulnerability scanning
- Secret detection in source code
- Configuration misconfiguration analysis
- License compliance checking
- Recursive directory scanning with exclusions

### CI/CD Security Gates

Integrates with CI/CD pipelines to enforce security policies:

```bash
# Run security gate with default policies
./ci-cd-security-gate.sh

# Custom thresholds
./ci-cd-security-gate.sh --max-critical 0 --max-high 5 --max-medium 20

# Different failure policies
./ci-cd-security-gate.sh --no-fail-secrets --fail-misconfig

# Integration with notifications
./ci-cd-security-gate.sh --slack-webhook $SLACK_WEBHOOK --teams-webhook $TEAMS_WEBHOOK
```

**CI/CD Integration:**
- GitHub Actions (included workflow)
- GitLab CI/CD
- Jenkins
- Azure Pipelines
- CircleCI

**Security Gates:**
- Zero critical vulnerabilities
- Limited high/medium vulnerabilities
- No secrets in codebase
- Configuration compliance
- License policy adherence

### Security Dashboard

Generates comprehensive security dashboards and metrics:

```bash
# Generate complete dashboard
python3 security-dashboard.py

# JSON metrics only
python3 security-dashboard.py --format json

# Custom input/output directories
python3 security-dashboard.py --reports-dir ./custom-reports --output-dir ./dashboard
```

**Dashboard Features:**
- Vulnerability analysis charts
- Security trends over time
- Compliance scoring
- Interactive HTML reports
- Machine-readable JSON metrics
- Executive summary reports

### Remediation Automation

Provides automated remediation suggestions and tools:

```bash
# Generate remediation suggestions
./remediation-automation.sh

# Create automated fix scripts
./remediation-automation.sh --generate-patches

# Enable automatic fixes for safe vulnerabilities
./remediation-automation.sh --auto-fix --update-dependencies

# Create JIRA tickets
./remediation-automation.sh --enable-jira

# Notifications
./remediation-automation.sh --slack-webhook $SLACK_WEBHOOK
```

**Remediation Features:**
- Automated vulnerability fix scripts
- Secret removal guidance
- Configuration hardening patches
- Dependency update automation
- JIRA ticket generation
- Git branch creation for fixes

## ⚙️ Configuration

### Trivy Configuration

The main configuration is in `trivy-config.yaml`:

```yaml
# Vulnerability scanning
vulnerability:
  severity: ["CRITICAL", "HIGH", "MEDIUM"]
  exit-code: 1

# Secret detection rules
secret:
  config: |
    rules:
      - id: "trading-api-keys"
        regex: '(?i)(coinbase|bluefin)[\w\-_]*[\s=:]+["\']?([a-z0-9]{32,})["\']?'
      - id: "openai-api-key"
        regex: 'sk-[a-zA-Z0-9]{48}'

# Compliance standards
compliance:
  spec:
    - "docker-cis"
    - "k8s-cis"
```

### Environment Variables

```bash
# Security gate thresholds
export MAX_CRITICAL_VULNS=0
export MAX_HIGH_VULNS=5
export MAX_MEDIUM_VULNS=20

# Notification settings
export SLACK_WEBHOOK="https://hooks.slack.com/services/..."
export TEAMS_WEBHOOK="https://outlook.office.com/webhook/..."

# Feature flags
export ENABLE_AUTO_FIX=false
export ENABLE_SBOM_GENERATION=true
export ENABLE_SECRET_SCANNING=true
```

## 🔗 CI/CD Integration

### GitHub Actions

The pipeline includes a comprehensive GitHub Actions workflow (`.github/workflows/security-scan.yml`):

```yaml
name: Security Scan
on: [push, pull_request, schedule]

jobs:
  container-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run Trivy container scan
        run: ./security/trivy/scan-images.sh --all --format sarif
      - name: Upload SARIF results
        uses: github/codeql-action/upload-sarif@v3
```

**Features:**
- Automated scanning on push/PR
- SARIF upload to GitHub Security tab
- Security gate enforcement
- Artifact generation and storage
- PR comments with results
- Compliance reporting

### GitLab CI/CD

```yaml
security_scan:
  stage: security
  script:
    - ./security/trivy/ci-cd-security-gate.sh
  artifacts:
    reports:
      sast: security/trivy/reports/sarif/*.sarif
    paths:
      - security/trivy/reports/
```

### Jenkins Pipeline

```groovy
pipeline {
    agent any
    stages {
        stage('Security Scan') {
            steps {
                sh './security/trivy/ci-cd-security-gate.sh'
                publishHTML([
                    allowMissing: false,
                    alwaysLinkToLastBuild: true,
                    keepAll: true,
                    reportDir: 'security/trivy/dashboard',
                    reportFiles: 'security_dashboard.html',
                    reportName: 'Security Dashboard'
                ])
            }
        }
    }
}
```

## 📊 Security Metrics

The pipeline generates comprehensive security metrics:

### Vulnerability Metrics
- Total vulnerabilities by severity
- Trends over time
- Package-specific vulnerability counts
- CVSS score distributions
- Fix availability statistics

### Compliance Metrics
- Security policy adherence
- Configuration compliance scores
- License compliance status
- Secret detection results
- Remediation progress tracking

### Performance Metrics
- Scan execution times
- Database update frequency
- False positive rates
- Coverage statistics

## 🛡️ Security Policies

### Default Security Gates

1. **Zero Critical Vulnerabilities**: No critical severity vulnerabilities allowed
2. **Limited High Vulnerabilities**: Maximum 5 high severity vulnerabilities
3. **No Secrets**: Zero tolerance for hardcoded secrets
4. **Configuration Compliance**: Docker and infrastructure hardening
5. **License Compliance**: Approved licenses only

### Customization

```bash
# Relaxed policy for development
./ci-cd-security-gate.sh --max-critical 2 --max-high 10 --no-fail-secrets

# Strict policy for production
./ci-cd-security-gate.sh --max-critical 0 --max-high 0 --fail-misconfig
```

## 🔧 Troubleshooting

### Common Issues

1. **Trivy Database Update Failures**
   ```bash
   # Manual database update
   trivy image --download-db-only
   ```

2. **Permission Errors**
   ```bash
   # Fix permissions
   sudo chown -R $USER:$USER security/trivy/reports
   chmod +x security/trivy/*.sh
   ```

3. **Missing Dependencies**
   ```bash
   # Install required tools
   pip install matplotlib pandas seaborn jinja2
   sudo apt-get install jq curl
   ```

4. **Large Report Files**
   ```bash
   # Use compression
   find security/trivy/reports -name "*.json" -exec gzip {} \;
   ```

### Debug Mode

```bash
# Enable debug output
export TRIVY_DEBUG=true
export LOG_LEVEL=DEBUG

# Run with verbose output
./scan-images.sh --all 2>&1 | tee debug.log
```

## 📚 Additional Resources

- [Trivy Documentation](https://aquasecurity.github.io/trivy/)
- [Docker Security Best Practices](https://docs.docker.com/engine/security/)
- [OWASP Container Security](https://owasp.org/www-project-container-security/)
- [CIS Benchmarks](https://www.cisecurity.org/cis-benchmarks/)
- [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework)

## 🤝 Contributing

To contribute to the security pipeline:

1. Test changes in a development environment
2. Validate with multiple scan targets
3. Update documentation and examples
4. Follow security best practices
5. Add appropriate test coverage

## 📄 License

This security pipeline is part of the AI Trading Bot project and follows the same licensing terms.

---

For questions or support, please refer to the main project documentation or create an issue in the project repository.
