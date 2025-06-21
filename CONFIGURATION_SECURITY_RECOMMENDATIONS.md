# Configuration Security Recommendations

## 🚨 IMMEDIATE SECURITY ACTIONS REQUIRED

### 1. API Key Security (CRITICAL)
The following API keys were found in local `.env` files and must be rotated immediately:

```bash
# Found in .env file - ROTATE THESE KEYS NOW:
TAVILY_API_KEY=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
PERPLEXITY_API_KEY=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
FIRECRAWL_API_KEY=fc-5ae56c3903cc47a19217e66a28d2f32d
BLUEFIN_SERVICE_API_KEY=trading-bot-secret
```

**Action Plan:**
1. Immediately rotate these API keys on their respective platforms:
   - [Tavily API Dashboard](https://app.tavily.com/)
   - [Perplexity API Settings](https://www.perplexity.ai/settings/api)
   - [Firecrawl Dashboard](https://www.firecrawl.dev/)
2. Update `.env` file with new keys
3. Audit any production deployments that may be using these keys

### 2. Environment File Management
**Current Status:** ✅ Good - `.env` files are properly excluded from git

**Verification:**
```bash
# Verify .env files are ignored
git ls-files | grep -E "\\.env"  # Should return nothing
```

## 🔒 SECURITY BEST PRACTICES IMPLEMENTATION

### 1. Implement Secrets Management

Create a secure secrets management workflow:

```bash
# Example: Using environment-specific secrets
cp .env.example .env.development
cp .env.example .env.production
cp .env.example .env.staging

# Secure file permissions
chmod 600 .env*
chown $USER:$USER .env*
```

### 2. API Key Validation Script

```python
#!/usr/bin/env python3
"""Validate API keys are not placeholder values"""

import os
import re
from pathlib import Path

def validate_api_keys():
    """Check for placeholder or weak API keys."""
    env_file = Path(".env")
    if not env_file.exists():
        print("❌ No .env file found")
        return False

    dangerous_patterns = [
        r'API_KEY=(your_|sk-proj-|your-)',
        r'PRIVATE_KEY=(your_|0x123)',
        r'SECRET=(your_|secret|password)',
        r'API_KEY=\s*$',  # Empty values
    ]

    content = env_file.read_text()
    for pattern in dangerous_patterns:
        if re.search(pattern, content):
            print(f"⚠️  Found placeholder or weak key: {pattern}")
            return False

    print("✅ API key validation passed")
    return True

if __name__ == "__main__":
    validate_api_keys()
```

### 3. Production Environment Hardening

For production environments, implement these additional security measures:

```json
{
  "security": {
    "enable_api_key_rotation": true,
    "api_key_rotation_days": 30,
    "enable_secure_headers": true,
    "enable_rate_limiting": true,
    "rate_limit_per_minute": 100,
    "enable_audit_logging": true,
    "max_login_attempts": 5,
    "login_lockout_duration": 300
  }
}
```

### 4. Container Security

When running in Docker, use these security practices:

```dockerfile
# Run as non-root user
USER 1000:1000

# Minimal base image
FROM python:3.12-slim

# No secrets in build args
ARG BUILD_VERSION
ENV BUILD_VERSION=${BUILD_VERSION}

# Secrets via environment or mounted volumes only
# NEVER: ENV API_KEY=secret_value
```

## ⚡ CONFIGURATION VALIDATION PIPELINE

### 1. Pre-commit Hook Setup

```bash
# Install pre-commit validation
cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
echo "🔍 Validating configuration files..."

# Check for secrets in staged files
if git diff --cached --name-only | xargs grep -l "API_KEY.*[a-zA-Z0-9]{20}" 2>/dev/null; then
    echo "❌ Potential API key found in staged files"
    exit 1
fi

# Validate JSON config files
python3 scripts/validate_config_comprehensive.py --quiet
if [ $? -ne 0 ]; then
    echo "❌ Configuration validation failed"
    exit 1
fi

echo "✅ Configuration validation passed"
EOF

chmod +x .git/hooks/pre-commit
```

### 2. CI/CD Security Validation

```yaml
# .github/workflows/security-validation.yml
name: Security Validation
on: [push, pull_request]

jobs:
  security-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Check for secrets
        run: |
          # Fail if any files contain real API keys
          if grep -r "sk-[a-zA-Z0-9]{48}" . --exclude-dir=.git; then
            echo "❌ OpenAI API key found in repository"
            exit 1
          fi

      - name: Validate configurations
        run: |
          python3 scripts/validate_config_comprehensive.py
```

## 🛡️ RUNTIME SECURITY MONITORING

### 1. Configuration Drift Detection

Implement monitoring to detect configuration changes:

```python
def monitor_config_changes():
    """Monitor for unauthorized configuration changes."""
    import hashlib
    import json

    # Calculate config file hashes
    config_hashes = {}
    for config_file in Path("config").glob("*.json"):
        content = config_file.read_text()
        config_hashes[str(config_file)] = hashlib.sha256(content.encode()).hexdigest()

    # Store in secure location and monitor for changes
    return config_hashes
```

### 2. API Key Usage Monitoring

Track API key usage to detect anomalies:

```python
class APIKeyMonitor:
    """Monitor API key usage patterns."""

    def __init__(self):
        self.usage_patterns = {}

    def log_api_usage(self, key_name: str, endpoint: str, timestamp: float):
        """Log API usage for pattern analysis."""
        if key_name not in self.usage_patterns:
            self.usage_patterns[key_name] = []

        self.usage_patterns[key_name].append({
            "endpoint": endpoint,
            "timestamp": timestamp,
            "ip": self.get_client_ip()
        })

    def detect_anomalies(self) -> List[str]:
        """Detect unusual API usage patterns."""
        anomalies = []

        for key_name, usage in self.usage_patterns.items():
            # Check for unusual volume
            recent_usage = [u for u in usage if time.time() - u["timestamp"] < 3600]
            if len(recent_usage) > 1000:  # More than 1000 calls per hour
                anomalies.append(f"High usage detected for {key_name}")

        return anomalies
```

## 🎯 RECOMMENDED SECURITY ARCHITECTURE

### 1. Multi-Environment Setup

```
environments/
├── development/
│   ├── .env.development
│   └── config/
├── staging/
│   ├── .env.staging
│   └── config/
└── production/
    ├── .env.production (secrets manager)
    └── config/
```

### 2. Secrets Management Integration

```python
# Example: AWS Secrets Manager integration
import boto3

class SecretsManager:
    """Secure secrets management."""

    def __init__(self, environment: str):
        self.environment = environment
        self.client = boto3.client('secretsmanager')

    def get_secret(self, secret_name: str) -> str:
        """Retrieve secret from AWS Secrets Manager."""
        try:
            response = self.client.get_secret_value(
                SecretId=f"trading-bot/{self.environment}/{secret_name}"
            )
            return response['SecretString']
        except Exception as e:
            logger.error(f"Failed to retrieve secret {secret_name}: {e}")
            raise

    def rotate_api_key(self, key_name: str) -> str:
        """Rotate API key and update secret."""
        # Implementation depends on the API provider
        pass
```

## 📊 SECURITY COMPLIANCE CHECKLIST

### Development Environment
- [ ] All `.env` files excluded from version control
- [ ] API keys are not placeholder values
- [ ] File permissions are restrictive (600)
- [ ] Pre-commit hooks validate configurations
- [ ] Local secrets are documented and tracked

### Production Environment
- [ ] Secrets stored in dedicated secrets manager
- [ ] API keys rotated regularly (30-90 days)
- [ ] Configuration changes require approval
- [ ] Audit logging enabled for all configuration access
- [ ] Network access restricted to necessary endpoints

### Monitoring & Alerting
- [ ] Configuration drift monitoring active
- [ ] API usage anomaly detection implemented
- [ ] Security incident response plan documented
- [ ] Regular security audits scheduled
- [ ] Backup and recovery procedures tested

## 🚀 IMPLEMENTATION TIMELINE

### Week 1: Immediate Security
- [ ] Rotate exposed API keys
- [ ] Implement secrets validation script
- [ ] Set up pre-commit hooks
- [ ] Document current security posture

### Week 2: Process Hardening
- [ ] Implement configuration validation pipeline
- [ ] Set up environment-specific secrets management
- [ ] Create security monitoring dashboards
- [ ] Establish key rotation procedures

### Week 3: Advanced Security
- [ ] Integrate with enterprise secrets manager
- [ ] Implement runtime security monitoring
- [ ] Set up security alerting
- [ ] Create incident response procedures

### Week 4: Compliance & Audit
- [ ] Complete security audit
- [ ] Document security procedures
- [ ] Train team on security practices
- [ ] Establish ongoing security review process

---

*Generated by Configuration Security Analysis Tool*
*Last Updated: 2025-06-21*
