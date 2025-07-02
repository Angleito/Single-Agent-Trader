# VPS and Docker Security Hardening Guide

## Table of Contents
1. [VPS-Level Security](#vps-level-security)
2. [Docker & Docker Compose Security](#docker--docker-compose-security)
3. [Secrets Management](#secrets-management)
4. [Network Security](#network-security)
5. [Monitoring & Logging](#monitoring--logging)
6. [Backup & Recovery](#backup--recovery)
7. [Security Checklist](#security-checklist)

---

## VPS-Level Security

### 1. Initial Server Setup

#### Create Non-Root User
```bash
# As root user
adduser cryptobot
usermod -aG sudo cryptobot

# Copy SSH keys to new user
rsync --archive --chown=cryptobot:cryptobot ~/.ssh /home/cryptobot
```

#### SSH Hardening
Edit `/etc/ssh/sshd_config`:
```bash
# Disable root login
PermitRootLogin no

# Disable password authentication
PasswordAuthentication no
PubkeyAuthentication yes

# Change default SSH port
Port 2222

# Limit users who can SSH
AllowUsers cryptobot

# Other security settings
Protocol 2
MaxAuthTries 3
ClientAliveInterval 300
ClientAliveCountMax 2
X11Forwarding no
AllowAgentForwarding no
PermitEmptyPasswords no
```

Restart SSH:
```bash
sudo systemctl restart sshd
```

### 2. Firewall Configuration

#### Using UFW (Uncomplicated Firewall)
```bash
# Install UFW
sudo apt-get update
sudo apt-get install ufw

# Default policies
sudo ufw default deny incoming
sudo ufw default allow outgoing

# Allow SSH on custom port
sudo ufw allow 2222/tcp

# Allow HTTP/HTTPS (if needed for monitoring)
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp

# Allow Docker Swarm ports (if using)
# sudo ufw allow 2377/tcp
# sudo ufw allow 7946/tcp
# sudo ufw allow 7946/udp
# sudo ufw allow 4789/udp

# Enable firewall
sudo ufw enable
```

### 3. Automatic Security Updates

```bash
# Install unattended-upgrades
sudo apt-get install unattended-upgrades

# Configure automatic updates
sudo dpkg-reconfigure --priority=low unattended-upgrades

# Edit /etc/apt/apt.conf.d/50unattended-upgrades
# Enable security updates only
```

### 4. Fail2ban Installation

```bash
# Install Fail2ban
sudo apt-get install fail2ban

# Create local config
sudo cp /etc/fail2ban/jail.conf /etc/fail2ban/jail.local

# Edit /etc/fail2ban/jail.local
[DEFAULT]
bantime = 3600
findtime = 600
maxretry = 3

[sshd]
enabled = true
port = 2222
filter = sshd
logpath = /var/log/auth.log
maxretry = 3
```

### 5. System Monitoring Tools

```bash
# Install monitoring tools
sudo apt-get install -y htop iotop nethogs

# Install security audit tools
sudo apt-get install -y lynis rkhunter

# Run security audit
sudo lynis audit system
```

---

## Docker & Docker Compose Security

### 1. Docker Installation Security

```bash
# Use official Docker repository only
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-compose-plugin
```

### 2. Docker Daemon Configuration

Create `/etc/docker/daemon.json`:
```json
{
  "icc": false,
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  },
  "userland-proxy": false,
  "disable-legacy-registry": true,
  "live-restore": true,
  "no-new-privileges": true,
  "seccomp-profile": "default",
  "userns-remap": "default"
}
```

### 3. Docker Compose Security Best Practices

#### Secure docker-compose.yml
```yaml
version: '3.8'

services:
  ai-trading-bot:
    image: ai-trading-bot:latest
    # Run as non-root user
    user: "1000:1000"
    
    # Read-only root filesystem
    read_only: true
    
    # Security options
    security_opt:
      - no-new-privileges:true
      - seccomp:unconfined
    
    # Resource limits
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 2G
        reservations:
          cpus: '1'
          memory: 1G
    
    # Health check
    healthcheck:
      test: ["CMD", "python", "-c", "import sys; sys.exit(0)"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    
    # Mount required volumes
    volumes:
      - ./logs:/app/logs:rw
      - ./data:/app/data:rw
      - /tmp:/tmp:rw  # For temporary files
    
    # Use secrets instead of environment variables
    secrets:
      - coinbase_api_key
      - coinbase_private_key
      - openai_api_key
    
    # Network isolation
    networks:
      - internal
    
    # Drop all capabilities and add only required ones
    cap_drop:
      - ALL
    cap_add:
      - NET_BIND_SERVICE

secrets:
  coinbase_api_key:
    file: ./secrets/coinbase_api_key.txt
  coinbase_private_key:
    file: ./secrets/coinbase_private_key.txt
  openai_api_key:
    file: ./secrets/openai_api_key.txt

networks:
  internal:
    driver: bridge
    internal: true
```

### 4. Dockerfile Security

```dockerfile
# Use specific version tags, not latest
FROM python:3.12-slim-bookworm

# Create non-root user
RUN groupadd -r cryptobot && useradd -r -g cryptobot cryptobot

# Install security updates
RUN apt-get update && apt-get upgrade -y && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=cryptobot:cryptobot . .

# Switch to non-root user
USER cryptobot

# Use ENTRYPOINT instead of CMD
ENTRYPOINT ["python", "-m", "bot.main"]
```

### 5. Container Image Scanning

```bash
# Install Trivy for vulnerability scanning
wget -qO - https://aquasecurity.github.io/trivy-repo/deb/public.key | sudo apt-key add -
echo "deb https://aquasecurity.github.io/trivy-repo/deb $(lsb_release -sc) main" | sudo tee -a /etc/apt/sources.list.d/trivy.list
sudo apt-get update
sudo apt-get install trivy

# Scan your images
trivy image ai-trading-bot:latest

# Use Docker Scout (built into Docker)
docker scout cves ai-trading-bot:latest
```

---

## Secrets Management

### 1. Docker Secrets Setup

```bash
# Create secrets directory
mkdir -p ./secrets
chmod 700 ./secrets

# Create secret files (DO NOT commit these!)
echo "your_coinbase_api_key" > ./secrets/coinbase_api_key.txt
echo "your_coinbase_private_key" > ./secrets/coinbase_private_key.txt
echo "your_openai_api_key" > ./secrets/openai_api_key.txt

# Set proper permissions
chmod 600 ./secrets/*.txt
```

### 2. Application Code for Reading Secrets

```python
# bot/config.py modifications
import os

def read_docker_secret(secret_name):
    """Read secret from Docker secret file or environment variable"""
    secret_path = f"/run/secrets/{secret_name}"
    if os.path.exists(secret_path):
        with open(secret_path, 'r') as f:
            return f.read().strip()
    # Fallback to environment variable for development
    return os.getenv(secret_name.upper())

# Usage
COINBASE_API_KEY = read_docker_secret('coinbase_api_key')
COINBASE_PRIVATE_KEY = read_docker_secret('coinbase_private_key')
OPENAI_API_KEY = read_docker_secret('openai_api_key')
```

### 3. External Secrets Management (Optional)

For production, consider using:
- **HashiCorp Vault**: Enterprise-grade secrets management
- **AWS Secrets Manager**: If using AWS infrastructure
- **DigitalOcean Spaces**: For encrypted secret storage

---

## Network Security

### 1. Docker Network Isolation

```yaml
# docker-compose.yml
networks:
  frontend:
    driver: bridge
  backend:
    driver: bridge
    internal: true  # No external access

services:
  ai-trading-bot:
    networks:
      - backend
  
  nginx:
    networks:
      - frontend
      - backend
```

### 2. DigitalOcean Cloud Firewall

Configure via DigitalOcean Control Panel:
- Create firewall rules
- Apply to your Droplet
- Restrict inbound traffic to only necessary ports

### 3. VPN Setup (Optional)

For additional security, set up WireGuard VPN:
```bash
# Install WireGuard
sudo apt-get install wireguard

# Generate keys
wg genkey | tee privatekey | wg pubkey > publickey

# Configure /etc/wireguard/wg0.conf
```

---

## Monitoring & Logging

### 1. Docker Logging Configuration

```yaml
# docker-compose.yml
services:
  ai-trading-bot:
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
        labels: "service=ai-trading-bot"
```

### 2. Log Aggregation with Promtail/Loki

```yaml
# docker-compose.monitoring.yml
services:
  promtail:
    image: grafana/promtail:latest
    volumes:
      - ./promtail-config.yml:/etc/promtail/config.yml:ro
      - /var/log:/var/log:ro
      - /var/lib/docker/containers:/var/lib/docker/containers:ro
    command: -config.file=/etc/promtail/config.yml
    
  loki:
    image: grafana/loki:latest
    ports:
      - "3100:3100"
    volumes:
      - ./loki-config.yml:/etc/loki/local-config.yaml
```

### 3. Security Monitoring

```bash
# Install AIDE (Advanced Intrusion Detection Environment)
sudo apt-get install aide
sudo aideinit

# Install auditd
sudo apt-get install auditd
sudo systemctl enable auditd

# Configure audit rules
sudo auditctl -w /etc/passwd -p wa -k passwd_changes
sudo auditctl -w /etc/docker/ -p wa -k docker_changes
```

### 4. Alerts and Notifications

```python
# bot/monitoring/alerts.py
import requests
import logging

class SecurityAlertManager:
    def __init__(self, webhook_url):
        self.webhook_url = webhook_url
        
    def send_alert(self, message, severity="warning"):
        """Send security alert to webhook"""
        payload = {
            "text": f"🚨 Security Alert [{severity.upper()}]: {message}",
            "severity": severity
        }
        try:
            requests.post(self.webhook_url, json=payload)
        except Exception as e:
            logging.error(f"Failed to send alert: {e}")
```

---

## Backup & Recovery

### 1. Automated Backups

```bash
#!/bin/bash
# backup.sh
BACKUP_DIR="/backup"
DATE=$(date +%Y%m%d_%H%M%S)

# Backup Docker volumes
docker run --rm \
  -v ai-trading-bot_data:/data \
  -v $BACKUP_DIR:/backup \
  alpine tar czf /backup/data_$DATE.tar.gz -C / data

# Backup configuration
tar czf $BACKUP_DIR/config_$DATE.tar.gz \
  docker-compose.yml \
  .env \
  config/

# Upload to DigitalOcean Spaces or S3
# s3cmd put $BACKUP_DIR/*_$DATE.tar.gz s3://your-backup-bucket/

# Clean old backups (keep last 7 days)
find $BACKUP_DIR -name "*.tar.gz" -mtime +7 -delete
```

### 2. Backup Cron Job

```bash
# Add to crontab
0 2 * * * /home/cryptobot/backup.sh >> /var/log/backup.log 2>&1
```

### 3. Disaster Recovery Plan

1. **Regular Testing**: Test restore procedures monthly
2. **Documentation**: Keep recovery procedures documented
3. **Off-site Backups**: Use DigitalOcean Spaces or S3
4. **Encryption**: Encrypt backups before uploading

---

## Security Checklist

### Initial Setup
- [ ] Created non-root user
- [ ] Configured SSH key authentication
- [ ] Disabled root login
- [ ] Changed default SSH port
- [ ] Configured UFW firewall
- [ ] Installed and configured Fail2ban
- [ ] Enabled automatic security updates
- [ ] Run initial security audit with Lynis

### Docker Security
- [ ] Using official Docker images
- [ ] Running containers as non-root
- [ ] Implemented resource limits
- [ ] Configured read-only root filesystem
- [ ] Dropped unnecessary capabilities
- [ ] Implemented health checks
- [ ] Regular vulnerability scanning with Trivy
- [ ] Using Docker secrets for sensitive data

### Network Security
- [ ] Configured Docker network isolation
- [ ] Set up DigitalOcean Cloud Firewall
- [ ] Restricted inbound traffic
- [ ] Implemented internal networks

### Monitoring
- [ ] Configured centralized logging
- [ ] Set up intrusion detection (AIDE)
- [ ] Configured audit logging
- [ ] Implemented security alerts

### Backup & Recovery
- [ ] Automated backup scripts
- [ ] Off-site backup storage
- [ ] Tested restore procedures
- [ ] Encrypted backup files

### Ongoing Maintenance
- [ ] Weekly security updates
- [ ] Monthly vulnerability scans
- [ ] Quarterly security audits
- [ ] Annual penetration testing

---

## Additional Resources

- [OWASP Docker Security Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Docker_Security_Cheat_Sheet.html)
- [CIS Docker Benchmark](https://www.cisecurity.org/benchmark/docker)
- [DigitalOcean Security Best Practices](https://www.digitalocean.com/community/tutorials/recommended-security-measures-to-protect-your-servers)
- [Docker Security Documentation](https://docs.docker.com/engine/security/)

---

## Emergency Response

### In Case of Compromise

1. **Isolate**: Immediately isolate the affected system
2. **Assess**: Determine the extent of the breach
3. **Contain**: Stop the attack from spreading
4. **Eradicate**: Remove the threat
5. **Recover**: Restore from clean backups
6. **Review**: Conduct post-incident analysis

### Important Commands

```bash
# Check for unauthorized access
last -a
who
w

# Check running processes
ps aux | grep -v "$(ps aux | grep -E '^(root|cryptobot)')"

# Check network connections
netstat -tulpn
ss -tulpn

# Check Docker containers
docker ps -a
docker logs <container_name>

# Emergency shutdown
docker-compose down
sudo ufw --force reset
sudo ufw default deny incoming
sudo ufw enable
```

Remember: Security is an ongoing process, not a one-time setup. Regular updates, monitoring, and audits are essential for maintaining a secure environment.