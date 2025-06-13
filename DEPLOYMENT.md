# AI Trading Bot - Production Deployment Guide

This guide provides comprehensive instructions for deploying the AI Trading Bot in production environments using Docker, Docker Compose, and Kubernetes.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Docker Deployment](#docker-deployment)
3. [Docker Compose Deployment](#docker-compose-deployment)
4. [Kubernetes Deployment](#kubernetes-deployment)
5. [CI/CD Pipeline](#cicd-pipeline)
6. [Monitoring and Observability](#monitoring-and-observability)
7. [Security Considerations](#security-considerations)
8. [Troubleshooting](#troubleshooting)
9. [Scaling and Performance](#scaling-and-performance)
10. [Backup and Recovery](#backup-and-recovery)

## Prerequisites

### System Requirements

- **CPU**: Minimum 2 cores, Recommended 4+ cores
- **Memory**: Minimum 4GB RAM, Recommended 8GB+ RAM
- **Storage**: Minimum 20GB available space, SSD recommended
- **Network**: Stable internet connection for API access

### Software Requirements

- Docker 24.0+ or containerd
- Docker Compose 2.20+ (for Docker Compose deployment)
- Kubernetes 1.25+ (for Kubernetes deployment)
- kubectl configured with cluster access
- Git for source code management

### External Services

- **Coinbase Advanced Trade API** credentials
- **OpenAI API** key
- **PostgreSQL** database (can be deployed with the stack)
- **Redis** instance (can be deployed with the stack)

## Docker Deployment

### Build the Docker Image

```bash
# Build with default settings
docker build -t ai-trading-bot:latest .

# Build with custom build arguments
docker build \
  --build-arg BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ') \
  --build-arg VCS_REF=$(git rev-parse HEAD) \
  --build-arg VERSION=v1.0.0 \
  -t ai-trading-bot:v1.0.0 .
```

### Run Single Container

```bash
# Create environment file
cat > .env << EOF
COINBASE_API_KEY=your_api_key
COINBASE_API_SECRET=your_api_secret
COINBASE_PASSPHRASE=your_passphrase
OPENAI_API_KEY=your_openai_key
DRY_RUN=true
LOG_LEVEL=INFO
EOF

# Run container
docker run -d \
  --name ai-trading-bot \
  --env-file .env \
  -p 8080:8080 \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/data:/app/data \
  --restart unless-stopped \
  ai-trading-bot:latest
```

## Docker Compose Deployment

### Development Environment

```bash
# Start development stack
docker-compose up -d

# View logs
docker-compose logs -f ai-trading-bot-dev

# Stop stack
docker-compose down
```

### Production Environment

```bash
# Create production environment file
cp .env.example .env.prod
# Edit .env.prod with production values

# Create secrets directory
mkdir -p secrets
echo "your_db_password" > secrets/db_password.txt
echo "your_grafana_password" > secrets/grafana_password.txt
echo "your_grafana_secret_key" > secrets/grafana_secret_key.txt
chmod 600 secrets/*

# Create volume directories
mkdir -p volumes/{logs,data,backups,redis,postgres,prometheus,grafana}

# Deploy production stack
docker-compose -f docker-compose.prod.yml --env-file .env.prod up -d

# Verify deployment
docker-compose -f docker-compose.prod.yml ps
docker-compose -f docker-compose.prod.yml logs ai-trading-bot
```

### Health Checks

```bash
# Check application health
curl http://localhost:8080/health

# Check detailed health status
curl http://localhost:8080/health/detailed

# Check metrics
curl http://localhost:8081/metrics
```

## Kubernetes Deployment

### Prerequisites

```bash
# Verify cluster access
kubectl cluster-info
kubectl get nodes

# Install Helm (optional, for easier management)
curl https://get.helm.sh/helm-v3.13.0-linux-amd64.tar.gz | tar xz
sudo mv linux-amd64/helm /usr/local/bin/
```

### Deploy to Kubernetes

1. **Create Namespace and RBAC**

```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/rbac.yaml
```

2. **Create Secrets**

```bash
# Create trading bot secrets
kubectl create secret generic ai-trading-bot-secrets \
  --namespace=ai-trading-bot \
  --from-literal=COINBASE_API_KEY="your_api_key" \
  --from-literal=COINBASE_API_SECRET="your_api_secret" \
  --from-literal=COINBASE_PASSPHRASE="your_passphrase" \
  --from-literal=OPENAI_API_KEY="your_openai_key" \
  --from-literal=JWT_SECRET="your_jwt_secret" \
  --from-literal=WEBHOOK_SECRET="your_webhook_secret"

# Create database secrets
kubectl create secret generic postgres-secrets \
  --namespace=ai-trading-bot \
  --from-literal=POSTGRES_DB="trading_bot" \
  --from-literal=POSTGRES_USER="bot_user" \
  --from-literal=POSTGRES_PASSWORD="your_db_password"

# Create Redis secrets
kubectl create secret generic redis-secrets \
  --namespace=ai-trading-bot \
  --from-literal=REDIS_PASSWORD="your_redis_password"

# Create monitoring secrets
kubectl create secret generic monitoring-secrets \
  --namespace=ai-trading-bot \
  --from-literal=GF_SECURITY_ADMIN_PASSWORD="your_grafana_password" \
  --from-literal=GF_SECURITY_SECRET_KEY="your_grafana_secret"
```

3. **Deploy ConfigMaps and Storage**

```bash
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/pvc.yaml
```

4. **Deploy Applications**

```bash
# Update image tag in deployment
sed -i 's|image: ai-trading-bot:latest|image: your-registry/ai-trading-bot:v1.0.0|g' k8s/deployment.yaml

# Apply deployments
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

5. **Configure Autoscaling (Optional)**

```bash
kubectl apply -f k8s/hpa.yaml
```

6. **Verify Deployment**

```bash
# Check pod status
kubectl get pods -n ai-trading-bot

# Check services
kubectl get svc -n ai-trading-bot

# View logs
kubectl logs -f deployment/ai-trading-bot -n ai-trading-bot

# Port forward for testing
kubectl port-forward -n ai-trading-bot service/ai-trading-bot-service 8080:8080
curl http://localhost:8080/health
```

### Ingress Configuration (Optional)

```bash
# Install ingress controller (if not already installed)
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/main/deploy/static/provider/cloud/deploy.yaml

# Create ingress for the trading bot
cat << EOF | kubectl apply -f -
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ai-trading-bot-ingress
  namespace: ai-trading-bot
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
spec:
  rules:
  - host: trading-bot.your-domain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: ai-trading-bot-service
            port:
              number: 8080
EOF
```

## CI/CD Pipeline

### GitHub Actions Setup

1. **Configure Repository Secrets**

Go to your GitHub repository → Settings → Secrets and variables → Actions, and add:

```
COINBASE_API_KEY
COINBASE_API_SECRET
COINBASE_PASSPHRASE
OPENAI_API_KEY
DB_PASSWORD
REDIS_PASSWORD
GRAFANA_PASSWORD
GRAFANA_SECRET_KEY
JWT_SECRET
WEBHOOK_SECRET
KUBE_CONFIG_DATA  # Base64 encoded kubeconfig
SLACK_WEBHOOK_URL  # For notifications
```

2. **Trigger Deployment**

```bash
# Push to main branch triggers staging deployment
git push origin main

# Create tag triggers production deployment
git tag v1.0.0
git push origin v1.0.0

# Manual deployment via workflow dispatch
# Go to Actions tab → Deploy workflow → Run workflow
```

3. **Monitor Deployment**

- View deployment progress in GitHub Actions
- Check Slack notifications (if configured)
- Verify application health endpoints

## Monitoring and Observability

### Access Monitoring Dashboards

1. **Grafana Dashboard**

```bash
# Docker Compose
curl http://localhost:3000

# Kubernetes
kubectl port-forward -n ai-trading-bot service/grafana-service 3000:3000
```

Default credentials: admin / (password from secrets)

2. **Prometheus Metrics**

```bash
# Docker Compose
curl http://localhost:9090

# Kubernetes
kubectl port-forward -n ai-trading-bot service/prometheus-service 9090:9090
```

3. **Application Metrics**

```bash
# Health endpoint
curl http://your-domain:8080/health

# Detailed health
curl http://your-domain:8080/health/detailed

# Prometheus metrics
curl http://your-domain:8081/metrics
```

### Key Metrics to Monitor

- **Application Health**: Bot uptime, health status
- **Trading Performance**: P&L, trade count, win rate
- **System Resources**: CPU, memory, disk usage
- **API Performance**: Response times, error rates
- **Database Performance**: Connection count, query times

### Alerting

Alerts are configured in Prometheus rules:

- Bot down or unhealthy
- High resource usage
- Trading errors
- API connectivity issues
- SLA breaches

## Security Considerations

### Container Security

- Non-root user execution
- Read-only root filesystem
- Minimal base image (distroless/alpine)
- No privileged containers
- Security contexts enforced

### Network Security

- Network policies restrict traffic
- Internal service communication only
- Rate limiting on public endpoints
- TLS/SSL encryption (when configured)

### Secrets Management

- Kubernetes secrets for sensitive data
- No secrets in container images
- Regular secret rotation
- Least privilege access

### Access Control

- RBAC policies for Kubernetes
- Service accounts with minimal permissions
- Authentication required for monitoring tools
- Audit logging enabled

## Troubleshooting

### Common Issues

1. **Pod/Container Won't Start**

```bash
# Check pod status
kubectl describe pod <pod-name> -n ai-trading-bot

# Check logs
kubectl logs <pod-name> -n ai-trading-bot

# Check events
kubectl get events -n ai-trading-bot --sort-by='.lastTimestamp'
```

2. **Configuration Issues**

```bash
# Validate configuration
kubectl exec -it <pod-name> -n ai-trading-bot -- python -c "
from bot.config import Settings
from bot.config_utils import StartupValidator
settings = Settings()
validator = StartupValidator(settings)
result = validator.run_comprehensive_validation()
print(result)
"
```

3. **Database Connection Issues**

```bash
# Test database connectivity
kubectl exec -it postgres-0 -n ai-trading-bot -- psql -U bot_user -d trading_bot -c "SELECT 1;"

# Check database logs
kubectl logs postgres-0 -n ai-trading-bot
```

4. **API Connectivity Issues**

```bash
# Test external API access
kubectl exec -it <pod-name> -n ai-trading-bot -- curl -I https://api.exchange.coinbase.com
kubectl exec -it <pod-name> -n ai-trading-bot -- curl -I https://api.openai.com
```

### Debug Mode

Enable debug mode for detailed logging:

```bash
# Set LOG_LEVEL=DEBUG in environment
kubectl set env deployment/ai-trading-bot LOG_LEVEL=DEBUG -n ai-trading-bot

# Or update ConfigMap
kubectl patch configmap ai-trading-bot-config -n ai-trading-bot -p '{"data":{"LOG_LEVEL":"DEBUG"}}'
kubectl rollout restart deployment/ai-trading-bot -n ai-trading-bot
```

### Log Analysis

```bash
# Follow logs in real-time
kubectl logs -f deployment/ai-trading-bot -n ai-trading-bot

# Get logs from specific time range
kubectl logs deployment/ai-trading-bot -n ai-trading-bot --since=1h

# Search logs for errors
kubectl logs deployment/ai-trading-bot -n ai-trading-bot | grep -i error
```

## Scaling and Performance

### Horizontal Scaling

⚠️ **Warning**: The trading bot should typically run as a single instance to avoid conflicts. Use vertical scaling instead.

### Vertical Scaling

1. **Increase Resource Limits**

```bash
# Update deployment resources
kubectl patch deployment ai-trading-bot -n ai-trading-bot -p '{
  "spec": {
    "template": {
      "spec": {
        "containers": [{
          "name": "ai-trading-bot",
          "resources": {
            "requests": {"memory": "2Gi", "cpu": "1000m"},
            "limits": {"memory": "4Gi", "cpu": "2000m"}
          }
        }]
      }
    }
  }
}'
```

2. **Enable Vertical Pod Autoscaler**

```bash
kubectl apply -f k8s/hpa.yaml
```

### Performance Optimization

1. **Database Optimization**

- Regular vacuum and analyze
- Proper indexing
- Connection pooling
- Query optimization

2. **Redis Optimization**

- Memory management
- Persistence settings
- Connection limits

3. **Application Optimization**

- Async operations
- Caching strategies
- Resource monitoring
- Algorithm optimization

## Backup and Recovery

### Database Backup

```bash
# Create backup
kubectl exec postgres-0 -n ai-trading-bot -- pg_dump -U bot_user trading_bot > backup_$(date +%Y%m%d_%H%M%S).sql

# Automated backup script
cat << 'EOF' > backup-script.sh
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
kubectl exec postgres-0 -n ai-trading-bot -- pg_dump -U bot_user trading_bot | gzip > "backup_${DATE}.sql.gz"
# Upload to cloud storage (AWS S3, Google Cloud Storage, etc.)
EOF
chmod +x backup-script.sh

# Schedule with cron
echo "0 2 * * * /path/to/backup-script.sh" | crontab -
```

### Configuration Backup

```bash
# Backup Kubernetes manifests
kubectl get all,cm,secrets,pvc -n ai-trading-bot -o yaml > k8s-backup.yaml

# Backup application configuration
kubectl get cm ai-trading-bot-config -n ai-trading-bot -o yaml > config-backup.yaml
```

### Disaster Recovery

1. **Recovery Plan**

- Document recovery procedures
- Test recovery regularly
- Maintain offsite backups
- Monitor backup integrity

2. **Recovery Steps**

```bash
# Restore database
kubectl exec -i postgres-0 -n ai-trading-bot -- psql -U bot_user trading_bot < backup.sql

# Restore configuration
kubectl apply -f config-backup.yaml

# Restart services
kubectl rollout restart deployment/ai-trading-bot -n ai-trading-bot
```

## Support and Maintenance

### Health Monitoring

- Set up automated health checks
- Configure alerting for critical issues
- Monitor resource usage trends
- Track performance metrics

### Updates and Maintenance

```bash
# Update application
kubectl set image deployment/ai-trading-bot ai-trading-bot=ai-trading-bot:v1.1.0 -n ai-trading-bot

# Rolling restart
kubectl rollout restart deployment/ai-trading-bot -n ai-trading-bot

# Rollback if needed
kubectl rollout undo deployment/ai-trading-bot -n ai-trading-bot
```

### Documentation

- Keep deployment documentation current
- Document configuration changes
- Maintain runbooks for common issues
- Record lessons learned

---

For additional support or questions, please refer to the project documentation or create an issue in the GitHub repository.